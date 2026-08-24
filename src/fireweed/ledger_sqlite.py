"""Sprint 5 (§4B weeks 1–2) — durable, hash-chained event ledger.

A SQLite-backed `Ledger` that is a drop-in for the in-memory `EventLog` (same `record(...)` signature,
so `GraphState.attach_ledger` takes it unchanged), but PERSISTS every event synchronously — commit
before return, so an ack never precedes durability. We never build our own WAL: SQLite's commit (like
Postgres's) is the durability guarantee. The Postgres impl is the same SQL over a DSN; this runs
everywhere (CI, dev, offline) with zero new dependencies.

Per-tenant, gap-free, hash-chained from entry #1. Checkpoints materialize the derived graph state into
the ledger so recovery is `latest checkpoint + tail replay`, never replay-from-zero.
"""
from __future__ import annotations

import json
import sqlite3
import threading

from .ledger import (
    LedgerEvent, GENESIS_PREV_HASH, WRITE_KINDS, apply_event,
    graph_state_dict, load_graph_state,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    tenant_id TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    event_id  TEXT NOT NULL,
    kind      TEXT NOT NULL,
    ts        TEXT NOT NULL,
    payload   TEXT NOT NULL,
    trace     TEXT,
    prev_hash TEXT NOT NULL,
    hash      TEXT NOT NULL,
    PRIMARY KEY (tenant_id, seq)
);
"""


class SQLiteLedger:
    """Durable ledger. `path=":memory:"` for tests; a file path for real persistence."""

    def __init__(self, path: str = ":memory:") -> None:
        # check_same_thread=False: the TenantLockManager serializes writers; an internal lock guards
        # the shared connection so multi-tenant writes from different threads can't interleave a stmt.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._wlock = threading.RLock()   # re-entrant: record() holds it while calling _tail()
        self._conn.execute("PRAGMA journal_mode=WAL")      # durable + concurrent-read friendly
        self._conn.execute("PRAGMA synchronous=FULL")      # fsync on commit — durability before ack
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── the EventLog-compatible write surface ────────────────────────────────────
    def _tail(self, tenant_id: str) -> tuple[str, int]:
        row = self._conn.execute(
            "SELECT hash, seq FROM ledger WHERE tenant_id=? ORDER BY seq DESC LIMIT 1", (tenant_id,)
        ).fetchone()
        return (row[0], row[1] + 1) if row else (GENESIS_PREV_HASH, 0)

    def record(self, tenant_id: str, kind: str, ts: str, event_id: str,
               payload: dict | None = None, trace: dict | None = None) -> LedgerEvent:
        with self._wlock:          # tail-read + insert atomic → gap-free seq + correct prev_hash chain
            prev_hash, seq = self._tail(tenant_id)
            ev = LedgerEvent(seq=seq, tenant_id=tenant_id, event_id=event_id, kind=kind, ts=ts,
                             payload=payload or {}, trace=trace, prev_hash=prev_hash).sealed()
            self._conn.execute(
                "INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?,?)",
                (tenant_id, seq, event_id, kind, ts,
                 json.dumps(ev.payload, sort_keys=True), json.dumps(ev.trace, sort_keys=True),
                 prev_hash, ev.hash),
            )
            self._conn.commit()    # durable BEFORE the caller proceeds — commit-before-ack
        return ev

    def events(self, tenant_id: str = "local") -> list[LedgerEvent]:
        with self._wlock:
            rows = self._conn.execute(
                "SELECT seq,tenant_id,event_id,kind,ts,payload,trace,prev_hash,hash "
                "FROM ledger WHERE tenant_id=? ORDER BY seq", (tenant_id,)
            ).fetchall()
        return [LedgerEvent(seq=r[0], tenant_id=r[1], event_id=r[2], kind=r[3], ts=r[4],
                            payload=json.loads(r[5]), trace=json.loads(r[6]) if r[6] else None,
                            prev_hash=r[7], hash=r[8]) for r in rows]

    def __len__(self) -> int:
        with self._wlock:
            return self._conn.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]

    def tenant_len(self, tenant_id: str) -> int:
        with self._wlock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM ledger WHERE tenant_id=?", (tenant_id,)).fetchone()[0]

    def graph_version(self, tenant_id: str) -> int:
        """Monotonic per-tenant version = next seq (= committed event count). A cache holding an older
        version must fold the missing tail before applying a new write (optimistic invalidation)."""
        with self._wlock:
            return self._tail(tenant_id)[1]

    def tenants(self) -> list[str]:
        with self._wlock:
            return [r[0] for r in self._conn.execute(
                "SELECT DISTINCT tenant_id FROM ledger ORDER BY tenant_id").fetchall()]

    # ── checkpoints + recovery ───────────────────────────────────────────────────
    def checkpoint(self, tenant_id: str, graph) -> LedgerEvent:
        """Append a CHECKPOINT event holding the materialized derived state — so recovery is bounded
        (latest checkpoint + tail), never O(all history)."""
        seq = self._tail(tenant_id)[1]
        return self.record(tenant_id, "CHECKPOINT", ts=f"ckpt:{seq}",
                           event_id=f"{tenant_id}:ckpt:{seq}",
                           payload={"state": graph_state_dict(graph)})

    def replay_graph(self, tenant_id: str, graph, keyring=None):
        """Reconstruct a tenant's derived graph = latest CHECKPOINT state + tail write-events. `graph`
        is a fresh GraphState to fold into (crash recovery). With a `keyring`, encrypted content is
        decrypted; shredded subjects reconstruct as tombstones. Returns the graph."""
        evs = self.events(tenant_id)
        ck_idx = None
        for i in range(len(evs) - 1, -1, -1):
            if evs[i].kind == "CHECKPOINT":
                ck_idx = i
                break
        if ck_idx is not None:
            load_graph_state(graph, evs[ck_idx].payload["state"])
            tail = evs[ck_idx + 1:]
        else:
            tail = evs
        for ev in tail:
            if ev.kind in WRITE_KINDS or ev.kind in ("ERASE", "PRUNE", "COMPRESS"):
                apply_event(ev, graph, keyring=keyring)
        return graph

    def close(self) -> None:
        self._conn.close()
