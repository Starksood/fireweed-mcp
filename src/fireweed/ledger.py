"""Sprint 5 — the event ledger: schema, canonical serialization, and hash-chain.

An append-only, hash-chained mutation ledger is the source of truth; the graph is a derived cache;
state = fold(ledger). This module is the SCHEMA + CHAIN layer (the week-1 foundation that cannot be
retrofitted): the event envelope, a deterministic canonical serialization the hash depends on, and
chain construction/verification. The `apply(event, graph)` chokepoint and Postgres persistence build
on this; see docs/sprint/SPRINT5_EVENT_LEDGER.md.

Design invariants pinned here:
  * canonical_bytes is order-insensitive and whitespace-stable — byte-identical replay + the hash both
    depend on it;
  * every event carries prev_hash from entry #1 — tamper of any historical row is detectable;
  * events carry the INGEST clock (payload/ts), never wall time — replay must reproduce hashes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

# ── Event kinds ─────────────────────────────────────────────────────────────────
# Resolver mutations map 1:1; the rest name the non-resolver write paths (§4B R1 inventory).
RESOLVER_KINDS = ("CREATE", "DEDUP", "REINFORCE", "MODIFY", "DISPUTE", "NOOP")
ENGINE_KINDS = ("SYNTHESIZE", "COMPRESS", "PRUNE", "SEED", "ALIAS_MERGE", "ENTITY_UPSERT")
LIFECYCLE_KINDS = ("ERASE", "CHECKPOINT")
# Write-grain kinds: the concrete graph mutation captured at the raw-writer boundary. Replay uses
# these (byte-identical because the object — with its already-realized uuids/timestamps — is stored,
# so a resolver change or wall-clock never rewrites history). The semantic RESOLVER/ENGINE kinds ride
# as audit metadata on the same events (payload.semantic / trace).
WRITE_KINDS = ("ADD_NODE", "UPDATE_NODE", "ADD_ENTITY", "UPDATE_ENTITY", "ADD_RELATION")
EVENT_KINDS = frozenset(RESOLVER_KINDS + ENGINE_KINDS + LIFECYCLE_KINDS + WRITE_KINDS)

GENESIS_PREV_HASH = ""   # prev_hash of seq 0


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic UTF-8 serialization: sorted keys, no whitespace variance, stable float repr.

    The hash and byte-identical replay both depend on this being order-insensitive and stable.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass
class LedgerEvent:
    """One append-only ledger entry. `hash` is derived (over every field except `hash` itself)."""
    seq: int                       # monotonic per tenant, gap-free
    tenant_id: str
    event_id: str                  # uuid (caller-supplied so it's deterministic in tests)
    kind: str                      # one of EVENT_KINDS
    ts: str                        # ISO-8601 UTC — the INGEST clock, never wall time
    payload: dict = field(default_factory=dict)
    trace: dict | None = None      # resolver DecisionTrace when present
    prev_hash: str = GENESIS_PREV_HASH
    hash: str = ""

    def _hashable(self) -> dict:
        d = asdict(self)
        d.pop("hash", None)        # the hash covers everything BUT itself
        return d

    def compute_hash(self) -> str:
        return "sha256:" + hashlib.sha256(canonical_bytes(self._hashable())).hexdigest()

    def sealed(self) -> "LedgerEvent":
        """Return a copy with `hash` filled in from the current fields."""
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind: {self.kind!r}")
        object.__setattr__(self, "hash", self.compute_hash())
        return self


class EventLog:
    """In-memory append-only log with hash-chaining. The reference implementation of the ledger
    interface; the Postgres-backed impl (synchronous commit-before-ack) will satisfy the same
    contract. Not thread-safe on its own — the per-tenant advisory lock (Part D) serializes writers.
    """

    def __init__(self) -> None:
        self._events: list[LedgerEvent] = []

    @property
    def tail_hash(self) -> str:
        return self._events[-1].hash if self._events else GENESIS_PREV_HASH

    def record(self, tenant_id: str, kind: str, ts: str, event_id: str,
               payload: dict | None = None, trace: dict | None = None) -> LedgerEvent:
        """Append a new event: assign seq, chain prev_hash to the tail, seal the hash."""
        ev = LedgerEvent(
            seq=len(self._events), tenant_id=tenant_id, event_id=event_id,
            kind=kind, ts=ts, payload=payload or {}, trace=trace,
            prev_hash=self.tail_hash,
        ).sealed()
        self._events.append(ev)
        return ev

    def events(self) -> list[LedgerEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)


def verify_chain(events: list[LedgerEvent]) -> bool:
    """True iff every event's hash is self-consistent AND its prev_hash links the predecessor,
    with gap-free monotonic seq. Any tampered/removed/reordered row makes this False."""
    prev = GENESIS_PREV_HASH
    for i, ev in enumerate(events):
        if ev.seq != i:
            return False
        if ev.prev_hash != prev:
            return False
        if ev.hash != ev.compute_hash():
            return False
        prev = ev.hash
    return True


def fold(events: list[LedgerEvent], apply_fn, state=None):
    """state = fold(ledger): replay events through a pure apply_fn(event, state) -> state.

    apply_fn is supplied by the engine (the single `apply(event, graph)` chokepoint). Recovery folds
    from the latest CHECKPOINT's materialized state + the tail; a full replay folds from `state=None`.
    """
    for ev in events:
        state = apply_fn(ev, state)
    return state


# ── The engine chokepoint: apply a write-grain event to a GraphState ────────────

_OP_KINDS = frozenset(WRITE_KINDS)


def apply_event(ev: "LedgerEvent", graph, keyring=None):
    """Replay one write-grain event into `graph` — the single `apply(event, graph)` chokepoint.

    Reconstructs the concrete object the raw writer stored and re-applies it under the graph's
    `_replaying` flag (so replay does not re-emit). Because the stored object already carries its
    realized uuids/timestamps, replay is byte-identical regardless of resolver/clock changes. With a
    `keyring`, node content is decrypted (crypto-shredding); if the subject's key was shredded, the
    content reconstructs as a tombstone — history is unrecoverable after erasure.
    """
    from . import fabric   # lazy: avoids a graph<->fabric import cycle at module load
    _REMOVAL = ("ERASE", "PRUNE", "COMPRESS")
    if ev.kind not in _OP_KINDS and ev.kind not in _REMOVAL:
        raise ValueError(f"apply_event only handles write-grain + ERASE/PRUNE/COMPRESS, got {ev.kind!r}")
    op = ev.kind

    def _node(obj):
        if keyring is not None:
            from .crypto import decrypt_node_content
            obj = decrypt_node_content(obj, keyring)
        return fabric._node_from_dict(obj)

    graph._replaying = True
    try:
        if op == "ADD_NODE":
            graph.add_node(_node(ev.payload["obj"]))
        elif op == "UPDATE_NODE":
            graph.update_node(_node(ev.payload["obj"]))
        elif op == "ADD_ENTITY":
            graph.add_entity(fabric._entity_from_dict(ev.payload["obj"]))
        elif op == "UPDATE_ENTITY":
            graph.update_entity(fabric._entity_from_dict(ev.payload["obj"]))
        elif op == "ADD_RELATION":
            graph.add_relation(fabric._relation_from_dict(ev.payload["obj"]))
        elif op == "ERASE":
            # Right-to-erasure: remove the subject's closure from derived state. Replayable, so a
            # fold reconstructs then erases → the post-erasure graph never carries the subject.
            m = ev.payload["closure"]
            for rid in m.get("relation_ids", []):
                graph.remove_relation(rid)
            for nid in m.get("node_ids", []):
                graph.remove_node(nid)
            for eid in m.get("entity_ids", []):
                graph.remove_entity(eid)
        elif op == "PRUNE":
            # Retention: drop nodes (+ their incident relations) the policy deemed eligible. Entities
            # are NOT removed (a pruned superseded claim doesn't un-know the person). Replayable.
            for rid in ev.payload.get("relation_ids", []):
                graph.remove_relation(rid)
            for nid in ev.payload.get("node_ids", []):
                graph.remove_node(nid)
        elif op == "COMPRESS":
            # Aggregate near-duplicates: the survivor absorbs the cluster's reinforcement; the folded
            # duplicates (+ their relations) are removed. No claim invented. Replayable.
            graph.update_node(_node(ev.payload["survivor"]))
            for rid in ev.payload.get("relation_ids", []):
                graph.remove_relation(rid)
            for nid in ev.payload.get("removed_node_ids", []):
                graph.remove_node(nid)
    finally:
        graph._replaying = False
    return graph


def _jsonable(o):
    """asdict output -> pure-JSON (sets become sorted lists), so payloads canonical-serialize and the
    reconstructors (which accept lists for set fields) round-trip."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, set):
        return sorted(_jsonable(v) for v in o)
    return o


def graph_state_dict(graph) -> dict:
    """The derived graph state (nodes+entities+relations) as an id-sorted, pure-JSON dict — the
    materialized form stored in a CHECKPOINT and reloaded on recovery."""
    from dataclasses import asdict
    return {
        "nodes": sorted((_jsonable(asdict(n)) for n in graph.all_nodes()), key=lambda d: d["node_id"]),
        "entities": sorted((_jsonable(asdict(e)) for e in graph.all_entities()), key=lambda d: d["entity_id"]),
        "relations": sorted((_jsonable(asdict(r)) for r in graph.all_relations()), key=lambda d: d["relation_id"]),
    }


def graph_fingerprint(graph) -> bytes:
    """Canonical bytes of the derived graph state. Two graphs with the same fingerprint are
    byte-identical for the fold-equivalence / crash-replay exit tests."""
    return canonical_bytes(graph_state_dict(graph))


def load_graph_state(graph, state: dict) -> None:
    """Materialize a CHECKPOINT's graph_state_dict into `graph` (under _replaying, so no re-emit).
    Recovery = load_graph_state(latest checkpoint) then apply_event over the tail."""
    from . import fabric
    graph._replaying = True
    try:
        for nd in state.get("nodes", []):
            graph.add_node(fabric._node_from_dict(nd))
        for ed in state.get("entities", []):
            graph.add_entity(fabric._entity_from_dict(ed))
        for rd in state.get("relations", []):
            graph.add_relation(fabric._relation_from_dict(rd))
    finally:
        graph._replaying = False
