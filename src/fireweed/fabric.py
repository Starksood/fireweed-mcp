"""Fireweed public API — experimental facade over ingestion + retrieval + reader.
Stability: EXPERIMENTAL. See docs/STABILITY.md.
v1 contract: Turn.text is a pre-extracted claim. Sync only: read() blocks on LLM.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from .pipeline import IngestContext, IngestResult
from .retrieval import RetrievalResult, query_graph
from .reader import ReadResponse, LLMCallable
from .percept_buffer import PerceptBuffer, AdmitResult
from .perceiver import perceive_into
from .consolidator import Consolidator, TwoClockScheduler, ConsolidationReport
from .graph import (Node, Entity, Relation, EntityRef, Predicate, Motivation, MemoryContext,
                    Temporal, Provenance, Reinforcement, NodeStatus, RelationRef,
                    EntityProvenance, RelationEvidence)

_FIREWEED_VERSION = "16.0.0-alpha"
_SNAPSHOT_VERSION = 2

@dataclass
class Turn:
    turn_id: str
    text: str
    evidence_span: str
    confidence: float = 0.9

@dataclass
class Session:
    session_id: str
    turns: list[Turn]
    timestamp: str = "1970-01-01T00:00:00Z"

class Fireweed:
    def __init__(
        self,
        llm: LLMCallable,
        *,
        perceiver_llm: LLMCallable | None = None,
        reflect_llm: LLMCallable | None = None,
        buffer: PerceptBuffer | None = None,
    ) -> None:
        self._ctx = IngestContext()
        self._llm = llm
        self._reflect_llm = reflect_llm  # W4: optional LLM for REFLECT (evidence->pattern)
        self._pending_sources: dict[str, str] = {}   # source_id -> text, for receipt binding
        # Two-clock streaming perception (optional). perceiver_llm is the tiny
        # "eyes" model for Clock 1; the scheduler is bound lazily so non-streaming
        # users pay nothing, and its Consolidator shares self._ctx so perceived
        # nodes and batch-ingested nodes live in one graph.
        self._perceiver_llm = perceiver_llm
        self._buffer = buffer
        self._scheduler: TwoClockScheduler | None = None
        self._ingested_session_ids: set[str] = set()

    def ingest(self, session: Session) -> list[IngestResult]:
        if session.session_id in self._ingested_session_ids:
            raise ValueError(f"Session {session.session_id!r} has already been ingested. "
                           "Each session_id must be unique per Fireweed instance.")
        self._ingested_session_ids.add(session.session_id)
        self._ctx.begin_session(session.session_id, session.timestamp)
        results: list[IngestResult] = []
        for turn in session.turns:
            results.append(self._ctx.ingest(turn.text, turn.evidence_span, turn.confidence,
                                           turn.turn_id, session.session_id))
        return results

    def query(self, question: str, ego_graph: bool = False) -> RetrievalResult:
        return query_graph(question, self._ctx.graph, ego_graph=ego_graph)

    def read(self, question: str, retrieval: RetrievalResult) -> ReadResponse:
        from .reader import read as _read
        return _read(question, retrieval, self._llm)

    def extract_and_ingest(self, text: str, source_id: str, timestamp: str = None) -> list[IngestResult]:
        """Extract claims from raw text and ingest them as a session.

        This is the end-to-end path: raw text → LLM extraction → validation → graph storage.
        Implements X2 (Claim Extractor) — normalizes claims before ingestion.

        Args:
            text: Free text containing claims (e.g., session transcript or paragraph)
            source_id: Session identifier (used for reinforcement tracking)
            timestamp: ISO timestamp (default: current UTC time)

        Returns:
            List of IngestResult objects (one per extracted claim)
        """
        if timestamp is None:
            timestamp = _utcnow_iso()
        from .extractor import extract_claims
        turns = extract_claims(text, source_id, timestamp, self._llm)
        return self.ingest(Session(session_id=source_id, turns=turns, timestamp=timestamp))

    def ingest_document(self, text: str, source_id: str,
                        timestamp: str = None) -> list[IngestResult]:
        """Sprint 3 — ingest a document and bind each derived claim to a verifiable byte range
        (doc_hash, byte_start, byte_end) in the hash-signed source. See receipts.bind_document."""
        from .receipts import bind_document
        results = self.extract_and_ingest(text, source_id, timestamp)
        bind_document(self._ctx.graph, text, source_id)
        return results

    def receipt(self, node):
        """Render a verifiable document receipt for a node, or None if it isn't document-bound."""
        from .receipts import receipt_for
        return receipt_for(node)

    # --- Two-clock streaming perception (Clock 1 -> buffer -> Clock 2) -----------
    def _ensure_scheduler(self) -> TwoClockScheduler:
        """Bind (once) a TwoClockScheduler whose Consolidator writes through this
        fabric's IngestContext. Sharing self._ctx is what makes query()/read() see
        perceived nodes. Rebuilt lazily after restore() swaps the context out."""
        if self._scheduler is None:
            self._scheduler = TwoClockScheduler(
                buffer=self._buffer,
                consolidator=Consolidator(self._ctx, reflect_llm=self._reflect_llm),
            )
        return self._scheduler

    def perceive(self, text: str, source_id: str,
                 timestamp: str | None = None, speaker: str | None = None) -> list[AdmitResult]:
        """Clock 1. Perceive `text` with the tiny model and stage every Percept in
        the buffer. NEVER touches the graph — consolidation happens on tick()/flush(),
        so a stalled or slow consolidator can never block perception.

        `speaker` is who is talking: when set, first-person references ("I moved to Seattle") are
        deterministically rewritten to the speaker ("Maya moved to Seattle") BEFORE perception, so
        self-facts anchor to the speaker's entity — the core personal-memory pattern.

        Requires a perceiver_llm at construction. Returns the AdmitResult list so
        callers can observe salience-based evictions under load.
        """
        if self._perceiver_llm is None:
            raise ValueError(
                "perceive() requires a perception model. Construct "
                "Fireweed(llm=..., perceiver_llm=<tiny model>) to use streaming perception."
            )
        if timestamp is None:
            timestamp = _utcnow_iso()
        if speaker:
            from .speaker import rewrite_first_person
            text = rewrite_first_person(text, speaker)
        # Hold the source bytes until consolidation so flush()/tick() can bind byte-range receipts,
        # exactly as ingest_document does. Without this the abstraction path had no verifiable
        # provenance at all (receipts 0/176 on the ops run) — which read as an architectural tradeoff
        # between abstraction and provenance, but was only an unbound source.
        self._pending_sources[source_id] = text
        return perceive_into(self._ensure_scheduler(), text, source_id, timestamp,
                             self._perceiver_llm)

    # --- Hot-swap (Stage 4): the model is interchangeable; the fabric is the identity -------
    def swap_reader(self, llm: LLMCallable) -> LLMCallable:
        """Swap the reader/batch-extractor model mid-session and return the old one. The graph
        is untouched — query()/read()/extract_and_ingest() use the new model immediately."""
        old, self._llm = self._llm, llm
        return old

    def swap_perceiver(self, perceiver_llm: LLMCallable) -> LLMCallable:
        """Swap the Clock-1 perception model ("eyes") mid-session and return the old one.
        perceive() reads it fresh, so the scheduler and graph carry over unchanged."""
        old, self._perceiver_llm = self._perceiver_llm, perceiver_llm
        return old

    swap_extractor = swap_perceiver  # canonical Stage-4 name for the perception/extraction model

    def _bind_pending_receipts(self) -> None:
        """Bind byte-range receipts for every source consolidated since the last cycle.

        Mirrors ingest_document's second step. Sources are cleared once bound: a claim whose evidence
        span is not a contiguous slice of its source is simply left unbound (bind_document never
        fabricates a coordinate), so nothing is retried forever.
        """
        if not self._pending_sources:
            return
        from .receipts import bind_document
        for source_id, text in self._pending_sources.items():
            bind_document(self._ctx.graph, text, source_id)
        self._pending_sources.clear()

    def tick(self, timestamp: str | None = None) -> ConsolidationReport | None:
        """Clock 2. Consolidate one batch into the graph iff the buffer is ready to
        drain (full, or idle past its timer); otherwise return None. Runs exactly
        one decay turn per consolidated batch."""
        if timestamp is None:
            timestamp = _utcnow_iso()
        report = self._ensure_scheduler().tick(timestamp=timestamp)
        if report is not None:
            self._bind_pending_receipts()
        return report

    def flush(self, timestamp: str | None = None) -> ConsolidationReport | None:
        """Consolidate everything currently buffered, ignoring the drain triggers
        (e.g. at shutdown or before snapshot). None if nothing was buffered."""
        if timestamp is None:
            timestamp = _utcnow_iso()
        report = self._ensure_scheduler().flush(timestamp=timestamp)
        if report is not None:
            self._bind_pending_receipts()
        return report

    def snapshot(self) -> bytes:
        data = {
            "fireweed_version": _FIREWEED_VERSION,
            "snapshot_version": _SNAPSHOT_VERSION,
            "nodes": [asdict(n) for n in self._ctx.graph.all_nodes()],
            "entities": [asdict(e) for e in self._ctx.graph.all_entities()],
            "relations": [asdict(r) for r in self._ctx.graph.all_relations()],
            "sessions_seen": {k: sorted(v) for k, v in self._ctx._sessions_seen.items()},
            "total_sessions": self._ctx._total_sessions,
            "seen_domains": sorted(self._ctx._seen_domains),
            "session_timestamp": self._ctx._session_timestamp,
            "session_anchor": self._ctx._session_anchor,
            "ingested_session_ids": sorted(self._ingested_session_ids),
        }
        return json.dumps(data, default=_json_default).encode("utf-8")

    def significance_snapshot(self) -> dict:
        """The grounded significance side-table (Stage 3, W1) as a JSON-able dict keyed by
        node_id. Symmetric with snapshot(); empty until percepts carry grounded rationale/
        cause. Lives outside the graph snapshot because significance is the M axis — an
        orthogonal side-table, not part of the Node schema."""
        from .significance import store_as_dict
        if self._scheduler is None:
            return {}
        return store_as_dict(self._scheduler.consolidator.significance)

    def restore(self, data: bytes) -> None:
        raw = json.loads(data.decode("utf-8"))
        sv = raw.get("snapshot_version")
        if sv == 1:
            raise ValueError(
                "Snapshot version 1 was generated before deterministic node IDs (X1). "
                "Restore not supported. Re-ingest sessions to produce a version 2 snapshot."
            )
        if sv != _SNAPSHOT_VERSION:
            raise ValueError(
                f"Snapshot version mismatch: expected {_SNAPSHOT_VERSION}, got {sv}. "
                "Restore not supported."
            )
        new_ctx = IngestContext()
        for nd in raw["nodes"]:
            new_ctx.graph.add_node(_node_from_dict(nd))
        for ed in raw["entities"]:
            new_ctx.graph.add_entity(_entity_from_dict(ed))
        for rd in raw["relations"]:
            new_ctx.graph.add_relation(_relation_from_dict(rd))
        new_ctx._sessions_seen = {k: set(v) for k, v in raw["sessions_seen"].items()}
        new_ctx._total_sessions = raw["total_sessions"]
        new_ctx._seen_domains = set(raw["seen_domains"])
        new_ctx._session_timestamp = raw["session_timestamp"]
        new_ctx._session_anchor = raw["session_anchor"]
        self._ctx = new_ctx
        self._ingested_session_ids = set(raw["ingested_session_ids"])
        # The scheduler's Consolidator referenced the old context; drop it so the
        # next perceive()/tick() rebinds against the restored graph. Any percepts
        # buffered but not yet consolidated are intentionally discarded — only
        # consolidated state is part of a snapshot.
        self._scheduler = None

def _utcnow_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _json_default(obj):
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def _node_from_dict(d: dict) -> Node:
    d = dict(d)
    d["domains"] = set(d["domains"])
    d["entities"] = [EntityRef(**e) for e in d["entities"]]
    d["predicate"] = Predicate(**d["predicate"])
    d["temporal"] = Temporal(**d["temporal"])
    d["provenance"] = Provenance(**d["provenance"])
    d["reinforcement"] = Reinforcement(**d["reinforcement"])
    d["status"] = NodeStatus(**d["status"])
    d["motivation"] = Motivation(**d["motivation"]) if d.get("motivation") else None
    d["context"] = MemoryContext(**d["context"]) if d.get("context") else None
    d["relations"] = [RelationRef(**r) for r in d.get("relations", [])]
    return Node(**d)

def _entity_from_dict(d: dict) -> Entity:
    d = dict(d)
    d["provenance"] = [EntityProvenance(**p) for p in d["provenance"]]
    return Entity(**d)

def _relation_from_dict(d: dict) -> Relation:
    d = dict(d)
    d["evidence"] = RelationEvidence(**d["evidence"])
    return Relation(**d)
