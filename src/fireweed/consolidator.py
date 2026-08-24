"""Consolidator + two-clock scheduler — Clock 2, and the loop that ties it to Clock 1.

This is the integration point of the two-clock architecture:

    Clock 1 (perception)  ->  PerceptBuffer  ->  Clock 2 (consolidation)
      scheduler.perceive()      (bridge)         scheduler.tick()

`Consolidator` is Clock 2: it drains a batch of percepts, writes them to the graph
through the EXISTING deterministic pipeline (firewall -> entity linker -> resolver),
maintains the decay side-table, and runs one decay turn per consolidation cycle.
One consolidation cycle == one `turn` == one drain — the unit decay counts in.

`TwoClockScheduler` owns a PerceptBuffer and a Consolidator and exposes the two
clocks as methods. Perception calls `perceive()` (never blocks). A driver loop
calls `tick()` at consolidation cadence; tick consolidates a batch iff the buffer
says it should drain (batch full, or idle past its timer).

Coherence-at-speed lives here: because nothing reaches the graph until a batch is
drained and run through the resolver, contradictions within a batch are reconciled
by the existing deterministic resolver before any write. The graph only ever sees
reconciled batches; intake never exposes inconsistency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pipeline import IngestContext
from .percept_buffer import PerceptBuffer, Percept
from .decay import DecayState, DecayReport, apply_decay
from .resolver import CreateMutation, DedupMutation, ReinforceMutation, ModifyMutation
from . import significance as sig
from . import self_model as sm
from .field_edges import materialize_field
from .consolidation_ops import run_consolidation_ops


def _subject(node) -> str:
    """The primary entity a node is about (first actor, else first entity, else "")."""
    for ref in node.entities:
        if ref.role in ("actor", "co_actor"):
            return ref.entity_id
    return node.entities[0].entity_id if node.entities else ""


def _mutation_node_id(mutation) -> str | None:
    """The node a mutation resolved to (for keying the significance side-table)."""
    if isinstance(mutation, CreateMutation):
        return mutation.node.node_id
    if isinstance(mutation, ModifyMutation):
        return mutation.new_node.node_id
    if isinstance(mutation, ReinforceMutation):
        return mutation.target_node_id
    if isinstance(mutation, DedupMutation):
        return mutation.existing_node_id
    return None


@dataclass
class ConsolidationReport:
    turn: int
    percepts_in: int
    accepted: int                 # claims that produced a write (ACCEPT/RESCUE)
    rejected: int                 # claims blocked by the firewall (REJECT/QUARANTINE)
    nodes_registered: int         # new decay-tracked nodes this cycle
    nodes_reinforced: int         # existing nodes whose r increased this cycle
    significance_recorded: int = 0  # nodes that gained grounded significance (W1)
    field_edges_created: dict | None = None  # typed field edges by type this cycle (W2)
    frozen_this_turn: int = 0        # nodes frozen by the consolidation scheduler this cycle (W4)
    reflections_created: int = 0     # evidence->pattern reflection nodes created this cycle (W4)
    compressions_created: int = 0    # COLD clusters folded into summaries this cycle (W4)
    predictions_made: int = 0        # forward-looking claims recorded as predictions this cycle (W5)
    predictions_resolved: int = 0    # open predictions resolved by observations this cycle (W5)
    decay: DecayReport | None = None

    def as_dict(self) -> dict:
        return {
            "turn": self.turn,
            "percepts_in": self.percepts_in,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "nodes_registered": self.nodes_registered,
            "nodes_reinforced": self.nodes_reinforced,
            "significance_recorded": self.significance_recorded,
            "field_edges_created": self.field_edges_created or {},
            "frozen_this_turn": self.frozen_this_turn,
            "reflections_created": self.reflections_created,
            "compressions_created": self.compressions_created,
            "predictions_made": self.predictions_made,
            "predictions_resolved": self.predictions_resolved,
            "decay": self.decay.as_dict() if self.decay else None,
        }


class Consolidator:
    """Clock 2. Drains percepts into the graph and runs decay once per turn."""

    def __init__(self, ctx: IngestContext | None = None, *, decay_enabled: bool = True,
                 reflect_llm=None) -> None:
        self.ctx = ctx or IngestContext()
        self.decay_enabled = decay_enabled
        # Stage 3 (W4): optional reasoning LLM for REFLECT (evidence->pattern). None => REFLECT
        # is skipped and consolidation stays fully deterministic (FREEZE only).
        self.reflect_llm = reflect_llm
        self.turn = 0
        self.bookkeeping: dict[str, DecayState] = {}
        # Stage 3 (W1): the M (significance) axis — a side-table mirroring `bookkeeping`,
        # populated from grounded perceiver proposals. Never touches r or the Node schema.
        self.significance: dict[str, sig.SignificanceState] = {}
        # Stage 3 (W4): node_ids FROZEN by the consolidation scheduler — made decay-immune
        # because they carry grounded significance (M shields against forgetting, T).
        self.frozen: set[str] = set()
        # Stage 3 (W5): recursive self-model — predictions keyed by node_id, resolved by later
        # observations; calibration measures whether the self's confidence matched reality.
        self.predictions: dict[str, sm.Prediction] = {}

    @property
    def graph(self):
        return self.ctx.graph

    def consolidate(
        self,
        percepts: list[Percept],
        *,
        session_id: str | None = None,
        timestamp: str = "1970-01-01T00:00:00Z",
    ) -> ConsolidationReport:
        """Process one batch: advance the turn, ingest, update decay state, decay."""
        self.turn += 1
        session_id = session_id or f"consolidate_turn_{self.turn:06d}"

        # Snapshot reinforcement before ingest so we can tell new vs. reinforced.
        before = {n.node_id: n.reinforcement.overall for n in self.graph.all_nodes()}

        accepted = rejected = significance_recorded = 0
        predictions_made = predictions_resolved = 0
        if percepts:
            self.ctx.begin_session(session_id, timestamp)
            for i, p in enumerate(percepts):
                # Carry the ORIGINATING source_id into the turn id. It used to be
                # f"{session_id}_t{i:03d}" with a synthetic session ("consolidate_turn_000001"), which
                # discarded the source identity at consolidation — so receipts.bind_document, which
                # matches nodes by f"{source_id}_" prefix, could never find a perceived node. That is
                # the whole reason the perceive path reported receipts 0/176 in the ops run while the
                # document path reported 100%: not a property of abstraction, a dropped identifier.
                # Turn number keeps it unique when the same source is consolidated across cycles.
                turn_id = (f"{p.source_id}_c{self.turn:04d}t{i:03d}" if p.source_id
                           else f"{session_id}_t{i:03d}")
                result = self.ctx.ingest(p.claim, p.evidence, p.confidence, turn_id, session_id)
                if result.firewall_decision in ("REJECT", "QUARANTINE"):
                    rejected += 1
                    continue
                accepted += 1
                nid = _mutation_node_id(result.mutation)
                if nid is None:
                    continue
                # Stage 3 (W1): attach grounded significance to the node this percept resolved
                # to. Grounding already happened in the perceiver; here we only accumulate.
                if p.rationale or p.cause:
                    sig.record(self.significance, nid,
                               rationale=p.rationale, rationale_grounding=p.rationale_grounding,
                               cause=p.cause, source_turn_id=turn_id)
                    significance_recorded += 1
                # Stage 3 (W5): a forward-looking claim becomes a prediction; any other claim is
                # an observation that may RESOLVE an earlier prediction (predict -> observe).
                node = self.graph.get_node(nid)
                if node is not None:
                    subject = _subject(node)
                    if sm.record_prediction(self.predictions, nid, node.claim, subject, self.turn):
                        predictions_made += 1
                    else:
                        predictions_resolved += len(
                            sm.resolve_with(self.predictions, node.claim, subject, self.turn, nid))

        # Register new nodes / refresh reinforced nodes in the decay side-table.
        registered = reinforced = 0
        for node in self.graph.all_nodes():
            nid = node.node_id
            r = node.reinforcement.overall
            state = self.bookkeeping.get(nid)
            if state is None:
                self.bookkeeping[nid] = DecayState(
                    created_turn=self.turn,
                    last_accessed_turn=self.turn,
                    r_at_last_access=r,
                    confidence=(r / 0.4 if node.node_type == "inference" and r > 0 else None),
                )
                registered += 1
            elif r > before.get(nid, r):
                # Reinforced this cycle — reset the decay baseline from the new value.
                state.last_accessed_turn = self.turn
                state.r_at_last_access = r
                state.hot_entry_turn = None  # re-evaluate HOT window from the new level
                reinforced += 1

        # Stage 3 (W2): build the typed field — causal/motivational/temporal edges. Cheap,
        # pure read of (graph, significance), idempotent across turns.
        field_edges = materialize_field(self.graph, self.significance)

        # Stage 3 (W4): opportunity-scored background metabolism. FREEZE protects meaningful
        # (high grounded significance) not-yet-permanent memories from decay. Runs BEFORE decay
        # so newly frozen nodes are immune this very turn.
        ops_report = run_consolidation_ops(self.graph, self.significance, self.frozen,
                                           turn=self.turn, reflect_llm=self.reflect_llm)

        decay_report = None
        if self.decay_enabled:
            decay_report = apply_decay(self.graph, self.bookkeeping, self.turn,
                                       frozen=self.frozen, significance=self.significance)

        return ConsolidationReport(
            turn=self.turn,
            percepts_in=len(percepts),
            accepted=accepted,
            rejected=rejected,
            nodes_registered=registered,
            nodes_reinforced=reinforced,
            significance_recorded=significance_recorded,
            field_edges_created=field_edges,
            frozen_this_turn=len(ops_report.frozen),
            reflections_created=len(ops_report.reflections),
            compressions_created=len(ops_report.compressions),
            predictions_made=predictions_made,
            predictions_resolved=predictions_resolved,
            decay=decay_report,
        )


class TwoClockScheduler:
    """Owns a PerceptBuffer (Clock 1 sink) and a Consolidator (Clock 2).

    perceive(): admit a percept — fast, never blocks.
    tick():     consolidate one batch iff the buffer says it should drain.
    flush():    force-consolidate whatever remains (e.g. at shutdown).
    """

    def __init__(
        self,
        buffer: PerceptBuffer | None = None,
        consolidator: Consolidator | None = None,
    ) -> None:
        # NB: use `is None`, not `or` — an empty PerceptBuffer is falsy (__len__==0),
        # so `buffer or PerceptBuffer()` would silently discard a passed-in empty buffer.
        self.buffer = buffer if buffer is not None else PerceptBuffer()
        self.consolidator = consolidator if consolidator is not None else Consolidator()

    @property
    def graph(self):
        return self.consolidator.graph

    def perceive(
        self,
        claim: str,
        evidence: str,
        confidence: float,
        salience: float,
        candidate_domains: tuple[str, ...] = (),
        polarity: int = 0,
        entity_hint: str | None = None,
        source_id: str = "",
        rationale: str | None = None,
        cause: str | None = None,
        rationale_grounding: float = 0.0,
    ):
        """Clock 1 sink. Returns the AdmitResult (incl. any eviction)."""
        return self.buffer.admit(
            claim=claim, evidence=evidence, confidence=confidence, salience=salience,
            candidate_domains=candidate_domains, polarity=polarity,
            entity_hint=entity_hint, source_id=source_id,
            rationale=rationale, cause=cause, rationale_grounding=rationale_grounding,
        )

    def tick(self, *, timestamp: str = "1970-01-01T00:00:00Z") -> ConsolidationReport | None:
        """Clock 2 driver. Consolidate one batch if the buffer is ready; else None."""
        if not self.buffer.should_drain():
            return None
        batch = self.buffer.drain()
        return self.consolidator.consolidate(batch, timestamp=timestamp)

    def flush(self, *, timestamp: str = "1970-01-01T00:00:00Z") -> ConsolidationReport | None:
        """Drain and consolidate everything remaining, regardless of triggers."""
        if len(self.buffer) == 0:
            return None
        batch = self.buffer.drain(max_items=len(self.buffer))
        return self.consolidator.consolidate(batch, timestamp=timestamp)
