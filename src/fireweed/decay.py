"""Decay — "forgetting as metabolism" for the v16 graph.

Ported from v15 memory_loop.py (MemoryFabric.decay / _compute_decayed_r), the
forgetting stage the original thesis called for ("Forgetting as metabolism").
This is the Clock 2 counterpart to graph growth: consolidation writes new memory
*and* lets idle, low-value memory fade, so the graph can grow continuously
without drowning in noise.

Why a side-table instead of new Node fields
-------------------------------------------
The v15 decay operated on dict nodes carrying r / last_accessed_turn /
created_turn / valence / confidence. The v16 Node dataclass has none of these
turn-based fields, and its Reinforcement is recurrence-based. Retrofitting Node
(and graph_serializer, and 400+ tests) to carry decay bookkeeping would be a
large, risky schema change for a substrate that is still being designed.

Instead, decay state lives in a side-table (`dict[node_id -> DecayState]`) owned
by the consolidator. The Node schema and its serialization are untouched; only
`node.reinforcement.overall` is mutated (which retrieval already reads). This is
reversible and keeps the blast radius at zero for the existing pipeline.

Faithful-port notes / deliberate adaptations
--------------------------------------------
* Identity immunity: nodes at/above CORE_THRESHOLD never decay (same as v15
  THRESHOLD_IDENTITY).
* Linear-from-last-access: decay uses r_at_last_access as the baseline, so it is
  linear from the last touch rather than compounding each turn (v15 behaviour).
* Survival floor: nodes younger than DECAY_SURVIVAL_FLOOR_TURNS cannot fall below
  DECAY_SURVIVAL_FLOOR (newborn grace).
* HOT stabilization: a node freshly in the HOT band decays slower for a few turns
  (post-tetanic-potentiation analogy), preventing it from leapfrogging back below
  HOT before reinforcement can accumulate.
* Valence -> polarity: v16 has no scalar valence. We map categorical
  Predicate.polarity to the v15 modulation: positive/negative (emotionally
  charged) persist; neutral fades fastest. This is the one genuine adaptation.
* Inference-confidence decay: v16 Node has no `confidence` field, so confidence is
  tracked in DecayState (initialized with the v15 r/0.4 fallback).

Three-axis forgetting (Stage 4, pillar 4)
-----------------------------------------
The original spec wanted T/R/M as distinct forgetting axes; the reinforcement formula collapsed
them. They are now explicit in this loop:
  * T (immutable): CORE (r>=CORE_THRESHOLD) and FROZEN nodes never decay (identity_protected).
  * R (reinforcement): the linear-from-last-access decay below.
  * M (slow drift): grounded significance (W1) dampens the per-turn rate, so a meaningful memory
    fades more slowly than a bare one of equal r — graded, distinct from FREEZE's binary immunity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .significance import significance_prior
from .constants import (
    CORE_THRESHOLD,
    HOT_THRESHOLD,
    DECAY_BASE_PER_TURN,
    DECAY_SURVIVAL_FLOOR,
    DECAY_SURVIVAL_FLOOR_TURNS,
    DECAY_HOT_STABILIZATION_TURNS,
    DECAY_HOT_STABILIZATION_FACTOR,
    DECAY_POLARITY_MODIFIER_POSITIVE,
    DECAY_POLARITY_MODIFIER_NEGATIVE,
    DECAY_POLARITY_MODIFIER_NEUTRAL,
    SIGNIFICANCE_DECAY_DAMPING,
    INFER_CONFIDENCE_DECAY_TURNS,
    INFER_CONFIDENCE_DECAY_RATE,
    INFER_CONFIDENCE_DECAY_INTERVAL,
)


@dataclass
class DecayState:
    """Per-node decay bookkeeping, kept in a side-table (not on the Node)."""
    created_turn: int
    last_accessed_turn: int
    r_at_last_access: float
    hot_entry_turn: int | None = None
    last_decay_interval: int = 0
    confidence: float | None = None  # tracked only for inference nodes


@dataclass
class DecayReport:
    turn: int
    nodes_examined: int = 0
    nodes_decayed: int = 0
    confidence_decayed: int = 0
    identity_protected: int = 0          # T axis: CORE + frozen nodes (immune)
    m_slowed: int = 0                    # M axis: nodes whose decay was dampened by significance
    cold_candidates: list[str] = field(default_factory=list)  # node_ids now below COLD

    def as_dict(self) -> dict:
        return {
            "turn": self.turn,
            "nodes_examined": self.nodes_examined,
            "nodes_decayed": self.nodes_decayed,
            "confidence_decayed": self.confidence_decayed,
            "identity_protected": self.identity_protected,
            "m_slowed": self.m_slowed,
            "n_cold_candidates": len(self.cold_candidates),
        }


def polarity_modifier(polarity: str) -> float:
    """Map categorical Predicate.polarity to a decay multiplier (lower = slower)."""
    if polarity == "positive":
        return DECAY_POLARITY_MODIFIER_POSITIVE
    if polarity == "negative":
        return DECAY_POLARITY_MODIFIER_NEGATIVE
    return DECAY_POLARITY_MODIFIER_NEUTRAL


def compute_decayed_r(
    state: DecayState,
    polarity: str,
    idle_turns: int,
    current_turn: int,
    significance_factor: float = 0.0,
) -> float:
    """Return the new reinforcement value after `idle_turns` of idleness.

    Linear from r_at_last_access (not compounding). May stamp state.hot_entry_turn
    as a side effect when a node is first seen in the HOT band. `significance_factor`
    (the M axis, in [0,~0.8]) dampens the decay rate — a meaningful memory drifts slower.
    """
    r_base = state.r_at_last_access
    base = DECAY_BASE_PER_TURN * idle_turns
    modifier = polarity_modifier(polarity)

    # M axis (Stage 4 pillar 4): grounded significance is slow-drift — scale down the rate in
    # proportion to significance (FREEZE already gives full immunity above its threshold).
    if significance_factor > 0.0:
        modifier *= max(0.0, 1.0 - SIGNIFICANCE_DECAY_DAMPING * significance_factor)

    # HOT stabilization window (post-tetanic potentiation analogy).
    if HOT_THRESHOLD <= r_base < CORE_THRESHOLD:
        if state.hot_entry_turn is None:
            state.hot_entry_turn = current_turn
        if current_turn - state.hot_entry_turn <= DECAY_HOT_STABILIZATION_TURNS:
            modifier *= DECAY_HOT_STABILIZATION_FACTOR

    proposed = r_base - (base * modifier)

    turns_alive = current_turn - state.created_turn
    if turns_alive <= DECAY_SURVIVAL_FLOOR_TURNS:
        return max(DECAY_SURVIVAL_FLOOR, proposed)
    return max(0.0, proposed)


def apply_decay(
    graph,
    bookkeeping: dict[str, DecayState],
    current_turn: int,
    *,
    identity_threshold: float = CORE_THRESHOLD,
    frozen: set[str] | None = None,
    significance: dict | None = None,
) -> DecayReport:
    """Decay every tracked, non-identity node by its idle time. Mutates
    node.reinforcement.overall in place. Untracked nodes are skipped (the
    consolidator is responsible for registering nodes on creation).

    `frozen` (Stage 3, W4) are node_ids made decay-immune by FREEZE — a meaningful memory
    protected from forgetting even though it has not (yet) reached CORE on its own.
    """
    frozen = frozen or set()
    report = DecayReport(turn=current_turn)
    for node in graph.all_nodes():
        state = bookkeeping.get(node.node_id)
        if state is None:
            continue
        report.nodes_examined += 1

        r = node.reinforcement.overall
        if r >= identity_threshold or node.node_id in frozen:
            report.identity_protected += 1
            continue

        idle = current_turn - state.last_accessed_turn
        if idle > 0:
            polarity = node.predicate.polarity
            sig_factor = significance_prior(significance.get(node.node_id)) if significance else 0.0
            if sig_factor > 0.0:
                report.m_slowed += 1
            new_r = compute_decayed_r(state, polarity, idle, current_turn, sig_factor)
            if new_r != r:
                node.reinforcement.overall = new_r
                report.nodes_decayed += 1
                if new_r < graph_cold_threshold():
                    report.cold_candidates.append(node.node_id)

        # Inference-confidence decay — guesses fade unless re-confirmed.
        if node.node_type == "inference" and idle > INFER_CONFIDENCE_DECAY_TURNS:
            intervals_past = (idle - INFER_CONFIDENCE_DECAY_TURNS) // INFER_CONFIDENCE_DECAY_INTERVAL
            if intervals_past > state.last_decay_interval:
                if state.confidence is None:
                    # v15 fallback: derive confidence from r when not tracked yet.
                    state.confidence = (node.reinforcement.overall / 0.4
                                        if node.reinforcement.overall > 0 else 0.0)
                old_conf = state.confidence
                new_conf = max(0.0, old_conf - INFER_CONFIDENCE_DECAY_RATE)
                state.confidence = new_conf
                turns_alive = current_turn - state.created_turn
                floor = DECAY_SURVIVAL_FLOOR if turns_alive <= DECAY_SURVIVAL_FLOOR_TURNS else 0.0
                node.reinforcement.overall = max(floor, new_conf * 0.4)
                state.last_decay_interval = intervals_past
                report.confidence_decayed += 1

    return report


def graph_cold_threshold() -> float:
    """The COLD layer boundary — nodes below this are pruning candidates."""
    from .constants import COLD_THRESHOLD
    return COLD_THRESHOLD
