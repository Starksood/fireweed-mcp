"""Typed field edges — turn a list of facts into a connected field (Stage 3, W2).

W1 grounds *significance* per node: a why-it-matters rationale and the source clauses a
claim is caused-by. Those causes live in the significance side-table as TEXT. W2's first
job is to make the field RELATIONAL: promote each grounded cause clause into a typed
`causes` edge between two nodes, so "I'm down to forty minutes each way" is not just
annotated with "the 38 route got rerouted" — it is *linked* to the node that states the
reroute. A field-like self is facts-in-relation, not facts-in-a-list.

LLM proposes, CODE decides — again. W1's perceiver proposed the cause (and the guard already
grounded it to the source). Here the CODE decides the LINK: a cause clause becomes an edge
only if it matches an existing active node well enough (content overlap >= CAUSE_EDGE_MIN_MATCH).
An ungrounded match is left as a dangling annotation in the side-table — better an unlinked
truth than an invented edge. Direction is cause -> effect: source = the node the clause
matches (the cause), target = the node that carries the significance (the effect).

Determinism + idempotency: materialization is a pure read of (graph, significance) that
appends edges; it skips any (cause, effect) pair already linked, so running it every
consolidation turn converges rather than duplicating.
"""
from __future__ import annotations

import uuid

from .graph import GraphState, Relation, RelationEvidence, RelationRef
from .significance import SignificanceState, _content_tokens
from .constants import CAUSE_EDGE_MIN_MATCH, MOTIVATES_EDGE_MIN_MATCH


def _content_jaccard(a: str, b: str) -> float:
    ta, tb = _content_tokens(a), _content_tokens(b)
    union = ta | tb
    return 0.0 if not union else len(ta & tb) / len(union)


def _best_match_node(clause: str, anchor_node_id: str, graph: GraphState):
    """The active node whose claim best matches `clause` (excluding the anchor node
    itself). Returns (node, score) or (None, 0.0). Shared by causes and motivates."""
    best, best_score = None, 0.0
    for node in graph.all_nodes():
        if node.node_id == anchor_node_id:
            continue
        if node.status.memory_state != "active":
            continue
        score = _content_jaccard(clause, node.claim)
        if score > best_score:
            best, best_score = node, score
    return best, best_score


def _has_edge(graph: GraphState, source_id: str, target_id: str,
              relation_type: str | None = None) -> bool:
    """True if an edge source->target exists. relation_type=None matches ANY type
    (used to keep causes and motivates from double-linking the same pair)."""
    for r in graph.all_relations():
        if r.source_id == source_id and r.target_id == target_id:
            if relation_type is None or r.relation_type == relation_type:
                return True
    return False


def _add_typed_edge(graph: GraphState, source_id: str, target_id: str, relation_type: str,
                    *, confidence: float, span: str, anchor_node) -> str:
    """Create a typed node->node Relation and link a RelationRef onto the anchor node so
    1-hop expansion can traverse it. Returns the new relation_id."""
    relation_id = "rel_" + uuid.uuid4().hex[:12]
    src, tgt = graph.get_node(source_id), graph.get_node(target_id)
    graph.add_relation(Relation(
        relation_id=relation_id, source_id=source_id, target_id=target_id,
        relation_type=relation_type, polarity="positive",
        claim=f"{src.claim} -> {tgt.claim}", confidence=round(confidence, 3),
        evidence=RelationEvidence(source_node_ids=[source_id, target_id], source_spans=[span]),
        status="validated",
    ))
    anchor_node.relations.append(RelationRef(
        relation_id=relation_id, relation_type=relation_type,
        target_node_id=(source_id if anchor_node.node_id == target_id else target_id),
    ))
    return relation_id


def materialize_causal_edges(
    graph: GraphState,
    significance: dict[str, SignificanceState],
    *,
    min_match: float = CAUSE_EDGE_MIN_MATCH,
) -> list[str]:
    """Promote grounded cause clauses into typed `causes` Relations (cause -> effect).

    For each node carrying grounded causes, find the active node each cause clause matches
    (content overlap >= min_match) and, unless already linked, create a `causes` edge. The
    effect node gets a RelationRef so retrieval/expansion can walk the field. Returns the
    list of created relation_ids (empty when nothing matched — the honest default)."""
    created: list[str] = []
    for effect_id in sorted(significance.keys()):  # sorted -> deterministic order
        state = significance[effect_id]
        effect = graph.get_node(effect_id)
        if effect is None or effect.status.memory_state != "active":
            continue
        for cause_clause in state.causes:
            cause_node, score = _best_match_node(cause_clause, effect_id, graph)
            if cause_node is None or score < min_match:
                continue  # dangling annotation: a grounded truth we can't yet link
            if _has_edge(graph, cause_node.node_id, effect_id, "causes"):
                continue  # idempotent
            created.append(_add_typed_edge(graph, cause_node.node_id, effect_id, "causes",
                                           confidence=score, span=cause_clause, anchor_node=effect))
    return created


def materialize_motivational_edges(
    graph: GraphState,
    significance: dict[str, SignificanceState],
    *,
    min_match: float = MOTIVATES_EDGE_MIN_MATCH,
) -> list[str]:
    """Promote grounded rationales into typed `motivates` Relations (motivator -> motivated).

    A node's grounded rationale ("why this matters") often names another fact that motivates
    it. When the rationale matches another active node's claim (overlap >= min_match), link
    that node as the motivator. Skips pairs already joined by ANY edge so a cause is not also
    relabelled a motive (causes runs first and is the stronger claim)."""
    created: list[str] = []
    for node_id in sorted(significance.keys()):
        state = significance[node_id]
        if not state.rationale:
            continue
        node = graph.get_node(node_id)
        if node is None or node.status.memory_state != "active":
            continue
        motivator, score = _best_match_node(state.rationale, node_id, graph)
        if motivator is None or score < min_match:
            continue
        # UNDIRECTED any-type guard: if the pair is already connected either way (e.g. a
        # causes edge — the stronger claim), don't add a motivates edge. This prevents the
        # reroute--causes-->commute / commute--motivates-->reroute 2-cycle a live run exposed.
        if (_has_edge(graph, motivator.node_id, node_id)
                or _has_edge(graph, node_id, motivator.node_id)):
            continue
        created.append(_add_typed_edge(graph, motivator.node_id, node_id, "motivates",
                                       confidence=score, span=state.rationale, anchor_node=node))
    return created


def materialize_temporal_edges(graph: GraphState) -> list[str]:
    """Create `before` Relations (earlier -> later) between active, same-entity nodes that
    both carry an event_time. Fully deterministic and grounded in the substrate — no LLM,
    no fabrication. ISO/`YYYY-MM` dates compare correctly lexicographically; equal times are
    skipped (no ordering to assert)."""
    created: list[str] = []
    dated = [n for n in graph.all_nodes()
             if n.status.memory_state == "active" and n.temporal.event_time]
    dated.sort(key=lambda n: (n.temporal.event_time, n.node_id))  # deterministic
    for i, earlier in enumerate(dated):
        for later in dated[i + 1:]:
            if earlier.temporal.event_time == later.temporal.event_time:
                continue
            if not ({e.entity_id for e in earlier.entities} & {e.entity_id for e in later.entities}):
                continue  # only order events that share a subject — otherwise it is noise
            if _has_edge(graph, earlier.node_id, later.node_id, "before"):
                continue
            created.append(_add_typed_edge(
                graph, earlier.node_id, later.node_id, "before",
                confidence=1.0, span=f"{earlier.temporal.event_time} < {later.temporal.event_time}",
                anchor_node=later))
    return created


def materialize_field(graph: GraphState, significance: dict[str, SignificanceState]) -> dict:
    """Build the typed field in one pass: causal (from grounded causes), motivational (from
    grounded rationale), and temporal (from event_time). Returns per-type created counts.
    Causal runs first so a stated cause is never also relabelled a weaker `motivates`."""
    return {
        "causes": len(materialize_causal_edges(graph, significance)),
        "motivates": len(materialize_motivational_edges(graph, significance)),
        "before": len(materialize_temporal_edges(graph)),
    }
