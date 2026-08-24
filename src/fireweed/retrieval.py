"""Phase 10 — Stateless retrieval engine. Pure: reads graph, never writes.
Determinism invariant: same question + same graph → identical RetrievalResult.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

from .graph import GraphState, Node, Entity
from .query_parser import QueryParsed, parse_query, _NEGATION_RE
from .scoring import jaccard_score, query_coverage
from .domain_classifier import classify_domains
from .constants import (
    RETRIEVAL_W_JACCARD,
    RETRIEVAL_W_ENTITY,
    RETRIEVAL_W_REINFORCE,
    RETRIEVAL_MIN_JACCARD_ABSTAIN,
    RETRIEVAL_MIN_COVERAGE_ABSTAIN,
    SIGNIFICANCE_RETRIEVAL_WEIGHT,
)
from .significance import SignificanceState, significance_prior
from .read_gate import read_gate, ReadGateVerdict


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class EvidencePath:
    node_ids: list[str]   # chain from context anchor to result node
    path_type: Literal["direct", "entity_match", "graph_hop"]


@dataclass
class ResultEntry:
    node: Node
    score: float
    is_inference: bool          # True when node.node_type == "inference"
    evidence_path: EvidencePath
    in_tension: bool = False     # Stage 3 (W3): node is `disputed` — held in standing contradiction


@dataclass
class RetrievalResult:
    query: str
    parsed_query: QueryParsed
    matched_nodes: list[ResultEntry]    # primary results, ranked by score desc
    expanded_nodes: list[ResultEntry]   # 1-hop entity expansion, not in matched_nodes
    entities_in_scope: list[Entity]     # Entity objects for all entity_ids in scope
    temporal_window_applied: tuple[str, str] | None
    polarity_filter_applied: Literal["positive", "negative", "any"]
    abstain: bool
    abstain_reason: str | None          # "no_evidence" | "entity_not_found"
                                        # | "unknown_predicate" | None
    # Read Gate (docs/DESIGN_read_gate.md): the adjudication behind the abstain flag, when the
    # gate is enabled. None when the gate is off, so pre-gate callers are byte-identical.
    gate_verdict: "ReadGateVerdict | None" = None


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(query: str, node: Node, query_entity_ids: list[str],
           significance: dict[str, SignificanceState] | None = None) -> float:
    """RETRIEVAL_W_JACCARD*jaccard + RETRIEVAL_W_ENTITY*entity_precision + RETRIEVAL_W_REINFORCE*reinforcement
    (+ SIGNIFICANCE_RETRIEVAL_WEIGHT*significance_prior when a significance side-table is supplied).

    entity_precision = min(1.0, |qe_expanded ∩ ne| / |qe|)
    qe_expanded = qe | {eid[:-2] for eid in qe if eid.endswith("'s")}

    Possessive expansion: a possessive query entity (e.g., ent_maya's) also matches the base-form
    node entity (ent_maya), because the graph may store claims using either form of the same name.
    Expansion is query-side only — node entity IDs are not modified — so a non-possessive query
    (ent_maya) still does NOT match possessive-only nodes (ent_maya's). min(1.0, ...) prevents
    precision from exceeding 1.0 when a node covers both forms of an entity.
    If query has no entity_ids, entity_precision = 0.0.

    The significance term is OPT-IN (Stage 3, W1): when `significance` is None (the default,
    and every pre-W1 caller) the score is byte-identical to before. When supplied, a node that
    carries grounded meaning (a why-it-matters rationale and/or causal links) gets a small,
    bounded prior — significance, not just frequency, shapes what surfaces.
    """
    # Lexical relevance is the BETTER of jaccard and query-coverage. Jaccard normalizes by the
    # union, so it reads a short query against a long claim as irrelevant however well the query
    # is covered; coverage asks only "are the query's terms here?". Taking the max means a node
    # that fully covers the query ranks like one that lexically resembles it, and no score is ever
    # lowered. Without this the abstain gate and the ranking disagreed: query("consolidation")
    # stopped abstaining (a candidate covered it) while the covering node -- a long reflection
    # with tiny jaccard -- stayed below ten HOT nodes and never entered the result.
    j = max(jaccard_score(query, node.normalized_claim),
            query_coverage(query, node.normalized_claim))
    qe = set(query_entity_ids)
    qe_expanded = qe | {eid[:-2] for eid in qe if eid.endswith("'s")}
    ne = {e.entity_id for e in node.entities}
    entity_precision = min(1.0, len(qe_expanded & ne) / len(qe)) if qe else 0.0
    base = (RETRIEVAL_W_JACCARD * j + RETRIEVAL_W_ENTITY * entity_precision
            + RETRIEVAL_W_REINFORCE * node.reinforcement.overall)
    if significance:
        base += SIGNIFICANCE_RETRIEVAL_WEIGHT * significance_prior(significance.get(node.node_id))
    return base


# ── Filters ───────────────────────────────────────────────────────────────────

def _expand_1hop(candidates: list[Node], graph: GraphState, ts: str | None = None,
                 ego_graph: bool = False) -> list[Node]:
    """Active nodes sharing ≥1 entity with candidates, excluding candidates.

    Only returns valid (non-superseded) nodes. With ego_graph, the sharing-entity scan is served
    by the inverted index (only nodes touching the candidates' entities are examined) rather than
    a full valid-node scan — same result set, bounded cost.
    """
    cand_ids = {n.node_id for n in candidates}
    entity_ids = {e.entity_id for n in candidates for e in n.entities}
    if not entity_ids:
        return []
    scan = graph.nodes_touching(entity_ids, ts) if ego_graph else graph.get_valid_nodes(ts)
    expanded: dict[str, Node] = {}
    for node in scan:
        if node.node_id in cand_ids:
            continue
        if node.status.memory_state not in ("active", "disputed"):
            continue
        if {e.entity_id for e in node.entities} & entity_ids:
            expanded[node.node_id] = node
    return list(expanded.values())


_FIELD_EDGE_TYPES = ("causes", "motivates", "before")


def _expand_field(candidates: list[Node], graph: GraphState, ts: str | None = None) -> dict[str, tuple[str, str]]:
    """1-hop expansion along typed FIELD edges (causes/motivates/before; Stage 3, W2),
    bidirectional. Returns {reached_node_id: (anchor_node_id, relation_type)} for active,
    valid nodes, excluding the candidates themselves. Unlike entity expansion this is NOT
    domain-filtered by the caller — a cause/motivation may live in another domain, which is
    exactly the cross-domain link the field exists to preserve."""
    cand_ids = {n.node_id for n in candidates}
    valid_ids = {n.node_id for n in graph.get_valid_nodes(ts)}
    reached: dict[str, tuple[str, str]] = {}
    for n in candidates:  # candidates are already score-sorted -> first anchor wins, deterministic
        for rel in graph.get_relations_for_node(n.node_id):
            if rel.relation_type not in _FIELD_EDGE_TYPES:
                continue
            other = rel.target_id if rel.source_id == n.node_id else rel.source_id
            if other in cand_ids or other in reached:
                continue
            tgt = graph.get_node(other)
            if tgt is None or tgt.status.memory_state not in ("active", "disputed") or other not in valid_ids:
                continue
            reached[other] = (n.node_id, rel.relation_type)
    return reached


def _apply_temporal_filter(nodes: list[Node], window: tuple[str, str] | None) -> list[Node]:
    """Keep nodes whose event_time falls within window. Nodes with event_time=None pass through."""
    if window is None:
        return nodes
    from_date, to_date = window
    return [n for n in nodes if n.temporal.event_time is None
            or (from_date <= n.temporal.event_time <= to_date)]


def _apply_polarity_filter(nodes: list[Node], polarity: str) -> list[Node]:
    """Filter to negative-polarity nodes when polarity='negative'; fall back if empty.

    Accepts a node if EITHER:
    - node.predicate.polarity == "negative"  (pipeline-tagged), OR
    - the node's normalized_claim matches _NEGATION_RE (claim-level negation detection)

    This catches claims like "Maya has stopped running..." where the pipeline does not
    annotate past-tense cessation as negative polarity, but the claim clearly contains
    a negation word ("stopped") matching the same regex used by the query parser.
    """
    if polarity != "negative":
        return nodes
    filtered = [
        n for n in nodes
        if n.predicate.polarity == "negative"
        or _NEGATION_RE.search(n.normalized_claim or n.claim)
    ]
    return filtered if filtered else nodes


# ── Entry point ───────────────────────────────────────────────────────────────

def query_graph(
    question: str,
    graph: GraphState,
    current_timestamp: str | None = None,
    session_context: dict | None = None,  # Phase 11+: ignored in Phase 10
    max_results: int = 10,
    min_score: float = 0.0,
    significance: dict[str, SignificanceState] | None = None,  # Stage 3 (W1): opt-in M prior
    traverse_field: bool = False,  # Stage 3 (W2): opt-in typed-edge (causes/motivates/before) expansion
    ego_graph: bool = False,  # Sprint 2: opt-in bounded ego-graph candidate generation via the index
    use_read_gate: bool = True,  # Read Gate: adjudicate admission into the conversation
) -> RetrievalResult:
    """Stateless retrieval: read-only, deterministic, never raises."""
    try:
        return _query(question, graph, current_timestamp, max_results, min_score,
                      significance, traverse_field, ego_graph, use_read_gate)
    except Exception:
        pq = parse_query(question, graph, current_timestamp)
        return RetrievalResult(
            query=question, parsed_query=pq,
            matched_nodes=[], expanded_nodes=[],
            entities_in_scope=[],
            temporal_window_applied=None,
            polarity_filter_applied="any",
            abstain=True, abstain_reason="no_evidence",
        )


def _query(
    question: str, graph: GraphState, ts: str | None,
    max_results: int, min_score: float,
    significance: dict[str, SignificanceState] | None = None,
    traverse_field: bool = False,
    ego_graph: bool = False,
    use_read_gate: bool = True,
) -> RetrievalResult:
    pq = parse_query(question, graph, ts)

    # Stage 1 — candidate generation (only valid/non-superseded nodes)
    # When entities are specified in the query, require either entity match or high semantic
    # similarity to prevent matching unrelated entities with similar names (e.g., Aria vs Ari).
    #
    # Sprint 2 (opt-in): for an entity-anchored query, seed from the entity->nodes index (bounded
    # ego-graph, O(Σ deg(eid))) instead of scanning the whole graph. This drops the entity-DISJOINT
    # jaccard>=0.05 tail that the full scan keeps — an intentional, documented difference: an
    # entity-anchored query surfaces its entities' neighborhood, not unrelated nodes that happen to
    # share 5% of words. Falls back to the full scan when the query resolves no entities.
    seed_nodes: list[Node]
    if ego_graph and pq.entity_ids:
        seed_nodes = graph.nodes_touching(set(pq.entity_ids), ts)
    else:
        seed_nodes = graph.get_valid_nodes(ts)
    candidates: list[Node] = []
    for n in seed_nodes:
        # active OR disputed: a disputed node is held in standing tension (W3), not retired —
        # both sides of an unresolved contradiction stay answerable. Only superseded is hidden.
        if n.status.memory_state not in ("active", "disputed"):
            continue
        score = _score(question, n, pq.entity_ids, significance)
        if score <= min_score:
            continue
        # If query specifies entities, require entity match or sufficient Jaccard similarity
        if pq.entity_ids:
            node_entity_ids = {e.entity_id for e in n.entities}
            has_entity_match = bool(node_entity_ids & set(pq.entity_ids))
            if not has_entity_match:
                # No exact entity match; require high semantic similarity to include
                # This prevents "Ari" from matching when "Aria" is queried
                claim_jaccard = jaccard_score(question, n.normalized_claim)
                if claim_jaccard < 0.05:  # Require at least 5% semantic overlap
                    continue
        candidates.append(n)

    # Stage 2 — temporal filter
    candidates = _apply_temporal_filter(candidates, pq.temporal_window)

    # Stage 3 — polarity filter
    candidates = _apply_polarity_filter(candidates, pq.polarity)

    # Stage 3.5 — domain filter: restrict candidates to query-relevant domains (permissive)
    # This prevents entity-based matching from pulling in unrelated facts, but allows
    # related domains (e.g., "to work" queries can match commute claims about work).
    # Only filters when there's a clear cross-domain mismatch.
    # Example: "What does Chris rent?" should only match housing claims, not budget.
    query_domains = classify_domains(question)
    if query_domains and query_domains != {"other"}:
        # Filter only when there's no domain/semantic match - keep permissive for edge cases
        candidates = [
            n for n in candidates
            if (n.domains & query_domains)  # Direct domain match
            or ("other" in n.domains)  # Always allow flexible "other" domain
            or jaccard_score(question, n.normalized_claim) > 0.05  # Allow with semantic similarity
        ]

    # Stage 4 — score, sort deterministically, cap
    scored: list[tuple[float, Node]] = [
        (_score(question, n, pq.entity_ids, significance), n) for n in candidates
    ]
    scored.sort(key=lambda x: (-x[0], x[1].node_id))
    scored = scored[:max_results]

    # Stage 5 — 1-hop expansion with domain-aware filtering
    top_nodes = [n for _, n in scored]
    expanded_raw = _expand_1hop(top_nodes, graph, ts, ego_graph)
    expanded_raw = _apply_temporal_filter(expanded_raw, pq.temporal_window)
    expanded_raw = _apply_polarity_filter(expanded_raw, pq.polarity)

    # Filter expanded nodes to preserve domain context: only include nodes that share
    # domain(s) with matched nodes, or have "other" domain (flexible domain fallback).
    # This prevents entity-based expansion from pulling in unrelated facts.
    # Example: query "What does Chris rent?" should match housing claim; when expanding
    # on entity Chris, only include other housing claims, not budget claims about car repairs.
    matched_domains = set()
    for node in top_nodes:
        matched_domains |= node.domains
    expanded_raw = [
        n for n in expanded_raw
        if (n.domains & matched_domains) or ("other" in n.domains)
    ]

    # Stage 5b — typed FIELD traversal (Stage 3, W2). Follow causes/motivates/before edges
    # from the matched nodes so a query about an effect surfaces its cause/motivation. Opt-in
    # (traverse_field) and deliberately NOT domain-filtered — the field's value is cross-domain
    # links. When traverse_field is False, field_provenance stays empty and Stage 5 is unchanged.
    field_provenance: dict[str, tuple[str, str]] = {}
    if traverse_field:
        already = {n.node_id for n in expanded_raw} | {n.node_id for n in top_nodes}
        field_nodes: list[Node] = []
        for nid, prov in _expand_field(top_nodes, graph, ts).items():
            field_provenance[nid] = prov
            if nid not in already:
                node = graph.get_node(nid)
                if node is not None:
                    field_nodes.append(node)
        field_nodes = _apply_temporal_filter(field_nodes, pq.temporal_window)
        field_nodes = _apply_polarity_filter(field_nodes, pq.polarity)
        expanded_raw = expanded_raw + field_nodes

    expanded_scored = sorted(
        [(_score(question, n, pq.entity_ids, significance), n) for n in expanded_raw],
        key=lambda x: (-x[0], x[1].node_id),
    )[:max_results]

    # Build ResultEntry lists
    matched: list[ResultEntry] = [
        ResultEntry(
            node=n, score=s, is_inference=(n.node_type == "inference"),
            evidence_path=EvidencePath(
                node_ids=[n.node_id],
                path_type="entity_match" if pq.entity_ids else "direct",
            ),
            in_tension=(n.status.memory_state == "disputed"),
        )
        for s, n in scored
    ]

    matched_entity_map: dict[str, str] = {}
    for _, mn in scored:
        for eref in mn.entities:
            if eref.entity_id not in matched_entity_map:
                matched_entity_map[eref.entity_id] = mn.node_id

    expanded: list[ResultEntry] = []
    for s, n in expanded_scored:
        if n.node_id in field_provenance:  # reached via a typed field edge (causes/motivates/before)
            anchor_id = field_provenance[n.node_id][0]
        else:                              # reached via shared-entity expansion (unchanged path)
            anchor_id = next(
                (matched_entity_map[e.entity_id]
                 for e in n.entities if e.entity_id in matched_entity_map),
                n.node_id,
            )
        expanded.append(ResultEntry(
            node=n, score=s, is_inference=(n.node_type == "inference"),
            evidence_path=EvidencePath(node_ids=[anchor_id, n.node_id], path_type="graph_hop"),
            in_tension=(n.status.memory_state == "disputed"),
        ))

    # Entities in scope
    all_entity_ids = (
        {e.entity_id for entry in matched + expanded for e in entry.node.entities}
        | set(pq.entity_ids)
    )
    entities_in_scope = [
        ent for eid in sorted(all_entity_ids)
        if (ent := graph.get_entity(eid)) is not None
    ]

    # Abstention — two conditions:
    # 1. No matched nodes (unchanged behaviour)
    # 2. Relevance gate: abstain when NEITHER the jaccard gate nor the coverage gate is cleared
    #    by any candidate, i.e. no node is meaningfully relevant to the query.
    #    This prevents entity-matched but off-topic nodes from producing spurious results
    #    (e.g., salary query matching commute nodes purely via the entity ent_maya).
    #    X4 RELAXATION: For "other" domain claims (structurally valid but domain-unknown),
    #    relax Jaccard threshold only if the entry has a high entity match score (≥0.8).
    max_jaccard = max(
        (jaccard_score(question, n.normalized_claim) for n in candidates),
        default=0.0,
    )
    # X4: For "other" domain claims with strong entity precision, relax Jaccard threshold
    # (entity matching is primary signal for domain-unknown structurally valid claims)
    # X5: Extended to multi-domain claims (work, health, hobby, family) with strong entity match
    # (multi-domain claims may have low semantic overlap with query but high entity relevance)
    multi_domain_relaxed = {"other", "work", "health", "hobby", "family"}
    has_strong_entity_match = any(
        (entry.node.domains & multi_domain_relaxed) and
        (len({e.entity_id for e in entry.node.entities}
             & set(pq.entity_ids)) / len(set(pq.entity_ids)) >= 0.8)
        for entry in matched
        if pq.entity_ids
    )
    jaccard_threshold = 0.05 if has_strong_entity_match else RETRIEVAL_MIN_JACCARD_ABSTAIN

    # Coverage is an OR with the jaccard gate, so it can only make retrieval MORE willing to
    # answer, never less -- anything jaccard already accepted is untouched. It exists because
    # jaccard's union normalization conflates "irrelevant" with "relevant but verbose": a
    # one-token query cannot clear 0.12 against a long claim even on a perfect hit.
    max_coverage = max(
        (query_coverage(question, n.normalized_claim) for n in candidates),
        default=0.0,
    )
    abstain = len(matched) == 0 or (max_jaccard < jaccard_threshold
                                    and max_coverage < RETRIEVAL_MIN_COVERAGE_ABSTAIN)
    abstain_reason: str | None = None
    if abstain:
        abstain_reason = (
            "entity_not_found" if pq.entity_mentions and not pq.entity_ids
            else "no_evidence"
        )

    # ── The Read Gate (docs/DESIGN_read_gate.md) ─────────────────────────────
    # The similarity thresholds above cannot express "the substrate does not know this": they weigh
    # every query token equally, so an entity match carries a question whose PREDICATE has no
    # support -- query("Priya's salary") answering with a hire date. The gate adjudicates instead of
    # scoring, and its verdict is authoritative in BOTH directions: it can abstain where the score
    # answered (the defect) and answer where the score abstained (a short query whose head is
    # grounded but whose jaccard is below the floor). Candidates are still returned either way --
    # an abstaining result exposes its weak candidates by documented contract.
    gate_verdict: ReadGateVerdict | None = None
    if use_read_gate:
        gate_verdict = read_gate(question, graph, ts)
        if len(matched) == 0:
            abstain, abstain_reason = True, (gate_verdict.reason or "no_evidence")
        else:
            abstain = gate_verdict.abstain
            abstain_reason = gate_verdict.reason if abstain else None

    # When the query IS answerable, reinforcement must not carry irrelevant nodes into the answer.
    # Scoring is 0.55*jaccard + 0.40*entity + 0.05*reinforcement over a `score > 0` gate, so a node
    # with NO lexical and NO entity relation to the query still scores 0.05*r > 0 and rides along.
    # Measured on the ops graph: query("vLLM") correctly surfaced "The vLLM upgrade fixed Gemma-3"
    # and then listed "DocRED is a DOCUMENT-level task", "The recall is 0.542" and "Receipts are
    # 121/121" beneath it -- all HOT, all unrelated. Honoring the abstain flag (4eb3b86) fixed the
    # case where NOTHING matches; this is the case where one genuine match legitimises the rest.
    #
    # Deliberately applied only when not abstaining: an abstaining result still returns its weak
    # candidates for the caller to inspect, which is the documented contract. `expanded` is also
    # left alone -- those are 1-hop neighbours offered as context, not as direct answers.
    if not abstain:
        relevant = [
            e for e in matched
            if jaccard_score(question, e.node.normalized_claim) > 0.0
            or ({en.entity_id for en in e.node.entities} & set(pq.entity_ids))
        ]
        if relevant:                      # never empty an answerable result
            matched = relevant

    return RetrievalResult(
        query=question,
        parsed_query=pq,
        matched_nodes=matched,
        expanded_nodes=expanded,
        entities_in_scope=entities_in_scope,
        temporal_window_applied=pq.temporal_window,
        polarity_filter_applied=pq.polarity,
        abstain=abstain,
        abstain_reason=abstain_reason,
        gate_verdict=gate_verdict,
    )
