"""Opportunity-scored consolidation ops (Stage 3, W4) — background metabolism.

v15's memory_loop ran REFLECT / FREEZE / COMPRESS / REPLAY as *background* work, fired by
surprise-adaptive gates rather than brittle turn counters. This ports that discipline into
v16's "code decides" world. Each op exposes:

  * an ELIGIBILITY gate — a pure read of (graph + side-tables) returning scored candidates,
    where the score is the OPPORTUNITY (how worth-doing this op is right now); and
  * an APPLY step — the deterministic mutation.

A scheduler fires the highest-opportunity eligible candidates each consolidation turn, under
a per-op budget (so expensive ops like REFLECT/COMPRESS stay bounded). The graph grows and
metabolizes continuously instead of only on explicit calls.

This increment lands the scheduler + FREEZE (deterministic). FREEZE states a W1<->W4 link
directly: the memories that MATTER (high grounded significance) but are not yet permanent
(below CORE) are made immune to decay. Significance (M) protects against forgetting (T).
REFLECT (evidence->pattern, LLM-proposed + grounded) and COMPRESS (cluster cold nodes ->
summary) plug into the same scheduler in later increments.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .graph import (
    GraphState, Node, Relation, RelationEvidence,
    Temporal, Provenance, Reinforcement, NodeStatus, normalize_text,
)
from .reinforcement import get_layer, compute_reinforcement
from .resolver import _extract_predicate
from .scoring import is_personality_claim, jaccard_score
from .domain_classifier import extract_facets
from .significance import significance_prior, SignificanceState, _content_tokens
from .constants import (
    FREEZE_MIN_SIGNIFICANCE, FREEZE_BUDGET_PER_TURN,
    REFLECT_MIN_CLUSTER, REFLECT_MIN_GROUNDING, REFLECT_CSR_FLOOR, REFLECT_BUDGET_PER_TURN,
    REFLECT_NOVELTY_MAX,
    COMPRESS_MAX_R, COMPRESS_MIN_GROUP, COMPRESS_MIN_GROUNDING, COMPRESS_CSR_FLOOR,
    COMPRESS_BUDGET_PER_TURN,
    SYNTHESIS_MIN_CONFIDENCE,
)


@dataclass
class ConsolidationOpsReport:
    frozen: list[str] = field(default_factory=list)         # node_ids frozen this turn
    reflections: list[str] = field(default_factory=list)    # reflection node_ids created this turn
    compressions: list[str] = field(default_factory=list)   # summary node_ids created this turn

    def as_dict(self) -> dict:
        return {"frozen": len(self.frozen), "reflections": len(self.reflections),
                "compressions": len(self.compressions)}


# ── FREEZE — significance protects against decay ───────────────────────────────

def freeze_candidates(
    graph,
    significance: dict[str, SignificanceState],
    already_frozen: set[str],
    *,
    min_significance: float = FREEZE_MIN_SIGNIFICANCE,
) -> list[tuple[str, float]]:
    """Nodes worth protecting from decay: active/disputed, carrying grounded significance at
    or above min_significance, and not yet CORE (CORE is already decay-immune, so freezing it
    is a no-op). Returned as (node_id, opportunity) sorted by opportunity desc — opportunity is
    the grounded significance prior (the more a memory means, the more worth protecting)."""
    out: list[tuple[str, float]] = []
    for n in graph.all_nodes():
        if n.node_id in already_frozen:
            continue
        if n.status.memory_state not in ("active", "disputed"):
            continue
        if get_layer(n.reinforcement.overall) == "CORE":
            continue
        prior = significance_prior(significance.get(n.node_id))
        if prior < min_significance:
            continue
        out.append((n.node_id, round(prior, 3)))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


# ── REFLECT — the self observes a pattern across its own facts ──────────────────

_REFLECT_PROMPT = (
    "Here are several facts about one person. Name ONE concrete pattern they SHARE.\n"
    "STRICT RULES:\n"
    "- Reuse the facts' OWN KEY WORDS (the specific nouns/verbs/places below). Do not paraphrase "
    "them into abstract synonyms.\n"
    "- Do NOT begin with 'The system', 'The facts', 'This person', 'There is a' — state the "
    "pattern directly, like a fact.\n"
    "- Do NOT invent traits, feelings, or motives that are not written below.\n"
    "- It must be true of at least TWO facts, and must not just repeat one fact.\n"
    "- If there is no real shared pattern, use confidence 0.0.\n\n"
    "Facts:\n{facts}\n\n"
    'Respond with JSON only: {{"observation": "<pattern, in the facts\' own words>", "confidence": <0.0-1.0>}}'
)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_REFLECTABLE = ("fact", "event", "state", "preference", "constraint")


def _stem(t: str) -> str:
    """Strip a common inflectional suffix. A reflection GENERALIZES, so it naturally varies
    word forms ("delays" for the source's "delayed"); stemming lets grounding measure semantic
    overlap, not surface morphology. A live gemma-3-4b run was rejecting good patterns on this."""
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    return t


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in _content_tokens(text)}


def reflect_candidates(graph, *, min_cluster: int = REFLECT_MIN_CLUSTER) -> list[tuple[str, list, int]]:
    """Clusters of related facts worth reflecting on: active reflectable nodes grouped by
    domain, where a domain holds >= min_cluster of them. Returns (domain, nodes, opportunity)
    sorted by opportunity (cluster size) desc — a bigger cluster is a stronger pattern signal.
    Reflections and inferences are excluded (we reflect on facts, not on prior reflections)."""
    by_domain: dict[str, list] = {}
    for n in graph.all_nodes():
        if n.status.memory_state not in ("active", "disputed"):
            continue
        if n.node_type not in _REFLECTABLE:
            continue
        for dom in sorted(n.domains):
            by_domain.setdefault(dom, []).append(n)
    out = [(dom, nodes, len(nodes)) for dom, nodes in by_domain.items() if len(nodes) >= min_cluster]
    out.sort(key=lambda x: (-x[2], x[0]))
    return out


def _parse_obj(raw: str, key: str):
    """Parse a fenced/plain JSON object, returning (value-at-key, confidence) or None."""
    text = _FENCE_RE.search(raw.strip())
    text = text.group(1) if text else raw.strip()
    try:
        d = json.loads(text)
        return str(d[key]).strip(), float(d["confidence"])
    except Exception:
        return None


def _parse_observation(raw: str):
    return _parse_obj(raw, "observation")


def _make_reflection_node(observation: str, cluster: list, confidence: float) -> Node:
    now = datetime.now(timezone.utc).isoformat()
    domains = set().union(*[n.domains for n in cluster]) if cluster else set()
    r = compute_reinforcement(0.0, REFLECT_CSR_FLOOR)
    return Node(
        node_id="node_" + uuid.uuid4().hex[:12], node_type="reflection",
        claim=observation, normalized_claim=normalize_text(observation),
        entities=[], domains=domains, facets=extract_facets(observation),
        predicate=_extract_predicate(observation),
        temporal=Temporal(now, now, None, None, None, None),
        provenance=Provenance("reflection", "+".join(n.node_id for n in cluster),
                              "consolidation_reflect", confidence),
        reinforcement=Reinforcement(0.0, REFLECT_CSR_FLOOR, r),
        status=NodeStatus("active", "ACCEPT", "provisional"),
        motivation=None, context=None, relations=[],
    )


# Causal/conditional markers. A reflection may SUMMARISE a cluster; it must not invent a causal or
# conditional relation the cluster never asserted. Observed failure (212-commit ops run): the cluster
# held "LM Studio is the local compute" and (separately) "if it needs a GPU, the design is wrong";
# REFLECT welded them into "Local/LM Studio needs a GPU if the design is wrong" — false, asserted by
# nobody, and it PASSED the stemmed-grounding guard because every content stem was present somewhere
# in the cluster. Set-intersection grounding cannot see an invented relation, exactly as it could not
# see a transposed one before `grounding.py` was added to the main gate.
_CAUSAL_MARKERS = (
    " if ", " because ", " since ", " therefore ", " thus ", " so that ", " causes ", " caused ",
    " requires ", " required ", " needs ", " needed ", " leads to ", " results in ", " due to ",
    " in order to ", " unless ", " otherwise ", " implies ", " means that ",
)


def _invents_causation(obs: str, claims: list[str]) -> bool:
    """True if the observation asserts a causal/conditional link that no SINGLE claim already made.

    The first version of this guard checked whether the marker word was absent from the cluster.
    It failed on the real example: "Local/LM Studio needs a GPU if the design is wrong" uses `needs`
    and `if`, and BOTH appear in the cluster — inside a different claim, joining different arguments.
    The fabrication does not invent a relation WORD, it re-wires existing ones onto new arguments,
    which is transposition wearing a causal hat.

    So the test is argument-level: split at the marker and require that ONE claim already covers both
    sides. A cluster where "needs a GPU" and "the design is wrong" live in separate claims cannot
    license an observation welding them together.
    """
    o = obs.lower()
    marker = next((m for m in _CAUSAL_MARKERS if m in f" {o} "), None)
    if marker is None:
        return False
    left, _, right = o.partition(marker.strip())
    ls, rs = _stems(left), _stems(right)
    if not ls or not rs:
        return False
    # 0.7, not 0.5. At 0.5 the real fabrication slipped through: the cluster claim supplied
    # "needs a GPU" (exactly half of the left side) while "LM Studio" was imported from a DIFFERENT
    # claim. Half-coverage is the signature of a welded argument, so a licensing claim must own most
    # of both sides. Strict by intent — reflections are the least-verified layer in the store.
    for c in claims:
        cs = _stems(c)
        if (len(ls & cs) / len(ls) >= 0.7) and (len(rs & cs) / len(rs) >= 0.7):
            return False
    return True


def _invents_numbers(obs: str, claims: list[str]) -> bool:
    """A generalisation must not introduce a figure the cluster never stated."""
    from .grounding import numerals
    return not (numerals(obs) <= numerals(" ".join(claims)))


def _too_similar_semantically(obs: str, existing: list[str], threshold: float = 0.90) -> bool:
    """Catch semantic duplicates that lexical novelty structurally cannot.

    The run produced three reflections asserting the same thing; two were 0.71 stem-similar (a lower
    lexical threshold catches those) but one pair sat at 0.44 — lexically distant, semantically
    identical. No lexical threshold reaches that case, so this uses the same pinned encoder the
    entity linker already depends on. Soft: if unavailable, lexical novelty alone still applies.
    """
    try:
        from . import semantic_encoder
    except Exception:
        return False
    for e in existing:
        try:
            if semantic_encoder.similarity(obs, e) >= threshold:
                return True
        except Exception:
            return False
    return False


def _add_derivation_edges(graph, derived, sources) -> None:
    """Record `derived_from` from a derived node to every source it was computed from.

    Erasure needs an UNAMBIGUOUS derivation graph. `supports` (REFLECT) and `supersedes` (COMPRESS)
    already exist, but `supersedes` is overloaded: pipeline._mark_superseded writes it for ordinary
    revision too, so following it during erasure would pull revision chains into a subject's
    closure and over-delete across subjects. A dedicated edge keeps "X was computed from Y"
    separate from "X replaced Y".

    Direction is derived -> source, matching the question erasure asks: given these sources are
    going away, which derived nodes lose their footing?
    """
    for src in sources:
        graph.add_relation(Relation(
            relation_id="rel_" + uuid.uuid4().hex[:12],
            source_id=derived.node_id, target_id=src.node_id,
            relation_type="derived_from", polarity="positive",
            claim=f"{derived.node_id} derived from {src.node_id}", confidence=1.0,
            evidence=RelationEvidence([derived.node_id, src.node_id], []), status="validated",
        ))


def run_reflect(graph: GraphState, llm: Callable[[str], str], *,
                budget: int = REFLECT_BUDGET_PER_TURN,
                min_cluster: int = REFLECT_MIN_CLUSTER) -> list[str]:
    """LLM PROPOSES an evidence->pattern observation over a fact cluster; CODE DECIDES via
    overreach + coverage + grounding + restatement + novelty guards. Creates a tentative
    `reflection` node (low r) with `supports` edges from each evidence node. Returns created ids."""
    created: list[str] = []
    existing_reflections = [n for n in graph.all_nodes()
                            if n.node_type == "reflection" and n.status.memory_state == "active"]
    for domain, cluster, _score in reflect_candidates(graph, min_cluster=min_cluster)[:budget]:
        claims = [n.claim for n in cluster]
        try:
            parsed = _parse_observation(llm(_REFLECT_PROMPT.format(
                facts="\n".join(f"- {c}" for c in claims))))
        except Exception:
            continue
        if parsed is None:
            continue
        obs, confidence = parsed
        if not obs or confidence < SYNTHESIS_MIN_CONFIDENCE:
            continue
        if is_personality_claim(obs):              # overreach guard (evidence->pattern, not trait-guess)
            continue
        obs_stems = _stems(obs)
        if not obs_stems:
            continue
        if sum(1 for c in claims if jaccard_score(obs, c) > 0) < 2:   # must be ABOUT the set (>=2)
            continue
        src_stems = set().union(*[_stems(c) for c in claims])
        if len(obs_stems & src_stems) / len(obs_stems) < REFLECT_MIN_GROUNDING:  # grounded in cluster (stemmed)
            continue
        if any(obs_stems <= _stems(c) for c in claims):     # not a restatement of one fact
            continue
        if _invents_causation(obs, claims):        # no fabricated "X needs Y if Z" over the cluster
            continue
        if _invents_numbers(obs, claims):          # no figures the cluster never stated
            continue
        # novelty: lexical threshold lowered 0.80 -> REFLECT_NOVELTY_MAX after three near-duplicate
        # reflections got through at 0.71, then a semantic check for the duplicates lexical misses.
        if any(jaccard_score(obs, r.claim) >= REFLECT_NOVELTY_MAX for r in existing_reflections):
            continue
        if _too_similar_semantically(obs, [r.claim for r in existing_reflections]):
            continue
        node = _make_reflection_node(obs, cluster, confidence)
        graph.add_node(node)
        for src in cluster:
            graph.add_relation(Relation(
                relation_id="rel_" + uuid.uuid4().hex[:12],
                source_id=src.node_id, target_id=node.node_id,
                relation_type="supports", polarity="positive",
                claim=f"{src.node_id} supports {node.node_id}", confidence=1.0,
                evidence=RelationEvidence([src.node_id], [src.claim]), status="validated",
            ))
        _add_derivation_edges(graph, node, cluster)
        existing_reflections.append(node)
        created.append(node.node_id)
    return created


# ── COMPRESS — fold COLD forgettable clutter into one grounded summary ──────────

_COMPRESS_PROMPT = (
    "Here are several minor, low-importance facts about one person that are fading from memory. "
    "Condense them into ONE short summary sentence that preserves their gist.\n"
    "RULES: reuse the facts' own key words; do not add new facts, motives, or feelings; do not "
    "begin with 'The system'/'The facts'. If they don't belong together, use confidence 0.0.\n\n"
    "Facts:\n{facts}\n\n"
    'Respond with JSON only: {{"summary": "<condensed sentence>", "confidence": <0.0-1.0>}}'
)


def compress_candidates(graph, significance: dict[str, SignificanceState], frozen: set[str], *,
                        max_r: float = COMPRESS_MAX_R,
                        min_group: int = COMPRESS_MIN_GROUP) -> list[tuple[str, list, int]]:
    """COLD, forgettable, MEANINGLESS fact clusters (grouped by domain, >= min_group). Excludes
    anything protected: frozen (W4), disputed (W3 — only `active` is taken), inference/reflection
    nodes, and nodes carrying grounded significance (W1). Sorted by cluster size desc."""
    by_domain: dict[str, list] = {}
    for n in graph.all_nodes():
        if n.status.memory_state != "active":           # not disputed/superseded
            continue
        if n.node_type in ("inference", "reflection"):
            continue
        if n.node_id in frozen:
            continue
        st = significance.get(n.node_id)
        if st is not None and not st.is_empty():        # meaningful -> never compressed
            continue
        if n.reinforcement.overall >= max_r:            # only genuinely COLD clutter
            continue
        for dom in sorted(n.domains):
            by_domain.setdefault(dom, []).append(n)
    out = [(dom, nodes, len(nodes)) for dom, nodes in by_domain.items() if len(nodes) >= min_group]
    out.sort(key=lambda x: (-x[2], x[0]))
    return out


def _make_summary_node(summary: str, cluster: list, confidence: float) -> Node:
    now = datetime.now(timezone.utc).isoformat()
    domains = set().union(*[n.domains for n in cluster]) if cluster else set()
    r = compute_reinforcement(0.0, COMPRESS_CSR_FLOOR)
    return Node(
        node_id="node_" + uuid.uuid4().hex[:12], node_type="summary",
        claim=summary, normalized_claim=normalize_text(summary),
        entities=[], domains=domains, facets=extract_facets(summary),
        predicate=_extract_predicate(summary),
        temporal=Temporal(now, now, None, None, None, None),
        provenance=Provenance("compression", "+".join(n.node_id for n in cluster),
                              "consolidation_compress", confidence),
        reinforcement=Reinforcement(0.0, COMPRESS_CSR_FLOOR, r),
        status=NodeStatus("active", "ACCEPT", "provisional"),
        motivation=None, context=None, relations=[],
    )


def run_compress(graph: GraphState, llm: Callable[[str], str],
                 significance: dict[str, SignificanceState], frozen: set[str], *,
                 budget: int = COMPRESS_BUDGET_PER_TURN,
                 min_group: int = COMPRESS_MIN_GROUP) -> list[str]:
    """LLM PROPOSES a condensed summary of a COLD cluster; CODE DECIDES via overreach +
    grounding (stemmed) + restatement guards. On admission: create the summary node and mark
    each source `superseded` by it (non-destructive — provenance kept, clutter hidden from
    retrieval), with a `supersedes` edge summary->source. Returns created summary ids."""
    from dataclasses import replace
    created: list[str] = []
    for domain, cluster, _score in compress_candidates(graph, significance, frozen, min_group=min_group)[:budget]:
        claims = [n.claim for n in cluster]
        try:
            parsed = _parse_obj(llm(_COMPRESS_PROMPT.format(
                facts="\n".join(f"- {c}" for c in claims))), "summary")
        except Exception:
            continue
        if parsed is None:
            continue
        summary, confidence = parsed
        if not summary or confidence < SYNTHESIS_MIN_CONFIDENCE:
            continue
        if is_personality_claim(summary):
            continue
        sm_stems = _stems(summary)
        if not sm_stems:
            continue
        src_stems = set().union(*[_stems(c) for c in claims])
        if len(sm_stems & src_stems) / len(sm_stems) < COMPRESS_MIN_GROUNDING:  # stay close to source
            continue
        node = _make_summary_node(summary, cluster, confidence)
        now = node.temporal.asserted_at
        graph.add_node(node)
        _add_derivation_edges(graph, node, cluster)
        for src in cluster:
            graph.update_node(replace(
                src, status=replace(src.status, memory_state="superseded"),
                temporal=replace(src.temporal, valid_to=now, superseded_by=node.node_id)))
            graph.add_relation(Relation(
                relation_id="rel_" + uuid.uuid4().hex[:12],
                source_id=node.node_id, target_id=src.node_id,
                relation_type="supersedes", polarity="positive",
                claim=f"{node.node_id} compresses {src.node_id}", confidence=1.0,
                evidence=RelationEvidence([node.node_id, src.node_id], [src.claim]), status="validated",
            ))
        created.append(node.node_id)
    return created


# ── Scheduler ──────────────────────────────────────────────────────────────────

def run_consolidation_ops(
    graph,
    significance: dict[str, SignificanceState],
    frozen: set[str],
    *,
    turn: int,
    freeze_budget: int = FREEZE_BUDGET_PER_TURN,
    reflect_llm: Callable[[str], str] | None = None,
    reflect_budget: int = REFLECT_BUDGET_PER_TURN,
    compress_budget: int = COMPRESS_BUDGET_PER_TURN,
) -> ConsolidationOpsReport:
    """Opportunity-scored background metabolism for one consolidation turn. Mutates `frozen`
    in place (the consolidator owns it, like the decay bookkeeping). Returns what fired.

    Every op follows the same shape: score candidates by opportunity -> rank -> spend a budget.
    FREEZE is deterministic (always on); REFLECT and COMPRESS run only when reflect_llm is
    supplied (the LLM proposes; the guards decide). Order: FREEZE (protect) -> REFLECT (observe)
    -> COMPRESS (forget clutter), so protected/meaningful nodes are never compressed."""
    report = ConsolidationOpsReport()
    for node_id, _opportunity in freeze_candidates(graph, significance, frozen)[:freeze_budget]:
        frozen.add(node_id)
        report.frozen.append(node_id)
    if reflect_llm is not None:
        report.reflections = run_reflect(graph, reflect_llm, budget=reflect_budget)
        report.compressions = run_compress(graph, reflect_llm, significance, frozen,
                                           budget=compress_budget)
    return report
