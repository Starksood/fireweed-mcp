"""Deterministic graph mutation resolver. No LLM calls. Same input → same output."""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Union

from .claim import Claim
from .graph import (
    GraphState, Node, normalize_text,
    EntityRef, Predicate, Motivation, MemoryContext,
    Temporal, Provenance, Reinforcement, NodeStatus, RelationRef,
)
from .domain_classifier import extract_facets
from .constants import (
    POLARITY_OPPOSITION_THRESHOLD,
    SCOPE_MATCH_THRESHOLD,
    CONTRADICTION_MAX_TIME_GAP_DAYS,
    MODIFY_CONTRADICTION_CONFIDENCE_THRESHOLD,
    CANON_MERGE_MIN_JACCARD,
)

# ── Mutation result types ─────────────────────────────────────────────────────

@dataclass
class DecisionTrace:
    decision_path: list[str]
    matched_node_id: str | None

@dataclass
class CreateMutation:
    node: Node
    trace: DecisionTrace

@dataclass
class DedupMutation:
    existing_node_id: str
    trace: DecisionTrace

@dataclass
class ReinforceMutation:
    target_node_id: str
    trace: DecisionTrace

@dataclass
class ModifyMutation:
    old_node_id: str
    new_node: Node
    trace: DecisionTrace

@dataclass
class DisputeMutation:
    """A contradiction held as STANDING TENSION rather than collapsed to supersession (Stage 3,
    W3). The existing claim stays live; the opposing claim is added; both are marked `disputed`
    and linked by a `contradicts` edge. Used when an opposition carries NO temporal-change
    signal — the person holds opposed things (ambivalence / context-dependence), not an update.
    "Opposites are linked." A change-signalled opposition still routes to ModifyMutation."""
    existing_node_id: str
    new_node: Node
    trace: DecisionTrace

@dataclass
class NoopMutation:
    reason: str
    trace: DecisionTrace

Mutation = Union[CreateMutation, DedupMutation, ReinforceMutation, ModifyMutation, DisputeMutation, NoopMutation]

# ── Private helpers ───────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    union = wa | wb
    return 0.0 if not union else len(wa & wb) / len(union)


_CONTENT_RE = re.compile(r"[^\w]+")


def _content_jaccard(a: str, b: str) -> float:
    """Jaccard over content tokens (len>2), dropping articles/copulas. Used by the
    entity-keyed merge to tell paraphrases of one fact ("nurse at Riverside" vs
    "nurse at the Riverside clinic", ~0.6) from different predications of the same
    entities ("nurse" vs "doctor", ~0.5). Mirrors retrieval/perceiver tokenizers."""
    ta = {t for t in _CONTENT_RE.sub(" ", a.lower()).split() if len(t) > 2}
    tb = {t for t in _CONTENT_RE.sub(" ", b.lower()).split() if len(t) > 2}
    union = ta | tb
    return 0.0 if not union else len(ta & tb) / len(union)

_TEMPORAL_SIGNALS = [
    "now ", "moved to", "changed from", "no longer", "used to",
    "switched to", "left ", "quit ", "started ", "recently ",
    "from now", "as of", "not anymore",
]

def _has_temporal_signal(text: str) -> bool:
    lower = text.lower()
    return any(s in lower for s in _TEMPORAL_SIGNALS)

_PREFERENCE_VERBS = {"likes", "prefers", "enjoys", "loves", "hates", "wants", "avoids", "craves", "dislikes", "favors"}
_CONSTRAINT_MARKERS = {"must", "cannot", "never", "only", "won't", "can't", "not allowed", "always has to", "required to"}
_STATE_VERBS = {"is", "are", "has", "lives", "works", "owns", "stays", "remains", "resides", "belongs"}
_TIME_MARKERS = {
    "yesterday", "last week", "last month", "last year",
    "on monday", "on tuesday", "on wednesday", "on thursday",
    "on friday", "on saturday", "on sunday",
    "in january", "in february", "in march", "in april", "in may", "in june",
    "in july", "in august", "in september", "in october", "in november", "in december",
    "this morning", "tonight", "earlier today",
}

def _classify_node_type(text: str) -> str:
    words = set(text.lower().split())
    lower = text.lower()
    if words & _PREFERENCE_VERBS:
        return "preference"
    if any(m in lower for m in _CONSTRAINT_MARKERS):
        return "constraint"
    if any(w.endswith("ed") for w in words) and any(m in lower for m in _TIME_MARKERS):
        return "event"
    if words & _STATE_VERBS:
        return "state"
    return "fact"

_NEGATION_WORDS = {"not", "no", "never", "doesn't", "don't", "won't", "can't", "cannot", "isn't", "aren't", "no longer"}
_NEGATIVE_SENTIMENT_WORDS = {"hates", "hate", "dislikes", "dislike", "despises", "despise", "abhors", "abhor", "rejects", "reject", "avoids", "avoid"}

def _extract_predicate(text: str) -> Predicate:
    words = text.split()
    lw = [w.lower().rstrip(".,!?") for w in words]
    # Detect negative polarity from explicit negation or negative sentiment verbs
    polarity = "positive"
    if set(lw) & _NEGATION_WORDS:
        polarity = "negative"
    elif set(lw) & _NEGATIVE_SENTIMENT_WORDS:
        polarity = "negative"
    lemma = "unknown"
    for w in lw:
        if w.endswith(("s", "ed", "ing", "es")) and len(w) > 3:
            if w.endswith("ing") and len(w) > 5:
                lemma = w[:-3]
            elif w.endswith("ed") and len(w) > 4:
                lemma = w[:-2]
            elif w.endswith("es") and len(w) > 4:
                lemma = w[:-2]
            elif w.endswith("s") and len(w) > 3:
                lemma = w[:-1]
            else:
                lemma = w
            break
    obj = None
    for i, w in enumerate(lw):
        if w in (lemma, lemma + "s", lemma + "ed"):
            if i + 1 < len(lw):
                obj = lw[i + 1]
            break
    return Predicate(lemma=lemma, polarity=polarity, object=obj)

_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
_WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

def _extract_temporal(text: str, session_timestamp: str) -> tuple[str | None, str | None]:
    """Return (event_time, valid_from). Both may be None.
    Pure function — no clock reads, no globals."""
    try:
        anchor = datetime.fromisoformat(session_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None, None
    lower = text.lower()
    if "yesterday" in lower:
        d = (anchor - timedelta(days=1)).date().isoformat(); return d, d
    if "last week" in lower:
        d = (anchor - timedelta(days=7)).date().isoformat(); return d, d
    if "last month" in lower:
        try:
            from dateutil.relativedelta import relativedelta
            d = (anchor - relativedelta(months=1)).strftime("%Y-%m")
        except ImportError:
            d = (anchor - timedelta(days=30)).strftime("%Y-%m")
        return d, d
    if "last year" in lower:
        d = str(anchor.year - 1); return d, d
    if any(p in lower for p in ("this morning", "earlier today", "tonight")):
        d = anchor.date().isoformat(); return d, d
    m = re.search(r"\bin\s+(" + "|".join(_MONTH_MAP.keys()) + r")\s+(\d{4})\b", lower)
    if m:
        d = f"{m.group(2)}-{_MONTH_MAP[m.group(1)]}"; return d, d
    m = re.search(r"\bon\s+(" + "|".join(_WEEKDAY_MAP.keys()) + r")\b", lower)
    if m:
        days_back = (anchor.weekday() - _WEEKDAY_MAP[m.group(1)]) % 7 or 7
        d = (anchor - timedelta(days=days_back)).date().isoformat(); return d, d
    return None, None

def _generate_node_id(normalized_claim: str,
                      entity_ids: list[str],
                      session_timestamp: str) -> str:
    """Deterministic node ID from claim content + entity context + session anchor.

    Same input → same node_id. Two ingests of identical claims in different sessions
    yield different IDs because session_timestamp differs (correct: each is its own
    evidence anchor). Two ingests of identical claims in the same session yield
    identical IDs (caught by DEDUP path; CREATE path should not encounter this).
    """
    payload = f"{normalized_claim}|{','.join(sorted(entity_ids))}|{session_timestamp}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"node_{digest}"


# ── Temporal contradiction detection (X5) ────────────────────────────────────

def _jaccard_similarity(a: str, b: str) -> float:
    """Lexical Jaccard over word sets — the zero-dependency default similarity backend."""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return intersection / union if union > 0 else 0.0


# ── Pluggable similarity backend (NorthStar guardrail 1, amended 2026-07-06) ─────────────────────
# The claim-level "are these the same fact?" similarity is swappable. Default = lexical Jaccard
# (unchanged behavior, zero deps). A PINNED, hash-verified open-weights encoder can be swapped in —
# deterministic != lexical — for robustness to paraphrase / morphology / negation / non-English, which
# fixes the brittleness LoCoMo exposed. The strict verbatim dedup gates (_jaccard / _content_jaccard)
# stay lexical on purpose. Process-global: set it once at startup (a benchmark or service init).
_similarity_backend = _jaccard_similarity


def similarity(a: str, b: str) -> float:
    return _similarity_backend(a, b)


def set_similarity_backend(fn) -> None:
    global _similarity_backend
    _similarity_backend = fn


def use_semantic_similarity() -> None:
    """Switch to the pinned semantic encoder (imports the soft dependency lazily)."""
    from . import semantic_encoder
    set_similarity_backend(semantic_encoder.similarity)


def use_lexical_similarity() -> None:
    set_similarity_backend(_jaccard_similarity)

def _extract_scope(text: str) -> str:
    """Extract object/scope from claim text (typically the object of the predicate).

    Heuristic: return the object part, skipping subject + verb.
    E.g., "Maya likes spicy food" → "spicy food"
    E.g., "Maya likes food" → "food"
    For phrases: "Subject Verb Object..." - return everything after the verb.
    """
    words = text.split()
    if len(words) <= 2:
        return text

    # For 3 words "Subject Verb Object" - return just the object
    if len(words) == 3:
        return words[2]

    # For 4+ words "Subject Verb Object[...]" - return object and beyond
    # Return last 2 words as a heuristic
    return ' '.join(words[-2:])

def _compute_polarity_opposition(claim_polarity: str | None, node_polarity: str | None) -> float:
    """Score how opposite the polarities are (0.0–1.0)."""
    ANTONYM_PAIRS = {
        'positive': {'negative', 'dislikes', 'hates', 'disdain'},
        'negative': {'positive', 'likes', 'loves', 'adores'},
        'warm': {'cold', 'cool'},
        'cold': {'warm', 'hot'},
        'likes': {'hates', 'dislikes', 'dislike'},
        'hates': {'likes', 'loves', 'adores'},
        'loves': {'hates', 'dislikes'},
        'dislikes': {'loves', 'likes', 'adores'},
    }

    if claim_polarity is None or node_polarity is None:
        return 0.0

    claim_p = claim_polarity.lower()
    node_p = node_polarity.lower()

    # Direct antonym
    if claim_p in ANTONYM_PAIRS:
        if node_p in ANTONYM_PAIRS[claim_p]:
            return 1.0

    # Same polarity
    if claim_p == node_p:
        return 0.0

    # Unknown polarity (partial opposition)
    if 'unknown' in (claim_p, node_p):
        return 0.3

    return 0.0

def _compute_scope_match(claim_text: str, node_text: str) -> float:
    """Score scope match (do the claims have the same scope/object?)."""
    claim_scope = _extract_scope(claim_text)
    node_scope = _extract_scope(node_text)

    if not claim_scope or not node_scope:
        return 0.5

    # Exact match
    if claim_scope.lower() == node_scope.lower():
        return 1.0

    # Substring match (spicy food vs spicy)
    if claim_scope.lower() in node_scope.lower() or \
       node_scope.lower() in claim_scope.lower():
        return 0.8

    # Semantic similarity
    semantic_sim = similarity(claim_scope, node_scope)
    return min(semantic_sim * 1.5, 1.0)

def _find_contradiction_candidates(claim_text: str, graph: GraphState,
                                   resolved_entities: list[EntityRef]) -> list[Node]:
    """Find nodes that might be contradicted by the new claim."""
    candidates = []

    # Get primary subject entity
    claim_subject = resolved_entities[0].entity_id if resolved_entities else None
    if not claim_subject:
        return []

    # Search for nodes with same subject
    for node in graph.all_nodes():
        # Must mention same entity
        if not any(e.entity_id == claim_subject for e in node.entities):
            continue

        # Skip non-active nodes
        if node.status.memory_state != "active":
            continue

        # Only predicates can contradict
        if node.node_type not in ("preference", "state", "constraint"):
            continue

        # Skip if node is too old (contradictions must be temporally close)
        try:
            node_time = datetime.fromisoformat(node.temporal.asserted_at.replace("Z", "+00:00"))
            # We don't have the claim timestamp here, so use a reasonable heuristic
            # Skip if older than the max gap (we'll refine this in resolve_mutation)
        except (ValueError, AttributeError):
            continue

        # Compute similarity
        # Lower bound relaxed to 0.30 to catch real contradictions like "hates spicy food" vs "likes food"
        sim = similarity(claim_text, node.claim)
        if sim < 0.30 or sim > 0.95:
            continue

        candidates.append(node)

    return candidates

def _detect_contradiction(claim_text: str, claim_polarity: str | None,
                         resolved_entities: list[EntityRef],
                         graph: GraphState,
                         session_timestamp: str) -> tuple[bool, str | None, float]:
    """
    Detect if claim contradicts an existing node.

    Returns:
        (is_contradiction, node_id, confidence)
    """
    candidates = _find_contradiction_candidates(claim_text, graph, resolved_entities)
    if not candidates:
        return (False, None, 0.0)

    try:
        claim_time = datetime.fromisoformat(session_timestamp.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return (False, None, 0.0)

    best_contradiction = None
    best_confidence = 0.0

    for candidate_node in candidates:
        # Check polarity opposition
        polarity_score = _compute_polarity_opposition(
            claim_polarity,
            candidate_node.predicate.polarity
        )

        if polarity_score < POLARITY_OPPOSITION_THRESHOLD:
            continue

        # Verify temporal ordering (new claim must be AFTER candidate node)
        try:
            node_time = datetime.fromisoformat(candidate_node.temporal.asserted_at.replace("Z", "+00:00"))
            temporal_valid = claim_time > node_time
            time_gap_days = (claim_time - node_time).days

            if not temporal_valid or time_gap_days > CONTRADICTION_MAX_TIME_GAP_DAYS:
                continue
        except (ValueError, AttributeError):
            continue

        # Check scope match
        scope_score = _compute_scope_match(claim_text, candidate_node.claim)
        if scope_score < SCOPE_MATCH_THRESHOLD:
            continue

        # Compute final confidence
        sim_score = similarity(claim_text, candidate_node.claim)
        confidence = (
            0.4 * polarity_score +
            0.3 * scope_score +
            0.2 * sim_score +
            0.1 * (1.0 if temporal_valid else 0.0)
        )

        if confidence > best_confidence:
            best_confidence = confidence
            best_contradiction = candidate_node.node_id

    if best_contradiction and best_confidence >= MODIFY_CONTRADICTION_CONFIDENCE_THRESHOLD:
        return (True, best_contradiction, best_confidence)

    return (False, None, 0.0)

def _grounding_class(claim) -> str | None:
    """The provenance class for a claim being committed. None when there is no evidence span to
    classify against (turn-based sources that predate evidence binding)."""
    from .grounding import classify
    span = getattr(claim, "evidence_span", None)
    return classify(claim.claim, span) if span else None


def _build_node(claim: Claim, resolved_domains: set[str], firewall_decision: str,
                resolved_entities: list[EntityRef], session_timestamp: str) -> Node:
    now = datetime.now(timezone.utc).isoformat()
    normalized = normalize_text(claim.claim)
    node_id = _generate_node_id(normalized, [e.entity_id for e in resolved_entities], session_timestamp)
    event_time, valid_from = _extract_temporal(claim.claim, session_timestamp)
    return Node(
        node_id=node_id, node_type=_classify_node_type(claim.claim),
        claim=claim.claim, normalized_claim=normalized,
        entities=resolved_entities, domains=resolved_domains, facets=extract_facets(claim.claim),
        predicate=_extract_predicate(claim.claim),
        temporal=Temporal(asserted_at=now, stored_at=now, event_time=event_time,
                          valid_from=valid_from, valid_to=None, superseded_by=None),
        provenance=Provenance(source_turn_id=claim.source_turn_id,
                              source_span=claim.evidence_span,
                              extraction_method="llm_candidate_plus_firewall",
                              confidence=claim.confidence,
                              # derived, not plumbed: the class is a pure function of the claim and
                              # the span, both already here, so no write path needs a new argument
                              grounding_class=_grounding_class(claim)),
        reinforcement=Reinforcement(local_frequency=0.0, cross_session_recurrence=0.0, overall=0.0),
        status=NodeStatus(memory_state="active", firewall_decision=firewall_decision,
                          validation_state="provisional"),
        motivation=None, context=None, relations=[],
    )

# ── Main resolver function ────────────────────────────────────────────────────

# Types that describe a CURRENT DISPOSITION, and can therefore be revised out of existence by a
# later change-signalled contradiction. Everything else (`fact`, `event`) is a historical record:
# it can be DISPUTED by a conflicting claim, but not un-happened.
_REVISABLE_TYPES = ("state", "preference", "constraint")


def resolve_mutation(
    claim: Claim,
    resolved_domains: set[str],
    resolved_entities: list[EntityRef],
    session_timestamp: str,
    graph: GraphState,
) -> Mutation:
    """Pure function: reads graph, returns a mutation. Never mutates the graph."""
    # Stage 1 — NOOP gates
    if claim.claim.strip() == "":
        return NoopMutation(reason="empty_claim", trace=DecisionTrace(["noop_empty_claim"], None))
    if claim.confidence < 0.30:
        return NoopMutation(reason="low_confidence", trace=DecisionTrace(["noop_low_confidence"], None))

    # Stage 1.5 — Check for temporal contradiction (X5)
    path: list[str] = ["noop_check_passed"]
    polarity = _extract_predicate(claim.claim).polarity
    (is_contradiction, contradicted_node_id, confidence) = _detect_contradiction(
        claim.claim, polarity, resolved_entities, graph, session_timestamp
    )
    if is_contradiction and contradicted_node_id:
        path.append(f"contradiction_detected_{contradicted_node_id}_conf_{confidence:.2f}")
        new_node = _build_node(claim, resolved_domains, "ACCEPT", resolved_entities, session_timestamp)
        # W3: a contradiction is an UPDATE only if the new claim signals a change ("now", "no
        # longer", "switched"...). Otherwise it is STANDING TENSION — hold both, don't collapse.
        #
        # The type gate is the same KIND Stage 4 uses, and for the same reason. Restricting Stage 4
        # alone left this path open: the 212-commit ops re-run still superseded 167 `fact` and 4
        # `constraint` nodes at 100 commits, and since `_mark_superseded` is reachable only from
        # ModifyMutation, and Stage 4 can no longer fire on a non-state node, every one of those
        # came through here. A contradiction against a revisable current-status is an update; a
        # contradiction against a fact or an event is a DISPUTE — the claim that "Run D achieved
        # 70%" is not erased by a later conflicting measurement, it is put in tension with it, and
        # holding both is exactly what the disputed state exists for.
        #
        # The line is current-disposition vs historical-record, NOT state-only: the suite caught
        # that "Maya likes spicy food" -> "Maya now hates spicy food" classifies as `preference`
        # and is exactly the X5 revision this branch exists to serve. Stage 4 stays state-only on
        # purpose — it fires on a merely-RELATED node with no conflict detected, which is far
        # weaker evidence than a detected contradiction and needs the tighter gate.
        contradicted = graph.get_node(contradicted_node_id)
        if _has_temporal_signal(claim.claim) and contradicted is not None \
                and contradicted.node_type in _REVISABLE_TYPES:
            return ModifyMutation(old_node_id=contradicted_node_id, new_node=new_node,
                                trace=DecisionTrace(path + ["modify_contradiction"], new_node.node_id))
        return DisputeMutation(existing_node_id=contradicted_node_id, new_node=new_node,
                               trace=DecisionTrace(path + ["dispute_contradiction"], new_node.node_id))

    # Stage 2 — Exact match → DEDUP
    exact = graph.find_exact_match(claim.claim)
    if exact:
        path.append("exact_match_found")
        return DedupMutation(existing_node_id=exact.node_id, trace=DecisionTrace(path, exact.node_id))
    path.append("no_exact_match")

    # Stage 3 — Find related candidates
    related = graph.find_related(claim.claim, resolved_domains, limit=5)
    if not related:
        path.append("no_related_found")
        node = _build_node(claim, resolved_domains, "ACCEPT", resolved_entities, session_timestamp)
        return CreateMutation(node=node, trace=DecisionTrace(path + ["create"], node.node_id))
    path.append("related_found")

    # Stage 4 — Evaluate candidates
    for node in related:
        # X4: For "other" domain (unknown but structurally valid), skip domain intersection check
        # Allow matching on entity identity and semantic similarity instead of domain overlap
        if resolved_domains != {"other"} and not (resolved_domains & node.domains):
            continue
        # A temporal signal supersedes a merely-RELATED node only when both sides are STATE claims.
        # Measured defect (212-commit ops-history run): 777 superseded vs 176 active — 94% of all
        # facts destroyed. 33% of commit messages contain a temporal word ("now", "already", "new"),
        # and this branch fired on the first lexically-related node with no conflict check at all, so
        # "The benchmark includes a runner" was superseded by an unrelated later claim about the
        # benchmark. A state can be revised ("X now lives in Y" replaces "X lives in Z"); a fact or
        # event cannot be un-happened, and conjunctive facts ("includes a runner", "includes a
        # scorer") are both true and must coexist. Restricting to state->state prevents the 504
        # fact/constraint supersessions in that run while keeping the 269 state revisions.
        # Only the OLD node's type gates this. Requiring the NEW claim to be a state too was too
        # strict and the suite caught it: "Maya moved to Seattle" classifies as `fact` (past tense,
        # no time marker) yet legitimately revises "Maya lives in Portland". What matters is whether
        # the thing being replaced is a revisable current-status, not how the replacement is phrased.
        if _has_temporal_signal(claim.claim) and node.node_type == "state":
            path.append(f"temporal_signal_on_{node.node_id}")
            new_node = _build_node(claim, resolved_domains, "ACCEPT", resolved_entities, session_timestamp)
            return ModifyMutation(old_node_id=node.node_id, new_node=new_node,
                                  trace=DecisionTrace(path + ["modify"], new_node.node_id))
        # Both merge paths below are entity-aware: a candidate that resolves to a
        # DIFFERENT non-empty entity set is a different referent and must not merge,
        # even at high lexical overlap. This is what keeps "Maya lives in Paris" and
        # "Maya lives in Paris, Texas" (distinct entities) apart — referent identity
        # is the entity linker's call. Empty-vs-anything is allowed (entity-free
        # claims still merge on text).
        claim_ents = {e.entity_id for e in resolved_entities}
        node_ents = {e.entity_id for e in node.entities}
        entities_conflict = bool(claim_ents) and bool(node_ents) and claim_ents != node_ents

        j = _jaccard(normalize_text(claim.claim), node.normalized_claim)
        if j >= 0.80 and not entities_conflict:
            path.append(f"semantic_match_{node.node_id}_jaccard_{j:.2f}")
            return ReinforceMutation(target_node_id=node.node_id,
                                     trace=DecisionTrace(path + ["reinforce"], node.node_id))
        # Entity-keyed canonical merge: when the candidate resolves to the SAME entity
        # set and shares polarity, surface paraphrases of one fact ("a nurse at
        # Riverside" / "a nurse at the Riverside clinic") should REINFORCE, not split,
        # so a recurring fact actually accumulates reinforcement. The content-Jaccard
        # floor keeps different predications of the same entities apart ("nurse" vs
        # "doctor"). Runs after the temporal and contradiction stages, so
        # updates/contradictions still route to MODIFY.
        if (claim_ents and claim_ents == node_ents
                and polarity == node.predicate.polarity
                and _content_jaccard(claim.claim, node.normalized_claim) >= CANON_MERGE_MIN_JACCARD):
            path.append(f"entity_canonical_match_{node.node_id}")
            return ReinforceMutation(target_node_id=node.node_id,
                                     trace=DecisionTrace(path + ["reinforce_entity_canonical"], node.node_id))

    # Stage 5 — Default CREATE (conservative bias)
    node = _build_node(claim, resolved_domains, "ACCEPT", resolved_entities, session_timestamp)
    return CreateMutation(node=node, trace=DecisionTrace(path + ["create_default"], node.node_id))
