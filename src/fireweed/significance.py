"""Significance — the meaning-bearing M axis of a memory (Stage 3, W1).

Where this sits in the thesis
-----------------------------
The original Fireweed design called for three-axis reinforcement, T/R/M:

    T (temporal)      how recently a memory was touched   -> decay side-table
    R (reinforcement) how often a fact recurs             -> Reinforcement.overall
    M (significance)  how much it MEANS                    -> this module

R says a fact came up a lot. M says *why it matters* and *what it causes / is caused
by*. A fact-store that only has R is a frequency table; a field-like self needs M.
This is the difference Stage 3 is built to make: significance, not just sign.

NB: this is NOT `bench/significance_population.py`. That harness "populates" the
fabric at SCALE (30 entities x 10 domains) to test read-consistency. This module
populates the MEANING of individual memories. Different sense of the word.

LLM proposes, CODE decides (the load-bearing invariant)
-------------------------------------------------------
The perceiver may PROPOSE, for a claim, a `rationale` ("why this matters") and a
`cause` ("the source clause this claim is caused-by"). Nothing it proposes is trusted
blind. This module is the CODE that DECIDES admission, by grounding against the source:

  * rationale — its content tokens (minus a small significance-framing stoplist) must
    be >= RATIONALE_MIN_GROUNDING present in the source text. A motive built from words
    the source never used is a fabrication and is REJECTED. "A fabricated motive is
    worse than a fact-list."
  * cause — admitted only if (1) a causal connective actually occurs in the source (we
    never invent causality the text does not state), (2) the cause clause is a real span
    of the source (>= CAUSE_MIN_CONTAINMENT of its content tokens occur there), and (3)
    it is a clause distinct from the claim itself.

Zero blast radius
-----------------
Significance lives in a side-table `dict[node_id -> SignificanceState]`, owned by the
consolidator, exactly like `decay.DecayState`. It NEVER mutates `Reinforcement.overall`
(the R axis) or the Node schema. `significance_weight(r, None)` returns `r` unchanged,
so every read/instrument that doesn't opt in behaves exactly as it did pre-W1 — M only
ever *adds* meaning on top of recurrence, it never overwrites it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .constants import (
    RATIONALE_MIN_GROUNDING,
    CAUSE_MIN_CONTAINMENT,
    SIGNIFICANCE_RATIONALE_WEIGHT,
    SIGNIFICANCE_CAUSAL_WEIGHT,
    SIGNIFICANCE_CAUSAL_SATURATION,
)

_TOKEN_RE = re.compile(r"[^\w]+")


def _content_tokens(text: str) -> set[str]:
    """Content tokens (len>2). Mirrors perceiver/retrieval tokenizers so perception,
    grounding, and recall all agree on what counts as 'content'."""
    return {t for t in _TOKEN_RE.sub(" ", text.lower()).split() if len(t) > 2}


# Words a rationale may use to FRAME significance without them counting toward (or
# against) grounding — they describe salience, not content. Kept small and explicit so
# the guard cannot be gamed by padding a rationale with importance-words.
_FRAMING_STOPWORDS = frozenset({
    "important", "matters", "because", "significant", "significance", "reflects",
    "shows", "showing", "means", "meaning", "key", "core", "central", "identity",
    "self", "this", "that", "the", "an", "are", "for", "and", "its", "their",
    "her", "his", "about", "why", "how", "what", "who", "to", "of", "it", "is",
})

# Causal connectives that LICENSE extracting a cause clause. If none appears in the
# source, there is no stated causality to capture and `cause` is refused outright.
# ", so " (comma-so) is the high-precision consequence marker that dominates natural
# speech ("the route got rerouted, so now I switch at Civic Center") — a live gemma-3-1b
# run showed the earlier narrow "so she/he" variants missed almost all real causality.
_CAUSAL_CONNECTIVES = (
    "because", "since", "due to", "as a result", "so that", "in order to",
    "therefore", "led to", "leads to", "caused", "causes", "resulted in",
    "owing to", "thanks to", ", so ", "which is why",
)


# ── Side-table state ──────────────────────────────────────────────────────────

@dataclass
class SignificanceState:
    """Per-node significance bookkeeping. Mirrors decay.DecayState: an orthogonal axis
    tracked OUTSIDE the Node schema so the existing pipeline is untouched."""
    rationale: str | None = None                       # admitted, grounded "why this matters"
    rationale_grounding: float = 0.0                   # frac of rationale content tokens found in source
    causes: list[str] = field(default_factory=list)    # admitted, grounded cause clauses (-> causes edges)
    source_turn_ids: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.rationale is None and not self.causes

    def as_dict(self) -> dict:
        return {
            "rationale": self.rationale,
            "rationale_grounding": round(self.rationale_grounding, 3),
            "causes": list(self.causes),
            "source_turn_ids": list(self.source_turn_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SignificanceState":
        return cls(
            rationale=d.get("rationale"),
            rationale_grounding=float(d.get("rationale_grounding", 0.0)),
            causes=list(d.get("causes", [])),
            source_turn_ids=list(d.get("source_turn_ids", [])),
        )


# ── Grounding guards (the "code decides" half) ─────────────────────────────────

def ground_rationale(rationale: str | None, source_text: str,
                     claim: str = "") -> tuple[str | None, float]:
    """Admit a proposed rationale iff it is GROUNDED and EXPLANATORY. Returns (admitted | None, score).

    Score = fraction of the rationale's content tokens (after dropping framing words)
    that occur in the source. Three rejections:
      * pure framing — no content left after the stoplist (asserts importance, grounds nothing);
      * fabrication — grounding < RATIONALE_MIN_GROUNDING (content the source never used);
      * restatement — when a `claim` is supplied, a rationale that contributes no grounded
        content token beyond the claim's own tokens is just the fact reworded, not a *why*.
        (A live gemma-3-1b run admitted several such restatements at grounding 1.0; this gate
        keeps "why: no sunlight" while dropping "why: <the claim, reordered>".)
    """
    if not isinstance(rationale, str) or not rationale.strip():
        return None, 0.0
    src = _content_tokens(source_text)
    rt = _content_tokens(rationale) - _FRAMING_STOPWORDS
    if not rt:
        return None, 0.0
    grounded = rt & src
    grounding = len(grounded) / len(rt)
    if grounding < RATIONALE_MIN_GROUNDING:
        return None, grounding
    if claim and not (grounded - _content_tokens(claim)):
        return None, grounding  # restatement: explains nothing the claim didn't already say
    return rationale.strip(), grounding


def ground_cause(cause: str | None, claim: str, source_text: str) -> str | None:
    """Admit a proposed cause clause iff it is GROUNDED. Returns admitted | None.

    Gate 1: a causal connective must occur in the source (no invented causality).
    Gate 2: the cause clause must be a real span (>= CAUSE_MIN_CONTAINMENT of its
            content tokens occur in the source).
    Gate 3: the cause must be a clause DISTINCT from the claim (a cause explains the
            claim; a restatement of the claim is not a cause).
    """
    if not isinstance(cause, str) or not cause.strip():
        return None
    if not any(c in source_text.lower() for c in _CAUSAL_CONNECTIVES):
        return None
    ct = _content_tokens(cause)
    if not ct:
        return None
    if len(ct & _content_tokens(source_text)) / len(ct) < CAUSE_MIN_CONTAINMENT:
        return None
    if ct <= _content_tokens(claim):  # cause is a subset of the claim -> restatement, not a cause
        return None
    return cause.strip()


def propose_significance(
    rationale: str | None, cause: str | None, claim: str, source_text: str,
) -> tuple[str | None, float, str | None]:
    """Single grounding entry point for the perceiver (which holds the full source).

    Returns (admitted_rationale, rationale_grounding, admitted_cause) — any of which may
    be None/0.0. The perceiver stamps these onto the Percept; the consolidator records
    them verbatim (grounding has already happened, against the full source, here)."""
    g_rat, g_score = ground_rationale(rationale, source_text, claim)
    g_cause = ground_cause(cause, claim, source_text)
    return g_rat, (g_score if g_rat is not None else 0.0), g_cause


# ── The M scalar (the "use it" half) ───────────────────────────────────────────

def significance_weight(r: float, state: SignificanceState | None) -> float:
    """The M value used to weight retrieval ranking and the identity field.

        significance = r * (1 + alpha*rationale_grounding + beta*causal_degree)

    causal_degree = min(n_causes, SAT)/SAT in [0,1]. With no state this is exactly `r`,
    so anything that doesn't opt in is identical to the pre-W1 r-proxy. Significance is
    a MULTIPLIER on recurrence, never a replacement: a meaningful fact and a bare fact of
    equal r are separated by how much grounded meaning the meaningful one carries.
    """
    if state is None or state.is_empty():
        return r
    return r * (1.0 + significance_prior(state))


def significance_prior(state: SignificanceState | None) -> float:
    """Bounded significance lift in [0, RATIONALE_WEIGHT + CAUSAL_WEIGHT], 0 when there is
    no grounded significance. Used as an additive prior in retrieval ranking (the
    rosetta_stone §16.2 "+0.10*motivation_relevance" term) and as the multiplier in
    significance_weight."""
    if state is None or state.is_empty():
        return 0.0
    causal_degree = min(len(state.causes), SIGNIFICANCE_CAUSAL_SATURATION) / SIGNIFICANCE_CAUSAL_SATURATION
    return (SIGNIFICANCE_RATIONALE_WEIGHT * (state.rationale_grounding if state.rationale else 0.0)
            + SIGNIFICANCE_CAUSAL_WEIGHT * causal_degree)


# ── Store accumulation (consolidator side) ─────────────────────────────────────

def record(
    store: dict[str, SignificanceState],
    node_id: str,
    *,
    rationale: str | None = None,
    rationale_grounding: float = 0.0,
    cause: str | None = None,
    source_turn_id: str = "",
) -> SignificanceState | None:
    """Merge already-admitted significance for `node_id` into the store. Idempotent-ish
    across re-encounters of the same node: keep the best-grounded rationale, union the
    grounded causes. Returns the resulting state, or None if there was nothing to record.

    The values passed here are assumed already grounded (the perceiver did that against
    the full source). This function does no grounding — it is pure accumulation, mirroring
    how the consolidator accumulates DecayState.
    """
    has_rationale = isinstance(rationale, str) and bool(rationale.strip())
    has_cause = isinstance(cause, str) and bool(cause.strip())
    if not has_rationale and not has_cause:
        return store.get(node_id)
    state = store.get(node_id) or SignificanceState()
    if has_rationale and rationale_grounding >= state.rationale_grounding:
        state.rationale = rationale.strip()
        state.rationale_grounding = rationale_grounding
    if has_cause and cause.strip() not in state.causes:
        state.causes.append(cause.strip())
    if source_turn_id and source_turn_id not in state.source_turn_ids:
        state.source_turn_ids.append(source_turn_id)
    store[node_id] = state
    return state


def store_as_dict(store: dict[str, SignificanceState]) -> dict:
    """Serialize a significance store for snapshotting alongside a graph snapshot."""
    return {nid: st.as_dict() for nid, st in store.items() if not st.is_empty()}


def store_from_dict(d: dict) -> dict[str, SignificanceState]:
    return {nid: SignificanceState.from_dict(sd) for nid, sd in (d or {}).items()}
