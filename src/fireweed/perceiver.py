"""Clock 1 — perception. The tiny-LLM "eyes" of the two-clock architecture.

  raw signal -> tiny LLM -> Percept   (this module; fast, must never block the graph)
  [ PerceptBuffer ]        staging / working memory
  Clock 2 (consolidation): batch -> firewall, entity-link, resolver, decay, write

This is the writer-side LLM tenant for the *perception* clock. It is the sibling
of extractor.py: extractor produces graph-bound Turns for the classic batch ingest
path; the perceiver produces Percepts for the streaming two-clock path. A Percept
is a superset of a Turn — it adds the perceptual hint fields (salience,
candidate_domains, polarity, entity_hint) that the buffer ranks on and Clock 2
re-verifies.

Why a small model is enough
---------------------------
Nothing a percept asserts is load-bearing. The only mandatory fields are the
*content* — claim plus the verbatim evidence span — validated exactly as
extractor.py validates them (the evidence must appear verbatim in the source).
Everything else is parsed *leniently*: hint fields (domains/polarity/entity) fall
back to safe defaults because Clock 2's deterministic components (firewall,
entity_linker, resolver) re-derive them from the claim text anyway, and confidence
is a self-report that defaults when omitted (tiny models routinely drop it on an
otherwise-good claim — going blind to that claim would be the wrong trade). A
malformed confidence (wrong type) is still rejected, since that signals a confused
model rather than a terse one. "Perception proposes; consolidation disposes." This
is precisely what lets the perceptual model be tiny and dumb without costing
identity coherence — the only thing it must get right is the claim/evidence pair,
the simplest part of the job. (Observed: gemma-3-1b, 1B params, extracts correct
verbatim claims but omits confidence under load — exactly this case.)

Three faithfulness guards keep a *creative* tiny model honest, because persistent
identity depends on canonical, source-bound claims (a paraphrasing model splinters a
recurring fact into separate nodes the resolver never merges, so reinforcement never
accumulates):
  1. verbatim-evidence — the cited span must occur in the source (as above);
  2. claim-grounding — most of the claim's content tokens must occur in the source,
     so a model that cites a real span ("Riverside clinic") cannot smuggle invented
     content into the claim ("...is a public health facility"). See
     PERCEIVER_MIN_CLAIM_GROUNDING. The prompt also asks for canonical, whole,
     faithful, stated-once claims; deterministic decoding (temperature 0) makes the
     same observation yield the same claim every time.
  3. claim-faithfulness (`grounding.claim_faithful`) — guard 2 compares SETS, so it is
     blind to a transposed relation ("Beta indemnifies Acme" citing "Acme indemnifies
     Beta"), and its >2-char filter drops digits, so amounts were never checked. This
     binds the claim to its own cited span: borrowed tokens must keep the evidence's
     order, and every number asserted must occur in the evidence.

Salience is the one genuinely new perceptual signal (extractor has no equivalent):
it governs eviction and drain order in the buffer. If the model omits it we fall
back to confidence as a proxy — a confident perception is provisionally more worth
keeping than an unsure one — so the buffer always has a usable ordering key.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable

from .constants import (
    EXTRACTION_MAX_INPUT_CHARS,
    EXTRACTION_MAX_CLAIMS_PER_CALL,
    EXTRACTION_MIN_CONFIDENCE,
    PERCEIVER_DEFAULT_CONFIDENCE,
    PERCEIVER_MIN_CLAIM_GROUNDING,
)
from .grounding import admissible
from .percept_buffer import Percept
from .significance import propose_significance

LLMCallable = Callable[[str], str]

_TOKEN_RE = re.compile(r"[^\w]+")


def _content_tokens(text: str) -> set[str]:
    """Content tokens (len>2) for grounding checks — drops articles/copulas like
    'is'/'a'/'at'. Mirrors retrieval's tokenizer so perception and recall agree."""
    return {t for t in _TOKEN_RE.sub(" ", text.lower()).split() if len(t) > 2}


# Evidence-span match policy. Default "verbatim" = the cited span must be an EXACT substring
# of the source (production behavior; preserved). "containment" admits a span whose content
# tokens are mostly present in the source (>= EVIDENCE_CONTAINMENT_MIN) — the same precision
# notion the Stage-4 harvester gate uses, to test whether paraphrasing perceivers (which cite
# near-verbatim spans the exact check rejects) become viable. Togglable via the
# FIREWEED_EVIDENCE_MATCH env var or by setting perceiver.EVIDENCE_MATCH_MODE directly.
EVIDENCE_MATCH_MODE = os.environ.get("FIREWEED_EVIDENCE_MATCH", "verbatim")
EVIDENCE_CONTAINMENT_MIN = 0.8


def _evidence_grounded(evidence: str, text: str, source_tokens: set[str]) -> bool:
    if EVIDENCE_MATCH_MODE == "containment":
        et = _content_tokens(evidence)
        return bool(et) and len(et & source_tokens) / len(et) >= EVIDENCE_CONTAINMENT_MIN
    return evidence in text

_PROMPT_TEMPLATE = """\
You are the perception layer of a memory system. Read the text below and extract
declarative claims about facts, events, preferences, or states. You are the "eyes":
spot what matters and tag it quickly. A later, careful stage re-checks your tags,
so approximate hints are fine — never omit a real claim for fear of mis-tagging it.

Output a JSON array of percept objects. Each object has these fields:
  "claim":      one single-sentence declarative statement of a single fact   (REQUIRED)
  "evidence":   the verbatim substring from the source text supporting it     (REQUIRED)
  "confidence": a number 0.0-1.0 — your certainty the claim is true           (REQUIRED)
  "salience":   a number 0.0-1.0 — how important/attention-worthy this is      (optional)
  "domains":    a list of short topic tags, e.g. ["work","transit"]           (optional)
  "polarity":   "positive", "negative", or "neutral" emotional charge         (optional)
  "entity":     the main person/place/thing the claim is about, as a string   (optional)
  "rationale":  one short phrase, in the SOURCE'S OWN WORDS, for why this fact (optional)
                matters to the person — only if the text makes it clear
  "cause":      the verbatim source clause that CAUSED this fact, only when    (optional)
                the source states a cause ("because ...", "since ...", "so ...")

Rules:
- Extract only declarative claims. No questions, commands, or hypotheticals.
- Do not extract meta-commentary about the conversation or the system itself.
- Each claim is a single sentence in past, present, or stative tense.
- The "evidence" string must appear verbatim in the source text.
- The evidence MUST CONTAIN THE SUBJECT of the claim by name. If the supporting sentence refers
  to the subject with a pronoun ("It needs a GPU"), EXTEND the evidence backwards to include the
  earlier sentence that names it. The span must stay one contiguous verbatim substring.
  A claim whose subject is not named inside its own evidence will be rejected.
- FAITHFUL: use only facts and words the source states. Never add detail, cause,
  consequence, or category the text does not contain (do not turn "a clinic" into
  "a public health facility", do not invent a department or a motive).
- CANONICAL: phrase each claim using the source's own wording, so the same fact
  always reads the same way across calls. Do not paraphrase what you can quote.
- WHOLE: keep a fact in one claim. "Maya is a nurse at Riverside" is ONE claim —
  do not split it into "Maya is a nurse" plus "Maya works at Riverside".
- ONCE: state each distinct fact a single time, even if the source repeats it.
- GROUNDED SIGNIFICANCE: "rationale" and "cause" are optional and must use only the
  source's own words. NEVER invent a motive or a cause the text does not state. If the
  text gives no reason and no cause, OMIT both fields — an empty field is always better
  than a guessed one. (A later stage discards any rationale/cause it cannot find in the
  source, so guessing only wastes them.)
- Maximum 20 percepts per call. If the text has more, keep the most salient.
- If the text contains no extractable claims, output [].

Source text:
{text}

Output only the JSON array. No prose, no preamble, no markdown fences."""


def _build_prompt(text: str) -> str:
    """Byte-deterministic: same text -> same bytes every call."""
    return _PROMPT_TEMPLATE.format(text=text)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _coerce_salience(item: dict, confidence: float) -> float:
    """Salience drives eviction/drain order. Lenient: fall back to confidence."""
    raw = item.get("salience")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return _clamp01(float(raw))
    return confidence  # already validated/clamped upstream


def _coerce_domains(item: dict) -> tuple[str, ...]:
    """Hint only — firewall re-classifies. Accept a list of non-empty strings."""
    raw = item.get("domains")
    if not isinstance(raw, list):
        return ()
    return tuple(d.strip() for d in raw if isinstance(d, str) and d.strip())


def _coerce_polarity(item: dict) -> int:
    """Hint only — resolver re-checks. Map to -1 / 0 / +1; default neutral (0)."""
    raw = item.get("polarity")
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, (int, float)):
        if raw > 0:
            return 1
        if raw < 0:
            return -1
        return 0
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("positive", "pos", "+", "+1"):
            return 1
        if s in ("negative", "neg", "-", "-1"):
            return -1
    return 0


def _coerce_entity(item: dict) -> str | None:
    """Hint only — entity_linker resolves. Accept a non-empty string."""
    raw = item.get("entity") or item.get("entity_hint")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def perceive_text(
    text: str,
    source_id: str,
    timestamp: str,
    llm: LLMCallable,
) -> list[Percept]:
    """Clock 1. Extract Percepts from free text via a single tiny-LLM call.

    Returns list[Percept] (not yet buffered). Core fields are validated exactly as
    extractor.extract_claims validates them; hint fields are coerced leniently.
    Never raises except on input-length violation, mirroring the extractor.
    """
    if len(text) > EXTRACTION_MAX_INPUT_CHARS:
        raise ValueError(
            f"Input text exceeds EXTRACTION_MAX_INPUT_CHARS ({EXTRACTION_MAX_INPUT_CHARS}): "
            f"got {len(text)} chars"
        )
    if not text.strip():
        return []

    source_tokens = _content_tokens(text)
    raw = llm(_build_prompt(text))
    try:
        parsed = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return []
    if not isinstance(parsed, list):
        return []

    percepts: list[Percept] = []
    n = 0
    for item in parsed:
        if n >= EXTRACTION_MAX_CLAIMS_PER_CALL:
            break
        if not isinstance(item, dict):
            continue
        # Content is what the perceiver must get right: a claim and the verbatim
        # span supporting it. Only these two are mandatory — confidence is a
        # self-report (defaultable), and the rest are re-verified hints.
        if not {"claim", "evidence"}.issubset(item.keys()):
            continue
        claim, evidence = item["claim"], item["evidence"]
        # --- content fields: validated strictly (same discipline as extractor.py) ---
        if not isinstance(claim, str) or not claim.strip():
            continue
        if not isinstance(evidence, str) or not _evidence_grounded(evidence, text, source_tokens):
            continue
        # --- faithfulness: the claim may rephrase the source but not INVENT content ---
        # Verbatim evidence proves the span exists; it does NOT prove the claim stays
        # within it. A creative model can cite "Riverside clinic" yet claim it "is a
        # public health facility". Require most of the claim's content tokens to occur
        # in the source; reject over-claims. (Empty-content claims skip the check.)
        claim_tokens = _content_tokens(claim)
        if claim_tokens:
            grounding = len(claim_tokens & source_tokens) / len(claim_tokens)
            if grounding < PERCEIVER_MIN_CLAIM_GROUNDING:
                continue
        # Overlap is a SET test, so it cannot see a transposed relation ("Beta indemnifies Acme"
        # from "Acme indemnifies Beta") and the >2-char token filter drops digits, so amounts went
        # unchecked entirely. Bind the claim to its own cited span, order- and numeral-aware.
        if not admissible(claim, evidence):
            continue
        # --- confidence: required-but-defaultable self-report ---
        # Absent  -> default (tiny models often drop it on a good claim; don't go blind).
        # Malformed (wrong type) -> reject the percept (a model-confusion signal).
        # Present & numeric -> clamp to [0,1]; honor an explicit low rating via the floor.
        if "confidence" not in item:
            confidence = PERCEIVER_DEFAULT_CONFIDENCE
        else:
            c = item["confidence"]
            if isinstance(c, bool) or not isinstance(c, (int, float)):
                continue
            confidence = _clamp01(float(c))
            if confidence < EXTRACTION_MIN_CONFIDENCE:
                continue
        # --- significance (Stage 3, W1): perceiver PROPOSES, code DECIDES via grounding ---
        # propose_significance admits a rationale/cause only if it traces to the source text;
        # a fabricated motive is dropped here, never reaching the graph.
        g_rationale, g_grounding, g_cause = propose_significance(
            item.get("rationale"), item.get("cause"), claim, text,
        )
        # --- hint fields: coerced leniently (Clock 2 re-verifies all of these) ---
        n += 1
        percepts.append(Percept(
            claim=claim.strip(),
            evidence=evidence,
            confidence=confidence,
            salience=_coerce_salience(item, confidence),
            candidate_domains=_coerce_domains(item),
            polarity=_coerce_polarity(item),
            entity_hint=_coerce_entity(item),
            source_id=source_id,
            received_at=timestamp,
            rationale=g_rationale,
            cause=g_cause,
            rationale_grounding=g_grounding,
        ))
    return percepts


def perceive_into(
    scheduler,
    text: str,
    source_id: str,
    timestamp: str,
    llm: LLMCallable,
) -> list:
    """Convenience: perceive `text` and admit every Percept into a TwoClockScheduler.

    This is the end-to-end Clock 1 entry point — raw text in, percepts staged in the
    buffer, returning the AdmitResult list (so callers can observe evictions). It
    never drains; Clock 2 (scheduler.tick) drives consolidation on its own cadence.
    """
    results = []
    for p in perceive_text(text, source_id, timestamp, llm):
        results.append(scheduler.perceive(
            claim=p.claim,
            evidence=p.evidence,
            confidence=p.confidence,
            salience=p.salience,
            candidate_domains=p.candidate_domains,
            polarity=p.polarity,
            entity_hint=p.entity_hint,
            source_id=p.source_id,
            rationale=p.rationale,
            cause=p.cause,
            rationale_grounding=p.rationale_grounding,
        ))
    return results
