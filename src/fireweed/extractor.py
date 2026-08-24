"""Writer-side LLM tenant: extracts declarative claims from free text.

No constitutional filtering is applied here. The firewall (firewall.py) handles
PII and injection gates during ingest. This module is responsible only for
structured claim extraction from raw text.
"""
from __future__ import annotations
import json
import re
from typing import Callable

from .grounding import admissible
from .constants import (
    EXTRACTION_MAX_INPUT_CHARS,
    EXTRACTION_MAX_CLAIMS_PER_CALL,
    EXTRACTION_MIN_CONFIDENCE,
)

LLMCallable = Callable[[str], str]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?|```", re.IGNORECASE)


def _sanitize_json_array(raw: str) -> str:
    """Coax an instruct model's reply into the bare JSON array we can parse. Models routinely wrap the
    array in ```json fences, prepend a sentence, or (thinking variants) emit a <think>…</think> block —
    a raw json.loads then fails and silently drops every claim. Strip those, then slice to the outermost
    [...] if there is surrounding prose. Returns "[]" when no array is present."""
    if not isinstance(raw, str):
        return "[]"
    text = _FENCE_RE.sub("", _THINK_RE.sub("", raw)).strip()
    if text.startswith("[") and text.endswith("]"):
        return text
    start, end = text.find("["), text.rfind("]")
    return text[start:end + 1] if (start != -1 and end > start) else "[]"

_PROMPT_TEMPLATE = """\
You are a claim extractor for a memory system. Read the text below and extract
declarative claims about facts, events, preferences, or states.

Output a JSON array of claim objects. Each object has exactly these fields:
  "claim": one single-sentence declarative statement of a single fact
  "evidence": the verbatim substring from the source text that supports this claim
  "confidence": a number from 0.0 to 1.0 indicating your certainty

Rules:
- Extract only declarative claims. Do not extract questions, commands, or hypotheticals.
- Do not extract meta-commentary about the conversation or the system itself.
- Each claim must be a single sentence in the past, present, or stative tense.
- The "evidence" string must appear verbatim in the source text.
- The evidence MUST CONTAIN THE SUBJECT of the claim by name. If the supporting sentence refers
  to the subject with a pronoun ("It is headquartered in Athens"), EXTEND the evidence backwards
  to include the earlier sentence that names it, so the span reads "Skai TV ... It is
  headquartered in Athens". The span must stay one contiguous verbatim substring.
  A claim whose subject is not named inside its own evidence will be rejected.
- Maximum 20 claims per call. If the text contains more, extract the most important ones.
- If the text contains no extractable claims, output [].

Source text:
{text}

Output only the JSON array. No prose, no preamble, no markdown fences."""


def _build_prompt(text: str) -> str:
    """Byte-deterministic: same text → same bytes every call."""
    return _PROMPT_TEMPLATE.format(text=text)


def extract_claims(
    text: str,
    source_id: str,
    timestamp: str,
    llm: LLMCallable,
) -> list:
    """Extract declarative claims from free text via a single LLM call.

    Returns list[Turn]. Never raises except on input length violation.
    """
    if len(text) > EXTRACTION_MAX_INPUT_CHARS:
        raise ValueError(
            f"Input text exceeds EXTRACTION_MAX_INPUT_CHARS ({EXTRACTION_MAX_INPUT_CHARS}): "
            f"got {len(text)} chars"
        )
    if not text.strip():
        return []

    raw = llm(_build_prompt(text))
    try:
        parsed = json.loads(_sanitize_json_array(raw))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []

    from .fabric import Turn  # lazy import — fabric imports extractor, not the reverse

    turns: list[Turn] = []
    seen_ids: set[str] = set()
    n = 0
    for item in parsed:
        if n >= EXTRACTION_MAX_CLAIMS_PER_CALL:
            break
        if not isinstance(item, dict):
            continue
        if not {"claim", "evidence", "confidence"}.issubset(item.keys()):
            continue
        claim, evidence, confidence = item["claim"], item["evidence"], item["confidence"]
        if not isinstance(claim, str) or not claim.strip():
            continue
        if not isinstance(evidence, str) or evidence not in text:
            continue
        # The span being verbatim does not make the CLAIM say what the span says: a transposed
        # relation ("Beta indemnifies Acme" citing "Acme indemnifies Beta") or an altered amount
        # would otherwise commit with a receipt that verifies against the contradicting bytes.
        # document_mode: this is the DOCUMENT path, where evidence is a byte span of a hashed
        # source. A claim that asserts more than its span ("signed the lease in Paris", "the
        # fraudulent lease") would otherwise commit AND receive a receipt that re-verifies forever
        # against bytes that never said it. docs/FINDING_predicate_fabrication.md.
        if not admissible(claim, evidence, document_mode=True):
            continue
        if not isinstance(confidence, (int, float)):
            continue
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0) or confidence < EXTRACTION_MIN_CONFIDENCE:
            continue
        n += 1
        turn_id = f"{source_id}_t{n:03d}"
        if turn_id in seen_ids:
            continue
        seen_ids.add(turn_id)
        turns.append(Turn(
            turn_id=turn_id,
            text=claim.strip(),
            evidence_span=evidence,
            confidence=confidence,
        ))
    return turns
