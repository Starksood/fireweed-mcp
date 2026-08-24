"""Phase 11 — LLM reader: renders RetrievalResult as attributed prose.
Single LLM call. No loops, no tool use, no graph writes.

X4.5 additions
--------------
* Reader JSON now includes ``is_evidence_sufficient: bool``.  The harness
  uses it as the primary abstention signal.
* Hedge-phrase backstop: if the LLM claims sufficiency but prose contains
  well-known hedge patterns the flag is overridden to abstained=True.
* ``_strip_json_markup`` strips accidental code-fence wrapping before parsing
  (fixes q036 formatting leak observed in X4.Claude run B).
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Callable, Literal

from .retrieval import RetrievalResult, ResultEntry

LLMCallable = Callable[[str], str]

_MAX_PROSE_CHARS = 800

_SYSTEM_TOKEN_RE = re.compile(
    r"\[MEMORY\]|\[CLAIM\]|CREATE:|MODIFY:|ACCEPT:|RESCUE:|REJECT:|QUARANTINE:"
    r"|node_[a-f0-9]{8,}",
    re.IGNORECASE,
)

_HEDGE_WORDS = frozenset({
    "suggests", "likely", "probably", "based on", "inferred",
    "may indicate", "appears to", "seems to", "might", "could indicate",
})

# Hedge-phrase backstop denylist (X4.5 + X4.5.1).
# If the LLM returns is_evidence_sufficient=True (or omits the field) but
# prose matches one of these patterns, abstained is forced to True.
# Each phrase is checked case-insensitively against the prose.
# X4.5.1 additions cover "is not stated/provided/described/mentioned" patterns
# (e.g. q024: "The exact monthly rent is not stated in the evidence").
_HEDGE_PHRASES: tuple[str, ...] = (
    # --- X4.5 original ---
    "evidence does not contain",
    "evidence does not specify",
    "evidence does not include",
    "evidence does not address",
    "evidence does not answer",
    "provided evidence does not",
    "no information about",
    "no specific information",
    "not addressed in the evidence",
    "not specified in the evidence",
    "not contained in the evidence",
    "not present in the available",
    "available evidence does not",
    # --- X4.5.1 additions (safe subset only) ---
    # "is not stated/provided in" patterns appear only as opening hedges,
    # not as trailing qualifiers in substantive answers (verified Run C/D).
    "is not stated in",
    "not stated in the",
    "is not provided in",
    "not provided in the",
    "not mentioned in the",
    "is not specified in",
    "does not state",
    "does not mention the",
    "evidence does not mention",
)

ABSTAIN_PROSE: dict[str, str] = {
    "no_evidence":      "I don't have any grounded information about that.",
    "entity_not_found": "I don't have any information about that person or topic in memory.",
}

_PROMPT_TEMPLATE = """\
You are a memory reader. Answer the question using ONLY the evidence nodes provided.

QUESTION: {question}

EVIDENCE:
{evidence}

INSTRUCTIONS:
- Answer in 1-3 sentences using only the facts listed above.
- For any node marked [INFERRED], use hedging language (e.g. "suggests", "likely").
- Do NOT include system tokens ([MEMORY], CREATE:, node_, ACCEPT, RESCUE) in your prose.
- Do NOT assert facts not present in the evidence.
- Your prose must be plain text — no JSON, no markdown fences, no preamble.
- Cite only node_ids from the list above.
- Set is_evidence_sufficient to false ONLY when the evidence contains NO information
  relevant to the question at all — i.e. the topic is completely absent from the nodes.
  If the evidence contains ANY relevant facts, partial context, or adjacent information,
  set is_evidence_sufficient to true and answer with what you have, hedging where needed.
  Partial evidence is sufficient — answer rather than abstain when evidence is partially relevant.
- When is_evidence_sufficient is false, briefly state that the topic is not present in memory.

Respond with a single valid JSON object only — no other text:
{{"prose": "<your answer>", "cited_node_ids": ["<node_id_1>", ...], "claim_status": "<asserted|inferred|uncertain>", "is_evidence_sufficient": true}}"""


@dataclass
class ValidationResult:
    passed: bool
    violations: list[str]


@dataclass
class ReadResponse:
    prose: str
    cited_node_ids: list[str]
    claim_status: Literal["asserted", "inferred", "uncertain"]
    abstained: bool
    validation: ValidationResult


def read(
    question: str,
    retrieval_result: RetrievalResult,
    llm: LLMCallable,
) -> ReadResponse:
    """Render RetrievalResult as attributed prose. Single call, never raises."""
    try:
        return _read(question, retrieval_result, llm)
    except Exception:
        return ReadResponse(
            prose=ABSTAIN_PROSE["no_evidence"],
            cited_node_ids=[],
            claim_status="uncertain",
            abstained=True,
            validation=ValidationResult(passed=False, violations=["read_exception"]),
        )


def _read(question: str, result: RetrievalResult, llm: LLMCallable) -> ReadResponse:
    # Priority 1: hard retrieval abstain (entity_not_found, no_evidence).
    if result.abstain:
        reason = result.abstain_reason or "no_evidence"
        prose = ABSTAIN_PROSE.get(reason, ABSTAIN_PROSE["no_evidence"])
        return ReadResponse(
            prose=prose, cited_node_ids=[],
            claim_status="uncertain", abstained=True,
            validation=ValidationResult(passed=True, violations=[]),
        )

    prompt = _build_prompt(question, result)
    raw = llm(prompt)
    prose, cited_ids, claim_status, is_sufficient = _parse_response(raw)

    # Priority 2: LLM's own is_evidence_sufficient field.
    abstained = False
    if not is_sufficient:
        abstained = True
    # Priority 3: hedge-phrase backstop (belt-and-suspenders).
    elif _hedge_phrase_detected(prose):
        abstained = True

    # System override: if any cited node is an inference, claim_status → "inferred".
    all_entries = result.matched_nodes + result.expanded_nodes
    if any(e.is_inference for e in all_entries if e.node.node_id in set(cited_ids)):
        claim_status = "inferred"

    # Priority 4: VALIDATION IS ENFORCING, not advisory.
    # `_validate` already detected `unsupported_citation` -- a citation to a node that was never
    # retrieved, i.e. a fabricated receipt -- and the result was merely ATTACHED to the response
    # while `abstained` stayed False. A reader could cite a node that does not exist and still have
    # its answer returned as asserted. Computing a violation and then ignoring it is worse than not
    # checking, because it looks like a check.
    validation = _validate(prose, cited_ids, result)
    if not validation.passed:
        abstained = True

    return ReadResponse(
        prose=prose,
        cited_node_ids=cited_ids,
        claim_status=claim_status,
        abstained=abstained,
        validation=validation,
    )


def _hedge_phrase_detected(prose: str) -> bool:
    """Return True if prose contains a known hedge pattern (case-insensitive)."""
    lower = prose.lower()
    return any(phrase in lower for phrase in _HEDGE_PHRASES)


def _strip_json_markup(text: str) -> str:
    """Strip accidental ```json / ``` wrapping before JSON parsing.

    Defense in depth: the prompt already forbids fences, but some models
    emit them anyway. Strip here so parsing succeeds regardless.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].lstrip()
    elif text.startswith("```"):
        text = text[3:].lstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text


def _build_prompt(question: str, result: RetrievalResult) -> str:
    """Deterministic: same result → same prompt."""
    all_entries = result.matched_nodes + result.expanded_nodes
    seen: set[str] = set()
    deduped: list[ResultEntry] = []
    for entry in all_entries:
        if entry.node.node_id not in seen:
            seen.add(entry.node.node_id)
            deduped.append(entry)

    lines = [
        f"  {i}. [{e.node.node_id}]{'  [INFERRED]' if e.is_inference else ''}"
        f" {e.node.claim}  (r={e.node.reinforcement.overall:.2f})"
        for i, e in enumerate(deduped, 1)
    ] or ["  (none)"]
    return _PROMPT_TEMPLATE.format(question=question, evidence="\n".join(lines))


def _parse_response(raw: str) -> tuple[str, list[str], str, bool]:
    """Parse LLM JSON. Returns (prose, cited_ids, claim_status, is_sufficient).

    is_sufficient defaults to True when field is absent (backwards compat:
    existing fake LLMs in tests don't emit the field; fall through to
    hedge-phrase detector).
    """
    cleaned = _strip_json_markup(raw)
    try:
        data = json.loads(cleaned)
        prose = str(data.get("prose", "")).strip()
        cited_ids = [str(x) for x in data.get("cited_node_ids", [])]
        cs = str(data.get("claim_status", "uncertain"))
        claim_status = cs if cs in ("asserted", "inferred", "uncertain") else "uncertain"
        # Default True when field is absent — preserves backwards compat.
        is_sufficient = bool(data.get("is_evidence_sufficient", True))
        return prose, cited_ids, claim_status, is_sufficient
    except (json.JSONDecodeError, TypeError):
        # FAIL CLOSED. This used to return is_sufficient=True and pass the RAW string through as
        # prose, so an unparseable reader response became an ASSERTED answer -- literally emitting
        # '```json\n{"prose": "Speaker 1 plays for basketball."...' to the caller as if it were an
        # answer. Measured on the 2026-08-22 re-run: 9 of 29 Fireweed fabrications (31%) were this
        # leak, not the model reasoning badly.
        #
        # A proposal the deterministic layer cannot parse is a proposal it cannot check, and the
        # doctrine is the same on both sides of the system: what cannot be verified is not admitted.
        return raw.strip()[:_MAX_PROSE_CHARS], [], "uncertain", False


def _validate(prose: str, cited_ids: list[str], result: RetrievalResult) -> ValidationResult:
    violations: list[str] = []
    if _SYSTEM_TOKEN_RE.search(prose):
        violations.append("operator_syntax")
    if len(prose) > _MAX_PROSE_CHARS:
        violations.append("prose_too_long")
    valid_ids = {e.node.node_id for e in result.matched_nodes + result.expanded_nodes}
    if any(cid not in valid_ids for cid in cited_ids):
        violations.append("unsupported_citation")
    cited_inference_ids = {
        e.node.node_id for e in result.matched_nodes + result.expanded_nodes
        if e.is_inference and e.node.node_id in set(cited_ids)
    }
    if cited_inference_ids and not any(hw in prose.lower() for hw in _HEDGE_WORDS):
        violations.append("inference_not_hedged")
    return ValidationResult(passed=not violations, violations=violations)
