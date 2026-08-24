r"""Constitutional rules: PII detection (C1), injection detection (C2). Called by firewall.evaluate()."""
from __future__ import annotations
import re
_PII_PATTERNS = [re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
                 re.compile(r'\b\d{10,}\b'), re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                 re.compile(r'\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b'),
                 re.compile(r'\b\d{3}[- ]\d{3}[- ]\d{4}\b'),  # XXX-XXX-XXXX
                 re.compile(r'\b\d{3}[- ]\d{4}\b'),              # XXX-XXXX
                 re.compile(r'\b[A-Z]{2}\d{6}[A-D]\b'),
                 re.compile(r'\b[A-Z]{1,2}\d[A-Z0-9]? \d[A-Z]{2}\b')]
_INJECTION_HARD = ["[memory]", "[/memory]", "<system_prompt>", "[inst]"]
_INJECTION_PHRASES = ["ignore previous instructions", "ignore all previous", "system prompt",
                      "forget everything", "you are now a", "jailbreak", "dan mode", "new persona"]
_INJECTION_INSTRUCTION_WORDS = ["rewrite", "override", "bypass", "change behavior"]
_INJECTION_CONTEXT = ["ignore", "forget", "you are now", "jailbreak"]

def _pii_detected(text: str) -> bool:
    upper = text.upper()
    return any(p.search(text) or p.search(upper) for p in _PII_PATTERNS)
def _injection_attempt(text: str) -> bool:
    # Hard patterns: non-negotiable markers of injection attempts
    if any(kw in text for kw in _INJECTION_HARD) or any(p in text for p in _INJECTION_PHRASES):
        return True

    # X3 Calibration: "prompt" detection with two-tier proximity + intent
    # Goal: eliminate false positives from legitimate technical contexts (prompt engineering, etc.)
    # while catching actual injection attempts
    if "prompt" in text:
        words = text.lower().split()
        prompt_indices = [i for i, w in enumerate(words) if "prompt" in w]

        # Tier 1: Inherently suspicious instruction words (rewrite, override, bypass, change behavior)
        # These are imperative verbs that indicate system manipulation on their own
        instruction_word_indices = [i for i, w in enumerate(words)
                                   if any(iw in w for iw in _INJECTION_INSTRUCTION_WORDS)]

        # Tier 2: Generic context words (ignore, forget, you are now, jailbreak)
        # These need additional evidence: proximity to "prompt" + imperative language
        context_indices = [i for i, w in enumerate(words)
                          if any(c in w for c in _INJECTION_CONTEXT)]

        # Imperative/instruction language that indicates system manipulation
        imperative_words = ["tell", "reveal", "output", "generate", "create", "show", "use"]
        has_imperative = any(imp in text.lower() for imp in imperative_words)

        # Tier 1: Instruction word + prompt within 8 words = injection
        if instruction_word_indices and prompt_indices:
            for p_idx in prompt_indices:
                for iw_idx in instruction_word_indices:
                    if abs(p_idx - iw_idx) <= 8:
                        return True  # Injection: instruction word near prompt

        # Tier 2: Context word + prompt + imperative language within 8 words = injection
        if has_imperative and context_indices and prompt_indices:
            for p_idx in prompt_indices:
                for c_idx in context_indices:
                    if abs(p_idx - c_idx) <= 8:
                        return True  # Injection: context word near prompt with imperative language

    # Standard "ignore previous" + instruction target pattern
    return ("ignore previous" in text or "disregard previous" in text) and any(
        kw in text for kw in ["instructions", "message", "rules"])
def check(text: str) -> str | None:
    """Constitutional check: return "constitutional:pii/injection" or None."""
    if _pii_detected(text):
        return "constitutional:pii"
    if _injection_attempt(text):
        return "constitutional:injection"
