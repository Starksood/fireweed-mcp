"""Memory Claim Firewall — pure decision pipeline.

ACCEPT  → known-domain claim, proceed to graph
RESCUE  → unknown-domain but valid, reclassify then proceed
REJECT  → greeting / placeholder / bare identity / artifact → drop
QUARANTINE → too unclear → log for review
"""
from __future__ import annotations

import re

from .claim import Claim, FirewallDecision, FirewallResult
from .constitutional import check as _constitutional_check
from .domain_classifier import classify_domains, DOMAIN_KEYWORDS

_GREETING_PREFIXES = ("hi ", "hello", "hey", "good morning", "good evening", "good afternoon")
_BARE_IDENTITY_LIST = {"maya", "user", "the user", "person", "they", "she", "he", "it"}
_DOMAIN_KW_SET: set[str] = (
    {kw for _, kws in DOMAIN_KEYWORDS for kw in kws}
    | {domain for domain, _ in DOMAIN_KEYWORDS}
)
_COPULAS = {"is", "are", "was", "were", "has", "have"}
_VERB_RE = re.compile(r"\b\w+(?:s|ed|ing|es)\b", re.IGNORECASE)
_IN_LOC_RE = re.compile(r"\bin\s+[A-Z][a-zA-Z]+\b")

_RESCUE_RULES: list[tuple[list[str], set[str]]] = [
    (["lives in", "apartment", "rent", "landlord", "lives at", "moving to", "move to", "relocat"], {"housing"}),
    (["shift", "closing", "10pm", "before work", "opening"], {"schedule"}),
    (["bus", "train", "station", "commute"], {"commute"}),
    (["chips", "sandwich", "pastry", "lunch", "dinner", "breakfast"], {"food"}),
    (["cost", "spending", "budget", "afford"], {"budget"}),
    (["rain", "cold", "hot", "snow"], {"weather"}),
    (["tired", "fatigue", "pain", "sleep"], {"fatigue"}),
]


def _reject(reason: str) -> FirewallResult:
    return FirewallResult(decision=FirewallDecision.REJECT, reason=reason, resolved_domains=set())


def _rescue_domains(text: str) -> set[str]:
    lower = text.lower()
    rescued: set[str] = set()
    for keywords, domains in _RESCUE_RULES:
        if any(kw in lower for kw in keywords):
            rescued |= domains
    if _IN_LOC_RE.search(text):
        rescued.add("housing")
    return rescued


def _has_specific_entity(text: str) -> bool:
    """X4: Check for capitalized noun, excluding vague/bare identity words."""
    vague = {"something", "thing", "situation", "issue", "problem", "stuff"}
    for w in text.split():
        clean = w.rstrip(".,!?\"'").rstrip("'s")
        base = clean.lower()
        if clean and clean[0].isupper() and base not in (vague | _BARE_IDENTITY_LIST):
            return True
    return False


def evaluate(claim: Claim) -> FirewallResult:
    """Evaluate a claim through the firewall. Pure function — no side effects."""
    text = claim.claim
    lower = text.strip().lower()

    # Step 1: REJECT gates — first match wins
    if any(lower.startswith(p) for p in _GREETING_PREFIXES):
        return _reject("greeting")

    stripped = text.strip()
    if re.fullmatch(r"<[^>]*>", stripped) or stripped == "updated content" or (
        stripped.startswith("<") and stripped.endswith(">")
    ):
        return _reject("placeholder")

    if lower in _BARE_IDENTITY_LIST or (" " not in stripped and stripped and stripped[0].isupper()):
        return _reject("bare_identity")

    # no_durable_predicate before too_short (single keyword like "food" is 4 chars)
    if lower in _DOMAIN_KW_SET:
        return _reject("no_durable_predicate")

    if len(stripped) < 10:
        return _reject("too_short")

    words = lower.split()
    if not any(w in _COPULAS for w in words) and not _VERB_RE.search(lower):
        return _reject("fragment")

    if "the user said" in lower or "user mentioned" in lower or re.search(r"\bcontent\b", lower):
        return _reject("parser_artifact")

    # Step 1c: Constitutional REJECT gates — PII and injection patterns
    constitutional_reason = _constitutional_check(lower)
    if constitutional_reason:
        return _reject(constitutional_reason)

    # Step 2: domain check
    domains = classify_domains(text)

    if domains:
        # Step 4: ACCEPT
        return FirewallResult(
            decision=FirewallDecision.ACCEPT,
            reason="known_domain" if len(domains) == 1 else "multi_domain",
            resolved_domains=domains,
        )

    # Step 3: RESCUE (including X4 structural validity fallback)
    rescued = _rescue_domains(text) or (_has_specific_entity(text) and {"other"}) or set()
    if rescued:
        return FirewallResult(decision=FirewallDecision.RESCUE, reason="unknown_domain_rescued", resolved_domains=rescued)
    return FirewallResult(decision=FirewallDecision.QUARANTINE, reason="too_unclear", resolved_domains=set())
