"""Domain classification for memory claims.

Ported from v15: diagnostics/domain_classifier.py
This module is unchanged from v15 — multi-domain classification is a validated principle.
"""
from __future__ import annotations

DOMAIN_KEYWORDS: list[tuple[str, list[str]]] = [
    ("schedule", ["opening shift", "closing shift", "early shift", "late shift", "morning shift", "shift", "schedule"]),
    ("body",     ["physio", "injury", "knee", "it band", "flare", "hurt", "ache", "pain", "running", "run", "back", "hurting"]),
    ("commute",  ["bus", "transfer", "van ness", "route", "platform", "commute", "transit", "train", "subway", "station", "bike", "bicycle"]),
    ("food",     ["breakfast", "coffee", "pastry", "pastries", "croissant", "oats", "groceries", "lunch", "dinner", "cook", "sandwich", "chips", "snack", "meal"]),
    ("housing",  ["landlord", "lease", "apartment", "flat", "move", "increase letter", "rent", "rents"]),
    ("budget",   ["savings", "spending", "bank", "afford", "expense", "expenses", "budget", "cost", "price", "save", "saving", "financial", "money"]),
    ("weather",  ["rain", "storm", "umbrella", "forecast", "wet", "soaked", "overcast", "jacket"]),
    ("fatigue",  ["tired", "exhausted", "nap", "energy", "fatigue", "drained", "worn out"]),
    ("planning", ["meal prep", "prepare", "planned", "routine", "schedule ahead", "plan ahead"]),
    ("work",     ["work", "works", "job", "analyst", "engineer", "code", "coding", "standing desk", "desk", "remote", "remotely", "colleague", "collaborate", "collaboration", "project", "backend", "frontend", "focus", "handle", "data analyst"]),
    ("health",   ["medical", "doctor", "therapy", "recovery", "health", "illness", "sick"]),
    ("hobby",    ["hobby", "hobby", "guitar", "band", "music", "play", "sport", "weekend", "leisure"]),
    ("family",   ["father", "mother", "parent", "brother", "sister", "family", "relative"]),
    ("other",    []),  # X4: Fallback domain for structurally valid claims that don't match known domains
]


def classify_domains(text: str) -> set[str]:
    """Returns a set of all domains that match keywords in the text.
    
    This is the emergent multi-domain classifier: a claim can activate multiple
    distinctions. For example, "pastries at the station" activates both {"food", "commute"}.
    
    Returns an empty set if no keywords match or text is empty/None.
    Does NOT include "unknown" in the result set.
    
    Examples:
        classify_domains("pastries at the station") → {"food", "commute"}
        classify_domains("after closing shift I bought chips") → {"schedule", "food"}
        classify_domains("letter from my landlord about rent") → {"housing", "budget"}
        classify_domains("random text") → set()
    """
    if not text:
        return set()
    
    lowered = text.lower()
    matched_domains = set()
    
    for domain, keywords in DOMAIN_KEYWORDS:
        for kw in keywords:
            if kw in lowered:
                matched_domains.add(domain)
                break  # Found a match for this domain, move to next domain
    
    return matched_domains


def classify_domain(text: str) -> str:
    """Returns one of: schedule, body, commute, food, budget, weather, fatigue,
    planning, housing, unknown. Lowercases the input and substring-matches keywords.
    First domain (in the declared order below) with any matching keyword wins.
    Returns 'unknown' if no keyword matches or text is empty/None.
    
    This is the backward-compatible single-domain classifier. For new code,
    prefer classify_domains() which returns all matching domains.
    """
    if not text:
        return "unknown"
    lowered = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS:
        for kw in keywords:
            if kw in lowered:
                return domain
    return "unknown"


FACET_KEYWORDS: dict[str, list[str]] = {
    "convenience":        ["corner shop", "chips", "ready meal", "grab", "station food",
                           "no preparation", "no interest in cooking", "quick", "easy meal"],
    "after_work":         ["after shift", "after late shift", "after her shift",
                           "on the way home", "late finish", "by the time i get back", "before bed"],
    "morning":            ["morning", "opening shift", "before work", "early",
                           "five o'clock", "five am", "before breakfast", "out the door"],
    "disruption":         ["delayed", "missed", "unreliable", "stuck",
                           "exposed platform", "no shelter", "cancel", "no connection", "running late"],
    "recurring":          ["every day", "repeatedly", "again", "keeps happening",
                           "each week", "three nights", "two or three"],
    "financial_strain":   ["adding up", "too much", "going up", "can't save",
                           "eighty pounds", "thirty to forty", "not expecting"],
    "health_management":  ["physio", "stopped running", "recovery", "appointment",
                           "rest", "ice", "stretch", "rehabilitation"],
    "weather_disruption": ["soaked", "wind", "wet through", "umbrella barely",
                           "overcast all week", "no sunlight", "heavy rain"],
    "budget_pressure":    ["spending diary", "track", "column", "batch cook", "save money",
                           "reduce", "cut back", "can't afford"],
    "physical_exposure":  ["standing", "exposed", "platform", "no shelter", "outside",
                           "wind", "fifteen minutes"],
}


def extract_facets(claim: str) -> list[str]:
    """Return all facet tags whose keywords appear in the claim. Pure function."""
    if not claim:
        return []
    lower = claim.lower()
    return [f for f, kws in FACET_KEYWORDS.items() if any(kw in lower for kw in kws)]
