"""Semantic scoring for synthesis evaluation.

Ported from v15: bench/exp040_organic_consolidation.py
Concept-bundle matching for evaluating INFER quality against hidden targets.
"""
from __future__ import annotations
import re

# Personality markers for filtering out trait-level inferences
PERSONALITY_MARKERS = [
    "is disciplined", "is resilient", "is a planner", "values ", "avoids ",
    "lacks motivation", "prefers efficiency", "is organised", "is organized",
    "is dedicated", "is hardworking", "is responsible",
]

# Semantic bundles for the Maya benchmark hidden inferences
# A claim is a semantic hit for target H_n if it contains terms from >= 2 bundles
SEMANTIC_BUNDLES: dict[str, dict[str, list[str]]] = {
    "H1": {
        "schedule_terms": ["early shift", "opening shift", "pre-dawn", "before 5",
                           "out the door", "five o'clock"],
        "commute_terms":  ["bus", "commute", "transit", "van ness", "transfer",
                           "journey", "platform"],
        "fatigue_terms":  ["tired", "fatigue", "exhausted", "energy", "drained",
                           "worn out"],
    },
    "H2": {
        "food_spend_terms": ["pastry", "pastries", "sandwich", "sandwiches",
                             "station food", "grab-and-go", "thirty", "forty",
                             "£30", "£40", "food spend", "weekly spend"],
        "meal_prep_terms":  ["batch cook", "cook at home", "prepare", "sunday",
                             "meal prep", "home cooking"],
        "budget_terms":     ["spending", "spend", "cost", "money", "budget",
                             "reduce", "save"],
    },
    "H3": {
        "weather_terms":  ["rain", "soaked", "wet", "umbrella", "storm",
                           "overcast", "jacket"],
        "commute_terms":  ["bus", "commute", "transit", "van ness", "transfer",
                           "platform", "journey"],
        "cascade_terms":  ["delay", "worse", "harder", "disrupts", "cascade",
                           "unreliable", "exposed"],
    },
    "H4": {
        "shift_terms": ["early shift", "opening shift", "before breakfast",
                        "pre-dawn", "out the door", "five o'clock", "5am",
                        "early morning"],
        "food_terms":  ["pastry", "pastries", "station", "grab-and-go",
                        "breakfast", "food purchase", "buying food",
                        "transit stop", "food at"],
    },
    "H5": {
        "rent_terms":     ["rent", "landlord", "eighty pounds", "increase",
                           "lease", "flat", "housing", "£80"],
        "spending_terms": ["spending", "purchases", "weekly", "thirty", "forty",
                           "£30", "£40", "budget", "financially", "cost"],
    },
    "H6": {
        "shift_terms":       ["closing shift", "late shift", "evening shift",
                              "10pm", "ten", "quarter past", "closing",
                              "late night"],
        "cooking_terms":     ["cook", "cooking", "cooker", "prepare meals",
                              "meal prep", "kitchen", "stand at the cooker",
                              "no interest in cooking"],
        "convenience_terms": ["corner shop", "chips", "ready meal", "convenience",
                              "buying lunch", "food purchase", "no interest in",
                              "convenient food", "grab"],
    },
}


def is_personality_claim(claim: str) -> bool:
    """Check if a claim is a personality-level inference (should be filtered out).
    
    Args:
        claim: The inference claim text
    
    Returns:
        True if the claim contains personality trait markers
    """
    low = claim.lower()
    return any(m in low for m in PERSONALITY_MARKERS)


def semantic_hit(claim_lower: str, bundles: dict[str, list[str]]) -> bool:
    """Check if a claim matches a target by containing terms from >= 2 bundles.
    
    Args:
        claim_lower: Lowercased claim text
        bundles: Dict of bundle_name -> list of terms
    
    Returns:
        True if claim contains terms from >= 2 bundles
    """
    bundles_matched = 0
    for terms in bundles.values():
        if any(t in claim_lower for t in terms):
            bundles_matched += 1
            if bundles_matched >= 2:
                return True
    return False


def semantic_score_claims(claims: list[dict]) -> dict:
    """Score claims against hidden targets using concept-bundle matching.
    
    A claim is a semantic hit for target H_n if it contains terms from >= 2
    of H_n's concept bundles (case-insensitive substring match).
    
    This is the PRIMARY metric for synthesis quality (exp039 finding).
    Semantic precision catches valid inferences that Jaccard misses due to
    paraphrase variation.
    
    Args:
        claims: List of claim dicts with "claim" key
    
    Returns:
        Dict with:
        - semantic_precision: Fraction of claims that hit a target
        - semantic_targets_hit: List of target IDs hit
        - semantic_unique_targets_hit: Count of unique targets hit
        - semantic_claim_scores: Per-claim scoring details
    """
    if not claims:
        return {
            "semantic_precision":          0.0,
            "semantic_targets_hit":        [],
            "semantic_unique_targets_hit": 0,
            "semantic_claim_scores":       [],
        }

    targets_hit: set[str] = set()
    semantic_claim_scores = []

    for c in claims:
        claim_lower = c["claim"].lower()
        best_h = None
        for h_id, bundles in SEMANTIC_BUNDLES.items():
            if semantic_hit(claim_lower, bundles):
                best_h = h_id
                break  # first matching target wins

        if best_h:
            targets_hit.add(best_h)

        semantic_claim_scores.append({
            "claim":        c["claim"],
            "semantic_hit": best_h,
        })

    valid_hits = [s for s in semantic_claim_scores if s["semantic_hit"] is not None]
    semantic_precision = len(valid_hits) / len(claims) if claims else 0.0

    return {
        "semantic_precision":          round(semantic_precision, 3),
        "semantic_targets_hit":        sorted(targets_hit),
        "semantic_unique_targets_hit": len(targets_hit),
        "semantic_claim_scores":       semantic_claim_scores,
    }


def _tokens(text: str) -> set[str]:
    """Content tokens for lexical scoring.

    The >2 filter drops stopword-ish noise, but it also silently deleted every SHORT ALPHANUMERIC
    IDENTIFIER -- "T4", "v2", "3B", "0.5" -- which in a technical corpus are the highest-signal
    terms in the query. Measured on the ops graph: query("T4") tokenized to the empty set and
    returned 0.0 against "the T4 is compute capability 7.5", so retrieval abstained while holding
    the answer. A short token is kept when it carries a digit; short pure-alpha tokens ("of", "is",
    "an") stay filtered.
    """
    out = set()
    for t in re.sub(r"[^\w]", " ", text.lower()).split():
        if len(t) > 2 or any(ch.isdigit() for ch in t):
            out.add(t)
    return out


def jaccard_score(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings.
    
    Tokenizes on word boundaries, filters tokens < 3 chars, lowercases.
    
    Args:
        a: First string
        b: Second string
    
    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def query_coverage(query: str, text: str) -> float:
    """Fraction of the QUERY's tokens present in `text` — asymmetric, unlike jaccard.

    Jaccard normalizes by the UNION, so it falls as the claim gets longer no matter how well the
    query is covered. A one-token query against a twenty-token claim tops out near 0.05 even on a
    perfect hit, which is below every useful abstain threshold. Measured on the ops graph:
    query("consolidation") abstained while three active nodes contained the word.

    Coverage asks the question the abstain gate actually means -- "is what the user asked about
    present here?" -- and is independent of how much else the claim says.
    """
    q = _tokens(query)
    if not q:
        return 0.0
    return len(q & _tokens(text)) / len(q)
