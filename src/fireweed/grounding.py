"""Faithfulness checks binding a proposed CLAIM to its cited EVIDENCE span.

Shared by both write paths — `perceiver.py` (conversational) and `extractor.py` (document) — so a
claim cannot enter the substrate through the weaker door.

Why this exists
---------------
Verbatim-evidence checking proves the cited span really occurs in the source. It does NOT prove the
claim *says what the span says*. The original token-overlap guard compared **sets**, so word order and
grammatical role were invisible and these all passed:

    source: "Acme indemnifies Beta"        claim: "Beta indemnifies Acme"      (roles reversed)
    source: "Alice paid Bob $5,000,000"    claim: "Alice paid Bob $9,000,000"  (amount altered)

Both are catastrophic in the domains this is sold into — a reversed indemnity is a reversed liability,
and the receipt still verified, pointing at the exact bytes of the contradicting sentence. The two
checks below close that, deterministically and with no model in the loop (resolver-purity guardrail):

  * ORDER — the claim's content tokens that appear in the evidence must appear there in the SAME
    RELATIVE ORDER. Rephrasing that preserves structure ("is a nurse at" -> "works as a nurse at")
    survives; transposing the arguments of a relation does not.
  * NUMERALS — every number in the claim must occur in the evidence. Digits were previously dropped
    by the >2-char token filter, so amounts were never checked at all.

Both are precision-first: they can reject a faithful-but-heavily-reordered paraphrase. That is the
intended trade — an unstated fact costs recall, a silently transposed one costs trust.
"""
from __future__ import annotations

from .constants import PREDICATE_MIN_SUPPORT

import re

_TOKEN_RE = re.compile(r"[^\w]+")
# A number: digits with optional , _ separators and optional decimal part. Currency/units are
# stripped by tokenization, so "$5,000,000" and "5000000" normalize alike.
_NUM_RE = re.compile(r"\d[\d,_]*(?:\.\d+)?")


def content_tokens(text: str) -> list[str]:
    """Content tokens in order (len>2) — drops articles/copulas like 'is'/'a'/'at'.
    Returns a LIST: order is the whole point of the order check."""
    return [t for t in _TOKEN_RE.sub(" ", text.lower()).split() if len(t) > 2]


def numerals(text: str) -> set[str]:
    """Normalized numeric literals ('$5,000,000' -> '5000000', '3.50' -> '3.5')."""
    out = set()
    for raw in _NUM_RE.findall(text):
        v = raw.replace(",", "").replace("_", "")
        if "." in v:
            v = v.rstrip("0").rstrip(".")          # 3.50 -> 3.5, 4.0 -> 4
        out.add(v or "0")
    return out


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(tok in it for tok in needle)


def order_preserved(claim: str, evidence: str) -> bool:
    """True if the claim's evidence-backed tokens keep the evidence's relative order.

    Only tokens present in the evidence are considered — a claim may add connective words, it may
    not transpose the ones it borrows. "Beta indemnifies Acme" against "Acme indemnifies Beta" gives
    [beta, indemnifies, acme] vs [acme, indemnifies, beta] -> not a subsequence -> False.
    """
    ev = content_tokens(evidence)
    ev_set = set(ev)
    shared = [t for t in content_tokens(claim) if t in ev_set]
    if not shared:
        return True                                 # nothing borrowed to transpose
    return _is_subsequence(shared, ev)


def numerals_grounded(claim: str, evidence: str) -> bool:
    """True if every number asserted by the claim also occurs in the evidence."""
    return numerals(claim) <= numerals(evidence)


# Auxiliaries and inflections that mark where the predicate starts. "s"/"es" endings are
# deliberately NOT treated as verbal: they match plural and Latin nouns ("corpus", "commits"),
# which made an earlier version read "The corpus is gitignored" as having the subject "the".
_AUX_VERBS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "does", "do", "did", "needs", "need", "uses", "use", "can", "will", "would",
    "should", "must", "may", "might",
})
_DETERMINERS = frozenset({"the", "a", "an", "this", "that", "these", "those",
                          "its", "their", "his", "her", "our", "your"})
# An expletive subject refers to nothing, so there is nothing to ground. "There were 199 commits"
# against "199 commits across 7 branches" is faithful; rejecting it would be a false positive.
_EXPLETIVES = frozenset({"there", "it", "they"})


def _verb_index(lowered: list[str]) -> int | None:
    for i, w in enumerate(lowered):
        if w in _AUX_VERBS:
            return i
        if w.endswith(("ed", "ing")) and len(w) > 4:
            return i
    return None


def subject_tokens(claim: str) -> set[str]:
    """The claim's subject: content tokens before the predicate, less determiners.

    Empty when no subject can be identified (no verb found, verb-initial, expletive-only, or
    determiner-only) — in which case there is nothing to check and grounding passes.
    """
    toks = claim.split()
    lowered = [w.lower().rstrip(".,!?;:") for w in toks]
    i = _verb_index(lowered)
    if not i:                                       # None, or verb-initial (imperative)
        return set()
    subj = set(content_tokens(" ".join(toks[:i]))) - _DETERMINERS
    return set() if subj <= _EXPLETIVES else subj - _EXPLETIVES


def subject_grounded(claim: str, evidence: str) -> bool:
    """True if the claim's subject is actually named in the evidence it cites.

    order_preserved constrains the ORDER of borrowed tokens but places no constraint on tokens the
    claim ADDS, so a subject could be invented wholesale and still verify. Measured on the
    212-commit ops run, where this was committed as an active node with a receipt over an exact
    byte range:

        claim    "Local/LM Studio needs a GPU if the design is wrong."
        evidence "If it needs a GPU the design is wrong"

    The source is a design CRITERION — requiring a GPU would mean the design is wrong. The claim
    asserts LM Studio requires one. Every prior check passed. Any subject could be substituted:
    claim_faithful("Kaggle needs a GPU if the design is wrong", ev) was True as well.

    Enrichment is still allowed — a subject that ADDS to a grounded one ("Fireweed Run D" against
    evidence naming "Run D") keeps an overlapping token and passes. Only wholesale invention,
    where no subject token appears in the evidence at all, is rejected.

    KNOWN COST — this rejects cross-sentence coreference, correct and incorrect alike. Checking the
    subject against the whole DOCUMENT instead would keep coreference, but it does not work: the
    source above reads "Local/LM Studio, under an hour. If it needs a GPU the design is wrong." The
    subject is present in the document, one sentence away, so a document-level check passes the
    fabrication. Structurally the bad case is identical to a good one ("Priya joined Acme. She was
    promoted in 2023." -> "Priya was promoted in 2023" cited to "promoted in 2023"): both resolve a
    subject from a neighbouring sentence, and no span-membership rule separates them.

    So this is a soundness/completeness trade, chosen deliberately: a receipt whose byte range does
    not name the subject does not prove the claim, and receipts are the product. The extraction-side
    remedy is for the perceiver to cite a span that INCLUDES the antecedent; that is a prompt
    change, not a gate change, and is not done here.
    """
    subj = subject_tokens(claim)
    if not subj:
        return True
    return bool(subj & set(content_tokens(evidence)))


def predicate_grounded(claim: str, evidence: str) -> bool:
    """Does the claim ASSERT MORE than its evidence does?

    `subject_grounded` constrains who a claim is ABOUT. `order_preserved` constrains the tokens it
    BORROWS. Neither asked whether it ADDS anything, so every one of these was admitted against
    "Marcus Delacroix signed the lease on 14 March 2024 for a term of 36 months." -- and the first
    one was committed WITH A VALID RECEIPT, because a receipt binds the evidence span, not the claim:

        signed the lease IN PARIS / the FRAUDULENT lease / UNDER DURESS / RELUCTANTLY signed

    Every content token of the claim must be supported by the evidence: present lexically, or within
    PREDICATE_MIN_SUPPORT cosine of some evidence token. Token-to-token, for the reason established
    in docs/DESIGN_read_gate.md §2b -- a claim is a bag of topics, a token is an assertion.

    MEASURED (docs/FINDING_predicate_fabrication.md): fabrications land at 0.240-0.359
    (paris 0.24, fraudulent 0.31, reluctantly 0.31, duress 0.34, lawyer/resigned 0.36) and legitimate
    rewording at 0.819+ (`monthly` for `per month`). PREDICATE_MIN_SUPPORT = 0.55 sits in a gap five
    times wider than the Read Gate's.

    NO SUBJECT EXEMPTION, deliberately. `subject_tokens` was tried and is unreliable in BOTH
    directions -- it captured `reluctantly` as part of the subject of "Marcus Delacroix reluctantly
    signed" (exempting the fabrication), and returned EMPTY for "Fireweed Run D achieved 70%
    quality" (exempting nothing). A check that depends on it inherits its failures.

    The consequence, stated rather than hidden: a subject QUALIFIER absent from the span
    ("Fireweed Run D" for "Run D", trap enr-01 at 0.249) is refused here too. In document mode that
    is correct -- the span does not say "Fireweed" -- and it is why this runs only in document mode,
    where a claim is supposed to stay close to its bytes. Turn-derived claims resolve subjects from
    conversational context by design and are NOT subject to this check.

    Degrades to lexical-only when the encoder is unavailable, which REFUSES MORE -- the safe
    direction, and the same trade the Read Gate makes.
    """
    from .read_gate import FUNCTION_WORDS

    ev = [t.lower() for t in content_tokens(evidence)]
    unsupported = [t.lower() for t in content_tokens(claim)
                   if t.lower() not in FUNCTION_WORDS and t.lower() not in ev]
    if not unsupported:
        return True
    ev_content = [t for t in ev if t not in FUNCTION_WORDS]
    if not ev_content:
        return False
    try:
        from .semantic_encoder import similarity
    except Exception:
        return False                      # no encoder -> refuse; safe direction
    try:
        for tok in unsupported:
            if max((similarity(tok, e) for e in ev_content), default=0.0) < PREDICATE_MIN_SUPPORT:
                return False
    except Exception:
        return False
    return True


def claim_faithful(claim: str, evidence: str, document_mode: bool = False) -> bool:
    """The full claim/evidence bind: the subject is real, structure is preserved, numbers are not
    invented. Measured cost of the subject check on the 212-commit ops corpus: 4 of 49 committed
    claims (8%) rejected, all four with a subject absent from their cited span.

    `document_mode` additionally requires that the claim assert no more than its evidence
    (predicate_grounded). It is OFF by default because the seam is the INGESTION PATH, not the
    grounding class: 12 of 22 conversational Maya claims classify `grounded_verbatim`, so scoping by
    class would have destroyed conversational recall while looking principled. Document extraction
    binds claims to byte ranges and must stay near them; turn extraction resolves and rewords by
    design. Measured: demo (document) 7/7 survive, Maya (conversational) 0/22 -- which is exactly
    why the check must never run on the conversational path.
    """
    return (subject_grounded(claim, evidence)
            and order_preserved(claim, evidence)
            and numerals_grounded(claim, evidence)
            and (not document_mode or predicate_grounded(claim, evidence)))


# ── Provenance classes ────────────────────────────────────────────────────────
# External review, 2026-08-19: relaxing the subject check must change the RECORD, not just the
# behaviour. A silent mode flag poisons determinism -- two substrates could fold identical ledgers
# and disagree about what is in them. A labelled class keeps `state = fold(ledger)` intact: the
# ledger remembers WHICH GATE admitted each claim, a review queue can filter on it, and the
# recall/soundness trade becomes a customer choice that is visible in the data rather than a
# configuration nobody can audit after the fact.

GROUNDED_VERBATIM = "grounded_verbatim"   # subject named in the cited span; admissible always
GROUNDED_RESOLVED = "grounded_resolved"   # subject resolved from OUTSIDE the span (coreference)

_ALLOW_RESOLVED = False


def set_resolved_subject_policy(allow: bool) -> bool:
    """Admit `grounded_resolved` claims as well. Returns the previous setting.

    OFF by default: the strict gate is the default because one member of this class is a
    meaning inversion ("Local/LM Studio needs a GPU if the design is wrong" from a span saying the
    opposite) and the gate cannot tell it from a correct pronoun resolution. Turning it on is a
    disclosed mode, and every claim it lets through is stamped GROUNDED_RESOLVED in provenance, so
    the relaxation is always recoverable by query and re-auditable later.
    """
    global _ALLOW_RESOLVED
    prev = _ALLOW_RESOLVED
    _ALLOW_RESOLVED = bool(allow)
    return prev


def resolved_subjects_allowed() -> bool:
    return _ALLOW_RESOLVED


def classify(claim: str, evidence: str) -> str | None:
    """Which gate admits this claim -- or None if no gate does.

    Pure function of (claim, evidence), so it can be recomputed for any stored node at any time.
    That is what makes the retroactive re-audit sweep possible: when the verifier improves, every
    historical claim can be re-classified against the new gate without its source document.
    """
    if not (order_preserved(claim, evidence) and numerals_grounded(claim, evidence)):
        return None
    return GROUNDED_VERBATIM if subject_grounded(claim, evidence) else GROUNDED_RESOLVED


def admissible(claim: str, evidence: str, document_mode: bool = False) -> bool:
    """The gate as the write paths apply it, under the current policy.

    `document_mode` is passed by the DOCUMENT extractor and not by the conversational perceiver.
    A document claim binds to a byte range and must assert no more than that span; a turn claim
    resolves subjects from context and rewords by design. See predicate_grounded for the measurement
    that rules out scoping this by grounding class instead.
    """
    cls = classify(claim, evidence)
    if cls is None:
        return False
    if document_mode and not predicate_grounded(claim, evidence):
        return False
    return _ALLOW_RESOLVED if cls is GROUNDED_RESOLVED else True
