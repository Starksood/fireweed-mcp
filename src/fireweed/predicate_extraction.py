"""The extraction gate — one-directional labelling of claims that are already admitted.

The rule this module exists to respect
--------------------------------------
`docs/FINDING_predicate_representation.md` recorded the blocker as "the extractor must be
deterministic, or it becomes a second place an ungrounded claim can enter." That was the wrong
requirement, and getting it wrong is what stalled the work for weeks.

The extractor does not sit on the admission path. By the time it runs, `grounding.py` has already
checked the claim against its evidence and already decided. This step has no power to admit
anything. It needs to be **one-directional**, not deterministic: it may only LABEL what is already
in, and its worst case must be silence.

    the model (or the bootstrap proposer) proposes:  a slot + the span it read the slot from
    deterministic code checks exactly two things:    slot is in the authored vocabulary
                                                     span lies inside the admitted evidence
    failing either:                                  the claim is admitted WITH NO PREDICATE

Never a refusal. Never a force-fit. An unlabelled claim is still stored, still receipted, still
retrievable by its literal surface form — which is the status quo for every claim today. So a bad
extraction costs a missed read and can never cost an ungrounded fact, and the floor cannot move
down. That asymmetry is the entire safety argument for this module.

What containment does NOT buy — stated here so the code is not read as claiming more
------------------------------------------------------------------------------------
Containment proves the cited span sits INSIDE text that already passed grounding. It does not prove
the span SUPPORTS the slot claimed for it. A proposer can point at grounded words that merely sit
near the concept — words about pay next to a claim that is actually about a job title — without
those words being what makes the slot true.

Containment is therefore necessary and not sufficient. It stops this step from admitting anything
new; it does not stop it from MISLABELLING something already admitted. The residual is a wrong
index entry over a right underlying claim: still literally retrievable, still correctly attributed
to its real span, receipt unchanged. That is strictly less severe than the silent-miss failure it
replaces, but it is real, and it is measured rather than assumed — see `bench/` and Phase 3.6.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .predicate_vocabulary import SLOTS, VOCABULARY_VERSION, is_slot

# Machine-readable outcomes. Every one of these except "typed" yields no predicate and no refusal.
TYPED = "typed"
NO_PROPOSAL = "no_proposal"                       # nothing proposed a slot for this claim
SLOT_NOT_IN_VOCABULARY = "slot_not_in_vocabulary"  # proposer invented a slot; rejected outright
SPAN_EMPTY = "span_empty"
SPAN_NOT_CONTAINED = "span_not_contained"          # cited span is not inside the admitted evidence


@dataclass(frozen=True)
class TypedPredicate:
    """A slot label anchored to the evidence that justified it."""
    slot: str
    span: str                   # the cited text, exactly as it appears in the evidence
    start: int                  # offsets into the ADMITTED EVIDENCE, not the source document
    end: int
    vocabulary_version: str
    proposer: str               # "bootstrap" | "model" — who proposed, for audit


@dataclass(frozen=True)
class Extraction:
    """Outcome of the gate. `predicate is None` is an ordinary result, never an error."""
    predicate: TypedPredicate | None
    reason: str

    @property
    def typed(self) -> bool:
        return self.predicate is not None


def locate(span: str, evidence: str) -> tuple[int, int] | None:
    """Where `span` sits inside `evidence`, or None if it does not sit inside it at all.

    Tolerates case and internal-whitespace differences, because a proposer re-emitting a span it
    read will routinely normalise both, and neither difference can introduce a word the evidence
    does not contain.

    Requires WORD-BOUNDARY alignment at both ends. Plain substring containment would accept "cat"
    against evidence reading "catastrophe" — the span would pass a containment check while pointing
    at nothing that means anything, which is precisely the kind of quiet nonsense this gate exists
    to keep out of the index.
    """
    if not isinstance(span, str) or not isinstance(evidence, str):
        return None
    toks = span.split()
    if not toks:
        return None
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(t) for t in toks) + r"(?!\w)"
    m = re.search(pattern, evidence, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def admit(slot: str | None, span: str | None, evidence: str,
          proposer: str = "model") -> Extraction:
    """The gate. Two checks, and failing either yields no predicate rather than a refusal."""
    if not slot or not span:
        return Extraction(None, NO_PROPOSAL)

    name = slot.strip().lower()
    if not is_slot(name):
        # An invented slot is dropped, never added. The vocabulary is authored by a person; a
        # proposer that could extend it would be authoring the index, which is the one thing the
        # closed list exists to prevent.
        return Extraction(None, SLOT_NOT_IN_VOCABULARY)

    if not span.strip():
        return Extraction(None, SPAN_EMPTY)

    at = locate(span, evidence)
    if at is None:
        return Extraction(None, SPAN_NOT_CONTAINED)

    start, end = at
    return Extraction(
        TypedPredicate(slot=name, span=evidence[start:end], start=start, end=end,
                       vocabulary_version=VOCABULARY_VERSION, proposer=proposer),
        TYPED,
    )


# ── The bootstrap proposer ────────────────────────────────────────────────────
# Deterministic, cue-driven, and explicitly temporary: it exists so the gate can be exercised and
# measured before any model is asked to propose anything. It is NOT the design — see the module
# docstring in predicate_vocabulary.py. Anything it proposes passes through `admit` exactly as a
# model proposal would; it gets no privileged path.

def _cue_hits(text: str) -> list[tuple[int, str, str]]:
    """(cue length, slot, cue) for every authored cue present in `text`, longest first."""
    low = text.lower()
    hits: list[tuple[int, str, str]] = []
    for slot in SLOTS.values():
        for cue in slot.cues:
            if re.search(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", low):
                hits.append((len(cue), slot.name, cue))
    # Longest cue wins; slot name breaks ties so the choice is total and reproducible rather than
    # dependent on dict ordering.
    hits.sort(key=lambda h: (-h[0], h[1], h[2]))
    return hits


def propose(claim: str, evidence: str) -> tuple[str, str] | None:
    """Propose (slot, span) for a claim, or None.

    The cue must appear in BOTH the claim and the evidence. Requiring it in the claim is what keeps
    the label about the claim rather than about the paragraph the claim was drawn from: evidence
    mentioning a cat, supporting a claim about someone's job, must not type that claim `pet`. This
    costs nothing — a cue absent from the claim was never evidence for the claim's own slot — and
    it removes the largest source of the mislabelling containment cannot catch.
    """
    if not claim or not evidence:
        return None
    for _, slot, cue in _cue_hits(claim):
        if locate(cue, evidence) is not None:
            return slot, cue
    return None


def extract(claim: str, evidence: str) -> Extraction:
    """Bootstrap end-to-end: propose, then submit the proposal to the same gate."""
    proposed = propose(claim, evidence)
    if proposed is None:
        return Extraction(None, NO_PROPOSAL)
    slot, span = proposed
    return admit(slot, span, evidence, proposer="bootstrap")
