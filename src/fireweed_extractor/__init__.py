"""fireweed-extractor — an UNTRUSTED companion that turns conversation into claim/evidence pairs.

WHY THIS IS A SEPARATE PACKAGE
------------------------------
Fireweed's write contract requires a caller to supply a claim AND the verbatim span supporting it.
Two independent reviewers identified this as the adoption blocker: every competing memory system
accepts `conversation in, memory out`, and requiring the caller to do extraction themselves forces
an architecture change before anyone sees any value.

The obvious fix -- put a model inside Fireweed and extract automatically -- would destroy the only
property Fireweed has. So the model goes OUTSIDE, and the trust boundary stays where it is:

    conversation
         |
         v
    extractor (a language model, UNTRUSTED, may hallucinate freely)
         |
         +--> proposed claim
         +--> proposed evidence span
                    |
                    v
         Fireweed's deterministic gate
                    |
            +-------+-------+
            v               v
         admitted        refused

**The extractor is never trusted and never needs to be.** It can hallucinate, misread, quote the
wrong sentence, or invent a fact wholesale. Every one of those failures is caught by the same
deterministic checks that already guard the write path, because a proposal is just a claim like any
other. A bad extraction costs a missed memory. It cannot cost a false one.

That is the whole design, and it is why this package is allowed to use a model at all.

WHAT THIS PACKAGE MAY NOT DO
----------------------------
- It may not write to a store directly. It returns proposals; the caller submits them.
- It may not be imported by `fireweed.*`. The dependency arrow points one way, enforced by a test.
- It may not soften, retry, or "fix" a rejection. A refusal is the system working.
"""

from .extract import Proposal, ExtractionResult, extract, LMStudioProposer, EchoProposer

__all__ = ["Proposal", "ExtractionResult", "extract", "LMStudioProposer", "EchoProposer"]
