"""First-person → speaker rewriting for conversational perception.

The core personal-memory pattern is a user talking about themselves ("I live in Seattle", "I moved
jobs"). The perceiver extracts third-person claims well but leaves first-person ones unanchored — the
subject "I" never resolves to the speaker, so self-facts aren't retrievable and updates don't supersede.

This is a DETERMINISTIC preprocessing step (pure code; resolver purity preserved): given the turn's
speaker, rewrite first-person references to that speaker's name BEFORE the LLM perceiver sees the text,
so "I moved to Seattle" (from Maya) becomes "Maya moved to Seattle" — which the perceiver then anchors
to entity Maya exactly like any third-person claim.
"""
from __future__ import annotations
import re

# Order matters: contractions before the bare pronoun. Case-insensitive, word-boundaried.
_RULES = [
    (r"\bI'm\b", "{s} is"), (r"\bI've\b", "{s} has"), (r"\bI'll\b", "{s} will"),
    (r"\bI'd\b", "{s} would"), (r"\bmyself\b", "{s}"),
    (r"\bI\b", "{s}"), (r"\bme\b", "{s}"), (r"\bmine\b", "{s}'s"),
    (r"\bmy\b", "{s}'s"),
]


def rewrite_first_person(text: str, speaker: str) -> str:
    """Rewrite first-person references in `text` to `speaker`. No-op if speaker is falsy.

    Conservative: only the fixed first-person tokens are touched (I/I'm/I've/I'll/I'd/me/my/mine/myself);
    everything else is left verbatim, so third-person content and other people's names are untouched.
    """
    if not speaker:
        return text
    out = text
    for pat, repl in _RULES:
        out = re.sub(pat, repl.format(s=speaker), out, flags=re.IGNORECASE)
    # tidy a doubled possessive if the speaker name already ended in s (rare); keep it simple otherwise
    return out
