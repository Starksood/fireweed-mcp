"""Phase 8 — deterministic pronoun coreference resolver. Pure functions only."""
from __future__ import annotations
import re
from .graph import EntityRef

SUBJECT_PRONOUNS: frozenset[str] = frozenset({"she", "he", "they", "i", "we"})
OBJECT_PRONOUNS:  frozenset[str] = frozenset({"her", "him", "them", "me", "us"})
ALL_PRONOUNS:     frozenset[str] = SUBJECT_PRONOUNS | OBJECT_PRONOUNS

# Anchor coreference resolves THIRD-PERSON pronouns ("she went..." → the established subject).
# FIRST-PERSON pronouns ("I"/"we"/"me"/"us") refer to the narrator, who in a first-person stream
# is the *implicit* self, NOT a named anchor entity. Attaching the anchor to every "I" claim was
# the over-attachment bug: in Maya's first-person corpus it pinned the (mis-roled) anchor onto
# claims that never name it ("I was doing three runs a week" → ent_van_ness). So the anchor is
# attached only when a third-person pronoun is present.
FIRST_PERSON_PRONOUNS: frozenset[str] = frozenset({"i", "we", "me", "us"})
THIRD_PERSON_PRONOUNS: frozenset[str] = frozenset({"she", "he", "they", "her", "him", "them"})

_PRONOUN_RE = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in ALL_PRONOUNS) + r')\b',
    re.IGNORECASE,
)
_THIRD_PERSON_RE = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in THIRD_PERSON_PRONOUNS) + r')\b',
    re.IGNORECASE,
)


def has_pronoun(claim: str) -> bool:
    """True iff claim contains a pronoun from ALL_PRONOUNS as a whole word (case-insensitive)."""
    try:
        return bool(_PRONOUN_RE.search(claim))
    except Exception:
        return False


def has_third_person_pronoun(claim: str) -> bool:
    """True iff claim contains a THIRD-person pronoun (she/he/they/her/him/them) as a whole word.
    First-person ('I'/'we'/'me'/'us') is excluded — it refers to the narrator, not the anchor."""
    try:
        return bool(_THIRD_PERSON_RE.search(claim))
    except Exception:
        return False


def resolve_pronouns(
    claim: str,
    anchor_entity_id: str | None,
    existing_refs: list[EntityRef],
) -> list[EntityRef]:
    """Append the anchor EntityRef only if the claim has a THIRD-PERSON pronoun, the anchor is set,
    and the anchor is not already linked. First-person claims get no anchor (the self is implicit)."""
    try:
        if not anchor_entity_id or not has_third_person_pronoun(claim):
            return list(existing_refs)
        if any(ref.entity_id == anchor_entity_id for ref in existing_refs):
            return list(existing_refs)
        return list(existing_refs) + [EntityRef(entity_id=anchor_entity_id, role="actor")]
    except Exception:
        return list(existing_refs)
