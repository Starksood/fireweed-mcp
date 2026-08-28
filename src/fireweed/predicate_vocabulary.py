"""The authored predicate vocabulary — a closed, versioned list of the slots a claim can be
indexed under.

Why this file is data and not code
----------------------------------
`read_gate.build_vocabulary` indexes the surface tokens a source document happened to contain.
Nobody wrote that vocabulary, nobody can audit it, and it changes silently with every ingest. So
`unknown_predicate` today conflates two completely different answers:

    "the substrate holds no fact of that kind"        <- a true, useful negative
    "the substrate holds it under different words"    <- a recall failure wearing the same label

A closed slot list separates them. A question whose head maps to a slot the substrate has never
filled is a *typed miss* — the schema has no compensation slot for this entity — and that is a
different sentence than "no claims ground 'salary'".

The list below is **edited by a person, never extended by a model.** That is the whole point: the
index vocabulary becomes an object you can read, diff, review and argue with, on the same standard
as every other decision surface in this project. `VOCABULARY_VERSION` is stamped onto every typed
predicate so a stored label can be traced to the list that produced it.

The two halves of an entry
--------------------------
`asks`   — the question heads that DEMAND this slot. Read side. "what pet", "which animal".
`cues`   — the phrases whose presence PROPOSES this slot. Write side, bootstrap proposer only.

On the bootstrap proposer's cues, honestly
------------------------------------------
For value-typed slots (`pet`, `sport`, `diet`) the cues are the filler terms, and they are pulled
straight from `lexical_relations.HYPERNYMS` rather than retyped — one table, no drift. So the
bootstrap proposer is, in part, the same curated filler enumeration the hypernym table already was.

That is deliberate and it is not the contribution. The contribution is the *gate* in
`predicate_extraction.py`: whatever proposes a slot — this cue table today, a model proposal at the
MCP surface tomorrow, a real parser later — passes the identical two checks and produces the
identical auditable label. The proposer is replaceable; the gate is not.

What does change immediately: once a slot is attached at write time, the READ side stops needing
the filler table at all. "What kind of pet?" resolves head -> slot -> "is any claim about this
entity typed `pet`?", instead of hoping the question's noun and the source's noun share a token.
The filler enumeration stops being load-bearing for retrieval and becomes a bootstrap detail.
"""
from __future__ import annotations

from dataclasses import dataclass

from .lexical_relations import HYPERNYMS

# Bumped whenever a slot is added, removed, or renamed, or its `asks` set changes. Stamped onto
# every TypedPredicate so a label in an old store can be traced to the list that produced it.
VOCABULARY_VERSION = "1"


@dataclass(frozen=True)
class Slot:
    """One canonical predicate slot."""
    name: str
    description: str            # what fills this slot, in a sentence — for review, not for code
    asks: frozenset[str]        # question heads that demand it
    cues: frozenset[str]        # bootstrap proposer: phrases that propose it


def _slot(name: str, description: str, asks: set[str], cues: set[str]) -> Slot:
    return Slot(name, description, frozenset(asks), frozenset(c.lower() for c in cues))


# Relation-marked slots: the cue is the RELATION phrase, not the value. These are the slots where
# an authored cue list is genuinely small and stable — there are few ways to say "is employed by"
# and unboundedly many employers.
_RELATION_SLOTS: list[Slot] = [
    _slot("employer", "the organisation a person works for",
          {"employer", "company", "workplace", "firm", "organisation", "organization"},
          {"works at", "works for", "working at", "working for", "employed by", "employed at",
           "joined", "hired by", "hired at", "works with", "is at"}),
    _slot("residence", "where a person lives",
          {"residence", "address", "home", "city", "town", "neighbourhood", "neighborhood",
           "flat", "apartment"},
          {"lives in", "lives at", "lives on", "living in", "living at", "moved to", "based in",
           "resides in", "relocated to", "rents in", "rents a"}),
    _slot("salary", "what a person is paid",
          {"salary", "pay", "wage", "compensation", "income", "earnings", "rate"},
          {"salary", "earns", "is paid", "paid a", "compensation", "wage", "annual pay",
           "per hour", "per year", "makes"}),
    _slot("shift", "the working hours a person is scheduled for",
          {"shift", "rota", "roster", "hours", "schedule"},
          {"shift", "rota", "roster", "on shift", "works nights", "works mornings",
           "opening shift", "closing shift"}),
    _slot("health_condition", "a diagnosed condition, injury or illness",
          {"condition", "illness", "diagnosis", "injury", "ailment", "disease", "health"},
          {"diagnosed with", "suffers from", "recovering from", "treated for", "was injured",
           "injury", "illness", "condition", "asthma", "diabetes", "migraine", "allergy",
           "allergic to", "fracture", "sprain"}),
    _slot("family_relation", "a named family or household relationship",
          {"family", "relative", "relation", "parent", "sibling", "spouse", "partner"},
          {"father", "mother", "brother", "sister", "son", "daughter", "wife", "husband",
           "partner", "spouse", "parents", "sibling", "cousin", "aunt", "uncle",
           "grandmother", "grandfather"}),
    _slot("commute_mode", "how a person travels to work",
          {"commute", "transport", "transit", "travel"},
          {"takes the bus", "takes the train", "commutes by", "commutes on", "cycles to",
           "walks to", "drives to", "rides the", "catches the"}),
]

# Value-marked slots: the slot is identified by its FILLER, so the bootstrap cues are the filler
# terms. Derived from HYPERNYMS rather than retyped — see the module docstring on why this is a
# bootstrap detail rather than the design.
#
# ORDERED MOST-SPECIFIC FIRST, and that ordering is load-bearing. HYPERNYMS deliberately unions the
# general categories for read-side rescue — `hobby` there contains every sport and every instrument,
# because a question about a hobby should be satisfied by a stored sport. That is right for widening
# a QUESTION and wrong for labelling a CLAIM: it typed "Dana plays basketball" as `hobby`, burying
# the more specific slot the substrate could have answered `sport` with. So each slot below drops
# every cue a more specific slot already claims, making the value cue sets disjoint. Widening stays
# where it belongs, on the read side, as a declared policy layer.
_VALUE_SLOTS: list[tuple[str, str, set[str], str]] = [
    ("pet",        "an animal a person keeps",          {"pet", "animal", "companion"},     "pet"),
    ("instrument", "an instrument a person plays",      {"instrument"},                     "instrument"),
    ("sport",      "a sport a person plays",            {"sport", "game", "athletics"},     "sport"),
    ("vehicle",    "a vehicle a person owns or drives", {"vehicle", "car", "bike"},         "vehicle"),
    ("language",   "a language a person speaks",        {"language", "tongue"},             "language"),
    ("diet",       "a dietary practice a person keeps", {"diet", "dietary", "eating"},      "diet"),
    ("music_taste", "music a person listens to",        {"music", "genre", "band", "artist"}, "music"),
    ("occupation", "the kind of work a person does",    {"occupation", "job", "profession", "role",
                                                         "title", "work", "career", "living",
                                                         "employment"},                     "job"),
    ("hobby",      "a leisure activity a person does",  {"hobby", "pastime", "activity", "doing"}, "hobby"),
]

SLOTS: dict[str, Slot] = {s.name: s for s in _RELATION_SLOTS}
_claimed: set[str] = set()
for _name, _desc, _asks, _hyp in _VALUE_SLOTS:
    _cues = {c.lower() for c in HYPERNYMS.get(_hyp, frozenset())} - _claimed
    _claimed |= _cues
    SLOTS[_name] = _slot(_name, _desc, _asks, _cues)

# head -> slot. Built once. A head demanding two slots would be a vocabulary bug, so the build
# asserts uniqueness rather than silently letting one win: an ambiguous head is exactly the case
# where a quiet choice produces a wrong refusal message nobody can explain later.
_ASK_INDEX: dict[str, str] = {}
for _s in SLOTS.values():
    for _a in _s.asks:
        _prior = _ASK_INDEX.get(_a)
        if _prior is not None and _prior != _s.name:
            raise ValueError(
                f"predicate vocabulary: ask term {_a!r} demands both {_prior!r} and {_s.name!r}; "
                "a head must map to exactly one slot"
            )
        _ASK_INDEX[_a] = _s.name


def is_slot(name: str) -> bool:
    """Is `name` a member of the authored vocabulary?"""
    return isinstance(name, str) and name.strip().lower() in SLOTS


def slot_for_head(head: str) -> str | None:
    """The slot a question head demands, or None if the head names no slot at all.

    None is meaningful and is NOT a failure: it means the question is not asking for anything this
    schema types, so a refusal cannot honestly be reported as a typed miss.
    """
    if not isinstance(head, str):
        return None
    return _ASK_INDEX.get(head.strip().lower())


def slot_names() -> list[str]:
    """Every slot, sorted. For the refusal message and for review tooling."""
    return sorted(SLOTS)
