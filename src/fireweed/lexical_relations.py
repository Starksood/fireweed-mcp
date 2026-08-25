"""Zero-dependency hypernym relations for the Read Gate's predicate check.

Why this exists
---------------
`pip install fireweed-mcp` declares no dependencies; the semantic encoder is an optional extra.
So a default install runs check 2 in lexical_only mode, where a question head must appear
*verbatim* in the substrate's vocabulary. Measured on the 410-item answerable corpus:

    semantic mode (encoder installed)   abstained 145/410 = 35.4%
    lexical-only  (DEFAULT install)     abstained 369/410 = 90.0%

Nine of ten answerable questions were refused, and the refusals were not subtle -- "what kind of
pet does X have?" was refused by a substrate whose grounded terms included *cat*. A first-run
developer reasonably concludes the software is broken.

The encoder is not the right fix for a default install: it costs a multi-gigabyte torch download
for a package whose selling point is that it has no dependencies. What the failing cases actually
need is not embedding similarity but a handful of ordinary hypernyms -- pet/cat, sport/basketball,
animal/dog. Those are enumerable, auditable, and deterministic, which suits a system whose thesis
is that deterministic code decides.

Scope and honesty
-----------------
This is a curated table, not a knowledge base. It covers the common attribute classes people ask
memory systems about. It will miss things, and a miss produces a refusal -- the safe direction.
It never invents an answer: it only permits the gate to consult claims it already holds, and
every returned claim still carries its own receipt. Extending it is a data edit, not a code change.
"""
from __future__ import annotations

# head term -> concrete terms that would satisfy a question about it.
# Deliberately one-directional: asking for "pet" is satisfied by "cat", but asking about "cat" is
# NOT satisfied by an unrelated pet. Broadening a specific question is how you get wrong answers.
HYPERNYMS: dict[str, frozenset[str]] = {
    "pet":      frozenset({"cat", "dog", "puppy", "kitten", "parrot", "hamster", "rabbit",
                           "bird", "fish", "snake", "lizard", "turtle", "ferret", "horse"}),
    "animal":   frozenset({"cat", "dog", "puppy", "kitten", "parrot", "hamster", "rabbit",
                           "bird", "fish", "snake", "lizard", "turtle", "ferret", "horse", "pet"}),
    "sport":    frozenset({"basketball", "football", "soccer", "baseball", "tennis", "hockey",
                           "volleyball", "golf", "running", "run", "swimming", "swim", "cycling",
                           "marathon", "boxing", "skiing", "surfing", "yoga", "climbing"}),
    "music":    frozenset({"metal", "rock", "jazz", "pop", "country", "classical", "blues",
                           "hiphop", "rap", "punk", "folk", "techno", "indie", "metallica"}),
    "genre":    frozenset({"metal", "rock", "jazz", "pop", "country", "classical", "blues",
                           "hiphop", "rap", "punk", "folk", "techno", "indie", "horror",
                           "romance", "fantasy", "thriller", "comedy"}),
    "food":     frozenset({"pizza", "pasta", "sushi", "burger", "salad", "steak", "cheese",
                           "bread", "rice", "soup", "curry", "tacos", "chocolate", "vegan",
                           "vegetarian", "seafood", "barbecue"}),
    "drink":    frozenset({"coffee", "tea", "beer", "wine", "juice", "soda", "water", "whisky",
                           "cocktail", "espresso", "latte"}),
    "instrument": frozenset({"guitar", "piano", "drums", "bass", "violin", "cello", "flute",
                             "saxophone", "trumpet", "keyboard", "ukulele"}),
    "job":      frozenset({"nurse", "doctor", "teacher", "engineer", "waiter", "waitress",
                           "lawyer", "chef", "driver", "artist", "writer", "developer",
                           "programmer", "manager", "accountant", "designer", "student",
                           "mechanic", "electrician", "plumber", "librarian", "scientist"}),
    "work":     frozenset({"nurse", "doctor", "teacher", "engineer", "waiter", "waitress",
                           "lawyer", "chef", "driver", "artist", "writer", "developer",
                           "programmer", "manager", "accountant", "designer", "job", "career"}),
    "hobby":    frozenset({"reading", "hiking", "painting", "gaming", "cooking", "baking",
                           "knitting", "sewing", "gardening", "fishing", "photography",
                           "drawing", "writing", "dancing", "camping", "climbing", "chess"}),
    "vehicle":  frozenset({"car", "truck", "motorcycle", "bike", "bicycle", "van", "scooter",
                           "jeep", "suv", "boat"}),
    "colour":   frozenset({"red", "blue", "green", "black", "white", "yellow", "orange",
                           "purple", "pink", "brown", "grey", "gray"}),
    "language": frozenset({"english", "spanish", "french", "german", "chinese", "japanese",
                           "korean", "italian", "portuguese", "hindi", "arabic", "russian"}),
}
HYPERNYMS["color"] = HYPERNYMS["colour"]
HYPERNYMS["profession"] = HYPERNYMS["job"]
HYPERNYMS["occupation"] = HYPERNYMS["job"]
HYPERNYMS["breed"] = HYPERNYMS["pet"]
HYPERNYMS["cuisine"] = HYPERNYMS["food"]


def hyponyms_of(head: str) -> frozenset[str]:
    """Concrete terms that would answer a question whose head is `head`. Empty when unknown."""
    return HYPERNYMS.get(head.lower().strip(), frozenset())


def grounded_hyponym(head: str, is_grounded) -> str | None:
    """The first term under `head` that `is_grounded(term)` accepts, or None.

    `is_grounded` is supplied by the caller (the gate's vocabulary), so this module never touches
    the substrate and stays a pure data lookup.
    """
    for term in sorted(hyponyms_of(head)):
        if is_grounded(term):
            return term
    return None


# The suffix heuristic above recognises a verb only by an -s/-ed/-ing/-es ending, so it misses every
# English irregular past tense. Measured on sixteen ordinary sentences, NINE were rejected as
# "fragment": "Ada Lovelace wrote the first algorithm", "Dana Kim went to Berlin last year", "Sam
# Okafor built a treehouse". Worse, the ones that passed did so by accident -- "Marcus Webb sold
# his bookshop" survives because *his* ends in s, not because *sold* was recognised.
#
# A claim rejected here is a claim the user is told nothing about, so this list is not cosmetic:
# it is the difference between a memory system storing an ordinary sentence and silently discarding
# it. Irregular verbs are a closed class, so enumerating them is a complete fix, not a heuristic.
IRREGULAR_VERBS = frozenset("""
ate awoke bade became began bent bet bit bled blew bore bought bred brought built burnt
came caught chose clung cost crept cut dealt dug did drank drew drove dwelt fed felt fought
found fled flew flung forbade forgot forgave froze got gave went ground grew hung had heard
hid hit held hurt kept knelt knew laid led leant leapt learnt left lent let lay lit lost made
meant met mistook mowed overcame paid put quit read rode rang rose ran said saw sought sold
sent set sewn shook shed shone shot showed shrank shut sang sank sat slept slid slit smelt
sowed spoke sped spelt spent spilt spun spat split spread sprang stood stole stuck stung
stank strode struck strung strove swore swept swelled swam swung took taught tore told thought
threw thrust trod understood upset woke wore wove wept won wound withdrew wrung wrote
""".split())


# base form -> inflected forms, for the Read Gate's predicate lookup. The firewall only needs to
# know that a token IS a verb; the gate needs to match "What did Ada Lovelace write?" against a
# stored "Ada Lovelace wrote the first algorithm", which the -s fold cannot do. Kept to verbs that
# plausibly appear in a durable claim about someone.
IRREGULAR_FORMS: dict[str, tuple[str, ...]] = {
    "write": ("wrote", "written"),      "take": ("took", "taken"),
    "go": ("went", "gone"),             "build": ("built",),
    "buy": ("bought",),                 "win": ("won",),
    "speak": ("spoke", "spoken"),       "fly": ("flew", "flown"),
    "make": ("made",),                  "sell": ("sold",),
    "grow": ("grew", "grown"),          "teach": ("taught",),
    "drive": ("drove", "driven"),       "run": ("ran",),
    "give": ("gave", "given"),          "find": ("found",),
    "lead": ("led",),                   "lose": ("lost",),
    "eat": ("ate", "eaten"),            "drink": ("drank", "drunk"),
    "see": ("saw", "seen"),             "know": ("knew", "known"),
    "hold": ("held",),                  "keep": ("kept",),
    "leave": ("left",),                 "meet": ("met",),
    "pay": ("paid",),                   "sit": ("sat",),
    "tell": ("told",),                  "think": ("thought",),
    "bring": ("brought",),              "catch": ("caught",),
    "choose": ("chose", "chosen"),      "wear": ("wore", "worn"),
    "sing": ("sang", "sung"),           "swim": ("swam", "swum"),
    "ride": ("rode", "ridden"),         "rise": ("rose", "risen"),
    "send": ("sent",),                  "spend": ("spent",),
    "stand": ("stood",),                "understand": ("understood",),
    "become": ("became",),              "begin": ("began", "begun"),
    "break": ("broke", "broken"),       "feel": ("felt",),
    "fight": ("fought",),               "forget": ("forgot", "forgotten"),
    "hear": ("heard",),                 "read": ("read",),
    "say": ("said",),                   "study": ("studied",),
}

_INVERSE: dict[str, str] = {}
for _b, _fs in IRREGULAR_FORMS.items():
    for _f in _fs:
        _INVERSE.setdefault(_f, _b)


def verb_forms(token: str) -> list[str]:
    """Other inflections of `token` when it is a known irregular verb, either direction."""
    t = token.lower().strip()
    out: list[str] = []
    if t in IRREGULAR_FORMS:
        out.extend(IRREGULAR_FORMS[t])
    base = _INVERSE.get(t)
    if base:
        out.append(base)
        out.extend(f for f in IRREGULAR_FORMS[base] if f != t)
    return [v for v in dict.fromkeys(out) if v != t]
