"""The Read Gate — admission control for the CONVERSATION, mirroring the firewall's admission
control for the GRAPH.

Design + measurements: `docs/DESIGN_read_gate.md`. Reviews: `docs/REVIEW_read_gate_proposal.md`.

The write side has always adjudicated: a claim enters the graph only if a pure function says its
evidence supports it. The read side scored instead — `jaccard >= 0.12 OR coverage >= 0.6` — and a
score cannot say "the substrate does not know this". Live, that answered "Priya's salary" with a
hire date: the entity matched, the predicate did not, and coverage weights every query token
equally.

A question is a claim with a hole in it. This module asks whether the hole has a filler:

    check 1  subject grounding    — every named subject resolves to a graph entity
    check 2  predicate grounding  — the demand head is grounded, lexically or semantically

Both are pure functions of (question, graph). Determinism holds: the encoder is pinned and
hash-verified (NorthStar guardrail 1, amended), and falls back to lexical-only when unavailable —
a fallback that can only ABSTAIN MORE, never answer more.

Check 3 (object typing) is deliberately absent; `docs/DESIGN_read_gate.md` §5 records why and what
would re-open it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

from .graph import GraphState
from .constants import (
    READ_GATE_MIN_PREDICATE_SIM,
    READ_GATE_DF_CAP,
)
from .lexical_relations import grounded_hyponym, verb_forms
from .predicate_vocabulary import slot_for_head, SLOTS

# ── The grammar layer — closed, declared, INSPECTABLE ─────────────────────────
# The review's Correction 2: a closed function-word list is unavoidable, because the same predicate
# cannot both discard function words and catch unsupported topic words. What it buys is that the
# list stops being a tuning knob buried in a similarity score and becomes a reviewable declaration.
# These words are consumed by the grammar layer and NEVER scored as content.
FUNCTION_WORDS: frozenset[str] = frozenset({
    # interrogatives
    "what", "which", "when", "where", "who", "whom", "whose", "why", "how",
    # auxiliaries and copulas
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # determiners and quantifiers
    "the", "a", "an", "this", "that", "these", "those", "any", "some", "each",
    "every", "no", "all", "both", "either", "neither", "such", "same", "other",
    # prepositions and particles
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "without", "about",
    "into", "onto", "over", "under", "after", "before", "between", "through",
    "during", "up", "down", "out", "off", "as", "than", "there", "here", "again",
    # pronouns and possessives
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "theirs",
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    # conjunctions and common adverbs with no predicate content
    "and", "or", "but", "if", "not", "so", "because", "while", "also", "just",
    "very", "too", "still", "yet", "ever", "never", "only", "even", "back",
    # deictic time and frequency adverbs — they modify WHEN a predicate holds, they are not
    # themselves predicates. "What does Priya prefer now?" demands `prefer`, not `now`; without
    # these the gate takes the trailing adverb as the demand head and refuses an answerable
    # question. Measured on tests/validity/test_temporal_contradiction_behavior.py.
    "now", "then", "today", "tomorrow", "yesterday", "currently", "recently",
    "already", "soon", "later", "earlier", "often", "always", "usually",
    "sometimes", "again", "anymore", "longer",
})

# Wh-words that take a nominal complement: "what BUS", "which SHIFT", "whose FLAT".
_WH_DETERMINER: frozenset[str] = frozenset({"what", "which", "whose"})
# PARTITIVE HEADS. "What KIND OF music" asks about music, not about kinds. Taking the token after
# the wh-word yields `kind`, which appears in no substrate, so the gate refuses EVERY question of
# this shape unconditionally. Measured before this fix: 4 of 6 answerable categories scored ~0
# (music 1/152, pet 0/118, work 0/156, diet 0/52) purely because of the phrasing — 478 of 734 rows.
# The demand is the noun the partitive governs.
_PARTITIVE: frozenset[str] = frozenset({"kind", "type", "sort", "form", "variety"})
# Wh-words that demand a typed object rather than naming one.
_WH_TYPED: dict[str, str] = {"where": "location", "when": "date", "who": "person"}
# "how many/much/long X" — the demand is the noun, not the degree word.
_HOW_DEGREE: dict[str, str] = {"many": "quantity", "much": "quantity", "long": "duration"}
_AUXILIARIES: frozenset[str] = frozenset({
    "is", "are", "was", "were", "do", "does", "did", "has", "have", "had",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
})

# Should a typed miss be allowed to OVERRULE the widening layers below it?
#
# Measured 2026-08-27 on 1,201 answerable + 594 held-out category traps, write path held fixed:
#
#     typed lookup only (this False)     answerable refusal  -0.2 pts    trap refusal  +0.2 pts
#     + authority       (this True)      answerable refusal +16.9 pts    trap refusal  +0.2 pts
#
# The authority rule fixes a real false accept -- an unscoped hypernym rescue answering "What pet
# does Dana have?" with Marcus's cat -- but at the CURRENT typed coverage (37% of nodes carry a
# slot) it refuses 203 answerable questions to do it. An entity usually has *some* typed claim, so
# the rule arms itself, while the specific claim holding the answer is untyped, so the miss is an
# artefact of sparse coverage rather than a fact about the store.
#
# That is precisely the failure the design predicted and said to measure rather than assume: a
# mislabel costs a missed read. It costs too many. The precision win is real and worth having, so
# this stays as a switch with its measurement attached rather than being deleted -- it becomes
# shippable when coverage is high enough that an untyped claim is the exception. Re-run
# bench/predicate_slots/run_phase3.py before flipping it.
TYPED_MISS_OVERRULES_RESCUE = False

# Should check 2 fold its vocabulary over the SUBJECT'S claims instead of the whole store?
#
# The gate resolves the subject in check 1 and then throws that resolution away: check 2 asks
# whether the CORPUS grounds the demand head, never whether this SUBJECT does. In a single-entity
# store the two are the same question, and every evaluation corpus this project has ever built puts
# one persona in one store — so the difference was structurally unmeasurable until
# bench/predicate_slots/cross_entity_trap_suite.jsonl existed.
#
# Measured 2026-08-27, 1,201 answerable + 594 category traps + 300 cross-entity traps:
#
#                       answerable refusal   trap refusal   cross-entity refusal
#     corpus-wide (off)        3.7%              96.1%             0.3%
#     subject-scoped (on)      4.8%              96.1%            95.3%
#
# 0.3% is not a typo. With the scope off, a store holding two people answers 299 of 300 questions
# about a person who has no such fact, using the OTHER person's. Scoping costs 1.0 point of
# answerable refusal and returns 95 points of that back, at no cost to trap refusal.
#
# This is the fourth instance of this codebase's signature defect: a decision is computed (check 1
# resolves the subject) and then discarded before the code that needs it (check 2 never asks who).
# Full measurement and the two harness defects found on the way:
# docs/FINDING_read_side_grounding_is_unscoped.md
SCOPE_PREDICATE_GROUNDING_TO_SUBJECT = True

# Let the encoder NAME A SLOT, and let deterministic code decide whether to answer.
#
# `_semantic_rescue` below compares the demand head against every token the corpus happens to
# contain -- an unbounded, corpus-dependent target set. Measured, that collapses trap refusal from
# 96.1% to 32.8%: with enough tokens in the store, something is always close enough to something.
#
# The target set is the defect, not the encoder. Constrained to the sixteen AUTHORED slots, the
# encoder can only ever answer "which named kind of fact is this question asking for?" -- and the
# answer is then handed to the same deterministic scope check that authorizes a lexically-typed
# demand. The encoder widens what the question can be UNDERSTOOD as; it never widens what counts
# as an answer, and it cannot produce one for an entity that carries no claim of that kind.
#
# This is the read-side form of the write-side rule the whole project runs on: the model proposes,
# deterministic code decides.
# Measured 2026-08-28. 900 held-out answerable, 594 held-out-phrasing traps, 300 cross-entity
# traps. Every item in the trap corpora forces this path to run -- the earlier category traps
# could not, because all five of their heads resolve lexically and the encoder never fired.
#
#   threshold   recall   held-out-phrasing trap refusal
#     lexical     0.8%        99.8%      (the safe, useless end)
#     0.45       41.1%        93.6%
#     0.50       38.7%        96.1%
#     0.60       35.2%        99.8%      <- +34.4 points of recall at NO measured safety cost
#     0.70       30.9%        99.8%
#   unbounded    39.8%        91.2%      and 32.8% on the category traps
#
# ON THE CHOICE OF 0.60 -- it was selected by sweeping the same corpora reported above, which is
# fitting to the test set, and this project has retracted two results for exactly that. The honest
# statement is that the SWEEP is the finding: a frontier exists, and it is not the one naive
# semantic expansion traces. A single operating point needs a fresh validation split before it can
# be quoted as a system property. 0.50 was the pre-registered value and also clears the bar.
SLOT_RESCUE_BY_ENCODER = False
SLOT_RESCUE_MIN_SIM = 0.60

_WORD_RE = re.compile(r"[^\w']+")


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens, possessives normalized to the base form."""
    out: list[str] = []
    for t in _WORD_RE.sub(" ", text.lower()).split():
        t = t.strip("'")
        if t.endswith("'s"):
            t = t[:-2]
        if t:
            out.append(t)
    return out


def _content(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in FUNCTION_WORDS]


# ── The demand — the hole in the claim ────────────────────────────────────────

@dataclass(frozen=True)
class Demand:
    """What the question asks the substrate to supply."""
    wh: str | None          # "what" | "how" | "why" | "bool" | None (bare keyword query)
    head: str | None        # the demand head — the token that must be grounded
    typed: str | None       # declared object-type demand, when the wh-word carries one


def parse_demand(question: str) -> Demand:
    """Extract the hole. Pure, never raises. See docs/DESIGN_read_gate.md §4 for the table."""
    toks = _tokens(question)
    if not toks:
        return Demand(None, None, None)
    content = _content(toks)
    first = toks[0]

    # "how many/much/long <NOUN>" — the demand is the noun before the first auxiliary, NOT the
    # degree word. "How many extra minutes can a Van Ness delay add?" demands `minutes`; taking the
    # first content token instead yields `extra`, which is ungrounded and abstains on a question the
    # substrate can answer.
    if first == "how" and len(toks) > 1 and toks[1] in _HOW_DEGREE:
        typed = _HOW_DEGREE[toks[1]]
        head = None
        for t in toks[2:]:
            if t in _AUXILIARIES:
                break
            if t not in FUNCTION_WORDS and t not in _HOW_DEGREE:
                head = t                      # keep advancing: the LAST noun before the aux
        return Demand("how", head, typed)

    # "what/which/whose <NOUN> …" — the nominal complement is the demand. The complement may be a
    # COMPOUND ("what health issue"), where the head is the LAST noun, not the first: taking toks[1]
    # demands `health` where the question demands `issue`. Bounded at two tokens because without a
    # POS tagger there is nothing to stop the scan at the verb ("affected" is not an auxiliary), and
    # an unbounded scan swallows the predicate.
    if first in _WH_DETERMINER and len(toks) > 1:
        nxt = toks[1]
        # "what kind of X" / "what type of X" -> the demand is X
        if nxt in _PARTITIVE and len(toks) > 3 and toks[2] == "of":
            for t in toks[3:]:
                if t not in FUNCTION_WORDS:
                    return Demand(first, t, None)
        if nxt not in FUNCTION_WORDS:
            np_tokens: list[str] = []
            for t in toks[1:]:
                if t in _AUXILIARIES or t in FUNCTION_WORDS or len(np_tokens) == 2:
                    break
                np_tokens.append(t)
            return Demand(first, np_tokens[-1] if np_tokens else nxt, None)
        # "what is/are <NP>" — the demand is the HEAD of the noun phrase, i.e. its last content
        # token: "What is Maya's annual salary?" demands `salary`, not `annual`.
        return Demand(first, content[-1] if content else None, None)

    if first in _WH_TYPED:
        return Demand(first, content[-1] if content else None, _WH_TYPED[first])
    if first in ("how", "why"):
        return Demand(first, content[-1] if content else None, None)
    if first in _AUXILIARIES:
        return Demand("bool", content[-1] if content else None, None)
    return Demand(None, content[-1] if content else None, None)


# ── The vocabulary fold — over the ledger, not a word list ────────────────────

def _morph_variants(t: str) -> list[str]:
    """The singular/plural pair for `t`, and nothing else.

    Motivating failure: "What does Marcus Webb own?" was refused by a substrate holding "Marcus
    Webb owns a bookshop." The vocabulary is an exact dict and own != owns, so a question that was
    entirely answerable was refused over a verb ending. A reader sees that and concludes the
    software is broken.

    Measured on 410 answerable + 1,185 absent-trap items, default (no encoder) install:

        none         answerable refusal 37.8%   trap refusal 75.1%
        +s/-s        answerable refusal 36.8%   trap refusal 73.8%
        +s/es/ies    answerable refusal 36.6%   trap refusal 73.8%

    So this is close to a one-for-one trade on THIS corpus, whose questions are templated and
    under-represent inflection mismatch; the failure above came from an ordinary question typed by
    hand. The wider -ed/-ing folding was dropped: it turns "car" into "cared" and "caring" and adds
    nothing the -s pair does not already get.
    """
    if len(t) < 3:
        return []
    out = [t + "s"]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        out.append(t[:-1])
    # "What did Ada Lovelace write?" against a stored "Ada Lovelace wrote ..." -- the -s fold
    # cannot reach an irregular past tense, and irregulars are exactly the verbs people use.
    out.extend(verb_forms(t))
    return [v for v in dict.fromkeys(out) if v != t and len(v) >= 3]


@dataclass
class Vocabulary:
    """Document frequency over ACTIVE claim content. A fold over the graph, recomputed per query:
    the substrate's own words are the only authority on what it can be asked about."""
    df: dict[str, int]
    n_docs: int
    # Document frequency over the AUTHORED SLOTS attached at admission — a second, much smaller
    # vocabulary whose members are enumerable in advance. `df` can only ever tell you which words
    # a corpus happened to use; `slots` tells you which KINDS of fact it holds, which is what makes
    # a typed miss distinguishable from a paraphrase miss. Defaulted so older callers constructing
    # `Vocabulary(df, n)` keep working.
    slots: dict[str, int] = field(default_factory=dict)
    # The same index, per entity. A corpus-wide slot index answers "does this store hold pay facts",
    # which is NOT the question "what is Dana's salary?" asks. Consulting the corpus-wide one for a
    # question with a named subject grounds the question because SOMEBODY has a salary, which is a
    # false accept of exactly the shape the typed lookup exists to eliminate. Entity scope first,
    # corpus scope only when the question names no subject.
    slots_by_entity: dict[str, dict[str, int]] = field(default_factory=dict)

    def status(self, token: str, df_cap: float = READ_GATE_DF_CAP) -> str:
        """Three-way, with the CORRECTED sign. The proposal dropped tokens absent from the corpus
        as 'non-discriminative'; absence is the strongest possible evidence that the substrate
        cannot answer, so absence must ABSTAIN. Only tokens ABOVE the cap are dropped.

        The lookup folds English inflection. "What does Marcus Webb own?" was refused by a
        substrate holding "Marcus Webb owns a bookshop" -- the vocabulary is an exact dict, and
        own != owns. A user reads that as broken, and is right to: nothing about the question was
        unanswerable, only its verb form. Folding is conservative (see _morph_variants) and can
        only ever match a token the substrate already contains.
        """
        n = self.df.get(token)
        if n is None:
            for v in _morph_variants(token):
                n = self.df.get(v)
                if n is not None:
                    break
        if n is None:
            return "ungrounded"
        if self.n_docs and n / self.n_docs > df_cap:
            return "non_discriminative"
        return "grounded"

    def tokens(self) -> list[str]:
        """Content vocabulary, for the semantic rescue. Sorted for determinism."""
        return sorted(t for t in self.df if t not in FUNCTION_WORDS and len(t) > 2)


def build_vocabulary(graph: GraphState, ts: str | None = None,
                     entity_id: str | None = None) -> Vocabulary:
    """Document frequency over active claim content, PLUS each claim's canonical predicate.

    The surface tokens alone are the literal words the source happened to use, which is why the
    read side needed a growing pile of query-time widening (morphology, hypernyms, an optional
    encoder) to match a question phrased any other way. Every node already carries a canonical
    `Predicate(lemma, polarity, object)` -- computed at admission, never read for matching. Indexing
    it puts the normalized form in the vocabulary at WRITE time: a store that recorded "owns a
    bookshop" grounds `own` because the lemma says so, not because a query-time rule guessed it.

    Additive, not a replacement: surface tokens stay indexed, so this can only ever raise the floor.
    """
    df: dict[str, int] = {}
    slots: dict[str, int] = {}
    by_entity: dict[str, dict[str, int]] = {}
    n = 0
    for node in graph.get_valid_nodes(ts):
        if node.status.memory_state not in ("active", "disputed"):
            continue
        # Entity scope. A vocabulary folded over the WHOLE store answers "does anyone here mention
        # a cat?", which is not the question "what pet does Dana have?" asks.
        if entity_id is not None and not any(e.entity_id == entity_id for e in node.entities):
            continue
        n += 1
        terms = set(_tokens(node.normalized_claim))
        pred = getattr(node, "predicate", None)
        if pred is not None:
            lemma = (getattr(pred, "lemma", None) or "").strip().lower()
            if lemma and lemma != "unknown":
                terms.add(lemma)
            obj = (getattr(pred, "object", None) or "").strip().lower()
            if obj:
                terms.update(_tokens(obj))
            slot = (getattr(pred, "slot", None) or "").strip().lower()
            if slot:
                slots[slot] = slots.get(slot, 0) + 1
                for e in node.entities:
                    per = by_entity.setdefault(e.entity_id, {})
                    per[slot] = per.get(slot, 0) + 1
        for t in terms:
            df[t] = df.get(t, 0) + 1
    return Vocabulary(df, n, slots, by_entity)


def _predicates_about(ent, graph, ts, limit: int = 8) -> list[str]:
    """Content words appearing in the active claims about `ent`, most frequent first.

    A refusal that names only what is missing leaves the caller nowhere to go. MCP callers are
    usually agents, which cannot browse the store the way a human would -- so an unqualified "no"
    is where the interaction ends. Naming the terms that ARE grounded lets the next question be
    the right one, and costs nothing the caller could not already obtain by querying the entity.
    """
    from collections import Counter
    name_toks = set(_tokens(ent.canonical_name))
    c: Counter = Counter()
    for node in graph.get_valid_nodes(ts):
        if node.status.memory_state not in ("active", "disputed"):
            continue
        if not any(e.entity_id == ent.entity_id for e in node.entities):
            continue
        for t in set(_tokens(node.normalized_claim)):
            if t in FUNCTION_WORDS or t in name_toks or len(t) < 3:
                continue
            c[t] += 1
    return [t for t, _ in c.most_common(limit)]


def _some_entity_names(graph, ts, limit: int = 5) -> list[str]:
    """A few entities the substrate actually holds, for an entity_not_found redirect."""
    seen: list[str] = []
    for node in graph.get_valid_nodes(ts):
        if node.status.memory_state not in ("active", "disputed"):
            continue
        for e in node.entities:
            ent = graph.get_entity(e.entity_id) if hasattr(graph, "get_entity") else None
            nm = getattr(ent, "canonical_name", None)
            if nm and nm not in seen:
                seen.append(nm)
                if len(seen) >= limit:
                    return seen
    return seen


# ── The verdict ───────────────────────────────────────────────────────────────

@dataclass
class ReadGateVerdict:
    abstain: bool
    reason: str | None            # "unknown_subject" | "unknown_predicate" | None
    detail: str                   # human-readable, demoable — the typed abstention
    demand: Demand
    unresolved_subjects: list[str] = field(default_factory=list)
    rescue_score: float | None = None   # cosine that grounded the head, when rescue fired
    # "semantic" when the encoder was available, "lexical_only" when it was not. A refusal issued in
    # lexical_only mode may well be answerable in a full deployment; the caller deserves to know.
    mode: str = "semantic"
    # What the caller could have asked instead. A refusal that only says "no" is a dead end: the
    # caller (often an agent, which cannot browse) has no way to discover the store's actual shape
    # and typically gives up or guesses. These turn a refusal into a redirect.
    known_predicates: list[str] = field(default_factory=list)   # grounded terms for this subject
    known_subjects: list[str] = field(default_factory=list)     # entities that DO exist
    remedy: str = ""                                            # the concrete next action


def _resolve_partial(name: str, graph: GraphState):
    """Resolve a name that is an unambiguous prefix of exactly one canonical entity name.

    `find_entity_by_name` matches the full canonical name only, so "Priya" does not resolve to
    "Priya Raman". Without this, check 1 fires `unknown_subject` on a person the substrate knows
    perfectly well — the gate would answer "no entity named Priya exists" while holding three
    claims about her, which is a worse failure than the one being fixed. Ambiguous prefixes
    (two entities sharing a first name) deliberately do NOT resolve: guessing between people is
    exactly the cross-wiring the entity linker's longest-match rule exists to prevent.
    """
    low = name.lower()
    hits = [e for e in graph.all_entities()
            if e.canonical_name.lower().split()[0] == low
            or low in [a.lower() for a in e.aliases]]
    return hits[0] if len(hits) == 1 else None


def _named_subjects(question: str, graph: GraphState,
                    vocab: "Vocabulary | None" = None) -> tuple[list[str], list[str]]:
    """(resolved, unresolved) named subjects.

    Interrogatives are excluded explicitly. `query_parser._is_name_token` treats any capitalized
    non-skip token as a name, so a sentence-initial "What"/"Where"/"Does" was being read as an
    entity mention — which made several abstentions correct for the WRONG reason (the gate fired on
    an unresolvable wh-word rather than on the missing subject). Measured on the eval fixture:
    "What shift does Dana work?" reported unresolved subjects ['What', 'Dana'].
    """
    from .query_parser import _is_name_token

    resolved: list[str] = []
    unresolved: list[str] = []
    raw = [t.rstrip(".,!?\"'") for t in question.split()]
    n = len(raw)
    i = 0
    while i < n:
        tok = raw[i]
        if not _is_name_token(tok) or tok.lower() in FUNCTION_WORDS:
            i += 1
            continue
        j = i
        while j < n and _is_name_token(raw[j]) and raw[j].lower() not in FUNCTION_WORDS:
            j += 1
        matched_to = None
        for end in range(j, i, -1):                     # greedy longest match, as query_parser does
            span = " ".join(raw[i:end])
            lookup = span[:-2] if span.endswith("'s") else span
            ent = graph.find_entity_by_name(lookup) or _resolve_partial(lookup, graph)
            if ent is not None:
                resolved.append(ent.canonical_name)
                matched_to = end
                break
        if matched_to is not None:
            i = matched_to
        else:
            bare = raw[i][:-2] if raw[i].endswith("'s") else raw[i]
            # A name absent from the ENTITY STORE may still be grounded in claim CONTENT: hand-built
            # and legacy graphs carry EntityRefs without registered Entity records, and a snapshot
            # restored from an older schema can too. Saying "no entity named Maya exists" while
            # holding four active claims that name Maya is a worse failure than the one this gate
            # fixes, so the vocabulary fold is the second authority before declaring a subject
            # unknown. Both authorities are the substrate's own record — neither is a word list.
            if vocab is not None and vocab.df.get(bare.lower()):
                resolved.append(bare)
            else:
                unresolved.append(bare)
            i += 1
    return resolved, unresolved


def rescue_available() -> bool:
    """Is the semantic rescue actually usable in THIS deployment?

    The encoder is a soft dependency, and its absence silently changes the gate's behaviour: every
    paraphrase question is refused. Measured on the demo corpus, the same container that answers
    "Who oversees the Northfield depot?" refuses "What illness does Priya Raman have?" (rescue 0.66)
    and "How many parcels did the depot handle?" (0.64) when the encoder is missing.

    That is a SAFE direction but a large behavioural difference, and shipping it invisibly means a
    pilot concludes the product is broken while our demo answers the same question. So the mode is
    reportable — /v1/health surfaces it — rather than something an operator has to infer.

    Checked by spec lookup, not by loading the model: this must stay cheap enough to call per query.
    """
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None


def slot_by_encoder(head: str, min_sim: float | None = None) -> tuple[str | None, float]:
    """Nearest AUTHORED slot for an untyped demand head, or (None, score) if nothing is close.

    Bounded by construction: the candidate set is the sixteen slots in `predicate_vocabulary`, never
    the corpus. The head is compared to each slot's declared question terms and to the slot name
    itself, and the best-scoring slot wins only if it clears `min_sim`.

    Returns a NAME, not a verdict. The caller still has to establish that the subject actually
    carries a claim of that kind, which is the check that keeps this from becoming the unbounded
    rescue it replaces.
    """
    if not head:
        return None, 0.0
    thr = SLOT_RESCUE_MIN_SIM if min_sim is None else min_sim
    try:
        from .semantic_encoder import similarity
    except Exception:
        return None, 0.0
    best_slot, best = None, 0.0
    try:
        for name, slot in SLOTS.items():
            for term in (name.replace("_", " "), *sorted(slot.asks)):
                sc = similarity(head, term)
                if sc > best:
                    best, best_slot = sc, name
    except Exception:
        return None, 0.0
    return (best_slot, best) if best >= thr else (None, best)


def _semantic_rescue(head: str, vocab: Vocabulary) -> tuple[bool, float | None]:
    """Is the demand head a paraphrase of something the substrate predicates on?

    DO NOT RECOMMEND THE `[semantic]` EXTRA. Measured 2026-08-27 against the category-trap corpus,
    which postdates this function's own measurements and had never been run against it:

        no encoder      held-out refusal 76.0%    trap refusal 97.7%
        encoder @0.55                    59.7%                 43.0%
        encoder @0.65                    76.0%                 79.3%
        encoder @0.75                    76.0%                 95.0%

    At 0.65 and above it delivers ZERO recall benefit — identical to no encoder — and still costs
    trap refusal. Below 0.65 it buys recall by answering questions whose answer is absent ("what
    diet does X follow?" rescued at 0.71 against a persona about teaching and volleyball). There is
    no threshold at which it beats being absent. See
    docs/FINDING_recall_was_measured_on_fitted_phrasings.md §3c.

    Left in place rather than deleted because the mechanism is not the problem — comparing a head to
    a corpus vocabulary is; a rescue scoped to the SUBJECT's own predicates may well behave. That is
    untested. Until it is tested this path should stay off.

    Compares the head to the corpus VOCABULARY, token to token — not to whole claims. Measured
    (docs/DESIGN_read_gate.md §2b): claim-level comparison interleaves the classes completely at
    every threshold, because a claim is a bag of topics while a token is a predicate. Token-level
    separates with a margin: discomfort→pain 0.76, living→lived 0.76, sport→injury 0.55 against
    salary→money 0.41, address→booked 0.44, revenue→annually 0.47.

    Degrades to (False, None) when the encoder is unavailable — the safe direction: more abstention.
    """
    try:
        from .semantic_encoder import similarity
    except Exception:
        return False, None
    best = 0.0
    try:
        for t in vocab.tokens():
            s = similarity(head, t)
            if s > best:
                best = s
        return best >= READ_GATE_MIN_PREDICATE_SIM, best
    except Exception:
        return False, None


def read_gate(
    question: str,
    graph: GraphState,
    ts: str | None = None,
    semantic_rescue: bool = True,
    vocab: Vocabulary | None = None,
) -> ReadGateVerdict:
    """Adjudicate admission into the conversation. Pure, deterministic, never raises."""
    try:
        return _read_gate(question, graph, ts, semantic_rescue, vocab)
    except Exception:
        # Fail SAFE: an unexpected shape abstains rather than answering.
        return ReadGateVerdict(True, "no_evidence", "read gate could not evaluate the question",
                               Demand(None, None, None))


def _read_gate(question, graph, ts, semantic_rescue, vocab) -> ReadGateVerdict:
    demand = parse_demand(question)
    vocab = vocab if vocab is not None else build_vocabulary(graph, ts)
    # A property of the DEPLOYMENT, not of this query's path — so every verdict carries it, including
    # the ones that return before the rescue is ever consulted.
    mode = "semantic" if (semantic_rescue and rescue_available()) else "lexical_only"

    # ── Check 1 — subject grounding ───────────────────────────────────────────
    resolved, unresolved = _named_subjects(question, graph, vocab)
    if unresolved:
        names = ", ".join(f'"{u}"' for u in unresolved)
        others = _some_entity_names(graph, ts)
        if others:
            hint = f"; the substrate does know: {', '.join(others)}"
            remedy = ("ask about one of the entities listed, or commit a claim about "
                      f"{unresolved[0]} first")
        else:
            hint = "; the substrate holds no entities yet"
            remedy = "commit a claim with evidence before querying"
        return ReadGateVerdict(
            True, "entity_not_found",
            f"no entity named {names} exists in the substrate{hint}",
            demand, unresolved_subjects=unresolved, mode=mode,
            known_subjects=others, remedy=remedy,
        )

    # ── Check 2 — predicate grounding ─────────────────────────────────────────
    head = demand.head
    if head is None:
        # Nothing but function words and resolved names: an entity-only query ("Priya Raman") is a
        # legitimate request for what is known about that entity.
        return ReadGateVerdict(False, None, "entity query", demand, mode=mode)

    if vocab.n_docs == 0:
        # Nothing has ever been committed. "no claims ground X" is true but misleading — the
        # substrate is empty, not selectively ignorant — and `no_evidence` is the shipped code for it.
        return ReadGateVerdict(True, "no_evidence", "the substrate holds no claims", demand,
                               mode=mode)

    status = vocab.status(head)
    if status in ("grounded", "non_discriminative"):
        return ReadGateVerdict(False, None, f'"{head}" is grounded in the substrate', demand,
                               mode=mode)

    # ── Typed lookup, against the AUTHORED slot vocabulary ────────────────────
    # The surface check just failed, which today means the question goes to the widening layers and
    # a refusal — if one comes — cannot say WHY. The slot index answers a different question: does
    # this substrate hold facts of the KIND being asked for? "What does Priya earn?" against a store
    # holding "Priya earns £40,000" grounds here without any query-time rescue, because the slot was
    # attached at admission from the same evidence.
    #
    # A head that names no slot leaves `demanded_slot` None, and nothing below changes. This can
    # only ever ground MORE questions than the surface check alone.
    demanded_slot = slot_for_head(head)
    if demanded_slot is None:
        for v in _morph_variants(head):
            demanded_slot = slot_for_head(v)
            if demanded_slot is not None:
                break
    # The encoder names a slot; it does not authorize an answer. Everything below is unchanged.
    slot_rescue_score = None
    if demanded_slot is None and semantic_rescue and SLOT_RESCUE_BY_ENCODER:
        demanded_slot, slot_rescue_score = slot_by_encoder(head)

    # Scope the lookup to the subject the question named. Asking the corpus-wide index whether it
    # holds a `salary` fact answers a question nobody asked: "What is Dana's salary?" would ground
    # because PRIYA has a salary, and the gate would hand back an answer about the wrong person or
    # an empty one. Scope is what turns the slot index from a topic detector into an answer to the
    # question actually asked.
    scope, scoped_to = vocab.slots, None
    if resolved:
        ent = graph.find_entity_by_name(resolved[0]) or _resolve_partial(resolved[0], graph)
        if SCOPE_PREDICATE_GROUNDING_TO_SUBJECT and ent is not None:
            sv = build_vocabulary(graph, ts, entity_id=ent.entity_id)
            if sv.n_docs:
                if sv.status(head) in ("grounded", "non_discriminative"):
                    return ReadGateVerdict(False, None,
                                           f'"{head}" is grounded in claims about '
                                           f'{ent.canonical_name}', demand, mode=mode)
                vocab = sv   # every layer below now answers about the subject, not the corpus
        if ent is not None and ent.entity_id in vocab.slots_by_entity:
            scope, scoped_to = vocab.slots_by_entity[ent.entity_id], ent.canonical_name
        elif ent is not None:
            # Known entity, nothing typed about it: an empty scope, not a missing one. Falling back
            # to the corpus index here would reintroduce the false accept this scoping removes.
            scope, scoped_to = {}, ent.canonical_name

    if demanded_slot is not None and scope.get(demanded_slot):
        where = f' for {scoped_to}' if scoped_to else ""
        how = (f' (slot named by encoder, {slot_rescue_score:.2f})'
               if slot_rescue_score is not None else "")
        return ReadGateVerdict(False, None,
                               f'"{head}" is grounded as typed predicate '
                               f'"{demanded_slot}"{where}{how}',
                               demand, rescue_score=slot_rescue_score, mode=mode)

    # ── HYPERNYMS as a declared policy layer, applied AFTER the typed lookup ──
    # The rescue below consults the CORPUS-WIDE vocabulary, so it answers "does anyone in this store
    # have a cat?" for a question that asked about one named person. Measured live: a store holding
    # Marcus's cat and Dana's basketball answered "What pet does Dana have?" by rescuing `pet` ->
    # `cat`, which is Marcus's. That false accept predates the typed index and is the exact failure
    # mode the widening layers were always going to produce, because a curated table cannot know who
    # a claim is about.
    #
    # A typed miss is now strong enough to overrule it, but only where it is actually informed:
    # this entity must already carry SOME typed claim, so we know the index covers it, and the
    # demanded slot must be absent from that coverage. An entity with nothing typed tells us
    # nothing, and there the rescue still runs exactly as before. Narrow on purpose — this is the
    # first rule in the system that makes a refusal STICK against a widening layer, and it should
    # earn any further reach by measurement rather than by assumption.
    #
    # The `demand.typed is None` condition is not a technicality, it is a correction. "Where does
    # Marco work now?" has head `work`, which maps to `occupation` — but the wh-word already
    # declared the question wants a LOCATION, so the head-to-slot mapping is answering on the wrong
    # axis, and the typed miss it produces is an artefact of that mismatch rather than a fact about
    # the store. Live, that suppressed a rescue which had been answering the question correctly. A
    # wh-word carrying its own object type is evidence the mapping may be off-axis, so authority
    # defers to it and the widening layers run exactly as they did before.
    typed_miss_is_authoritative = TYPED_MISS_OVERRULES_RESCUE and (
        demanded_slot is not None
        and demand.typed is None              # the wh-word does not demand a different axis
        and scoped_to is not None
        and bool(scope)                       # the entity participates in the typed index
        and not scope.get(demanded_slot)      # and carries no claim of the kind demanded
    )

    # Lexical hypernym rescue, BEFORE the encoder. A default `pip install fireweed-mcp` has no
    # encoder (zero declared dependencies), and lexical_only mode refused 369/410 answerable
    # questions -- including "what kind of pet?" against a substrate grounding *cat*. These are
    # ordinary hypernyms, not embedding similarity, so they need no model and stay auditable.
    hyp = None if typed_miss_is_authoritative else grounded_hyponym(
        head, lambda t: vocab.status(t) in ("grounded", "non_discriminative"))
    if hyp is not None:
        return ReadGateVerdict(False, None,
                               f'"{head}" is grounded via "{hyp}"', demand, mode=mode)

    rescued, score = (False, None)
    if semantic_rescue and not typed_miss_is_authoritative:
        rescued, score = _semantic_rescue(head, vocab)
    if rescued:
        return ReadGateVerdict(False, None, f'"{head}" paraphrases a grounded predicate',
                               demand, rescue_score=score, mode=mode)

    # The typed abstention the review asked for: refusal stops being an empirical tendency and
    # becomes a stated, gated property with a machine-readable reason.
    about = ""
    known: list[str] = []
    remedy = f'commit a claim that grounds "{head}", with evidence'
    if resolved:
        subj = resolved[0]
        ent = graph.find_entity_by_name(subj) or _resolve_partial(subj, graph)
        if ent is not None:
            n_about = sum(
                1 for node in graph.get_valid_nodes(ts)
                if node.status.memory_state in ("active", "disputed")
                and any(e.entity_id == ent.entity_id for e in node.entities)
            )
            if n_about:
                # "1 claim ... exist" shipped for months. Subject-verb agreement is not cosmetic in
                # a refusal message: it is the sentence the user is most likely to read closely.
                about = (f'; {n_about} claim{"s" if n_about != 1 else ""} '
                         f'about {ent.canonical_name} {"exist" if n_about != 1 else "exists"}')
                known = _predicates_about(ent, graph, ts)
                if known:
                    # a participle sidesteps the agreement problem for both 1 and n claims
                    about += f', grounding: {", ".join(known)}'
                    remedy = (f'ask about one of: {", ".join(known[:5])} — or commit a claim '
                              f'grounding "{head}"')
    caveat = ("" if mode == "semantic"
              else " (paraphrase matching unavailable: the semantic encoder is not installed, "
                   "so this refusal is stricter than a full deployment's)")
    # A TYPED miss and a VOCABULARY miss are different answers and deserve different sentences.
    # "no claims ground 'salary'" is what the gate can say when it has no schema: it cannot tell a
    # store that holds no pay facts from one that records them in words the question did not use.
    # When the head names a slot the substrate has never filled, the stronger statement is true and
    # worth making — it is the difference between "I could not match your wording" and "I hold no
    # fact of this kind", and only the second tells the caller to stop rephrasing.
    typed = ""
    if demanded_slot is not None and not scope.get(demanded_slot):
        subj_txt = f' about {scoped_to}' if scoped_to else " in the substrate"
        typed = (f'; no claim{subj_txt} is typed "{demanded_slot}"'
                 ', so this is a typed miss rather than a wording mismatch')
    return ReadGateVerdict(
        True, "unknown_predicate",
        f'no claims ground "{head}"{about}{typed}{caveat}',
        demand, rescue_score=score, mode=mode,
        known_predicates=known, remedy=remedy,
    )
