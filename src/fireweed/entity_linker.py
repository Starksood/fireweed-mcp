"""Entity linker: resolve or create Entity records from claim text.
The only place in the codebase that writes to the entity store.
"""
from __future__ import annotations
from .graph import GraphState, Entity, EntityRef, EntityProvenance

_SKIP_WORDS = {
    "I", "The", "A", "An", "This", "That", "These", "Those",
    "It", "He", "She", "They", "We", "You", "My", "His", "Her",
    "Their", "Our", "Its", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}
_LOCATION_PREPOSITIONS = ("in ", "at ", "near ")
_MULTI_WORD_SKIP = _SKIP_WORDS | {
    "Project", "Street", "Road", "Avenue", "Lane", "Drive", "Court",
}

# Words that are frequently capitalized only because they begin a sentence —
# connectives, determiners, hedges, modals, common verbs, time words. Without
# this filter the linker promotes sentence-initial tokens ("If", "There",
# "Started", "Rent", "Sundays") to entities and contaminates entity-set metrics.
# Checked lowercase and ONLY at sentence-initial single-token positions, so a
# genuine proper noun appearing mid-sentence (or as a multi-word phrase) is kept.
_COMMON_WORDS = frozenset({
    # honorifics / titles — never an entity on their own ("Mr. Skilling" -> not entity "Mr")
    "mr", "mrs", "ms", "dr", "mister", "sir", "madam", "madame", "prof", "professor",
    "rev", "reverend", "hon", "sen", "rep", "gov", "pres", "capt", "lt", "sgt", "col",
    # connectives / conjunctions / adverbial joiners
    "if", "when", "while", "then", "there", "here", "but", "and", "or", "so",
    "because", "since", "after", "before", "during", "although", "though",
    "however", "therefore", "thus", "meanwhile", "also", "plus", "yet", "still",
    "just", "even", "only", "anyway", "besides", "instead", "otherwise",
    "regardless", "despite", "unless", "until", "whenever", "wherever",
    "whatever", "whoever", "whether", "as", "by", "for", "from", "with",
    "without", "into", "onto", "about", "over", "under", "between",
    "in", "on", "at", "of", "to", "up", "off", "out", "per", "via",
    "twice",
    # determiners / quantifiers
    "some", "many", "most", "much", "more", "less", "few", "several", "each",
    "every", "all", "both", "either", "neither", "another", "other", "none",
    "such", "any", "no", "this", "that", "these", "those",
    # time / sequence
    "now", "today", "yesterday", "tomorrow", "soon", "later", "recently",
    "currently", "lately", "once", "again", "always", "never", "sometimes",
    "often", "usually", "eventually", "finally", "first", "next", "last",
    "earlier", "afterward", "tonight", "everyday", "weekends", "weekdays",
    "mornings", "evenings", "nights", "sundays", "mondays", "tuesdays",
    "wednesdays", "thursdays", "fridays", "saturdays",
    # adverbs / hedges
    "maybe", "perhaps", "probably", "definitely", "certainly", "actually",
    "basically", "generally", "mostly", "really", "very", "quite", "rather",
    "honestly", "frankly", "clearly", "obviously", "apparently", "hopefully",
    # modals / auxiliaries
    "can", "could", "would", "should", "will", "shall", "might", "must", "may",
    "do", "does", "did", "have", "has", "had", "was", "were", "been", "being",
    "am", "are", "is", "be", "got",
    # wh- / question words
    "what", "who", "where", "why", "how", "which", "whose", "whom",
    # common sentence-initial verbs / nouns that are not proper names
    "rent", "yes", "ok", "okay", "well", "oh", "thanks", "please", "went",
    "made", "took", "came", "saw", "said", "told", "asked", "kept", "felt",
    "found", "gave", "left", "let", "put", "set", "run", "ran", "started",
    "stopped", "things", "stuff", "people", "everyone", "everything",
    "someone", "something", "nothing", "nobody", "anyone", "anything",
    # personal / possessive pronouns — never proper entities, at any position
    "i", "you", "your", "yours", "we", "our", "ours", "they", "their", "theirs",
    "them", "us", "me", "my", "mine", "he", "him", "his", "she", "her", "hers",
    "it", "its", "one",
    # descriptive / property adjectives & generic abstractions — capitalized in Title-Case headings
    # and predicates of technical prose ("the store is Atomic", "Optimistic concurrency") but never a
    # proper entity. Frequency alone can't catch these (many are low-zipf: immutable 2.8, recursive 3.0)
    # and a zipf gate would wrongly kill real domain nouns (Memory, Identity, Graph), so they are listed
    # explicitly. Kept disjoint from genuine Fireweed concepts (Semantic, Temporal, Supersession, …).
    "atomic", "bare", "dense", "stable", "forbidden", "optimistic", "recursive",
    "immutable", "deterministic", "symbolic", "transient", "periodic", "named",
    "inline", "abstraction", "architectural", "historical", "constitutional",
    "compound", "affective", "twenty", "not",
})
# I-contractions are never proper entities; filtered regardless of position.
_CONTRACTIONS = frozenset({"i'm", "i've", "i'll", "i'd"})
_NOISE_SUFFIXES = ("ed", "ing", "ly")

# Sentence-initial common-word filter (external-review fix #1, entity-J). A token capitalized only
# because it opens a sentence ("Gonna", "Great", "Looks", "Take") is a common English word; a genuine
# proper noun is not. We detect "common" with a PINNED frequency resource (wordfreq — deterministic,
# guardrail-aligned) rather than growing a bespoke list. Threshold 5.0 is false-positive-free on real
# entities (max observed: "van" 4.78; Portland 4.18) while catching the junk (gonna 5.29, take 5.92).
_COMMON_ZIPF = 5.0


def _zipf(word: str) -> float:
    """Zipf frequency of `word` in English, or -1.0 if wordfreq (a soft dependency) is unavailable."""
    try:
        from wordfreq import zipf_frequency
    except Exception:
        return -1.0
    return zipf_frequency(word, "en")


def _is_contraction(clean: str) -> bool:
    """A contraction ("It'll", "You're", "Don't") — apostrophe followed by a lowercase letter.
    Keeps apostrophe-names like O'Brien (apostrophe + capital) AND possessives like "Maya's"
    (suffix 's, handled separately so the base name survives)."""
    j = clean.find("'")
    if not (0 < j < len(clean) - 1) or not clean[j + 1].islower():
        return False
    return clean[j + 1:].lower() != "s"                 # possessive 's is not a contraction


def _is_global_noise(clean: str) -> bool:
    """A capitalized token that is never a proper entity regardless of position:
    a function/common word (incl. plural day names) or a 1-2 letter all-caps token
    (IT, PS, OK). These leak even mid-sentence — a day name is correctly capitalized
    inside a sentence ("...on Sundays"), and "works in IT" capitalizes a non-entity —
    so the filter cannot be limited to sentence-initial position."""
    if clean.lower() in _COMMON_WORDS:
        return True
    if clean.isupper() and len(clean) <= 2:
        return True
    return False


def _is_initial_noise(clean: str) -> bool:
    """A capitalized token that is noise ONLY when it begins a sentence: a verb/adverb
    inflection (Started, Walking, Quickly) or a 3-letter all-caps token. Mid-sentence
    these are more likely genuine names (e.g. an -ing/-ed surname), so they are kept
    unless they open the sentence."""
    lw = clean.lower()
    if len(lw) >= 6 and lw.endswith(_NOISE_SUFFIXES):
        return True
    if clean.isupper() and len(clean) == 3:
        return True
    # a common English word (by pinned frequency) that opens a sentence is not a proper noun
    if _zipf(lw) >= _COMMON_ZIPF:
        return True
    return False


def _is_sentence_initial(tokens: list[str], i: int) -> bool:
    if i == 0:
        return True
    prev = tokens[i - 1].rstrip("\"')]’”")
    return prev.endswith((".", "!", "?"))


def _extract_mentions(claim_text: str) -> list[str]:
    """Return ordered unique entity mentions (multi-word capitalized phrases first)."""
    tokens = claim_text.split()
    mentions: list[str] = []
    i = 0
    while i < len(tokens):
        clean = tokens[i].rstrip(".,!?\"'")
        if not clean or clean[0].isupper() is False:
            i += 1
            continue
        if clean.lower() in _CONTRACTIONS or _is_contraction(clean):
            i += 1
            continue
        if clean.endswith("'s"):
            clean = clean[:-2]
        if clean in _MULTI_WORD_SKIP:
            i += 1
            continue
        if _is_global_noise(clean):
            i += 1
            continue
        if _is_sentence_initial(tokens, i) and _is_initial_noise(clean):
            i += 1
            continue
        if not clean[1:].replace("'", "").isalpha():
            i += 1
            continue
        phrase = [clean]
        j = i + 1
        while j < len(tokens):
            # do not extend a phrase across a sentence boundary
            if tokens[j - 1].rstrip("\"')]’”").endswith((".", "!", "?")):
                break
            nxt = tokens[j].rstrip(".,!?\"'")
            if nxt.endswith("'s"):
                nxt = nxt[:-2]
            if not nxt or nxt[0].isupper() is False or nxt in _MULTI_WORD_SKIP:
                break
            if not nxt[1:].replace("'", "").isalpha():
                break
            phrase.append(nxt)
            j += 1
        mention = " ".join(phrase)
        if mention not in mentions:
            mentions.append(mention)
        i = j if j > i + 1 else i + 1
    return mentions


def _is_place_mention(mention: str, claim_text: str) -> bool:
    """True if THIS claim uses a location preposition before the mention ('...at/in/near X')."""
    lower, ml = claim_text.lower(), mention.lower()
    return any((p + ml) in lower for p in _LOCATION_PREPOSITIONS)


def _infer_entity_type(mention: str, claim_text: str) -> str:
    return "place" if _is_place_mention(mention, claim_text) else "person"


def _infer_role(mention: str, claim_text: str) -> str:
    return "location" if _is_place_mention(mention, claim_text) else "actor"


# ── Sprint 1: entity canonicalization under noise (master plan §4.1) ─────────────────────────────
# OFF by default so the baseline is unchanged; enabled for benchmark/production runs. When on, a
# nickname/abbreviation/paraphrase variant is deterministically ABSORBED as an alias of the existing
# entity (alias-only mutation — provenance + the 0-fabrication invariant are untouched). See
# docs/sprint/SPRINT1_ENTITY_CANONICALIZATION.md.
_CANON_MODE: str | None = None          # None (off) | "prefix" | "semantic"
_CANON_PREFIX_MIN = 3
_CANON_SIM_WITH_PREFIX = 0.45
_CANON_SIM_ALONE = 0.62


def enable_entity_canonicalization(semantic: bool = False) -> None:
    global _CANON_MODE
    _CANON_MODE = "semantic" if semantic else "prefix"


def disable_entity_canonicalization() -> None:
    global _CANON_MODE
    _CANON_MODE = None


def _is_nick(a: str, b: str) -> bool:
    """One name is a case-insensitive prefix of the other, shorter ≥ 3 chars (Mel↔Melanie,
    Van↔Van Ness). Rejects Mel↔Marcus."""
    al, bl = a.lower(), b.lower()
    if al == bl:
        return False
    short, long = (al, bl) if len(al) <= len(bl) else (bl, al)
    return len(short) >= _CANON_PREFIX_MIN and long.startswith(short)


def _is_name_part(a: str, b: str) -> bool:
    """One name is a whole WORD of the other multi-word name — the surname case.

    `_is_nick` is left-anchored, so "Phillip" merges into "Phillip Allen" but "Allen" never does.
    On real corpora that splits one person in two: measured 2.8-3.0x entity inflation (28-30
    entities for 10 actors) on the synthetic run, present even with a perfect extractor, so it is
    the linker rather than extraction.

    Deliberately conservative:
      * the short name must be a single token of >= 3 chars,
      * the long name must be multi-token,
      * the token must not be a common English word — otherwise "Design" would swallow
        "Kestrel Design" and every other firm with Design in the name.
    The caller's ambiguity guard still applies: a part that fits two entities merges into neither.
    """
    al, bl = a.lower().strip(), b.lower().strip()
    if al == bl:
        return False
    short, long = (al, bl) if len(al) <= len(bl) else (bl, al)
    if " " in short or len(short) < _CANON_PREFIX_MIN:
        return False
    parts = long.split()
    if len(parts) < 2 or short not in parts:
        return False
    return short not in _COMMON_WORDS and _zipf(short) < _COMMON_ZIPF


def _name_variant(a: str, b: str) -> bool:
    """Either variant relation: prefix/nickname, or a whole-word part (surname)."""
    return _is_nick(a, b) or _is_name_part(a, b)


def _name_sim(a: str, b: str) -> float:
    try:
        from . import semantic_encoder
        return semantic_encoder.similarity(a, b)
    except Exception:
        return -1.0


def _canonicalize(mention: str, graph: GraphState) -> Entity | None:
    """Conservative fallback merge after exact name/alias fails. Prefix + semantic (belt-and-
    suspenders) for short names, or high semantic alone for paraphrases. Deterministic (entity_id
    order breaks ties)."""
    if _CANON_MODE is None:
        return None
    best, best_sim = None, -1.0
    for e in sorted(graph.all_entities(), key=lambda x: x.entity_id):
        names = [e.canonical_name] + list(e.aliases)
        # variant = prefix/nickname (Mel<->Melanie) OR whole-word part (Allen <-> Phillip Allen)
        matched = [n for n in names if _name_variant(mention, n)]
        prefix = bool(matched)
        if _CANON_MODE == "prefix":
            merge, sim = prefix, (1.0 if prefix else -1.0)
        else:  # semantic
            sim = max((_name_sim(mention, n) for n in names), default=-1.0)
            merge = (prefix and sim >= _CANON_SIM_WITH_PREFIX) or (sim >= _CANON_SIM_ALONE)
        # AMBIGUITY GUARD: merge a nickname/partial only when its stem expands to EXACTLY ONE entity.
        # "Phillip" -> "Phillip Allen" (unique) merges; "Theodore Roosevelt" -> {Jr, Sr, …} abstains, and
        # "Theodore Roosevelt Jr" won't collapse into the bare "Theodore Roosevelt" stem. Distinct people
        # who merely share a stem stay distinct (the #1-robust constraint); real name variants merge (#3).
        if merge and prefix:
            stem = min([mention] + matched, key=len)
            if _stem_expansions(stem, graph) > 1:
                merge = False
        if merge and sim > best_sim:
            best, best_sim = e, sim
    return best


def _stem_expansions(stem: str, graph: GraphState) -> int:
    """How many DISTINCT entities the `stem` could expand to (it equals, or is a nickname/prefix of,
    their canonical name). >1 means the stem is ambiguous and must not drive a merge."""
    sl = stem.lower()
    n = 0
    for e in graph.all_entities():
        # the guard must cover the SAME relation the merge uses, or a surname would bypass it:
        # "Allen" fitting both "Phillip Allen" and "Sarah Allen" has to abstain, not pick one.
        if sl == e.canonical_name.lower() or _name_variant(stem, e.canonical_name):
            n += 1
    return n


def _resolve(mention: str, graph: GraphState) -> Entity | None:
    entity = graph.find_entity_by_name(mention)
    if entity is not None:
        return entity
    ml = mention.lower()
    for e in graph.all_entities():
        if any(a.lower() == ml for a in e.aliases):
            return e
    cand = _canonicalize(mention, graph)                    # Sprint 1 fallback (off by default)
    if cand is not None:
        if ml != cand.canonical_name.lower() and not any(a.lower() == ml for a in cand.aliases):
            cand.aliases.append(mention)                    # absorb as alias (name-set only)
            graph.update_entity(cand)
        return cand
    return None


def _create_entity(mention: str, claim_text: str, source_turn_id: str,
                   source_span: str, graph: GraphState) -> Entity:
    # OPAQUE IDS. This was `"ent_" + mention.lower().replace(" ", "_")`, so an entity id embedded the
    # person's name -- and ids are structural, appearing in relations and node entity-refs throughout
    # an append-only ledger that erasure cannot rewrite. Every content field could be encrypted and
    # shredded and `ent_jane_doe` would still be sitting in the chain afterwards. An identifier
    # derived from a name is personal data.
    #
    # A bare hash would not fix it: sha256("Jane Doe") is computable by anyone who guesses the name.
    # The salt is per-install and lives with the keys rather than the store, so a party holding a
    # copy of the substrate cannot confirm a guess. Within one install the mapping stays
    # deterministic, which resolver purity requires.
    salt = getattr(graph, "_id_salt", "") or ""
    if salt:
        import hashlib
        digest = hashlib.sha256((salt + "\x00" + mention.lower().strip()).encode("utf-8")).hexdigest()
        base_id = "ent_" + digest[:20]
    else:
        # No salt configured: keep the legacy readable form. Existing stores stay loadable and the
        # engine remains usable standalone, but the name is recoverable from the id.
        base_id = "ent_" + mention.lower().replace(" ", "_")
    entity_id, suffix = base_id, 2
    while True:
        try:
            entity = Entity(
                entity_id=entity_id, canonical_name=mention,
                entity_type=_infer_entity_type(mention, claim_text),
                aliases=[], scopes=[], attributes={}, confidence=0.80,
                provenance=[EntityProvenance(source_turn_id=source_turn_id,
                                             source_span=source_span)],
            )
            graph.add_entity(entity)
            return entity
        except ValueError:
            entity_id = f"{base_id}_{suffix}"
            suffix += 1


def link_entities(claim_text: str, source_turn_id: str, source_span: str,
                  graph: GraphState) -> list[EntityRef]:
    """Resolve or create entities from claim_text. Never raises."""
    try:
        results: list[EntityRef] = []
        actor_seen = False
        for entity_name in _extract_mentions(claim_text):
            entity = _resolve(entity_name, graph)
            if entity is None:
                entity = _create_entity(entity_name, claim_text, source_turn_id, source_span, graph)
            elif entity.entity_type == "person" and _is_place_mention(entity_name, claim_text):
                # Aggregate place-typing (Stage 3, W2): an entity is a PLACE if EVER used with a
                # location preposition — not just on its first mention. Promotion is monotonic
                # (person -> place, never the reverse), matching the robust signal the merge-time
                # homonym discriminator uses, so "Van Ness" first seen prepositionless ("the Van
                # Ness transfer") corrects to place once "at Van Ness" appears. canonical_name is
                # unchanged, so the name index is unaffected.
                entity.entity_type = "place"
                graph.update_entity(entity)
            role = _infer_role(entity_name, claim_text)
            if role == "actor":
                if actor_seen:
                    role = "co_actor"
                else:
                    actor_seen = True
            results.append(EntityRef(entity_id=entity.entity_id, role=role))
        return results
    except Exception:
        return []
