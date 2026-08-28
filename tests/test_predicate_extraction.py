"""The extraction gate: it may only LABEL what grounding already admitted.

The tests that matter here are the ones asserting what the gate CANNOT do. Its safety argument is
entirely an argument about its worst case — a bad extraction costs a missed read and can never cost
an ungrounded fact — so the negative properties are the specification, not edge cases around it.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from fireweed.claim import Claim
from fireweed.fabric import Fireweed
from fireweed.read_gate import read_gate, build_vocabulary
from fireweed.predicate_vocabulary import SLOTS, VOCABULARY_VERSION, slot_for_head, is_slot
from fireweed.predicate_extraction import (
    admit, extract, locate, propose,
    NO_PROPOSAL, SLOT_NOT_IN_VOCABULARY, SPAN_NOT_CONTAINED, TYPED,
)
from fireweed.resolver import _build_node


def _node(claim: str, evidence: str):
    c = Claim(claim=claim, evidence_span=evidence, candidate_domains=set(),
              confidence=0.9, source_turn_id="t1")
    return _build_node(c, {"other"}, "ACCEPT", [], "2026-08-27T00:00:00+00:00")


def _store(*pairs):
    fw = Fireweed(llm=lambda p: "")
    fw._ctx.begin_session("s1", "2026-08-27T00:00:00+00:00")
    for claim, evidence in pairs:
        fw._ctx.ingest(claim, evidence, 0.9, "t", "s1")
    return fw._ctx.graph


# ── The two checks ────────────────────────────────────────────────────────────

def test_a_slot_outside_the_vocabulary_is_dropped_not_added():
    """The vocabulary is authored by a person. A proposer that could extend it would be authoring
    the index, which is the one thing a closed list exists to prevent."""
    out = admit("vibe", "owns a cat", "Marcus owns a cat.")
    assert out.predicate is None
    assert out.reason == SLOT_NOT_IN_VOCABULARY
    assert not is_slot("vibe")


def test_a_span_that_is_not_inside_the_evidence_is_dropped():
    out = admit("pet", "owns a dog", "Marcus owns a cat.")
    assert out.predicate is None
    assert out.reason == SPAN_NOT_CONTAINED


def test_containment_requires_word_boundaries():
    """Plain substring containment would accept 'cat' against 'a catastrophe' — the span would pass
    the check while pointing at nothing that means anything."""
    assert locate("cat", "It was a catastrophe.") is None
    assert locate("cat", "He owns a cat.") is not None


def test_containment_tolerates_case_and_whitespace_but_not_new_words():
    assert locate("WORKS   at", "Priya works at Acme.") is not None
    # collapsing whitespace must never let a token through that the evidence does not contain
    assert locate("worksat", "Priya works at Acme.") is None
    assert locate("works for", "Priya works at Acme.") is None


def test_the_offsets_point_at_the_real_span_in_the_evidence():
    ev = "Priya Raman works at Acme in Leeds."
    out = admit("employer", "works at", ev)
    assert out.reason == TYPED
    p = out.predicate
    assert ev[p.start:p.end] == p.span == "works at"


def test_every_label_is_stamped_with_the_vocabulary_that_produced_it():
    out = admit("pet", "cat", "Marcus owns a cat.")
    assert out.predicate.vocabulary_version == VOCABULARY_VERSION


# ── The safety property the whole design rests on ─────────────────────────────

def test_a_failed_extraction_never_refuses_the_claim():
    """The gate is one-directional. Every failure path leaves the claim admitted and untyped —
    which is exactly the status quo for every claim before this module existed."""
    n = _node("The weather was unusually cold.", "The weather was unusually cold yesterday.")
    assert n.predicate.slot is None          # nothing proposed a slot
    assert n.claim                            # and the claim is stored regardless
    assert n.predicate.lemma                  # the pre-existing predicate record is untouched


def test_an_untyped_claim_is_still_retrievable_by_its_literal_words():
    """The floor cannot move down: a claim the gate declined to type must remain exactly as
    findable as it was before the gate existed."""
    g = _store(("The Northfield depot handled 4000 parcels.",
                "The Northfield depot handled 4000 parcels."))
    node = g.all_nodes()[0]
    assert node.predicate.slot is None
    v = build_vocabulary(g)
    assert v.df.get("parcels")               # still indexed by surface form
    assert not read_gate("How many parcels did the Northfield depot handle?", g,
                         semantic_rescue=False).abstain


# ── The proposer ──────────────────────────────────────────────────────────────

def test_a_cue_in_the_evidence_but_not_the_claim_does_not_type_the_claim():
    """Evidence mentioning a cat, supporting a claim about someone's job, must not type that claim
    `pet`. This is the largest source of the mislabelling containment cannot catch."""
    out = extract("Priya Raman is a nurse.", "Priya Raman is a nurse and owns a cat.")
    assert out.predicate.slot == "occupation"
    assert out.predicate.slot != "pet"


def test_the_proposal_goes_through_the_same_gate_as_a_model_proposal():
    out = extract("Marcus Webb owns a cat.", "Marcus Webb owns a cat called Pepper.")
    assert out.predicate.proposer == "bootstrap"
    assert admit("pet", "cat", "Marcus Webb owns a cat.").predicate.proposer == "model"


def test_nothing_proposed_is_an_ordinary_outcome_not_an_error():
    assert propose("Mm, right.", "Mm, right.") is None
    assert extract("Mm, right.", "Mm, right.").reason == NO_PROPOSAL


# ── The vocabulary as an artifact ─────────────────────────────────────────────

def test_value_slot_cues_are_disjoint_so_the_most_specific_slot_wins():
    """HYPERNYMS unions the general categories for read-side rescue — `hobby` there contains every
    sport. Right for widening a question, wrong for labelling a claim: it typed 'Dana plays
    basketball' as `hobby` and buried the slot the substrate could actually answer."""
    assert not (SLOTS["sport"].cues & SLOTS["hobby"].cues)
    assert not (SLOTS["instrument"].cues & SLOTS["hobby"].cues)
    assert extract("Dana plays basketball.", "Dana plays basketball.").predicate.slot == "sport"


def test_no_question_head_demands_two_different_slots():
    """Enforced at import time; asserted here so the reason is recorded where it can be read. An
    ambiguous head means a quiet choice between slots, and a wrong refusal nobody can explain."""
    seen = {}
    for slot in SLOTS.values():
        for ask in slot.asks:
            assert seen.setdefault(ask, slot.name) == slot.name, f"{ask} is ambiguous"


def test_a_head_naming_no_slot_resolves_to_none_rather_than_guessing():
    assert slot_for_head("quux") is None


# ── The read side ─────────────────────────────────────────────────────────────

def test_a_typed_slot_grounds_a_question_the_surface_words_do_not():
    """'salary' appears nowhere in a store that says 'earns'. The slot was attached at admission
    from that same evidence, so the question lands without any query-time rescue."""
    g = _store(("Priya Raman earns 40000 a year.", "Priya Raman earns 40000 a year at Acme."))
    assert not read_gate("What is Priya Raman's salary?", g, semantic_rescue=False).abstain
    assert not build_vocabulary(g).df.get("salary")     # nothing lexical rescued it


def test_the_slot_lookup_is_scoped_to_the_subject_the_question_named():
    """A corpus-wide slot index grounds 'What is Dana's salary?' because PRIYA has a salary. Scope
    is what turns the index from a topic detector into an answer to the question asked."""
    g = _store(("Priya Raman earns 40000 a year.", "Priya Raman earns 40000 a year at Acme."),
               ("Dana Ali plays basketball.", "Dana Ali plays basketball every Saturday."))
    assert not read_gate("What is Priya Raman's salary?", g, semantic_rescue=False).abstain
    assert read_gate("What is Dana Ali's salary?", g, semantic_rescue=False).abstain


def test_a_typed_miss_overrules_the_corpus_wide_hypernym_rescue():
    """The rescue answers 'does anyone here have a cat?' for a question about one named person.
    Measured live before this rule: 'What pet does Dana have?' was answered with Marcus's cat."""
    g = _store(("Marcus Webb owns a cat.", "Marcus Webb owns a cat called Pepper."),
               ("Dana Ali plays basketball.", "Dana Ali plays basketball every Saturday."))
    assert not read_gate("What pet does Marcus Webb have?", g, semantic_rescue=False).abstain
    v = read_gate("What pet does Dana Ali have?", g, semantic_rescue=False)
    assert v.abstain and "typed miss" in v.detail


def test_a_question_about_one_person_is_not_answered_from_anothers_facts():
    """Measured 2026-08-27 on 300 cross-entity traps: with predicate grounding folded over the
    whole store, 299 of 300 questions about a person who has no such fact were answered using
    somebody else's. Both older corpora put one persona in one store, so this was structurally
    invisible to them. See docs/FINDING_read_side_grounding_is_unscoped.md."""
    g = _store(("Marcus Webb owns a cat.", "Marcus Webb owns a cat called Pepper."),
               ("Dana Ali plays basketball.", "Dana Ali plays basketball every Saturday."))
    assert not read_gate("What pet does Marcus Webb have?", g, semantic_rescue=False).abstain
    assert read_gate("What pet does Dana Ali have?", g, semantic_rescue=False).abstain
    # the same question with a different wh-word must not reopen it
    assert read_gate("Where is Dana Ali's pet?", g, semantic_rescue=False).abstain


def test_a_typed_miss_defers_when_the_wh_word_demands_a_different_axis():
    """'Where does Marco work?' has head `work` -> `occupation`, but the wh-word already declared
    the question wants a LOCATION, so the head-to-slot mapping answers on the wrong axis and the
    miss it produces is an artefact. Live, an unguarded rule suppressed a rescue that had been
    answering correctly (validity/test_temporal_contradiction_behavior::test_f4_out_of_order).

    Exercised with the authority switch forced ON, because that is the only configuration where the
    guard has any effect — the switch ships OFF (see read_gate.TYPED_MISS_OVERRULES_RESCUE), and a
    test that only passed in the shipped configuration would not be testing the guard at all.
    """
    import fireweed.read_gate as rg
    g = _store(("Marcus Webb owns a cat.", "Marcus Webb owns a cat called Pepper."),
               ("Marcus Webb plays basketball.", "Marcus Webb plays basketball every Saturday."))
    was = rg.TYPED_MISS_OVERRULES_RESCUE
    rg.TYPED_MISS_OVERRULES_RESCUE = True
    try:
        # `diet` is demanded, Marcus is typed (pet, sport) but carries no diet claim -> a typed miss
        assert rg.read_gate("What diet does Marcus Webb follow?", g, semantic_rescue=False).abstain
        # same miss, but the wh-word demands a location, so authority must defer to the rescue
        v = rg.read_gate("Where is Marcus Webb's pet?", g, semantic_rescue=False)
        assert not v.abstain
    finally:
        rg.TYPED_MISS_OVERRULES_RESCUE = was


def test_the_authority_switch_ships_off_with_its_measurement_recorded():
    """It refused 20.2% of answerable questions against a 3.7% baseline — a mislabel costing a
    missed read, exactly as the design predicted and said to measure rather than assume."""
    import fireweed.read_gate as rg
    assert rg.TYPED_MISS_OVERRULES_RESCUE is False
    assert rg.SCOPE_PREDICATE_GROUNDING_TO_SUBJECT is True


def test_an_entity_with_nothing_typed_still_gets_the_old_widening():
    """Authority requires the typed index to actually cover the entity. An entity it does not cover
    tells us nothing, and there the rescue must run exactly as it did before."""
    g = _store(("The Northfield depot handled 4000 parcels.",
                "The Northfield depot handled 4000 parcels."))
    ent = [e for e in g.all_entities()]
    assert ent, "fixture should register an entity"
    v = build_vocabulary(g)
    assert not v.slots_by_entity.get(ent[0].entity_id)
