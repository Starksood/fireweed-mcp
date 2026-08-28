"""Bounded slot naming: the encoder may say WHAT a question asks for, never WHETHER to answer.

The safety argument is entirely about the encoder's output space. These tests pin that boundary,
because the whole result collapses if the encoder ever regains an unbounded target set.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from fireweed.fabric import Fireweed
from fireweed.predicate_vocabulary import SLOTS
from fireweed import read_gate as rg


def _store(*pairs):
    fw = Fireweed(llm=lambda p: "")
    fw._ctx.begin_session("s1", "2026-08-27T00:00:00+00:00")
    for claim, evidence in pairs:
        fw._ctx.ingest(claim, evidence, 0.9, "t", "s1")
    return fw._ctx.graph


# The encoder is an optional extra and CI installs only pytest. Tests that exercise it must SKIP
# rather than pass, because `slot_by_encoder` returns None without it -- so an unguarded assertion
# like "only ever names an authored slot" would pass vacuously on a machine where the mechanism
# cannot run at all. Same failure shape as a trap corpus the mechanism never reaches.
def _encoder_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None


needs_encoder = pytest.mark.skipif(not _encoder_available(),
                                   reason="sentence-transformers not installed; "
                                          "bounded slot naming cannot run")


@pytest.fixture()
def bounded():
    was = rg.SLOT_RESCUE_BY_ENCODER
    rg.SLOT_RESCUE_BY_ENCODER = True
    yield
    rg.SLOT_RESCUE_BY_ENCODER = was


@needs_encoder
def test_the_encoder_can_only_ever_name_an_authored_slot():
    """The output space IS the safety property. An encoder that can return anything else is the
    unbounded rescue that collapsed trap refusal from 96.1% to 32.8%."""
    for head in ("creature", "trade", "craft", "quux", "supercalifragilistic", ""):
        slot, _ = rg.slot_by_encoder(head)
        assert slot is None or slot in SLOTS, f"{head!r} named {slot!r}, which is not a slot"


@needs_encoder
def test_a_head_that_matches_nothing_names_nothing():
    for head in ("quux", "weather", "asdfgh"):
        assert rg.slot_by_encoder(head, min_sim=0.5)[0] is None


@needs_encoder
def test_naming_a_slot_does_not_authorize_an_answer(bounded):
    """The crux. The encoder maps `creature` to `pet` with high confidence -- and the subject still
    has no pet, so the gate still refuses. The encoder widens what the question is UNDERSTOOD as;
    deterministic code decides whether anything answers it."""
    assert rg.slot_by_encoder("creature", min_sim=0.5)[0] == "pet"
    g = _store(("Dana Ali plays basketball.", "Dana Ali plays basketball every Saturday."))
    assert rg.read_gate("What creature does Dana Ali look after?", g).abstain


@needs_encoder
def test_naming_a_slot_does_answer_when_the_subject_actually_holds_one(bounded):
    g = _store(("Marcus Webb owns a cat.", "Marcus Webb owns a cat called Pepper."))
    v = rg.read_gate("What creature does Marcus Webb look after?", g)
    assert not v.abstain
    assert "pet" in v.detail


@needs_encoder
def test_the_encoder_cannot_reach_across_entities(bounded):
    """Subject scope still applies after the encoder names the slot -- otherwise bounded naming
    would reintroduce the cross-entity confusion that scoping closed."""
    g = _store(("Marcus Webb owns a cat.", "Marcus Webb owns a cat called Pepper."),
               ("Dana Ali plays basketball.", "Dana Ali plays basketball every Saturday."))
    assert rg.read_gate("What creature does Dana Ali look after?", g).abstain
    assert not rg.read_gate("What creature does Marcus Webb look after?", g).abstain


def test_the_shipped_default_is_off_with_its_measurement_recorded():
    """It ships off because the operating threshold was chosen by sweeping the reported corpora,
    which is fitting to the test set. The frontier is the finding; the point needs validation."""
    assert rg.SLOT_RESCUE_BY_ENCODER is False
    assert rg.SLOT_RESCUE_MIN_SIM == 0.60


@needs_encoder
def test_the_known_mislabel_is_pinned(bounded):
    """`habits` names `hobby`, not `diet`. At 0.50 that answered 11 diet traps with hobby facts;
    at the shipped 0.60 it falls below threshold. Pinned so a vocabulary edit that revives it fails
    here rather than in a corpus run nobody re-ran."""
    slot, score = rg.slot_by_encoder("habits", min_sim=0.0)
    assert slot == "hobby"
    assert score < 0.60, "the mislabel is back above threshold"
