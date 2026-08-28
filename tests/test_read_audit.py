"""Read auditing: on only when asked, and hashed unless asked twice.

The privacy default is the design decision under test. A system whose pitch is "trust neither the
agent nor the server" must not quietly begin recording every question asked of it, so the tests
that matter are the ones asserting what is NOT written.
"""
import json, pathlib, sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireweed import read_audit as ra   # noqa: E402


class _V:
    """A gate verdict, shaped like the real one."""
    def __init__(self, abstain=True, reason="unknown_predicate", head="salary", mode="lexical_only"):
        self.abstain, self.reason, self.mode = abstain, reason, mode
        self.demand = type("D", (), {"head": head})()


@pytest.fixture()
def on(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "ENABLED", True)
    monkeypatch.setattr(ra, "RECORD_QUERY_TEXT", False)
    return tmp_path / "read_audit.jsonl"


def _rows(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_auditing_is_off_unless_an_operator_turns_it_on(tmp_path):
    """A privacy-relevant surface must never appear because somebody upgraded."""
    assert ra.ENABLED is False and ra.RECORD_QUERY_TEXT is False
    p = tmp_path / "a.jsonl"
    assert ra.record(p, ra.build_event("What is X's salary?", _V())) is False
    assert not p.exists()


def test_the_query_text_is_not_written_by_default(on):
    ra.record(on, ra.build_event("What is Priya's salary?", _V()))
    row = _rows(on)[0]
    assert "query" not in row
    assert "salary" not in json.dumps(row).replace('"demand_head": "salary"', "")


def test_the_query_text_is_written_when_explicitly_enabled(on, monkeypatch):
    monkeypatch.setattr(ra, "RECORD_QUERY_TEXT", True)
    ra.record(on, ra.build_event("What is Priya's salary?", _V()))
    assert _rows(on)[0]["query"] == "What is Priya's salary?"


def test_repeat_queries_are_linkable_without_being_disclosed(on):
    for q in ("What is X's salary?", "  what  is  X's   SALARY? ", "Something else?"):
        ra.record(on, ra.build_event(q, _V()))
    fps = [r["fingerprint"] for r in _rows(on)]
    assert fps[0] == fps[1], "normalised repeats must share a fingerprint"
    assert fps[0] != fps[2]


def test_the_fingerprint_is_salted_against_dictionary_attack(on):
    """Unsalted, a log of short queries is trivially reversible -- the same reasoning that made
    entity identifiers salted rather than derived from names."""
    a = ra.fingerprint("What is X's salary?", "install-a")
    b = ra.fingerprint("What is X's salary?", "install-b")
    assert a != b
    assert ra.fingerprint("q") != "q"


def test_an_unwritable_log_never_breaks_a_read(on, monkeypatch):
    """The query already succeeded. Losing its audit row is the lesser harm."""
    def boom(*a, **k):
        raise OSError("read-only filesystem")
    monkeypatch.setattr(pathlib.Path, "mkdir", boom)
    assert ra.record(on, ra.build_event("q", _V())) is False


def test_the_summary_says_so_when_auditing_is_off(tmp_path):
    out = ra.summarise(tmp_path / "missing.jsonl")
    assert "off" in out.lower() and "FIREWEED_MCP_READ_AUDIT" in out


def test_the_summary_aggregates_abstention_reasons(on):
    ra.record(on, ra.build_event("a", _V(reason="unknown_predicate")))
    ra.record(on, ra.build_event("b", _V(reason="entity_not_found", head="pet")))
    ra.record(on, ra.build_event("c", _V(abstain=False, reason=None, head="pet")))
    out = ra.summarise(on)
    assert "1 answered" in out and "2 abstained" in out
    assert "unknown_predicate" in out and "entity_not_found" in out


def test_reads_are_not_recorded_in_the_mutation_ledger():
    """Reads change no state. Putting non-events into a chain whose guarantee is that it replays
    to the live state would break that guarantee for no benefit."""
    from fireweed.ledger import EVENT_KINDS
    assert not any("READ" in k or "QUERY" in k for k in EVENT_KINDS)
