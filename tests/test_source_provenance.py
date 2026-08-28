"""Auditing backwards from a stored memory to the arrival of its evidence.

The property under test is a SEPARATION, not a capability: three facts the store can prove, and two
it cannot, kept visibly apart. A provenance trail that mixes verified and asserted fields is worse
than none — it is the erasure-certificate mistake in a new place.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_mcp_server import Client, DOC        # noqa: E402 — same stdio harness, same protocol edge


@pytest.fixture()
def mcp(tmp_path):
    c = Client(tmp_path / "store")
    c.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
    yield c
    c.close()


def _register(c, **kw):
    args = {"source_id": "memo", "text": DOC}
    args.update(kw)
    return c.tool("add_source", args)


def _remember(c):
    return c.tool("remember", {
        "claim": "Priya Raman joined Acme in 2019 as a logistics analyst.",
        "evidence": "Priya Raman joined Acme in 2019 as a logistics analyst.",
        "source_id": "memo"})


# ── the arrival is recorded at all ────────────────────────────────────────────

def test_a_source_arrival_lands_in_the_append_only_ledger(mcp):
    """Before this, `add_source` wrote a file and returned a hash, and the ledger never learned the
    document existed — every claim binding was chained while the evidence's arrival was not."""
    out = _register(mcp, origin="s3://hr/staff.txt", origin_kind="file",
                    supplied_by="hr-sync", validated_by="none")
    assert "ledger seq" in out
    assert "chained to" in out


def _graph_lines(stats: str) -> list[str]:
    """Only the GRAPH rows. `sources held` legitimately changes when a source is registered; nodes
    and entities must not."""
    return [l for l in stats.splitlines()
            if l.startswith(("claims", "entities"))]


def test_the_arrival_event_does_not_change_graph_state(mcp):
    """ADD_SOURCE is audit-grain, not write-grain. Replay skips it (SQLiteLedger.replay filters on
    WRITE_KINDS), because applying it would materialise nothing while breaking the live-vs-replay
    fingerprint equivalence `seal()` exists to guarantee."""
    _register(mcp)
    _remember(mcp)
    before = _graph_lines(mcp.tool("memory_stats"))
    _register(mcp, source_id="memo2", text="Unrelated document about nothing at all.")
    after = _graph_lines(mcp.tool("memory_stats"))
    assert before == after, "registering a source changed graph state"
    assert any("1" in l for l in before), "fixture should have stored a claim"


# ── the separation between attested and declared ──────────────────────────────

def test_declared_provenance_is_labelled_as_unverified_on_every_registration(mcp):
    out = _register(mcp, origin="https://example.test/feed", origin_kind="url",
                    supplied_by="ingest-bot", validated_by="nothing")
    assert "NOT verified" in out
    assert "does not prove the assertion is true" in out


def test_an_unknown_origin_kind_is_recorded_as_unknown_rather_than_accepted(mcp):
    out = _register(mcp, origin_kind="telepathy")
    assert "unknown" in out


def test_the_trace_separates_what_is_proved_from_what_is_asserted(mcp):
    _register(mcp, origin="s3://hr/staff.txt", origin_kind="file",
              supplied_by="hr-sync", validated_by="none")
    _remember(mcp)
    t = mcp.tool("trace_evidence", {"claim": "Priya Raman joined"})
    assert "attested" in t and "declared" in t
    assert "s3://hr/staff.txt" in t                       # the assertion is shown
    assert "WHAT THIS DOES NOT PROVE" in t                # and bounded
    assert "ordering" in t.lower() or "order" in t.lower()


# ── the four questions the trace answers ──────────────────────────────────────

def test_the_trace_joins_a_stored_claim_to_its_sources_arrival(mcp):
    _register(mcp, origin="s3://hr/staff.txt", origin_kind="file")
    _remember(mcp)
    t = mcp.tool("trace_evidence", {"claim": "Priya Raman joined"})
    assert "MATCH" in t                 # bytes still what they were
    assert "ledger seq" in t            # arrival is in the chain
    assert "VERIFIES" in t              # and the chain itself holds


def test_tampering_with_the_source_fails_the_bytes_but_keeps_the_arrival_record(mcp, tmp_path):
    """The useful audit property: you can still see what the document was SUPPOSED to be, and that
    it no longer is. Losing the arrival record along with the bytes would destroy the evidence that
    the tamper happened."""
    _register(mcp, origin="s3://hr/staff.txt", origin_kind="file")
    _remember(mcp)
    src = tmp_path / "store" / "sources" / "memo.txt"
    src.write_text(src.read_text().replace("2019", "2021"))
    t = mcp.tool("trace_evidence", {"claim": "Priya Raman joined"})
    assert "SOURCE MISSING OR CHANGED" in t
    assert "ledger seq" in t            # the arrival, with the ORIGINAL hash, survives
    assert "real failure, not a warning" in t


def test_a_turn_bound_claim_says_there_is_nothing_to_audit_back_to(mcp):
    """Honest negative: with no registered source the chain does not exist, and the trace must say
    so rather than implying an audit happened."""
    mcp.tool("remember", {"claim": "Marcus Webb runs the Leeds depot.",
                          "evidence": "Marcus Webb runs the Leeds depot."})
    t = mcp.tool("trace_evidence", {"claim": "Marcus Webb"})
    assert "TURN-BOUND" in t
    assert "nothing to audit backwards to" in t


def test_an_unknown_claim_is_not_found_rather_than_silently_empty(mcp):
    _register(mcp)
    assert "NOT FOUND" in mcp.tool("trace_evidence", {"claim": "zzz nonexistent zzz"})


# ── the joinable-key regression ───────────────────────────────────────────────

def test_the_source_record_hashes_in_the_same_format_as_the_receipt():
    """The first version returned bare hex while receipts use `sha256:<hex>`, so every join between
    a claim and its source's arrival failed SILENTLY — and the failure looked exactly like the
    tamper detection working correctly. Two formats for one concept is a joinable-key bug."""
    sys.path.insert(0, str(ROOT / "src"))
    from fireweed.source_provenance import doc_hash, normalize_hash
    from fireweed.receipts import hash_document
    assert doc_hash("hello") == hash_document(b"hello")
    assert doc_hash("hello").startswith("sha256:")
    # and an older bare-hex record still joins
    bare = doc_hash("hello").split(":", 1)[1]
    assert normalize_hash(bare) == doc_hash("hello")
