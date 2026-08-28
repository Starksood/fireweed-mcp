"""The untrusted extractor: it may hallucinate freely, and the gate must not care.

The safety argument is a claim about the WORST case, so these tests are mostly about what the
extractor cannot achieve no matter how badly it behaves. A test suite that only shows the happy
path would be evidence for the model, not for the trust boundary.
"""
import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireweed_extractor import extract, EchoProposer, Proposal          # noqa: E402
from fireweed_extractor.extract import _why_refused                     # noqa: E402

SOURCE = ("Priya Raman joined Acme in 2019 as a logistics analyst. "
          "Marcus Webb runs the Leeds depot and cycles to work every morning.")


class _Fixed:
    """A proposer that emits exactly what a test tells it to. The worst-case extractor."""
    name = "fixed"

    def __init__(self, pairs):
        self.pairs = pairs

    def propose(self, text):
        return [Proposal(claim=c, evidence=e, proposer=self.name) for c, e in self.pairs]


# ── the trust boundary ────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind,claim,evidence", [
    ("overreach",
     "Priya Raman joined Acme in 2019 under duress.",
     "Priya Raman joined Acme in 2019 as a logistics analyst."),
    ("unsupported_inference",
     "Priya Raman is senior at Acme.",
     "Priya Raman joined Acme in 2019 as a logistics analyst."),
    ("transposed_relation",
     "Acme was joined by Priya Raman's employer in 2019.",
     "Priya Raman joined Acme in 2019 as a logistics analyst."),
    ("invented_numeral",
     "Priya Raman joined Acme in 2017 as a logistics analyst.",
     "Priya Raman joined Acme in 2019 as a logistics analyst."),
    ("wrong_subject",
     "Marcus Webb joined Acme in 2019 as a logistics analyst.",
     "Priya Raman joined Acme in 2019 as a logistics analyst."),
    ("fabricated_quote",
     "Priya Raman was promoted to director in 2021.",
     "Priya Raman was promoted to director in 2021."),
])
def test_every_fabrication_class_is_refused(kind, claim, evidence):
    """Six ways a real extraction model goes wrong. None may reach memory.

    `overreach` and `unsupported_inference` are here because the first implementation admitted both
    -- it used `claim_faithful`, whose document_mode defaults off and therefore skips the check that
    rejects a claim asserting more than its span.
    """
    r = extract(SOURCE, _Fixed([(claim, evidence)]))
    assert not r.admitted, f"{kind} was admitted — the trust boundary leaked"
    assert r.rejected[0][1], "a refusal must name the check that failed"


def test_a_faithful_proposal_is_admitted():
    """The boundary has to let real work through, or it is just an off switch."""
    r = extract(SOURCE, _Fixed([("Priya Raman joined Acme in 2019 as a logistics analyst.",
                                 "Priya Raman joined Acme in 2019 as a logistics analyst.")]))
    assert len(r.admitted) == 1 and not r.rejected


def test_a_quote_absent_from_the_source_cannot_be_used_as_evidence():
    """The most basic hallucination: citing text that was never there. Checked against the source
    passed to `extract`, not against the proposer's own assertion that it was there."""
    assert _why_refused.__name__          # imported for the sake of the check below
    r = extract(SOURCE, _Fixed([("Priya Raman was promoted in 2021.",
                                 "Priya Raman was promoted in 2021.")]))
    assert r.rejected and r.rejected[0][1] == "span_not_in_source"


# ── the extractor may not be privileged ───────────────────────────────────────

def test_the_extractor_applies_exactly_the_servers_checks():
    """An extractor held to a weaker standard than a hand-written caller is not an untrusted
    client, it is a privileged one. Asserted structurally so the two cannot drift apart: both must
    call the same four grounding predicates."""
    server_path = ROOT / "mcp_server" / "server.py"
    if not server_path.exists():                      # packaged layout
        server_path = ROOT / "src" / "fireweed_mcp" / "server.py"
    server = server_path.read_text()
    required = ["subject_grounded", "order_preserved", "numerals_grounded", "predicate_grounded"]
    for fn in required:
        assert f"not {fn}(claim, evidence)" in server, f"server no longer calls {fn}"
    src = (ROOT / "src" / "fireweed_extractor" / "extract.py").read_text()
    for fn in required:
        assert f"not {fn}(claim, evidence)" in src, f"extractor does not apply {fn}"


def test_the_engine_does_not_import_the_extractor():
    """The dependency arrow points one way. If any `fireweed.*` module imported the extractor, the
    extractor would be inside the trust boundary and the entire argument would be circular."""
    offenders = []
    for path in (ROOT / "src" / "fireweed").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mod = ""
            if isinstance(node, ast.Import):
                mod = " ".join(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            if "fireweed_extractor" in mod:
                offenders.append(path.name)
    assert not offenders, f"engine modules import the extractor: {offenders}"


def test_the_extractor_never_writes_to_a_store():
    """It returns proposals; the caller submits them. A package that both proposes and stores is
    back inside the boundary it exists to stay outside of."""
    src = (ROOT / "src" / "fireweed_extractor" / "extract.py").read_text()
    for forbidden in ("attach_ledger", "ctx.ingest", "add_node", "Fireweed("):
        assert forbidden not in src, f"extractor touches {forbidden}"


# ── ordinary behaviour ────────────────────────────────────────────────────────

def test_the_deterministic_control_needs_no_model():
    """Nothing here may depend on a server being up, or the suite is untestable offline."""
    r = extract(SOURCE, EchoProposer())
    assert len(r.admitted) == 2 and not r.rejected


def test_a_proposer_that_returns_nothing_is_an_ordinary_outcome():
    r = extract(SOURCE, _Fixed([]))
    assert r.proposed == 0 and r.rejection_rate == 0.0


def test_rejections_are_returned_rather_than_discarded():
    """The record of the extractor being caught is the evidence the design works, so it is part of
    the result rather than a log line."""
    r = extract(SOURCE, _Fixed([("Priya Raman joined Acme in 2019 under duress.",
                                 "Priya Raman joined Acme in 2019 as a logistics analyst.")]))
    assert r.rejection_rate == 1.0
    assert "REFUSED" in r.render()
