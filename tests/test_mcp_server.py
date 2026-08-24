"""The MCP server, driven over the actual JSON-RPC protocol.

Not unit tests of the tool functions — a subprocess speaking real stdio MCP, because the thing that
breaks in an MCP server is the protocol edge (a stray print to stdout, a notification answered with
a reply, a crash that kills the session), and none of that is visible from inside the process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "src" / "fireweed_mcp" / "server.py"

DOC = ("Priya Raman joined Acme in 2019 as a logistics analyst. "
       "Dana Whitfield manages the Northfield depot.")


class Client:
    """Minimal MCP stdio client."""

    def __init__(self, store: Path):
        env = dict(os.environ)
        env["FIREWEED_MCP_STORE"] = str(store)
        env["TOKENIZERS_PARALLELISM"] = "false"
        self.p = subprocess.Popen(
            [sys.executable, str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, env=env, bufsize=1)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self._id,
                                       "method": method, "params": params or {}}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def notify(self, method):
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.p.stdin.flush()

    def tool(self, name, args=None):
        r = self.call("tools/call", {"name": name, "arguments": args or {}})
        return r["result"]["content"][0]["text"]

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=20)


@pytest.fixture()
def mcp(tmp_path):
    c = Client(tmp_path / "store")
    c.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
    c.tool("add_source", {"source_id": "memo", "text": DOC})
    yield c
    c.close()


def test_handshake_and_tool_list(tmp_path):
    c = Client(tmp_path / "s")
    init = c.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
    assert init["result"]["serverInfo"]["name"] == "fireweed"
    assert "tools" in init["result"]["capabilities"]
    names = {t["name"] for t in c.call("tools/list")["result"]["tools"]}
    assert {"remember", "recall", "forget", "verify_receipts", "export_memory"} <= names
    c.close()


def test_initialized_notification_gets_no_reply(tmp_path):
    """A notification has no id; replying to one corrupts the stream. Asserted because it is the
    single easiest way to break an MCP server and it is invisible from inside the process."""
    c = Client(tmp_path / "s")
    c.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
    c.notify("notifications/initialized")
    r = c.call("tools/list")            # the NEXT reply must be this one, not a stale notification ack
    assert r["id"] == 2 and "tools" in r["result"]
    c.close()


# ── the gate: the agent proposes, the server decides ─────────────────────────

def test_grounded_claim_is_admitted_with_a_receipt(mcp):
    out = mcp.tool("remember", {
        "claim": "Priya Raman joined Acme in 2019 as a logistics analyst.",
        "evidence": "Priya Raman joined Acme in 2019 as a logistics analyst.",
        "source_id": "memo"})
    assert out.startswith("ADMITTED")
    assert "receipt   : bytes [" in out


@pytest.mark.parametrize("claim,reason", [
    ("Priya Raman joined Acme in 2019 under duress.", "asserts_more_than_evidence"),
    ("Priya Raman joined Acme in 2021.", "numeral_invented"),
])
def test_fabrications_are_refused_with_a_typed_reason(mcp, claim, reason):
    """A refusal that does not say WHY cannot be worked with by the agent on the other side."""
    out = mcp.tool("remember", {
        "claim": claim,
        "evidence": "Priya Raman joined Acme in 2019 as a logistics analyst.",
        "source_id": "memo"})
    assert out.startswith("REFUSED")
    assert reason in out


def test_recall_abstains_with_the_ungrounded_term_named(mcp):
    mcp.tool("remember", {"claim": "Dana Whitfield manages the Northfield depot.",
                          "evidence": "Dana Whitfield manages the Northfield depot.",
                          "source_id": "memo"})
    out = mcp.tool("recall", {"query": "Dana Whitfield"})
    assert "grounded result" in out
    ab = mcp.tool("recall", {"query": "Dana's salary"})
    assert "ABSTAINED" in ab and "salary" in ab


def test_receipts_are_tamper_evident(mcp, tmp_path):
    mcp.tool("remember", {"claim": "Dana Whitfield manages the Northfield depot.",
                          "evidence": "Dana Whitfield manages the Northfield depot.",
                          "source_id": "memo"})
    assert "1/1 checkable" in mcp.tool("verify_receipts")
    src = tmp_path / "store" / "sources" / "memo.txt"
    src.write_text(src.read_text().replace("Northfield", "Southfield"))
    after = mcp.tool("verify_receipts")
    assert "0/1 checkable" in after and "FAILED" in after


def test_forget_issues_a_signed_certificate_and_spares_bystanders(mcp):
    for c, e in [("Priya Raman joined Acme in 2019 as a logistics analyst.",
                  "Priya Raman joined Acme in 2019 as a logistics analyst."),
                 ("Dana Whitfield manages the Northfield depot.",
                  "Dana Whitfield manages the Northfield depot.")]:
        mcp.tool("remember", {"claim": c, "evidence": e, "source_id": "memo"})
    out = mcp.tool("forget", {"subject": "Priya"})
    assert "certificate issued" in out
    assert "hmac-sha256:" in out
    assert "every probe abstains : True" in out
    assert "bystanders surviving : 1" in out          # Dana must survive Priya's erasure


def test_export_is_readable_by_the_public_reference_reader(mcp, tmp_path):
    """Portability is only real if something OUTSIDE this server can read the export."""
    mcp.tool("remember", {"claim": "Dana Whitfield manages the Northfield depot.",
                          "evidence": "Dana Whitfield manages the Northfield depot.",
                          "source_id": "memo"})
    dest = tmp_path / "export.json"
    mcp.tool("export_memory", {"path": str(dest)})
    sys.path.insert(0, str(ROOT / "open_format"))
    from reference_reader import read_snapshot
    fmp = read_snapshot(dest.read_bytes())
    assert any("Northfield" in c.claim for c in fmp.active_claims())


def test_unknown_tool_is_an_error_not_a_crash(mcp):
    r = mcp.call("tools/call", {"name": "nope", "arguments": {}})
    assert "error" in r
    assert mcp.tool("memory_stats")          # the session survives it
