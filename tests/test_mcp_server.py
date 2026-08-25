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
        env["FIREWEED_MCP_KEYS"] = str(Path(store).parent / "keys")
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

    def tool_result(self, name, args=None):
        """The whole result object, for assertions about `isError` rather than prose."""
        return self.call("tools/call", {"name": name, "arguments": args or {}})["result"]

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
    assert "hmac-sha256:" in out or "ed25519:" in out, "the scheme must be named in the signature"
    assert "verifiable by others" in out, "the caller must be told what the signature is worth"
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


# ── Phase 0 hardening: enforcement and disclosure, both raised in review ──────


def test_refusal_is_signalled_structurally_not_only_in_prose(mcp):
    """A refusal must set isError, so enforcement does not depend on the model reading the text.

    Every verdict used to come back in an identical success envelope, so a harness that summarises
    tool output could soften or drop a REFUSED with nothing at the protocol level to stop it.
    """
    r = mcp.tool_result("remember", {
        "claim": "Priya Raman joined Acme in 2019 under duress.",
        "evidence": "Priya Raman joined Acme in 2019 as a logistics analyst.",
        "source_id": "memo"})
    assert r["content"][0]["text"].startswith("REFUSED")
    assert r.get("isError") is True


def test_admission_is_not_an_error(mcp):
    r = mcp.tool_result("remember", {
        "claim": "Priya Raman joined Acme in 2019 as a logistics analyst.",
        "evidence": "Priya Raman joined Acme in 2019 as a logistics analyst.",
        "source_id": "memo"})
    assert r["content"][0]["text"].startswith("ADMITTED")
    assert r.get("isError") is False


def test_abstention_is_deliberately_not_an_error(mcp):
    """A grounded refusal to answer is a CORRECT result, not a failed call.

    Flagging it would invite the retry-until-a-row-returns behaviour abstention exists to prevent,
    and would contradict recall's own tool description.
    """
    mcp.tool("remember", {"claim": "Priya Raman joined Acme in 2019 as a logistics analyst.",
                          "evidence": "Priya Raman joined Acme in 2019 as a logistics analyst.",
                          "source_id": "memo"})
    r = mcp.tool_result("recall", {"query": "What is Priya Raman's salary?"})
    assert r["content"][0]["text"].startswith("ABSTAINED")
    assert r.get("isError") is False


def test_erasure_certificate_discloses_its_own_scope(mcp):
    """The certificate computes a scope/cipher/key_destroyed caveat; the server must print it.

    tool_forget used to print the signature, counts and hashes -- the flattering fields -- drop the
    caveat, and assert the certificate was "the artifact a compliance reviewer asks for". Without a
    keyring no content key is destroyed, and the caller has to be told that.
    """
    mcp.tool("remember", {"claim": "Priya Raman joined Acme in 2019 as a logistics analyst.",
                          "evidence": "Priya Raman joined Acme in 2019 as a logistics analyst.",
                          "source_id": "memo"})
    out = mcp.tool("forget", {"subject": "Priya Raman"})
    assert "SCOPE" in out, "the certificate's own scope statement must reach the caller"
    assert "cipher" in out and "content key destroyed" in out
    # a keyring IS now attached, so node content is genuinely crypto-shredded
    assert "content key destroyed: True" in out
    # The cipher must be NAMED, whatever it is. Asserting "AES" specifically passed locally, where
    # `cryptography` happens to be installed, and failed in CI, where it is not and the keystream
    # fallback runs — an environment-dependent assertion masquerading as a property.
    cipher_line = next(l for l in out.splitlines() if l.strip().startswith("cipher"))
    named = cipher_line.split(":", 1)[1].strip()
    assert named and named != "none", f"the cipher actually used must be named, got {named!r}"


def test_signing_key_is_per_install_not_a_source_literal(tmp_path):
    """The key was b"fireweed-mcp-key", a constant in public source: anyone could forge a cert."""
    store = tmp_path / "store"
    c = Client(store)
    c.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
    c.tool("add_source", {"source_id": "memo", "text": DOC})
    c.tool("remember", {"claim": "Priya Raman joined Acme in 2019 as a logistics analyst.",
                        "evidence": "Priya Raman joined Acme in 2019 as a logistics analyst.",
                        "source_id": "memo"})
    c.tool("forget", {"subject": "Priya Raman"})
    c.close()
    # Either scheme is acceptable; what must never recur is a key that ships in the source.
    kdir = store.parent / "keys"
    keys = [p for p in (kdir / "signing_key", kdir / "ed25519_key") if p.exists()]
    assert keys, "a per-install signing key must be generated"
    key = keys[0].read_bytes()
    assert key != b"fireweed-mcp-key"
    assert len(key) == 32
    if (store / "ed25519_key").exists():
        pub = store / "ed25519_public_key"
        assert pub.exists(), "the public half must be published — it is what a verifier needs"
        assert pub.read_text().strip() != ""


# ── Phase 1: the ledger is attached, and erasure leaves a durable trace ───────


def test_mutations_are_recorded_in_a_persistent_ledger(mcp, tmp_path):
    """attach_ledger and seal() had no caller anywhere: every mutation recorded nothing, ever."""
    mcp.tool("remember", {"claim": "Priya Raman joined Acme in 2019 as a logistics analyst.",
                          "evidence": "Priya Raman joined Acme in 2019 as a logistics analyst.",
                          "source_id": "memo"})
    assert (tmp_path / "store" / "ledger.db").exists(), "the ledger must be persisted, not in-memory"


def test_erasure_is_remembered_so_a_reproposal_cannot_slip_back_in(mcp):
    """Erase a subject, re-propose the identical claim: it used to be silently re-admitted.

    Confirmed live before the fix -- identical claim, identical evidence, ADMITTED again with no
    reference to the erasure. There was no durable record for the write path to consult, because
    forget wrote its ERASE event to a throwaway in-memory ledger.
    """
    claim = "Priya Raman joined Acme in 2019 as a logistics analyst."
    mcp.tool("remember", {"claim": claim, "evidence": claim, "source_id": "memo"})
    assert mcp.tool("forget", {"subject": "Priya Raman"}).startswith("ERASED")
    again = mcp.tool("remember", {"claim": claim, "evidence": claim, "source_id": "memo"})
    assert again.startswith("NOT STORED (previously_erased)")
    assert "acknowledge_erasure" in again, "the refusal must name the way through"


def test_reproposal_is_possible_but_must_be_deliberate(mcp):
    """Not a permanent ban: re-consent happens, and two people share a name. It must be a choice."""
    claim = "Priya Raman joined Acme in 2019 as a logistics analyst."
    mcp.tool("remember", {"claim": claim, "evidence": claim, "source_id": "memo"})
    mcp.tool("forget", {"subject": "Priya Raman"})
    out = mcp.tool("remember", {"claim": claim, "evidence": claim, "source_id": "memo",
                                "acknowledge_erasure": True})
    assert out.startswith("ADMITTED")


def test_private_keys_do_not_live_in_the_store(mcp, tmp_path):
    """Copying, syncing or backing up the store must not hand over the keys with it.

    They used to sit beside substrate.json, which voided both guarantees they provide: anyone who
    could read the data could mint a certificate, and any backup taken before an erasure carried the
    content keys alongside the ciphertext they were meant to shred.
    """
    mcp.tool("remember", {"claim": "Priya Raman joined Acme in 2019 as a logistics analyst.",
                          "evidence": "Priya Raman joined Acme in 2019 as a logistics analyst.",
                          "source_id": "memo"})
    mcp.tool("forget", {"subject": "Priya Raman"})
    store = tmp_path / "store"
    for secret in ("signing_key", "ed25519_key", "keyring.json"):
        assert not (store / secret).exists(), f"{secret} must not live in the store directory"
    # the PUBLIC key is not a secret and belongs with the data — it is what a verifier needs
    if (tmp_path / "keys" / "ed25519_key").exists():
        assert (store / "ed25519_public_key").exists()
