#!/usr/bin/env python3
"""Fireweed MCP server — agent memory where every fact carries a receipt.

    claude mcp add fireweed -- python3 /path/to/fireweed-v16/mcp_server/server.py

WHAT MAKES THIS DIFFERENT FROM EVERY OTHER MEMORY SERVER

Vector-store memories accept whatever the model says and return whatever is nearest. This one
adjudicates. The calling agent is the PROPOSER; a deterministic gate decides what is admitted, and
the record can be checked afterwards by someone who does not trust either of us:

    remember   a claim is admitted ONLY if its evidence supports it. Refusals are TYPED
               ("the evidence names no subject for this claim"), not silent drops.
    recall     answers carry the byte range they came from. When the substrate cannot answer it
               SAYS SO, and says which term it could not ground.
    verify     re-hash the source, re-slice the bytes. Change one byte and it fails.
    forget     erasure with exact closure and a SIGNED CERTIFICATE. Bystanders survive.
    export     the whole substrate as a portable blob. You can leave whenever you want.

THE PROTOCOL IS THE ARCHITECTURE. "The model proposes, deterministic code decides" stops being a
slogan the moment the model is on the other side of an RPC boundary: the agent proposes a claim and
its evidence, and this server refuses the ones that are not grounded. The agent cannot talk its way
past the gate, because the gate is not a prompt.

Stdlib only — MCP over JSON-RPC on stdin/stdout, no SDK (the official one needs Python 3.10+ and
this engine runs on 3.9). Same discipline as open_format/reference_reader.py: a dependency list is a
promise, and short promises are keepable.

⚠️  IP BOUNDARY: this process imports the engine (src/fireweed/**). Shipping it to a developer's
machine ships the engine. See mcp_server/README.md — this is a real decision, not an oversight.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

# Installed as a package: `fireweed` is a sibling module, no path surgery needed.
ROOT = Path(__file__).resolve().parent

PROTOCOL_VERSION = "2024-11-05"
STORE = Path(os.environ.get("FIREWEED_MCP_STORE", Path.home() / ".fireweed" / "mcp"))
SUBSTRATE = STORE / "substrate.json"
SOURCES = STORE / "sources"


# ── engine access ─────────────────────────────────────────────────────────────

_fw = None


def fabric():
    """The substrate, loaded once and persisted after every mutation."""
    global _fw
    if _fw is None:
        from fireweed.fabric import Fireweed
        STORE.mkdir(parents=True, exist_ok=True)
        SOURCES.mkdir(parents=True, exist_ok=True)
        _fw = Fireweed(llm=lambda _p: "{}")      # no reader model: nothing here needs one
        if SUBSTRATE.exists():
            _fw.restore(SUBSTRATE.read_bytes())
    return _fw


def persist():
    SUBSTRATE.write_bytes(fabric().snapshot())


# ── tools ─────────────────────────────────────────────────────────────────────

def tool_remember(args: dict) -> str:
    """The gate. The agent proposes; this decides."""
    from fireweed.grounding import classify, predicate_grounded, subject_grounded, order_preserved, numerals_grounded
    from fireweed.receipts import bind_document, receipt_for

    claim = (args.get("claim") or "").strip()
    evidence = (args.get("evidence") or "").strip()
    source_id = (args.get("source_id") or "agent").strip()
    source_text = args.get("source_text") or ""
    if source_text:
        # Register the document inline. Without this, byte-range receipts -- the headline property --
        # only appeared if you had already read the docs and called add_source first. A first-run
        # developer would see "turn-bound (no source document registered)" and conclude the feature
        # did not exist. The best feature you have should not be behind a second tool call.
        STORE.mkdir(parents=True, exist_ok=True)
        SOURCES.mkdir(parents=True, exist_ok=True)
        (SOURCES / f"{source_id}.txt").write_text(source_text)
    if not claim or not evidence:
        return "REFUSED — both `claim` and `evidence` are required. Evidence must be text you are " \
               "quoting, not a summary of it."

    # Why-it-failed, not just that-it-failed. A gate that only says "no" cannot be worked with.
    if not subject_grounded(claim, evidence):
        return (f"REFUSED (unknown_subject) — the evidence does not name the subject this claim is "
                f"about.\n  claim   : {claim}\n  evidence: {evidence}\n"
                f"Quote a span that names the subject, or state the claim about who the span names.")
    if not order_preserved(claim, evidence):
        return (f"REFUSED (relation_transposed) — the claim rearranges the evidence's relation.\n"
                f"  claim   : {claim}\n  evidence: {evidence}")
    if not numerals_grounded(claim, evidence):
        return (f"REFUSED (numeral_invented) — the claim contains a number the evidence does not.\n"
                f"  claim   : {claim}\n  evidence: {evidence}")
    if not predicate_grounded(claim, evidence):
        return (f"REFUSED (asserts_more_than_evidence) — the claim adds something the evidence does "
                f"not say.\n  claim   : {claim}\n  evidence: {evidence}\n"
                f"This is the check that catches 'signed the FRAUDULENT lease' cited to a span that "
                f"never uses the word.")

    fw = fabric()
    ctx = fw._ctx
    turn = f"{source_id}_{len(ctx.graph.all_nodes())}"
    if source_id not in getattr(ctx, "_sessions_seen", {}):
        try:
            ctx.begin_session(source_id)
        except Exception:
            pass
    ctx.ingest(claim, evidence, 0.9, turn, source_id)

    # Bind a receipt when we hold the source document this evidence came from.
    src = SOURCES / f"{source_id}.txt"
    if src.exists():
        bind_document(ctx.graph, src.read_text(), source_id)
    node = next((n for n in ctx.graph.all_nodes() if n.claim == claim), None)
    persist()

    cls = classify(claim, evidence)
    rec = receipt_for(node) if node is not None else None
    out = [f"ADMITTED — {claim}", f"  grounding : {cls}"]
    if rec:
        out.append(f"  receipt   : bytes [{rec.byte_start}:{rec.byte_end}] of {rec.doc_hash[:19]}…")
    else:
        out.append("  receipt   : turn-bound (no source document registered for this source_id — "
                   "use `add_source` first to get byte-range receipts)")
    return "\n".join(out)


def tool_add_source(args: dict) -> str:
    """Register a source document so claims from it can carry byte-range receipts."""
    import hashlib
    source_id = (args.get("source_id") or "").strip()
    text = args.get("text") or ""
    if not source_id or not text:
        return "REFUSED — `source_id` and `text` are required."
    STORE.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)
    (SOURCES / f"{source_id}.txt").write_text(text)
    h = hashlib.sha256(text.encode()).hexdigest()
    return (f"registered {source_id} — {len(text)} bytes, sha256:{h[:16]}…\n"
            f"Claims remembered against this source_id now bind to byte ranges in it.")


def tool_recall(args: dict) -> str:
    """Retrieval. Deterministic — no model runs here."""
    from fireweed.retrieval import query_graph
    from fireweed.receipts import receipt_for, verify as verify_receipt

    q = (args.get("query") or "").strip()
    if not q:
        return "REFUSED — `query` is required."
    fw = fabric()
    r = query_graph(q, fw._ctx.graph)
    if r.abstain:
        g = r.gate_verdict
        detail = g.detail if g else "no grounded evidence"
        return (f"ABSTAINED ({r.abstain_reason}) — {detail}\n"
                f"This is a refusal, not an empty result. The substrate does not hold an answer to "
                f"this, and is telling you rather than guessing.")
    lines = [f"{len(r.matched_nodes)} grounded result(s):"]
    for e in r.matched_nodes[:5]:
        n = e.node
        rec = receipt_for(n)
        lines.append(f"  · {n.claim}")
        if rec:
            src = SOURCES / f"{n.provenance.source_turn_id.rsplit('_', 1)[0]}.txt"
            ok = verify_receipt(rec, src.read_bytes()) if src.exists() else None
            mark = {True: "verified", False: "FAILED", None: "source not held"}[ok]
            lines.append(f"      receipt bytes [{rec.byte_start}:{rec.byte_end}] — {mark}")
        lines.append(f"      node {n.node_id[:16]}… · {n.provenance.source_turn_id}")
    return "\n".join(lines)


def tool_verify(args: dict) -> str:
    """Re-hash the source and re-slice the bytes. The point of a receipt is that it CAN fail."""
    from fireweed.receipts import receipt_for, verify as verify_receipt
    fw = fabric()
    checked = ok = unheld = 0
    failures = []
    for n in fw._ctx.graph.all_nodes():
        rec = receipt_for(n)
        if rec is None:
            continue
        checked += 1
        src = SOURCES / f"{n.provenance.source_turn_id.rsplit('_', 1)[0]}.txt"
        if not src.exists():
            unheld += 1
            continue
        if verify_receipt(rec, src.read_bytes()):
            ok += 1
        else:
            failures.append(n.claim)
    out = [f"receipts re-verified: {ok}/{checked - unheld} checkable ({unheld} whose source is not held)"]
    for f in failures:
        out.append(f"  FAILED — {f}")
    if failures:
        out.append("A failure means the source document changed after the claim was bound to it.")
    return "\n".join(out)


def tool_forget(args: dict) -> str:
    """Erasure with exact closure and a signed certificate."""
    from fireweed.erasure import erase, ErasureIncomplete
    from fireweed.ledger_sqlite import SQLiteLedger
    from fireweed import retrieval

    subject = (args.get("subject") or "").strip()
    if not subject:
        return "REFUSED — `subject` is required."
    fw = fabric()
    g = fw._ctx.graph
    ent = next((e for e in g.all_entities() if subject.lower() in e.canonical_name.lower()), None)
    if ent is None:
        return f"no subject matching {subject!r} is present in the substrate — nothing to erase."
    probes = [ent.canonical_name, subject]
    try:
        cert = erase(g, SQLiteLedger(":memory:"), "mcp", ent.entity_id, probes,
                     lambda gg, q: retrieval.query_graph(q, gg), b"fireweed-mcp-key")
    except ErasureIncomplete as e:
        return f"NO CERTIFICATE ISSUED — erasure incomplete: {e}\nNothing was removed."
    persist()
    survivors = [n.claim for n in g.all_nodes() if n.status.memory_state in ("active", "disputed")]
    return "\n".join([
        f"ERASED {ent.canonical_name} — certificate issued",
        f"  signature            : {cert.signature}",
        f"  nodes in closure     : {len(cert.closure_manifest['node_ids'])}",
        f"  derived invalidated  : {len(cert.closure_manifest['derived_invalidated'])}",
        f"  every probe abstains : {cert.battery_all_abstained}",
        f"  state {cert.state_hash_before[:18]}… -> {cert.state_hash_after[:18]}…",
        f"  bystanders surviving : {len(survivors)}",
        "",
        "This certificate is the artifact a compliance reviewer asks for. The closure is exact: "
        "facts derived from the subject go too, and facts about everyone else do not.",
    ])


def tool_export(args: dict) -> str:
    """The portable substrate. Leaving is a feature."""
    fw = fabric()
    blob = fw.snapshot()
    dest = Path(args.get("path") or (STORE / "export.json"))
    dest.write_bytes(blob)
    n = len([x for x in fw._ctx.graph.all_nodes()
             if x.status.memory_state in ("active", "disputed")])
    return (f"exported {len(blob)} bytes -> {dest}\n"
            f"  {n} active claims, open format (open_format/SPEC.md), readable with a stdlib-only "
            f"reference reader. It outlives this server and any model.")


def tool_stats(args: dict) -> str:
    from fireweed.read_gate import rescue_available
    fw = fabric()
    g = fw._ctx.graph
    active = [n for n in g.all_nodes() if n.status.memory_state in ("active", "disputed")]
    sup = sum(1 for n in g.all_nodes() if n.status.memory_state == "superseded")
    _mode = "on" if rescue_available() else "OFF (encoder not installed — recall refuses more)"
    return "\n".join([
        f"claims (active)   : {len(active)}",
        f"claims (superseded): {sup}   — retained, never deleted; belief revision is part of the record",
        f"entities          : {len(g.all_entities())}",
        f"sources held      : {len(list(SOURCES.glob('*.txt'))) if SOURCES.exists() else 0}",
        f"store             : {STORE}",
        f"paraphrase matching: {_mode}",
    ])


TOOLS = [
    {"name": "remember", "description":
        "Commit a fact to memory. The claim is admitted ONLY if the evidence you cite supports it — "
        "you are the proposer, a deterministic gate decides. Refusals are typed and explain what to "
        "fix. Evidence must be text you are quoting verbatim, not a paraphrase.",
     "inputSchema": {"type": "object", "required": ["claim", "evidence"], "properties": {
         "claim": {"type": "string", "description": "the fact to remember"},
         "evidence": {"type": "string", "description": "verbatim text supporting it"},
         "source_id": {"type": "string", "description": "source this came from (default: agent)"},
         "source_text": {"type": "string", "description":
             "the full document this evidence was quoted from. Pass it and the claim binds to a "
             "verifiable BYTE RANGE in it — the receipt. Optional, but this is the point."}}},
     "fn": tool_remember},
    {"name": "add_source", "description":
        "Register a source document so claims remembered against it bind to verifiable byte ranges.",
     "inputSchema": {"type": "object", "required": ["source_id", "text"], "properties": {
         "source_id": {"type": "string"}, "text": {"type": "string"}}},
     "fn": tool_add_source},
    {"name": "recall", "description":
        "Search memory. Returns grounded claims with the byte ranges they came from. If the "
        "substrate cannot answer, it ABSTAINS and says which term it could not ground — treat that "
        "as a real answer, not an empty result.",
     "inputSchema": {"type": "object", "required": ["query"], "properties": {
         "query": {"type": "string"}}},
     "fn": tool_recall},
    {"name": "verify_receipts", "description":
        "Re-hash every held source and re-slice every receipt. Tamper-evident: change one byte of a "
        "source and its receipts stop verifying.",
     "inputSchema": {"type": "object", "properties": {}}, "fn": tool_verify},
    {"name": "forget", "description":
        "Erase everything about a subject and issue a SIGNED CERTIFICATE: exact closure, a probe "
        "battery that must all abstain, and bystanders left intact. This is the artifact for a "
        "'delete me and prove it' request.",
     "inputSchema": {"type": "object", "required": ["subject"], "properties": {
         "subject": {"type": "string"}}},
     "fn": tool_forget},
    {"name": "export_memory", "description":
        "Export the whole substrate as a portable open-format blob. Readable without this server, "
        "without any model, with a stdlib-only reference reader.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
     "fn": tool_export},
    {"name": "memory_stats", "description": "Substrate size, entities, sources held, and mode.",
     "inputSchema": {"type": "object", "properties": {}}, "fn": tool_stats},
]


# ── JSON-RPC / MCP plumbing ───────────────────────────────────────────────────

def respond(msg_id, result=None, error=None):
    m = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        m["error"] = error
    else:
        m["result"] = result
    sys.stdout.write(json.dumps(m) + "\n")
    sys.stdout.flush()


def handle(msg: dict) -> None:
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        respond(mid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fireweed", "version": "1.0.0"},
        })
    elif method in ("notifications/initialized", "initialized"):
        return                                   # notification: no reply
    elif method == "tools/list":
        respond(mid, {"tools": [{k: t[k] for k in ("name", "description", "inputSchema")}
                                for t in TOOLS]})
    elif method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        tool = next((t for t in TOOLS if t["name"] == name), None)
        if tool is None:
            respond(mid, error={"code": -32601, "message": f"unknown tool {name!r}"})
            return
        try:
            text = tool["fn"](params.get("arguments") or {})
        except Exception:
            # Surface the failure to the agent rather than dying: a memory server that vanishes
            # mid-conversation is worse than one that reports it could not do something.
            text = "ERROR — " + traceback.format_exc(limit=3)
        respond(mid, {"content": [{"type": "text", "text": text}]})
    elif method == "ping":
        respond(mid, {})
    elif mid is not None:
        respond(mid, error={"code": -32601, "message": f"unknown method {method!r}"})


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            handle(msg)
        except Exception:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
