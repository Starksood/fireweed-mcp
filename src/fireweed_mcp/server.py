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
import re
import sys
import traceback
from pathlib import Path

# Installed as a package: `fireweed` is a sibling module, no path surgery needed.
ROOT = Path(__file__).resolve().parent

PROTOCOL_VERSION = "2024-11-05"


def signer():
    """The signer for erasure certificates: Ed25519 when available, HMAC otherwise.

    Ed25519 is what makes a certificate mean anything to someone who does not already trust this
    server: the public key is published beside the store, and a third party holding it can verify a
    certificate without being able to produce one. HMAC cannot do that -- verification needs the same
    secret as signing -- so under HMAC the certificate is a tamper-detection checksum and says so.

    `cryptography` is an optional extra (`pip install fireweed-mcp[signing]`), so the default install
    stays dependency-free and gets the honest weaker guarantee rather than a hand-rolled strong one.
    """
    from fireweed.signing import Ed25519Signer, HmacSigner, ed25519_available
    if ed25519_available():
        kp = KEYS / "ed25519_key"
        if kp.exists():
            return Ed25519Signer(kp.read_bytes())
        KEYS.mkdir(parents=True, exist_ok=True)
        raw = Ed25519Signer.generate()
        import os as _os
        try:
            fd = _os.open(str(kp), _os.O_WRONLY | _os.O_CREAT | _os.O_EXCL, 0o600)
        except FileExistsError:
            return Ed25519Signer(kp.read_bytes())
        try:
            _os.write(fd, raw)
        finally:
            _os.close(fd)
        sg = Ed25519Signer(raw)
        # The public half is written in the clear ON PURPOSE: it is the artifact a verifier needs,
        # and publishing it is what separates this from the symmetric scheme.
        STORE.mkdir(parents=True, exist_ok=True)
        (STORE / "ed25519_public_key").write_text(sg.public_key() + "\n")
        return sg
    return HmacSigner(signing_key())


def signing_key() -> bytes:
    """The per-install key that signs erasure certificates.

    This was the literal b"fireweed-mcp-key" -- a constant in public source. HMAC-SHA256 over a
    canonical encoding is the right construction and `verify_signature` is constant-time, but a key
    every reader of the repository already has is not a secret: anyone could mint a validly-signed
    certificate for any subject and any closure manifest. Against accidental corruption that is a
    checksum; against the adversary the README invokes ("someone who trusts neither your agent nor
    this server") it establishes nothing.

    Generated once per install, 32 bytes from `secrets`, stored beside the substrate with owner-only
    permissions and never in the source tree. Certificates issued before this change do not verify
    against the new key -- correct, since they were forgeable by construction.
    """
    import os as _os
    import secrets
    kp = KEYS / "signing_key"
    if kp.exists():
        return kp.read_bytes()
    KEYS.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    # O_EXCL so two servers racing on a fresh store cannot each mint a key and clobber the other --
    # the loser would then sign certificates that do not verify against the key left on disk. First
    # writer wins; everyone else reads what it wrote.
    try:
        fd = _os.open(str(kp), _os.O_WRONLY | _os.O_CREAT | _os.O_EXCL, 0o600)
    except FileExistsError:
        return kp.read_bytes()
    try:
        _os.write(fd, key)
    finally:
        _os.close(fd)
    return key


def safe_source_id(raw: str) -> str:
    """`source_id` becomes a filename, so it is attacker-controlled path input.

    "../../etc/passwd" raised an unhandled traceback at the caller (it did not escape the store --
    the write simply failed -- but a stack trace is not an answer). Reduce to a single path
    component and keep only characters that are safe on POSIX and Windows alike.
    """
    import re as _re
    raw = (raw or "").strip().replace("\\", "/").split("/")[-1]
    raw = _re.sub(r'[^A-Za-z0-9._-]', "_", raw).lstrip(".")
    return raw[:120] or "agent"
STORE = Path(os.environ.get("FIREWEED_MCP_STORE", Path.home() / ".fireweed" / "mcp"))
SUBSTRATE = STORE / "substrate.json"
SOURCES = STORE / "sources"
LEDGER_DB = STORE / "ledger.db"

# KEYS LIVE APART FROM THE DATA THEY PROTECT.
#
# They used to sit in the store directory next to substrate.json and ledger.db, which quietly voided
# both guarantees they exist to provide: anyone who could read the data could read the signing key
# and mint a certificate for any subject, and any backup or `cp -r` of the store taken before an
# erasure carried the content keys alongside the ciphertext they were meant to shred.
#
# A separate directory does not defeat an attacker who already has the whole home directory, and it
# is not a substitute for a KMS or an OS keychain -- crypto.py's own docstring says as much. What it
# does fix is the ordinary case: copying, syncing, archiving or sharing the STORE no longer hands
# over the keys with it. Override with FIREWEED_MCP_KEYS.
KEYS = Path(os.environ.get("FIREWEED_MCP_KEYS", Path.home() / ".fireweed" / "keys"))
KEYRING = KEYS / "keyring.json"


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

        # ATTACH THE LEDGER. ledger.py implements a complete append-only hash-chained event log --
        # gap-free seq, prev_hash chain, canonical serialization, resolver_version stamped into
        # every payload -- and graph.py has attach_ledger, seal(), and an _emit guard that raises on
        # an unlogged mutation. None of it had a caller: `attach_ledger` and `seal()` were dead code,
        # `_sealed` was never True, so _emit took the silent early return and EVERY mutation this
        # server ever made recorded nothing. Found by an independent audit; see
        # docs/FINDING_ledger_unwired_no_tombstone.md.
        #
        # The keyring goes on here too, because attach_ledger takes it: with a keyring, node CONTENT
        # is encrypted in the persisted payload, so erasure destroying the key makes the history
        # unrecoverable rather than merely unreachable.
        from fireweed.crypto import Keyring
        from fireweed.ledger_sqlite import SQLiteLedger
        _keyring = Keyring.deserialize(KEYRING.read_bytes() if KEYRING.exists() else None)
        _fw._ctx.graph.attach_ledger(SQLiteLedger(str(LEDGER_DB)), tenant_id="mcp",
                                     keyring=_keyring)
        # SEAL. After this, a graph mutation with no ledger attached raises instead of silently
        # passing. The guard existed and was never armed -- `_sealed` was never True anywhere in the
        # repository -- which is precisely why the unattached ledger went unnoticed for the whole
        # life of the project. Arming it means the failure can never recur silently: it becomes a
        # crash on the write, not an absence discovered by an auditor months later.
        _fw._ctx.graph.seal()

        if SUBSTRATE.exists():
            # A truncated or hand-edited substrate must not take the server down on every call.
            # Quarantine it, say so once, and continue with an empty store -- losing the file is
            # already bad, and compounding it with an unreadable stack trace helps nobody.
            try:
                _fw.restore(SUBSTRATE.read_bytes())
            except Exception as exc:
                bad = SUBSTRATE.with_suffix(".corrupt")
                try:
                    SUBSTRATE.replace(bad)
                except OSError:
                    bad = None
                print(f"fireweed: could not read {SUBSTRATE} ({type(exc).__name__}: {exc}); "
                      + (f"moved to {bad}; " if bad else "")
                      + "starting from an empty store.", file=sys.stderr, flush=True)
    return _fw


def persist():
    SUBSTRATE.write_bytes(fabric().snapshot())
    kr = getattr(fabric()._ctx.graph, "_keyring", None)
    if kr is not None:
        KEYS.mkdir(parents=True, exist_ok=True)
        # The mutable half of crypto-shredding. Erasure deletes a key from here; if this is not
        # written back, the deletion does not survive the process and the shred is undone on restart.
        KEYRING.write_bytes(kr.serialize())


# ── tools ─────────────────────────────────────────────────────────────────────

def tool_remember(args: dict) -> str:
    """The gate. The agent proposes; this decides."""
    from fireweed.grounding import classify, predicate_grounded, subject_grounded, order_preserved, numerals_grounded
    from fireweed.receipts import bind_document, receipt_for

    claim = (args.get("claim") or "").strip()
    evidence = (args.get("evidence") or "").strip()
    source_id = safe_source_id(args.get("source_id") or "agent")
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

    # TOMBSTONE. Erasure used to leave no durable trace a later write could consult: erase a
    # subject, re-propose the identical claim with identical evidence, and it was silently admitted
    # again as if nothing had happened. Predicted by an independent audit and confirmed live before
    # the ledger was attached. Now that ERASE events are durable, the write path can ask.
    #
    # Deliberately an OVERRIDE, not a hard block: erasure is not a permanent ban on a person ever
    # being mentioned again -- someone may lawfully re-consent, or the same name may be a different
    # person. The requirement is that re-admission be a DECISION SOMEONE MAKES, recorded as such,
    # rather than something that happens quietly because nothing was looking.
    if not args.get("acknowledge_erasure"):
        from fireweed.erasure import name_fingerprint
        erased = _erased_fingerprints(ctx.graph)
        for name in (_candidate_names(claim) if erased else []):
            if name_fingerprint(name) in erased:
                return (f"NOT STORED (previously_erased) — \"{name}\" was erased from this "
                        f"substrate, and this claim names them again.\n  claim   : {claim}\n"
                        f"If this is intentional -- re-consent, or a different person with the same "
                        f"name -- pass acknowledge_erasure=true to admit it. That choice is recorded "
                        f"in the ledger.")

    turn = f"{source_id}_{len(ctx.graph.all_nodes())}"
    if source_id not in getattr(ctx, "_sessions_seen", {}):
        try:
            ctx.begin_session(source_id)
        except Exception:
            pass
    # NARROW THE STORED SPAN TO THE PART THAT SUPPORTS THE CLAIM.
    #
    # A caller may quote a whole document as evidence, and the stored provenance span was whatever
    # they passed. That is how one subject's sentence ended up inside a DIFFERENT subject's stored
    # record: erase Priya, and her text survived in the span attached to Marcus's claim, because his
    # evidence blob contained her sentence.
    #
    # The checks above have already run against the FULL evidence, so nothing is weakened. This only
    # decides what is retained, and it re-verifies that the narrowed span still supports the claim
    # before using it -- if it does not, the full evidence is kept and correctness wins over tidiness.
    stored_evidence = evidence
    if len(evidence) > len(claim):
        from fireweed.merkle import split_parts
        parts = split_parts(evidence)
        if len(parts) > 1:
            ctoks = {t for t in re.split(r"[^a-z0-9]+", claim.lower()) if len(t) > 2}
            best, score = None, 0.0
            for part in parts:
                ptoks = {t for t in re.split(r"[^a-z0-9]+", part.lower()) if len(t) > 2}
                if not ptoks:
                    continue
                v = len(ctoks & ptoks) / len(ctoks | ptoks)
                if v > score:
                    best, score = part, v
            if best and score >= 0.4 and subject_grounded(claim, best) \
                    and order_preserved(claim, best) and numerals_grounded(claim, best) \
                    and predicate_grounded(claim, best):
                stored_evidence = best

    result = ctx.ingest(claim, stored_evidence, 0.9, turn, source_id)

    # REPORT WHAT ACTUALLY HAPPENED. This returned "ADMITTED" unconditionally while the engine's
    # firewall was rejecting the claim, so `remember` told callers their fact was stored when the
    # substrate had recorded nothing -- silent data loss, in the one product where the record is
    # the entire promise. Computing a decision and then ignoring it is the same defect this project
    # has now hit in reader.py, run_opsgraph.py and here.
    decision = str(getattr(result, "firewall_decision", "") or "").upper()
    if "REJECT" in decision:
        reason = getattr(getattr(result, "mutation", None), "reason", None) or "rejected"
        persist()
        return (f"NOT STORED ({reason}) — the claim passed the evidence checks but the substrate's "
                f"firewall declined it.\n  claim   : {claim}\n"
                f"Nothing was written. Rephrase as a complete statement about a named subject, "
                f"or use `add_source` for material that is not a single durable fact.")

    # Bind a receipt when we hold the source document this evidence came from.
    src = SOURCES / f"{source_id}.txt"
    if src.exists():
        bind_document(ctx.graph, src.read_text(), source_id)
    node = next((n for n in ctx.graph.all_nodes() if n.claim == claim), None)
    persist()

    # OCCURRENCE, NOT JUST CONTENT. `verify_receipts` re-hashes a source and re-slices a range,
    # which proves stored content has not drifted -- it structurally cannot catch a write that never
    # landed, because a dropped fact has no receipt to verify. This read-back is the occurrence
    # proof, and it was already being computed here (to build the receipt) and its `None` case
    # discarded: the code printed ADMITTED plus a "no source document registered" line, which is
    # actively misleading, since it implies the fact was stored but unreceipted when in fact nothing
    # was stored at all. Reported by review; see docs/FINDING_write_omission_blind_spot.md.
    if node is None:
        return (f"NOT STORED (omission) — the claim passed every evidence check and the firewall, "
                f"but no matching node is present in the substrate on read-back.\n"
                f"  claim   : {claim}\n"
                f"Nothing was written. This is a bug in the write path, not a rejection of your "
                f"claim -- please report it with this claim and evidence.")

    cls = classify(claim, evidence)
    rec = receipt_for(node)
    out = [f"ADMITTED — {claim}", f"  grounding : {cls}"]
    if rec:
        out.append(f"  receipt   : bytes [{rec.byte_start}:{rec.byte_end}] of {rec.doc_hash[:19]}…")
    else:
        out.append("  receipt   : turn-bound (no source document registered for this source_id — "
                   "use `add_source` first to get byte-range receipts)")
    return "\n".join(out)


def _erased_fingerprints(graph) -> set:
    """Hashes of subjects with an ERASE event. Empty when no ledger is attached.

    Reads the durable record rather than the live graph, which is the point: the live graph is
    exactly where an erased subject is NOT, so it cannot answer "was this erased?". Hashes rather
    than names, because the ledger is append-only and a name written into it would survive the very
    erasure it records.
    """
    led = getattr(graph, "_ledger", None)
    if led is None:
        return set()
    out = set()
    try:
        for ev in led.events(getattr(graph, "_ledger_tenant", "local")):
            if ev.kind == "ERASE":
                fp = (ev.payload or {}).get("subject_name_hash")
                if fp:
                    out.add(fp)
    except Exception:
        return set()
    return out


def _candidate_names(claim: str) -> list:
    """Capitalised runs in a claim — the things that might name a previously erased subject."""
    return [m.group(0) for m in re.finditer(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", claim or "")]


def tool_add_source(args: dict) -> str:
    """Register a source document so claims from it can carry byte-range receipts."""
    import hashlib
    source_id = safe_source_id(args.get("source_id") or "")
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
        out = [f"ABSTAINED ({r.abstain_reason}) — {detail}",
               "A refusal, not an empty result: the substrate is telling you it has nothing "
               "rather than guessing."]
        # A refusal with no way forward is where an agent caller stops. Give it the next move.
        remedy = getattr(g, "remedy", "") if g else ""
        if remedy:
            out.append(f"Next: {remedy}.")
        if (g is not None and getattr(g, "mode", "semantic") == "lexical_only"
                and "encoder is not installed" not in detail):
            out.append("Note: running without the semantic encoder, so paraphrases of a grounded "
                       "term are refused. `pip install fireweed-mcp[semantic]` widens recall.")
        return "\n".join(out)
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


def load_source(source_id: str):
    """The source document, as a RedactableDoc, whether or not it has been redacted.

    A source lives as `<id>.txt` until an erasure redacts it, at which point it becomes
    `<id>.redacted.json` -- a list of parts where the erased subject's sentences are present only as
    their leaf hash. Both forms produce the SAME Merkle root, which is what lets a bystander's
    receipt keep verifying across the change.
    """
    from fireweed.merkle import RedactableDoc
    red = SOURCES / f"{source_id}.redacted.json"
    if red.exists():
        return RedactableDoc.from_dict(json.loads(red.read_text())), True
    plain = SOURCES / f"{source_id}.txt"
    if plain.exists():
        return RedactableDoc.from_text(plain.read_text()), False
    return None, False


def merkle_receipt_for(node):
    """A redaction-safe receipt, computed on demand from the source rather than persisted.

    Deliberately not stored on the node: the proof is derivable from the document at any time, and
    adding fields to Provenance would change every existing snapshot's schema for no gain.
    """
    from fireweed.merkle import inclusion_proof, split_parts
    from fireweed.receipts import Receipt
    prov = node.provenance
    source_id = prov.source_turn_id.rsplit("_", 1)[0]
    doc, _ = load_source(source_id)
    if doc is None:
        return None
    # Bind the PART THAT SUPPORTS THE CLAIM, not the whole evidence blob a caller happened to pass.
    # `source_span` is often the entire document; a receipt over all of it necessarily breaks when
    # any other subject in that document is erased, which would defeat the point. The claim's own
    # supporting sentence is the right unit, and it survives an unrelated redaction.
    claim_tokens = {t for t in re.split(r"[^a-z0-9]+", node.claim.lower()) if len(t) > 2}
    best, best_score = None, 0.0
    for i, e in enumerate(doc.entries):
        if "text" not in e:
            continue
        toks = {t for t in re.split(r"[^a-z0-9]+", e["text"].lower()) if len(t) > 2}
        if not toks:
            continue
        score = len(claim_tokens & toks) / len(claim_tokens | toks)
        if score > best_score:
            best, best_score = i, score
    if best is None or best_score < 0.4:
        return None            # no part clearly supports this claim: mint nothing rather than guess
    e = doc.entries[best]
    return Receipt(quote=e["text"], doc_hash=prov.doc_hash or "", byte_start=0, byte_end=0,
                   merkle_root=doc.root(), leaf_index=best,
                   proof=tuple(inclusion_proof(doc.hashes(), best)))


def tool_verify(args: dict) -> str:
    """Re-hash the source and re-slice the bytes. The point of a receipt is that it CAN fail."""
    from fireweed.receipts import receipt_for, verify as verify_receipt
    fw = fabric()
    checked = ok = unheld = redacted_ok = 0
    failures = []
    for n in fw._ctx.graph.all_nodes():
        rec = receipt_for(n)
        if rec is None:
            continue
        checked += 1
        source_id = n.provenance.source_turn_id.rsplit("_", 1)[0]
        doc, was_redacted = load_source(source_id)
        if doc is None:
            unheld += 1
            continue
        if was_redacted:
            # The flat byte-range check cannot survive a redaction by construction; the Merkle proof
            # can, and that is the whole reason it exists.
            from fireweed.receipts import verify_redactable
            mrec = merkle_receipt_for(n)
            if mrec is not None and verify_redactable(mrec, doc):
                ok += 1
                redacted_ok += 1
            else:
                failures.append(n.claim)
        elif verify_receipt(rec, (SOURCES / f"{source_id}.txt").read_bytes()):
            ok += 1
        else:
            failures.append(n.claim)
    out = [f"receipts re-verified: {ok}/{checked - unheld} checkable ({unheld} whose source is not held)"]
    if redacted_ok:
        out.append(f"  {redacted_ok} verified against a REDACTED source — the erased subject's text "
                   f"is gone and these receipts still hold.")
    for f in failures:
        out.append(f"  FAILED — {f}")
    if failures:
        out.append("A failure means the source document changed after the claim was bound to it.")
    return "\n".join(out)


def tool_forget(args: dict) -> str:
    """Erasure with exact closure and a signed certificate."""
    from fireweed.erasure import erase, ErasureIncomplete
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
        # USE THE REAL LEDGER, NOT A THROWAWAY. This passed SQLiteLedger(":memory:") -- so the ERASE
        # event, the append-only hash-chained record that makes a from-zero fold reconstruct and then
        # erase, was written to a database discarded when the call returned. After a forget the store
        # held the closure's absence and no record that an erasure had ever happened. Passing the
        # attached keyring is what turns this from a removal into a crypto-shred: the subject's
        # content key is destroyed, so historical ciphertext is unrecoverable.
        cert = erase(g, g._ledger, "mcp", ent.entity_id, probes,
                     lambda gg, q: retrieval.query_graph(q, gg), signer(),
                     keyring=g._keyring)
    except ErasureIncomplete as e:
        return f"NO CERTIFICATE ISSUED — erasure incomplete: {e}\nNothing was removed."
    # CLEAR A DANGLING SESSION ANCHOR. The anchor is the entity pronouns resolve against ("she" ->
    # the first person seen). After erasing that person it still held `ent_priya_raman`, which both
    # leaks the name through the identifier -- an id derived from a name is still personal data --
    # and points later pronoun resolution at an entity that no longer exists. It lives on the ingest
    # context rather than in graph_state_dict, so clearing it does not affect the live-vs-replay
    # fingerprint.
    erased_ids = set(cert.closure_manifest.get("entity_ids") or []) | {ent.entity_id}
    if getattr(ctx_of := fw._ctx, "_session_anchor", None) in erased_ids:
        ctx_of._session_anchor = None

    # REDACT THE SOURCE DOCUMENTS. `forget` operated on the graph and never touched SOURCES/*.txt,
    # so the erased subject's sentences survived in plaintext in a file beside the substrate -- one
    # grep recovered them after a "provable erasure".
    #
    # An earlier attempt at this was reverted because overwriting the bytes changed the document
    # hash and broke every OTHER party's receipt into the same file. The Merkle binding removes that
    # obstacle: a redacted part keeps its leaf hash, the root is unchanged, and a bystander's
    # inclusion proof still verifies. Their claim survives, their receipt survives, and the erased
    # text is gone -- all three, which the flat hash could not do.
    redacted_files = 0
    redacted_parts = 0
    if SOURCES.exists():
        from fireweed.merkle import RedactableDoc
        needle = ent.canonical_name.lower()
        for f in sorted(SOURCES.glob("*.txt")):
            source_id = f.stem
            doc, _ = load_source(source_id)
            if doc is None:
                continue
            hits = sum(1 for e in doc.entries if "text" in e and needle in e["text"].lower())
            if not hits:
                continue
            before_root = doc.root()
            new_doc = doc.redact(lambda t: needle in t.lower())
            assert new_doc.root() == before_root, "redaction must not move the root"
            (SOURCES / f"{source_id}.redacted.json").write_text(json.dumps(new_doc.as_dict()))
            f.unlink()          # the plaintext original is the leak; it does not survive
            redacted_files += 1
            redacted_parts += hits
        # A source already redacted by an earlier erasure may still name this subject.
        for f in sorted(SOURCES.glob("*.redacted.json")):
            doc = RedactableDoc.from_dict(json.loads(f.read_text()))
            if not any("text" in e and needle in e["text"].lower() for e in doc.entries):
                continue
            hits = sum(1 for e in doc.entries if "text" in e and needle in e["text"].lower())
            f.write_text(json.dumps(doc.redact(lambda t: needle in t.lower()).as_dict()))
            redacted_files += 1
            redacted_parts += hits

    persist()
    survivors = [n.claim for n in g.all_nodes() if n.status.memory_state in ("active", "disputed")]
    # PRINT THE CAVEAT THE CERTIFICATE ALREADY COMPUTES. `erase()` builds a `scope` string stating
    # exactly what is and is not certified, and with no keyring it says residual plaintext may
    # persist. This function used to print the signature, the counts and the state hashes -- the
    # flattering fields -- drop `scope`, `cipher` and `key_destroyed`, and then assert "this
    # certificate is the artifact a compliance reviewer asks for". The engine wrote the sentence
    # that prevents the overstatement and the server declined to display it. Same defect as the
    # ADMITTED-while-rejected bug, in the one place where it changes what a reader believes about a
    # legal artifact.
    return "\n".join([
        f"ERASED {ent.canonical_name} — certificate issued",
        f"  signature            : {cert.signature}",
        f"  nodes in closure     : {len(cert.closure_manifest['node_ids'])}",
        f"  derived invalidated  : {len(cert.closure_manifest['derived_invalidated'])}",
        f"  every probe abstains : {cert.battery_all_abstained}",
        f"  state {cert.state_hash_before[:18]}… -> {cert.state_hash_after[:18]}…",
        f"  bystanders surviving : {len(survivors)}",
        f"  cipher               : {cert.cipher}",
        f"  content key destroyed: {cert.key_destroyed}",
        f"  source parts redacted: {redacted_parts} in {redacted_files} document(s)",
        (f"  verifiable by others : yes — ed25519, public key {cert.public_key[:16]}…"
         if cert.adversary_checkable else
         "  verifiable by others : NO — this signature is symmetric (HMAC), so only this server "
         "can check it. It detects tampering; it does not prove anything to someone who does not "
         "already trust this server. `pip install fireweed-mcp[signing]` for ed25519."),
        "",
        f"SCOPE — what this certificate does and does not certify:\n  {cert.scope}",
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


# Verdicts meaning THE REQUESTED OPERATION DID NOT HAPPEN. Signalled structurally via `isError`,
# not only as prose the caller may or may not read.
_FAILED_PREFIXES = ("REFUSED", "NOT STORED", "ERROR", "NO CERTIFICATE ISSUED")


def _is_error(text: str) -> bool:
    """True when the call did not do what was asked.

    Review raised this directly: "I'd want the refuse in the handler, not only in the tool the model
    sees." Previously every verdict -- admitted, refused, and an unhandled traceback alike -- came
    back in an identical success envelope, so enforcement depended entirely on the calling model
    reading the prose and choosing to honour it. A harness that summarises tool output before a
    human sees it could soften or drop a REFUSED with nothing at the protocol level to stop it.

    ABSTAINED is deliberately NOT an error. A grounded refusal to answer is a CORRECT and complete
    result -- the whole thesis of this system -- and flagging it as a failure would invite exactly
    the retry-until-a-row-comes-back behaviour the abstention exists to prevent. `recall`'s tool
    description already tells callers to treat abstention as a real answer; marking it isError
    would contradict that.
    """
    return text.lstrip().startswith(_FAILED_PREFIXES)


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
        respond(mid, {"content": [{"type": "text", "text": text}], "isError": _is_error(text)})
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
