"""Sprint 3 — document receipts: byte-range provenance binding.

A document is a deterministic coordinate space (its bytes). A claim derived from a document binds to
`(doc_hash, byte_start, byte_end)`; the receipt renders the exact quote + hash + range, verifiable by
anyone who re-hashes the source and re-slices those bytes. Pure and dependency-free — the engine's
provenance guarantee, made checkable.

Invariant: a receipt is minted only when the verbatim span is a real contiguous byte slice of the
hashed document. We never fabricate a coordinate — 0-fabrication extends to provenance.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


def hash_document(doc: bytes) -> str:
    """Stable content hash of a source document. `sha256:<hex>`."""
    return "sha256:" + hashlib.sha256(doc).hexdigest()


def locate_span(doc: bytes, quote: str) -> tuple[int, int] | None:
    """UTF-8 byte offsets (start, end) of `quote` within `doc`, or None if not a contiguous slice.

    Exact byte match first (deterministic first occurrence). Falls back to a whitespace-normalized
    search — LLM evidence spans sometimes collapse runs of whitespace — mapping the match back to the
    original document's byte offsets. Returns None rather than guessing when the quote isn't present.
    """
    if not quote:
        return None
    q = quote.encode("utf-8")
    idx = doc.find(q)
    if idx != -1:
        return (idx, idx + len(q))

    # Whitespace-tolerant fallback: build a regex from the quote's non-space tokens, allowing any
    # whitespace run between them, and search the original bytes so the returned offsets are exact.
    try:
        text = doc.decode("utf-8")
    except UnicodeDecodeError:
        return None
    tokens = quote.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    m = re.search(pattern, text)
    if m is None:
        return None
    # char offsets -> byte offsets (UTF-8)
    start = len(text[: m.start()].encode("utf-8"))
    end = len(text[: m.end()].encode("utf-8"))
    return (start, end)


@dataclass(frozen=True)
class Receipt:
    """A verifiable provenance receipt for a document-derived claim.

    `doc_hash` + byte range is the original binding and still verifies exactly as before. The Merkle
    fields are additive: they bind the same quote to a hash tree over the document's parts, so the
    document can be REDACTED (an erased subject's sentence replaced by its leaf hash) without
    invalidating this receipt. Under the flat hash that was impossible -- any edit changed the hash
    and broke every receipt into the file, which is why erasure had to leave the text in place.
    """
    quote: str
    doc_hash: str
    byte_start: int
    byte_end: int
    merkle_root: str = ""
    leaf_index: int = -1
    proof: tuple = ()

    @property
    def redaction_safe(self) -> bool:
        """True when this receipt survives lawful redaction of another party's text."""
        return bool(self.merkle_root) and self.leaf_index >= 0

    def as_dict(self) -> dict:
        d = {
            "quote": self.quote,
            "doc_hash": self.doc_hash,
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
        }
        if self.redaction_safe:
            d["merkle_root"] = self.merkle_root
            d["leaf_index"] = self.leaf_index
            d["proof"] = [list(step) for step in self.proof]
        return d


def verify_redactable(receipt: Receipt, doc) -> bool:
    """Verify against a possibly-redacted document.

    `doc` may be raw bytes/str, or a RedactableDoc in which some parts have been reduced to their
    leaf hash. The quote must still be a surviving part, and its inclusion proof must reproduce the
    root. Redacting SOMEONE ELSE'S part leaves this untouched, which is the entire point.
    """
    from .merkle import RedactableDoc, leaf_hash, verify_inclusion
    if not receipt.redaction_safe:
        return False
    if isinstance(doc, (bytes, bytearray)):
        doc = RedactableDoc.from_text(doc.decode("utf-8"))
    elif isinstance(doc, str):
        doc = RedactableDoc.from_text(doc)
    if doc.root() != receipt.merkle_root:
        return False
    entries = doc.entries
    if not (0 <= receipt.leaf_index < len(entries)):
        return False
    entry = entries[receipt.leaf_index]
    if "text" not in entry:
        return False                      # our own part was redacted; nothing left to attest
    if entry["text"] != receipt.quote:
        return False
    # The leaf's own nonce must go into the hash, or a nonced document never verifies. Missing this
    # call site when the nonce parameter was added made every receipt fail against a redacted source
    # while the root still matched -- a confusing failure, because the document was provably intact.
    return verify_inclusion(leaf_hash(entry["text"], entry.get("nonce", "")),
                            tuple(map(tuple, receipt.proof)), receipt.merkle_root)


def verify(receipt: Receipt, doc: bytes) -> bool:
    """True iff `doc` hashes to the receipt's hash AND its `[start:end]` slice equals the quote.

    Any edit to the source flips the hash or the slice → False. Tamper-evident by construction.
    """
    if hash_document(doc) != receipt.doc_hash:
        return False
    if not (0 <= receipt.byte_start <= receipt.byte_end <= len(doc)):
        return False
    try:
        return doc[receipt.byte_start:receipt.byte_end].decode("utf-8") == receipt.quote
    except UnicodeDecodeError:
        return False


# ── engine binding (kept out of the fabric.py facade) ───────────────────────────

def bind_document(graph, text: str, source_id: str) -> None:
    """Treat `text` as a deterministic coordinate space: hash it once, then bind every node from
    `source_id` whose verbatim evidence span is a real contiguous slice of the doc to
    (doc_hash, byte_start, byte_end). Nodes whose span can't be located are left unbound — no
    fabricated coordinate. Mutates provenance in place; additive, so nothing else changes.
    """
    doc = text.encode("utf-8")
    doc_hash = hash_document(doc)
    prefix = f"{source_id}_"
    for node in graph.all_nodes():
        prov = node.provenance
        if prov.doc_hash is not None or not prov.source_turn_id.startswith(prefix):
            continue
        span = locate_span(doc, prov.source_span)
        if span is None:
            continue
        prov.doc_hash, prov.byte_start, prov.byte_end = doc_hash, span[0], span[1]


def receipt_for(node) -> Receipt | None:
    """Render a verifiable receipt from a node's provenance, or None if it isn't document-bound."""
    prov = node.provenance
    if prov.doc_hash is None or prov.byte_start is None or prov.byte_end is None:
        return None
    return Receipt(quote=prov.source_span, doc_hash=prov.doc_hash,
                   byte_start=prov.byte_start, byte_end=prov.byte_end)
