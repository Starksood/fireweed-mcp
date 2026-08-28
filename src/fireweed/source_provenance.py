"""Provenance for the EVIDENCE itself — a different trust boundary from the receipt.

The distinction, because conflating them is the whole risk
---------------------------------------------------------
A receipt binds a claim to bytes: `claim X occupies doc[i:j] of a document hashing to H`, and
`verify_receipts` re-hashes and re-slices to prove those bytes have not moved since. That is
**immutability since write**, and it is real.

It says nothing about whether H was *legitimate at* write time. The write path enforces
"the claim does not say more than the quote does" — entailment. It has never enforced, or even
recorded, "the quote came from somewhere real" — provenance. For a single-user local agent the two
collapse, because the evidence IS the conversation. They come apart the moment evidence is a log
file, an API response, or anything a third party handed over.

Before this module, a source document arrived through `add_source`, was written to disk, and left
**no trace in the append-only ledger at all** — there was no `ADD_SOURCE` event kind. The chain
recorded every claim binding and never recorded the document's arrival, so "audit backwards from a
stored memory to the evidence's arrival" had no record to land on.

What a SourceRecord does and does not attest
--------------------------------------------
`doc_hash` and `byte_length` are COMPUTED from the bytes. Everything else — `origin`, `origin_kind`,
`supplied_by`, `validated_by` — is **declared by the caller and never verified.** The record proves
that a caller *asserted* this provenance at this point in the chain. It does not prove the assertion
is true.

That gap is deliberate and disclosed rather than papered over, because a provenance field that looks
verified and is not would be worse than no field: it is the same failure as a signed certificate
that only its own signer can check. `disclosure()` prints the boundary, and the MCP tool prints it
on every registration.

And one boundary this module CANNOT close from inside the system: `ingested_at` is the ingest clock,
supplied by the caller, and the ledger's signing key sits in the same custody as the data. So the
chain proves ORDER relative to itself, never position in real time — an operator who holds both can
backdate the whole thing coherently. Proving "at write time" to someone who does not already trust
the operator needs an external anchor (a transparency log, or an RFC 3161 timestamp). See
`docs/FINDING_certificates_prove_nothing_to_an_adversary.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

# What the caller says the bytes are. A closed list, so the field is reviewable rather than free
# text — and `unknown` is the honest default, not an error.
ORIGIN_KINDS = ("conversation", "file", "url", "api", "user_submitted", "unknown")

# Fields the system COMPUTES from the bytes it was handed.
ATTESTED_FIELDS = ("source_id", "doc_hash", "byte_length")
# Fields the CALLER declares and nothing checks.
DECLARED_FIELDS = ("origin", "origin_kind", "supplied_by", "validated_by", "ingested_at")


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    doc_hash: str            # sha256 of the exact bytes registered — computed
    byte_length: int         # computed
    ingested_at: str         # the ingest clock, caller-supplied — DECLARED
    origin: str              # DECLARED
    origin_kind: str         # DECLARED, one of ORIGIN_KINDS
    supplied_by: str         # DECLARED
    validated_by: str        # DECLARED — what checked these bytes before ingest, if anything

    def to_payload(self) -> dict:
        return asdict(self)

    def disclosure(self) -> str:
        """The sentence that has to appear wherever this record is shown."""
        return (
            "attested (computed from the bytes): "
            + ", ".join(ATTESTED_FIELDS)
            + "\ndeclared (caller-supplied, NOT verified): "
            + ", ".join(DECLARED_FIELDS)
            + "\nThis record proves a caller ASSERTED this provenance at this point in the chain. "
              "It does not prove the assertion is true, and the ingest clock is self-attested — "
              "ordering is provable, real-world timing is not."
        )


def doc_hash(text: str) -> str:
    """`sha256:<hex>` — the SAME format `receipts.doc_hash` produces, deliberately.

    The first version returned bare hex. Every join between a claim's receipt and its source's
    arrival event then failed silently: the trace tool reported "source missing or changed" and
    "arrival not in the ledger" for a document sitting in the store with its event in the chain.
    Two hash formats for one concept is a joinable-key bug, and the failure looked exactly like the
    tamper detection working.
    """
    from .receipts import hash_document as _canonical
    return _canonical(text.encode("utf-8"))


def normalize_hash(h: str) -> str:
    """Accept either format on the way in, so an older record still joins."""
    if not isinstance(h, str):
        return ""
    return h if h.startswith("sha256:") else ("sha256:" + h if h else "")


def make_record(source_id: str, text: str, ingested_at: str, origin: str = "",
                origin_kind: str = "unknown", supplied_by: str = "",
                validated_by: str = "") -> SourceRecord:
    """Build a record. Unknown declared fields become the literal string "undeclared" rather than
    an empty one, so an audit can tell "nobody said" apart from "said nothing"."""
    kind = origin_kind if origin_kind in ORIGIN_KINDS else "unknown"
    return SourceRecord(
        source_id=source_id,
        doc_hash=doc_hash(text),
        byte_length=len(text.encode("utf-8")),
        ingested_at=ingested_at,
        origin=origin or "undeclared",
        origin_kind=kind,
        supplied_by=supplied_by or "undeclared",
        validated_by=validated_by or "undeclared",
    )
