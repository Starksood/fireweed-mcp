"""FMP reference reader — the open half of the open-core boundary.

PUBLIC ARTIFACT. This file is written to be extracted verbatim into the public repository, so it
carries NO engine logic: no resolver, no firewall, no gate, no scoring, no consolidation. It reads a
Fireweed snapshot and hands back what it contains. Everything that DECIDES is private; everything
that DESCRIBES is here.

Standard library only. No third-party imports, ever — a reader that needs a package to be installed
is not a format guarantee, and the Multi-Centennial Heirloom claim ("readable by whatever compute
exists in 2126") is only as good as this file's dependency list.

    from reference_reader import read_snapshot
    fmp = read_snapshot(open("snapshot.json","rb").read())
    for claim in fmp.active_claims():
        print(claim.claim, claim.receipt)

Conformance: `python open_format/conformance.py <snapshot.json>` — the suite any independent
implementation must pass to call itself an FMP reader.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator

SUPPORTED_SNAPSHOT_VERSIONS = (2,)


class FMPError(ValueError):
    """The bytes are not a snapshot this reader can honestly claim to understand."""


@dataclass(frozen=True)
class Receipt:
    """A claim's binding to a byte range of a hashed source document."""
    doc_hash: str
    byte_start: int
    byte_end: int
    quote: str

    def verify(self, doc: bytes) -> bool:
        """Re-hash the source and re-slice the bytes. Tamper-evident by construction: change any
        byte of the document and either the hash or the slice stops matching."""
        if "sha256:" + hashlib.sha256(doc).hexdigest() != self.doc_hash:
            return False
        if not (0 <= self.byte_start <= self.byte_end <= len(doc)):
            return False
        return doc[self.byte_start:self.byte_end].decode("utf-8", "replace") == self.quote


@dataclass(frozen=True)
class Claim:
    node_id: str
    claim: str
    node_type: str
    memory_state: str
    domains: tuple[str, ...]
    entity_ids: tuple[str, ...]
    receipt: Receipt | None
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def is_active(self) -> bool:
        # `disputed` is ACTIVE: both sides of an unresolved contradiction stay readable. Only
        # `superseded` is hidden, and it is retained in the file rather than deleted, so a reader
        # can always reconstruct what was once believed.
        return self.memory_state in ("active", "disputed")


@dataclass(frozen=True)
class Entity:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...]


@dataclass
class FMP:
    """A parsed Fireweed snapshot. Descriptive only — this object decides nothing."""
    fireweed_version: str
    snapshot_version: int
    claims: list[Claim]
    entities: list[Entity]
    relations: list[dict]

    def active_claims(self) -> Iterator[Claim]:
        return (c for c in self.claims if c.is_active)

    def entity(self, entity_id: str) -> Entity | None:
        return next((e for e in self.entities if e.entity_id == entity_id), None)

    def receipts(self) -> Iterator[tuple[Claim, Receipt]]:
        return ((c, c.receipt) for c in self.claims if c.receipt is not None)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise FMPError(msg)


def _receipt_from(prov: dict[str, Any] | None) -> Receipt | None:
    if not isinstance(prov, dict):
        return None
    doc_hash = prov.get("doc_hash")
    start, end = prov.get("byte_start"), prov.get("byte_end")
    # `source_span` is the verbatim evidence the claim was admitted on. The alternatives are
    # accepted so a future writer may rename it without orphaning existing readers.
    quote = prov.get("source_span") or prov.get("evidence_span") or prov.get("quote")
    # A receipt exists only when the full coordinate is present. A partial coordinate is NOT a
    # weak receipt — it is no receipt, and reporting it as one would be exactly the fabrication the
    # format exists to make impossible.
    if doc_hash is None or start is None or end is None or quote is None:
        return None
    return Receipt(str(doc_hash), int(start), int(end), str(quote))


def read_snapshot(data: bytes) -> FMP:
    """Parse snapshot bytes into an FMP. Raises FMPError on anything it cannot honestly read."""
    try:
        raw = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise FMPError(f"not valid UTF-8 JSON: {exc}") from exc
    _require(isinstance(raw, dict), "top level must be an object")

    version = raw.get("snapshot_version")
    _require(version in SUPPORTED_SNAPSHOT_VERSIONS,
             f"unsupported snapshot_version {version!r}; this reader supports "
             f"{SUPPORTED_SNAPSHOT_VERSIONS}")
    for key in ("nodes", "entities", "relations"):
        _require(isinstance(raw.get(key), list), f"missing or non-list '{key}'")

    claims: list[Claim] = []
    for n in raw["nodes"]:
        _require(isinstance(n, dict), "each node must be an object")
        for key in ("node_id", "claim"):
            _require(key in n, f"node missing '{key}'")
        status = n.get("status") or {}
        claims.append(Claim(
            node_id=str(n["node_id"]),
            claim=str(n["claim"]),
            node_type=str(n.get("node_type", "fact")),
            memory_state=str(status.get("memory_state", "active")),
            domains=tuple(sorted(n.get("domains") or [])),
            entity_ids=tuple(e.get("entity_id") for e in (n.get("entities") or [])
                             if isinstance(e, dict) and e.get("entity_id")),
            receipt=_receipt_from(n.get("provenance")),
            raw=n,
        ))

    entities = [
        Entity(
            entity_id=str(e["entity_id"]),
            canonical_name=str(e.get("canonical_name", "")),
            entity_type=str(e.get("entity_type", "concept")),
            aliases=tuple(e.get("aliases") or []),
        )
        for e in raw["entities"] if isinstance(e, dict) and "entity_id" in e
    ]

    return FMP(
        fireweed_version=str(raw.get("fireweed_version", "unknown")),
        snapshot_version=int(version),
        claims=claims,
        entities=entities,
        relations=list(raw["relations"]),
    )
