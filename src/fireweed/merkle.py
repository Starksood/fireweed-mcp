"""Redactable provenance: a Merkle tree over a document's parts.

Why
---
A receipt binds `(doc_hash, byte_start, byte_end)` where `doc_hash` is a flat SHA-256 of the whole
document. That is what makes erasure and verification mutually exclusive: change one byte to remove
an erased subject's name, and the hash changes, and every OTHER party's receipt into that document
fails. Measured directly -- redacting a source took a bystander from `1/1 checkable` to
`0/1 checkable — FAILED`.

The standard answer is a redactable signature over a hash tree, and it predates this project by two
decades (Steinfeld/Bull/Zheng 2001; Johnson et al. 2002; see
docs/FINDING_erasure_leaks_via_bystander_spans.md for the references).

  1. Split the document into PARTS and hash each as a leaf.
  2. A receipt binds the ROOT, plus the part it quotes, plus an inclusion proof.
  3. To redact a part: drop its plaintext and keep its leaf HASH.
  4. A verifier recomputes the root from surviving plaintext plus the retained hashes. The root is
     unchanged, so every other receipt still verifies -- and the verifier never learns what was
     redacted.

All three horns of the trilemma are satisfied at once: the erased text is gone, the bystander's
claim survives, and the bystander's receipt still verifies.

Pure stdlib. A verifier needs a hash function and nothing else -- no engine, no key.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# Domain separation: a leaf and an interior node must never hash identically, or an attacker could
# present an interior digest as if it were a leaf (the classic second-preimage attack on naive
# Merkle trees).
_LEAF = b"\x00"
_NODE = b"\x01"

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def split_parts(text: str) -> list[str]:
    """Sentence-granular leaves.

    Granularity is the one real design choice here. Too coarse and a redaction takes a bystander's
    quote with it; too fine and proofs grow. Sentences match how claims are quoted in practice --
    `locate_span` binds a verbatim sentence -- so a bystander's quote usually lands inside one leaf.
    """
    parts = [p for p in _SENTENCE.split(text.strip()) if p.strip()]
    return parts or ([text] if text else [])


def new_nonce() -> str:
    """16 random bytes, hex. One per leaf, destroyed when that leaf is redacted."""
    import secrets
    return secrets.token_hex(16)


def leaf_hash(part: str, nonce: str = "") -> str:
    """H(0x00 || nonce || part).

    The nonce is what makes a redaction actually hide its content. Without it the retained leaf hash
    is H(0x00 || part), and a redacted sentence drawn from a small space -- which sentences about
    people are -- can simply be guessed and confirmed against the hash. The redaction would hide the
    text from a casual reader and not from anyone motivated.

    A surviving leaf keeps its nonce, because a verifier needs it to recompute the hash. A redacted
    leaf keeps only the hash: the nonce is destroyed with the plaintext, so there is nothing left to
    check a guess against.
    """
    return hashlib.sha256(_LEAF + nonce.encode("utf-8") + part.encode("utf-8")).hexdigest()


def _pair(a: str, b: str) -> str:
    return hashlib.sha256(_NODE + bytes.fromhex(a) + bytes.fromhex(b)).hexdigest()


def merkle_root(hashes: list[str]) -> str:
    """Root over leaf hashes. An odd node is promoted unchanged rather than duplicated."""
    if not hashes:
        return hashlib.sha256(_LEAF).hexdigest()
    level = list(hashes)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_pair(level[i], level[i + 1]))
        if len(level) % 2:
            nxt.append(level[-1])          # promote, do not duplicate
        level = nxt
    return level[0]


def inclusion_proof(hashes: list[str], index: int) -> list[tuple[str, str]]:
    """Sibling path for `index`, as (side, hash) where side is which side the SIBLING is on."""
    if not (0 <= index < len(hashes)):
        raise IndexError(index)
    proof: list[tuple[str, str]] = []
    level, i = list(hashes), index
    while len(level) > 1:
        # Build the next level EXACTLY as merkle_root does, promotion included. An earlier version
        # omitted the promoted node here, so the two functions walked different trees and no proof
        # over an odd-sized level ever verified.
        nxt = [_pair(level[j], level[j + 1]) for j in range(0, len(level) - 1, 2)]
        promoted = len(level) % 2 == 1
        if promoted:
            nxt.append(level[-1])
        if promoted and i == len(level) - 1:
            i = len(nxt) - 1               # promoted unchanged; it has no sibling at this level
        else:
            sib = i ^ 1
            proof.append(("right" if sib > i else "left", level[sib]))
            i //= 2
        level = nxt
    return proof


def verify_inclusion(leaf: str, proof: list[tuple[str, str]], root: str) -> bool:
    cur = leaf
    for side, sib in proof:
        cur = _pair(cur, sib) if side == "right" else _pair(sib, cur)
    return cur == root


@dataclass(frozen=True)
class RedactableDoc:
    """A document as leaves, some of which may be redacted to their hash alone."""

    entries: tuple  # each: {"text": str} for a surviving part, {"hash": str} for a redacted one

    @classmethod
    def from_text(cls, text: str, nonces: list | None = None) -> "RedactableDoc":
        """Build from plaintext, minting a nonce per part unless supplied.

        `nonces` exists so a caller can rebuild an identical document deterministically -- verifying
        a stored receipt requires reproducing the same leaf hashes, and a fresh random nonce would
        produce a different root every time.
        """
        parts = split_parts(text)
        if nonces is None:
            nonces = [new_nonce() for _ in parts]
        return cls(tuple({"text": p, "nonce": n} for p, n in zip(parts, nonces)))

    def hashes(self) -> list[str]:
        return [e["hash"] if "hash" in e else leaf_hash(e["text"], e.get("nonce", ""))
                for e in self.entries]

    def nonces(self) -> list:
        """Per-leaf nonces; empty string where a leaf has been redacted and its nonce destroyed."""
        return [e.get("nonce", "") for e in self.entries]

    def root(self) -> str:
        return merkle_root(self.hashes())

    def redact(self, predicate) -> "RedactableDoc":
        """Replace every part for which `predicate(text)` is true with its hash. Root is unchanged."""
        out = []
        for e in self.entries:
            if "text" in e and predicate(e["text"]):
                # Hash only. The nonce is NOT carried over -- retaining it would let anyone who
                # guesses the sentence confirm the guess, which is the whole thing being prevented.
                out.append({"hash": leaf_hash(e["text"], e.get("nonce", ""))})
            else:
                out.append(dict(e))
        return RedactableDoc(tuple(out))

    def text(self) -> str:
        """Surviving plaintext. Redacted parts render as a marker, never as their content."""
        return " ".join(e["text"] if "text" in e else "[redacted]" for e in self.entries)

    def as_dict(self) -> dict:
        return {"entries": [dict(e) for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict) -> "RedactableDoc":
        return cls(tuple(dict(e) for e in d.get("entries", [])))
