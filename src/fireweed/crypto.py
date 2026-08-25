"""Sprint 5 (§4B weeks 4–5, 4b) — crypto-shredding for right-to-erasure.

The GDPR-defensible way to erase from an append-only, hash-chained ledger: encrypt a subject's content
at write time, store only CIPHERTEXT in the immutable payload (so the hash + chain stay valid forever),
keep the KEYS in a separate mutable keyring, and erase = delete the key. The historical content then
cannot be recovered — even by a from-zero replay — while structure and integrity survive untouched.

Cipher: **AES-256-GCM** (a recognized standard — what a customer security review expects; via
`cryptography`) with authenticated encryption. Falls back to a keyed HMAC-SHA256 keystream only if the
library is absent (dev/CI without the dep), tagged distinctly so the two never mix. Integrity of the
LEDGER is the hash-chain; GCM additionally authenticates each ciphertext. Key destruction is the erase
act; the certificate attests the key lifecycle (see erasure.Certificate).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAVE_AESGCM = True
    CIPHER = "AES-256-GCM"
except Exception:                                     # pragma: no cover - env without cryptography
    _HAVE_AESGCM = False
    CIPHER = "HMAC-SHA256-keystream (fallback)"

# node content fields that carry the subject's PII — encrypted at rest in the ledger payload.
# Structure (ids, entity links, domains, timestamps, hashes) stays plaintext so closure + chain work.
CONTENT_FIELDS = ("claim", "normalized_claim")
_PROV_FIELDS = ("source_span",)


def new_key() -> bytes:
    return secrets.token_bytes(32)                    # 256-bit; valid for both AES-256-GCM and the fallback


def _keystream(key: bytes, nonce: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        counter += 1
    return bytes(out[:n])


def encrypt(key: bytes, plaintext: str) -> str:
    pt = plaintext.encode("utf-8")
    if _HAVE_AESGCM:
        nonce = secrets.token_bytes(12)
        ct = AESGCM(key).encrypt(nonce, pt, None)     # ct includes the GCM auth tag
        return "enc2:" + base64.b64encode(nonce + ct).decode("ascii")
    nonce = secrets.token_bytes(16)
    ct = bytes(a ^ b for a, b in zip(pt, _keystream(key, nonce, len(pt))))
    return "enc1:" + base64.b64encode(nonce + ct).decode("ascii")


def decrypt(key: bytes, blob: str) -> str:
    """Decrypt a blob. A wrong/absent key never crashes — returns a tombstone (AES-GCM auth fails) or
    garbage (keystream), so a shredded subject reconstructs as unreadable, not as an exception."""
    if blob.startswith("enc2:"):
        raw = base64.b64decode(blob[5:])
        nonce, ct = raw[:12], raw[12:]
        try:
            return AESGCM(key).decrypt(nonce, ct, None).decode("utf-8", errors="replace")
        except Exception:
            return "[erased]"                         # wrong/destroyed key -> unrecoverable
    if blob.startswith("enc1:"):
        raw = base64.b64decode(blob[5:])
        nonce, ct = raw[:16], raw[16:]
        pt = bytes(a ^ b for a, b in zip(ct, _keystream(key, nonce, len(ct))))
        return pt.decode("utf-8", errors="replace")
    return blob


class Keyring:
    """subject entity_id → content key. The mutable side of crypto-shredding: deleting a key makes
    every ciphertext encrypted under it permanently unrecoverable. In production this is a KMS/HSM."""

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}

    def get_or_create(self, subject_id: str) -> bytes:
        k = self._keys.get(subject_id)
        if k is None:
            k = self._keys[subject_id] = new_key()
        return k

    def get(self, subject_id: str) -> bytes | None:
        return self._keys.get(subject_id)

    def shred(self, subject_id: str) -> bool:
        """Delete a subject's key — the irreversible act of erasure. Returns True if a key existed."""
        return self._keys.pop(subject_id, None) is not None

    def has(self, subject_id: str) -> bool:
        return subject_id in self._keys

    def serialize(self) -> bytes:
        """Persist the keyring (keys base64) — the mutable side of crypto-shredding. In prod this is a
        KMS/HSM, not a blob; here it lives alongside the immutable ledger so reload can decrypt."""
        import json
        return json.dumps({sid: base64.b64encode(k).decode("ascii")
                           for sid, k in self._keys.items()}).encode("utf-8")

    @classmethod
    def deserialize(cls, blob: bytes | None) -> "Keyring":
        kr = cls()
        if blob:
            import json
            kr._keys = {sid: base64.b64decode(v) for sid, v in json.loads(blob.decode("utf-8")).items()}
        return kr


_TOMBSTONE = "[erased]"


def _subject_of(node_dict: dict) -> str | None:
    """The entity whose key encrypts this node: the first actor, else the first entity. Deterministic."""
    ents = node_dict.get("entities", [])
    for e in ents:
        if e.get("role") == "actor":
            return e.get("entity_id")
    return ents[0].get("entity_id") if ents else None


def encrypt_node_content(node_dict: dict, keyring: Keyring) -> dict:
    """Return a copy of the node dict with content fields encrypted under its subject's key.
    Records `_enc_subject` so decrypt/shred knows which key. No subject → unchanged (nothing to key)."""
    subject = _subject_of(node_dict)
    if subject is None:
        return node_dict
    key = keyring.get_or_create(subject)
    d = dict(node_dict)
    for f in CONTENT_FIELDS:
        if isinstance(d.get(f), str):
            d[f] = encrypt(key, d[f])
    if isinstance(d.get("provenance"), dict):
        prov = dict(d["provenance"])
        for f in _PROV_FIELDS:
            if isinstance(prov.get(f), str):
                prov[f] = encrypt(key, prov[f])
        d["provenance"] = prov
    d["_enc_subject"] = subject
    return d


# Entity payloads are shaped differently from node payloads: there is no `entities` list to derive a
# subject from, and `provenance` is a LIST of records rather than one dict. `encrypt_node_content`
# therefore cannot be pointed at them, which is why ADD_ENTITY events shipped unencrypted and an
# erased subject's name and sentence survived in the ledger after a completed erasure.
_ENTITY_FIELDS = ("canonical_name",)
_ENTITY_PROV_FIELDS = ("source_span",)


def encrypt_entity_content(entity_dict: dict, keyring: Keyring) -> dict:
    """Encrypt an entity's name and provenance spans under ITS OWN key.

    Keying on the entity itself is what makes erasure reach this: destroying that entity's key
    renders its historical payloads unreadable. An entity that survives erasure keeps its key and
    its payloads, which is correct -- and is also why the closure now removes entities left with no
    surviving grounding, since those hold the erased subject's text and nothing else.
    """
    subject = entity_dict.get("entity_id")
    if not subject:
        return entity_dict
    key = keyring.get_or_create(subject)
    d = dict(entity_dict)
    for f in _ENTITY_FIELDS:
        if isinstance(d.get(f), str):
            d[f] = encrypt(key, d[f])
    prov = d.get("provenance")
    if isinstance(prov, list):
        out = []
        for entry in prov:
            if isinstance(entry, dict):
                e = dict(entry)
                for f in _ENTITY_PROV_FIELDS:
                    if isinstance(e.get(f), str):
                        e[f] = encrypt(key, e[f])
                out.append(e)
            else:
                out.append(entry)
        d["provenance"] = out
    d["_enc_subject"] = subject
    return d


def decrypt_entity_content(entity_dict: dict, keyring: Keyring) -> dict:
    """Inverse. A shredded key yields tombstones, exactly as for nodes."""
    subject = entity_dict.get("_enc_subject")
    if not subject:
        return entity_dict
    key = keyring.get(subject)
    d = {k: v for k, v in entity_dict.items() if k != "_enc_subject"}
    for f in _ENTITY_FIELDS:
        if isinstance(d.get(f), str):
            d[f] = decrypt(key, d[f]) if key else _TOMBSTONE
    prov = d.get("provenance")
    if isinstance(prov, list):
        out = []
        for entry in prov:
            if isinstance(entry, dict):
                e = dict(entry)
                for f in _ENTITY_PROV_FIELDS:
                    if isinstance(e.get(f), str):
                        e[f] = decrypt(key, e[f]) if key else _TOMBSTONE
                out.append(e)
            else:
                out.append(entry)
        d["provenance"] = out
    return d


def decrypt_node_content(node_dict: dict, keyring: Keyring) -> dict:
    """Inverse of encrypt_node_content. If the subject's key was shredded, content becomes a tombstone
    — proving the historical record is unrecoverable after erasure."""
    subject = node_dict.get("_enc_subject")
    if subject is None:
        return node_dict
    key = keyring.get(subject)
    d = {k: v for k, v in node_dict.items() if k != "_enc_subject"}
    for f in CONTENT_FIELDS:
        if isinstance(d.get(f), str) and d[f].startswith(("enc1:", "enc2:")):
            d[f] = decrypt(key, d[f]) if key else _TOMBSTONE
    if isinstance(d.get("provenance"), dict):
        prov = dict(d["provenance"])
        for f in _PROV_FIELDS:
            if isinstance(prov.get(f), str) and prov[f].startswith(("enc1:", "enc2:")):
                prov[f] = decrypt(key, prov[f]) if key else _TOMBSTONE
        d["provenance"] = prov
    return d
