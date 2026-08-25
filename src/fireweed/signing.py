"""How an erasure certificate is signed, and what that signature is worth.

Two schemes, and the difference is not cosmetic.

HMAC-SHA256 (default, zero-dependency)
    Symmetric. Verification requires the same secret used to sign, so the only party who can check
    the certificate is the party who could equally have forged it. Against accidental corruption or
    a later mistake by the same operator this is a useful checksum. Against a reader who does not
    already trust the operator -- the adversary the README invokes -- it establishes NOTHING. This
    is stated plainly rather than left for the reader to work out.

Ed25519 (optional, needs `cryptography`)
    Asymmetric. The private key signs, the PUBLIC key verifies, and the public key can be published
    without weakening anything. A third party who holds it can confirm the certificate came from
    this install and has not been altered, without being able to produce one. That is the difference
    between a checksum and evidence.

Note the contrast with byte-range receipts, which never had this problem: a receipt is verified by
re-hashing the source document and re-slicing the range, so anyone holding the document can check it
with no key at all. Receipts were always adversary-checkable; the certificate was not, and the
project treated them as the same class of artifact.

Deliberately NOT hand-rolled. A pure-Python Ed25519 would preserve the zero-dependency promise, but
hand-written cryptography in a product whose entire pitch is verifiability is the wrong trade -- so
the strong scheme is an optional extra and the weak one is honestly labelled.
"""
from __future__ import annotations

import hashlib
import hmac


def ed25519_available() -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: F401
        return True
    except Exception:
        return False


class HmacSigner:
    """Symmetric. A tamper-detection checksum, not an attestation."""

    scheme = "hmac-sha256"
    adversary_checkable = False

    def __init__(self, key: bytes) -> None:
        self._key = key

    def sign(self, payload: bytes) -> str:
        return f"{self.scheme}:" + hmac.new(self._key, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        return hmac.compare_digest(signature, self.sign(payload))

    def public_key(self) -> str | None:
        return None          # there is no public half; that is the point


class Ed25519Signer:
    """Asymmetric. The public key can be published; only the holder of the private key can sign."""

    scheme = "ed25519"
    adversary_checkable = True

    def __init__(self, private_bytes: bytes) -> None:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        self._sk = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)

    @staticmethod
    def generate() -> bytes:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
        sk = ed25519.Ed25519PrivateKey.generate()
        return sk.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def sign(self, payload: bytes) -> str:
        return f"{self.scheme}:" + self._sk.sign(payload).hex()

    def verify(self, payload: bytes, signature: str) -> bool:
        return verify_with_public_key(payload, signature, self.public_key())

    def public_key(self) -> str:
        from cryptography.hazmat.primitives import serialization
        raw = self._sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        return raw.hex()


def verify_with_public_key(payload: bytes, signature: str, public_key_hex: str) -> bool:
    """Verify an ed25519 certificate holding only the PUBLIC key — the point of the scheme."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import ed25519
    scheme, _, hexsig = signature.partition(":")
    if scheme != "ed25519":
        return False
    try:
        pk = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        pk.verify(bytes.fromhex(hexsig), payload)
        return True
    except (InvalidSignature, ValueError):
        return False
