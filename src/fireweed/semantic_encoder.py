"""Pinned, hash-verified deterministic semantic encoder — part of the substrate spec.

NorthStar guardrail 1 (amended 2026-07-06): *deterministic ≠ lexical.* A pinned open-weights encoder,
run locally at a fixed revision with its weights content-hash verified, is a pure function — same
input, same output, forever, no cloud, readable by any future compute. It is therefore permitted inside
the resolver and preferred over brittle word-set Jaccard for paraphrase / morphology / negation and
non-English input (the LoCoMo entity-J degradation is exactly this brittleness).

The engine core stays dependency-light: this module is imported ONLY when the semantic similarity
backend is enabled (`resolver.set_similarity_backend`), and sentence-transformers is a soft dependency
with a clear error if the semantic path is used without it.
"""
from __future__ import annotations
import hashlib
from functools import lru_cache
from pathlib import Path

# --- the pin: a specific model, a specific revision, a specific weights fingerprint --------------
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EXPECTED_WEIGHTS_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
_ROUND = 6  # round cosine so a fixed threshold is stable across trivial float noise

_model = None
_verified_fingerprint: str | None = None


class WeightsFingerprintError(RuntimeError):
    """Raised when the loaded encoder's weights do not match the pinned content hash (tamper/version)."""


def _weights_path() -> Path | None:
    import glob
    hits = glob.glob(str(Path.home() / ".cache/huggingface/hub"
                        / "models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
                        / PINNED_REVISION / "**/model.safetensors"), recursive=True)
    return Path(hits[0]) if hits else None


def _verify_weights() -> str | None:
    """Hash the pinned-revision weights and check them against EXPECTED. Raise on MISMATCH; return the
    fingerprint if verified; return None (with no error) if the file can't be located in this cache."""
    p = _weights_path()
    if p is None:
        return None
    fp = hashlib.sha256(p.read_bytes()).hexdigest()
    if fp != EXPECTED_WEIGHTS_SHA256:
        raise WeightsFingerprintError(
            f"{MODEL_ID}@{PINNED_REVISION} weights fingerprint {fp[:12]}… != pinned "
            f"{EXPECTED_WEIGHTS_SHA256[:12]}… — refusing to use unverified weights in the substrate")
    return fp


def _get_model():
    global _model, _verified_fingerprint
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # soft dependency
            raise RuntimeError(
                "semantic similarity backend needs sentence-transformers "
                "(`pip install sentence-transformers`); the lexical backend has no such dependency") from e
        _verified_fingerprint = _verify_weights()               # tamper/version check
        _model = SentenceTransformer(MODEL_ID, revision=PINNED_REVISION)
    return _model


@lru_cache(maxsize=8192)
def _embed(text: str):
    import numpy as np
    v = _get_model().encode(text or "", normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype=float)


def similarity(a: str, b: str) -> float:
    """Deterministic cosine in [0,1]-ish (rounded). Same inputs → same output, forever."""
    if not a or not b:
        return 0.0
    import numpy as np
    s = float(np.dot(_embed(a), _embed(b)))
    return round(s, _ROUND)


def fingerprint() -> dict:
    """The encoder's identity for the substrate spec / audit."""
    _get_model()
    return {"model_id": MODEL_ID, "revision": PINNED_REVISION,
            "weights_sha256": _verified_fingerprint or "unverified(cache-miss)",
            "expected_sha256": EXPECTED_WEIGHTS_SHA256}
