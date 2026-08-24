"""Verify a Fireweed evidence bundle — WITHOUT the engine.

PUBLIC ARTIFACT. Standard library only, and no Fireweed import anywhere: the whole point of an
evidence bundle is that checking it must not require the thing being checked. If you needed our
code to verify our claims, you would just be trusting us with extra steps.

    python open_format/verify_bundle.py evidence/

Exit 0 = every source hashed as declared, every receipt re-verified, and tampering was detected.
Exit 1 = something did not hold. The failing path is the load-bearing one: corrupt a byte of
evidence/sources/ and this must go red.

WHAT THIS CANNOT CHECK, stated plainly: a bundle's manifest may declare `recompute` entries (the
trap corpus, the read-gate bench). Those are instruments that run INSIDE the engine, so they are
not verifiable from a public bundle and are reported as `not verifiable here` rather than silently
skipped or, worse, counted as passes.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reference_reader import read_snapshot          # noqa: E402
import conformance                                  # noqa: E402


def verify(bundle: Path) -> int:
    failures: list[str] = []

    def check(name: str, ok: bool, note: str = "") -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} {note}")
        if not ok:
            failures.append(name)

    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        print(f"  no manifest.json in {bundle} — not a Fireweed evidence bundle")
        return 2
    manifest = json.loads(manifest_path.read_text())

    print(f"\n=== verify-bundle (engine-free) — {bundle} ===\n")
    print(f"  bundle    {manifest.get('name', '?')}  ({manifest.get('created', '?')})\n")

    # 1 — sources must hash to what the manifest declares
    sources: dict[str, bytes] = {}
    for source_id, declared in (manifest.get("sources") or {}).items():
        path = bundle / "sources" / source_id
        if not path.exists():
            check(f"source present: {source_id}", False)
            continue
        blob = path.read_bytes()
        actual = "sha256:" + hashlib.sha256(blob).hexdigest()
        check(f"source hashes as declared: {source_id}", actual == declared,
              "" if actual == declared else "TAMPERED OR REPLACED")
        sources[declared] = blob

    # 2 — every receipt must re-verify against those bytes, read through the public reader
    fmp = read_snapshot((bundle / "snapshot.json").read_bytes())
    total = verified = 0
    for _claim, receipt in fmp.receipts():
        total += 1
        doc = sources.get(receipt.doc_hash)
        if doc is not None and receipt.verify(doc):
            verified += 1
    check("every receipt re-verifies", total > 0 and verified == total, f"{verified}/{total}")
    expected = (manifest.get("expect") or {}).get("receipts_verified")
    if expected is not None:
        check("receipt count matches the manifest", verified == expected,
              f"declared {expected}")

    # 3 — tamper: the bundle MUST fail when the source changes. A verifier that only passes is not
    #     evidence of anything, so this asserts the negative directly.
    if sources:
        doc_hash, blob = next(iter(sources.items()))
        tampered = blob.replace(b"41208", b"41209") if b"41208" in blob else blob[:-1]
        survived = sum(1 for _c, r in fmp.receipts()
                       if r.doc_hash == doc_hash and r.verify(tampered))
        check("tampering breaks the receipts", survived == 0,
              f"{survived} survived a corrupted source (must be 0)")

    # 4 — the format itself
    c = conformance.run((bundle / "snapshot.json").read_bytes())
    check("open-format conformance", c.passed == len(c.rows), f"{c.passed}/{len(c.rows)}")

    # 5 — declared numbers this bundle cannot check on its own. Named, never silently dropped.
    for name in (manifest.get("recompute") or {}):
        print(f"  ----  {'recompute: ' + name:<44} not verifiable here (needs the engine)")

    print()
    if failures:
        print(f"  FAILED — {len(failures)}: {', '.join(failures)}\n")
        return 1
    print("  OK — sources hash, receipts re-verify, tampering is detected.\n")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    return verify(Path(argv[1]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
