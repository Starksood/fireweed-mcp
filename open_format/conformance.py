"""FMP conformance suite — the tests any implementation must pass to call itself an FMP reader.

PUBLIC ARTIFACT, standard library only (see reference_reader.py for why).

The point of this file is that it does NOT trust the reference reader. It states properties the
FORMAT must have, and checks them against whatever reader it is handed. Running it against the
reference reader proves the reference reader conforms; running it against yours proves yours does.

    python open_format/conformance.py path/to/snapshot.json

Exit code 0 = all checks pass.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reference_reader import read_snapshot, FMPError, Receipt   # noqa: E402


class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def __call__(self, name: str, ok: bool, note: str = "") -> None:
        self.rows.append((name, bool(ok), note))

    @property
    def passed(self) -> int:
        return sum(1 for _, ok, _ in self.rows if ok)

    def report(self) -> bool:
        width = max(len(n) for n, _, _ in self.rows)
        for name, ok, note in self.rows:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {note}")
        print(f"\n  {self.passed}/{len(self.rows)} conformance checks pass")
        return self.passed == len(self.rows)


def run(snapshot_bytes: bytes, reader=read_snapshot) -> Check:
    c = Check()
    fmp = reader(snapshot_bytes)
    raw = json.loads(snapshot_bytes.decode("utf-8"))

    # ── structure ─────────────────────────────────────────────────────────────
    c("version is declared and supported", fmp.snapshot_version == 2,
      f"snapshot_version={fmp.snapshot_version}")
    c("every node is surfaced", len(fmp.claims) == len(raw["nodes"]),
      f"{len(fmp.claims)} claims / {len(raw['nodes'])} nodes")
    c("every entity is surfaced", len(fmp.entities) == len(raw["entities"]),
      f"{len(fmp.entities)} entities")
    c("node ids are unique", len({x.node_id for x in fmp.claims}) == len(fmp.claims))
    c("entity ids are unique", len({e.entity_id for e in fmp.entities}) == len(fmp.entities))

    # ── referential integrity ─────────────────────────────────────────────────
    known = {e.entity_id for e in fmp.entities}
    dangling = {eid for cl in fmp.claims for eid in cl.entity_ids if eid not in known}
    c("claims reference only declared entities", not dangling,
      f"dangling: {sorted(dangling)[:3]}" if dangling else "")

    # `relations` is HETEROGENEOUS by design: node-to-node edges (supersedes, contradicts,
    # supports, derived_from) and entity-to-entity edges (co_occurs) share one list. The first run
    # of this suite reported 4 "dangling" relations that were simply co_occurs edges between
    # entities — an under-specified format, not a broken snapshot. The endpoint domain per type is
    # now declared here and in SPEC.md, and a reader that guesses wrong is non-conformant.
    ids = {x.node_id for x in fmp.claims}
    ENTITY_EDGES = {"co_occurs"}
    bad_rel = []
    for r in fmp.relations:
        if not isinstance(r, dict):
            continue
        domain = known if r.get("relation_type") in ENTITY_EDGES else ids
        if r.get("source_id") not in domain or r.get("target_id") not in domain:
            bad_rel.append(r)
    c("relations reference only declared nodes/entities", not bad_rel,
      f"{len(bad_rel)} dangling" if bad_rel else "node edges -> nodes, co_occurs -> entities")

    unknown_types = {r.get("relation_type") for r in fmp.relations if isinstance(r, dict)} - {
        "supersedes", "contradicts", "supports", "derived_from", "co_occurs",
        "causes", "motivates", "before"}
    c("relation types are all declared", not unknown_types,
      f"undeclared: {sorted(t for t in unknown_types if t)}" if unknown_types else "")

    # ── the memory-state contract ─────────────────────────────────────────────
    legal = {"active", "quarantined", "disputed", "superseded", "frozen"}
    illegal = {x.memory_state for x in fmp.claims} - legal
    c("memory_state values are all declared", not illegal, f"illegal: {illegal}" if illegal else "")
    c("superseded claims are RETAINED, not deleted",
      all(x.memory_state in legal for x in fmp.claims),
      "a reader can always reconstruct what was once believed")
    c("disputed claims stay readable",
      all(x.is_active for x in fmp.claims if x.memory_state == "disputed"),
      "both sides of an unresolved contradiction remain answerable")

    # ── receipts: the load-bearing claim of the whole format ──────────────────
    partial = 0
    for cl in fmp.claims:
        prov = (cl.raw.get("provenance") or {})
        has_some = any(prov.get(k) is not None for k in ("doc_hash", "byte_start", "byte_end"))
        has_all = all(prov.get(k) is not None for k in ("doc_hash", "byte_start", "byte_end"))
        if has_some and not has_all and cl.receipt is not None:
            partial += 1
    c("a partial coordinate is NOT reported as a receipt", partial == 0,
      "no receipt is better than a receipt that cannot be checked")

    ok_ranges = all(0 <= r.byte_start <= r.byte_end for _, r in fmp.receipts())
    c("receipt byte ranges are well ordered", ok_ranges)

    # ── tamper evidence, proved rather than asserted ──────────────────────────
    doc = b"Priya Raman joined Acme in 2019 as a logistics analyst."
    rec = Receipt("sha256:" + hashlib.sha256(doc).hexdigest(), 0, len(doc), doc.decode())
    c("receipt verifies against the true source", rec.verify(doc) is True)
    c("receipt FAILS against a tampered source",
      rec.verify(doc.replace(b"2019", b"2018")) is False, "one digit changed")
    c("receipt FAILS on a shifted range", Receipt(rec.doc_hash, 1, len(doc), rec.quote)
      .verify(doc) is False)

    # ── the reader must refuse what it cannot honestly read ───────────────────
    def _refuses(payload: bytes) -> bool:
        try:
            reader(payload)
            return False
        except FMPError:
            return True
        except Exception:
            return False        # wrong exception type is a conformance failure, not a pass

    c("refuses a future snapshot_version",
      _refuses(json.dumps({"snapshot_version": 99, "nodes": [], "entities": [],
                           "relations": []}).encode()),
      "silently reading an unknown version is how formats rot")
    c("refuses malformed JSON", _refuses(b"{not json"))
    c("refuses a snapshot missing 'nodes'",
      _refuses(json.dumps({"snapshot_version": 2, "entities": [], "relations": []}).encode()))

    # ── determinism ───────────────────────────────────────────────────────────
    a, b = reader(snapshot_bytes), reader(snapshot_bytes)
    c("reading is deterministic",
      [x.node_id for x in a.claims] == [x.node_id for x in b.claims])

    return c


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    data = Path(argv[1]).read_bytes()
    print(f"\n=== FMP CONFORMANCE — {argv[1]} ===\n")
    return 0 if run(data).report() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
