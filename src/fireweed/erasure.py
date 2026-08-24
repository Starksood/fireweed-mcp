"""Sprint 5 (§4B weeks 4–5) — right-to-erasure: exact closure + signed certificate.

The wedge P2 showpiece: "forget this person — here is the proof." What no vector store can do is
**enumerate exactly** what it knows about a subject; the graph structure makes the closure exact.
Erasure records an ERASE event (append-only, hash-chain intact), removes the closure from derived
state so every probe about the subject structurally abstains, and emits a signed certificate:
{state hash before, state hash after, closure manifest, post-erasure query-battery result}.

Scope of THIS layer: closure + ERASE + certificate + abstention, on the derived graph and the ledger.
Making the subject's content unrecoverable in the HISTORICAL ledger payloads (so a from-zero replay
can't reconstruct it) is crypto-shredding — the next increment — which encrypts erasable content under
a per-subject key and deletes the key on erasure. This layer is honest about that boundary.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Closure:
    """The exact set the erasure touches — computed from graph structure, not similarity search."""
    subject_entity_id: str
    node_ids: list[str]
    relation_ids: list[str]
    entity_ids: list[str]
    derived_ids: list[str] = field(default_factory=list)   # TOMBSTONE: derived nodes invalidated

    def manifest(self) -> dict:
        return {"subject": self.subject_entity_id,
                "node_ids": sorted(self.node_ids), "relation_ids": sorted(self.relation_ids),
                "entity_ids": sorted(self.entity_ids),
                # Listed SEPARATELY from directly-owned nodes: an auditor needs to see what
                # abstraction was destroyed, not only which facts. These carry no entities of their
                # own, so without this they would be invisible in the certificate.
                "derived_invalidated": sorted(self.derived_ids)}


def compute_closure(graph, subject_entity_id: str) -> Closure:
    """Exact transitive closure of a data subject: every node whose entities include the subject, the
    relations incident to those nodes or the subject, and the subject entity itself. Uses the inverted
    index — O(deg(subject)) — and is EXACT (structure, not embeddings)."""
    node_ids = {n.node_id for n in graph.nodes_touching({subject_entity_id})}
    # also catch superseded/disputed nodes touching the subject (nodes_touching is valid-only)
    for n in graph.all_nodes():
        if subject_entity_id in {e.entity_id for e in n.entities}:
            node_ids.add(n.node_id)

    # DERIVED NODES. REFLECT and COMPRESS output carries entities=[], so an entity-keyed closure
    # never reaches it and a summary restating the subject's facts survived their erasure —
    # measured with a canary in test_erasure_derived_nodes. Derived nodes are CACHE, not record:
    # state = fold(ledger), so a node computed from sources that are leaving is a stale cache
    # entry, and keeping it means keeping the subject's content.
    #
    # Deliberately DELETE, not re-derive. Re-deriving from surviving sources would need an LLM
    # call inside erase(), and a non-deterministic step in the middle of a signed certificate is
    # disqualifying. Re-derivation happens for free on the next consolidation cycle if the
    # surviving evidence still supports it.
    #
    # This over-deletes: a reflection over four facts loses its footing when ONE source is erased,
    # even though three remain. That is the intended trade — over-deletion is recoverable through
    # normal operation, a leak is not.
    derived_ids: set[str] = set()
    frontier = set(node_ids)
    while frontier:                                  # transitive: reflection over summary over fact
        rising = set()
        for rel in graph._relations.values():
            if rel.relation_type != "derived_from":
                continue
            if rel.target_id in frontier and rel.source_id not in node_ids:
                rising.add(rel.source_id)
        if not rising:
            break
        derived_ids |= rising
        node_ids |= rising
        frontier = rising

    relation_ids = []
    for rid, rel in graph._relations.items():
        if (rel.source_id == subject_entity_id or rel.target_id == subject_entity_id
                or rel.source_id in node_ids or rel.target_id in node_ids):
            relation_ids.append(rid)
    return Closure(subject_entity_id=subject_entity_id, node_ids=sorted(node_ids),
                   relation_ids=sorted(relation_ids), entity_ids=[subject_entity_id],
                   derived_ids=sorted(derived_ids))


@dataclass(frozen=True)
class Certificate:
    """A signed, verifiable erasure receipt.

    `scope` states EXACTLY what is certified so the certificate can never overstate (a certificate that
    claims more than the system delivers is a compliance liability, not a feature). The key-lifecycle
    fields attest the crypto-shred: which cipher, whether the content key was destroyed, and when.
    """
    subject_entity_id: str
    state_hash_before: str
    state_hash_after: str
    closure_manifest: dict
    battery_all_abstained: bool
    scope: str = ""
    cipher: str = ""
    key_destroyed: bool = False
    key_destroyed_at: str | None = None
    co_subject_note: str = ("shared records in the closure are shredded on erasure of any co-subject "
                            "(conservative default); a surviving co-subject's own account remains "
                            "re-derivable from their un-erased turns.")
    signature: str = ""

    def _signable(self) -> bytes:
        from .ledger import canonical_bytes
        return canonical_bytes({
            "subject": self.subject_entity_id,
            "before": self.state_hash_before, "after": self.state_hash_after,
            "closure": self.closure_manifest, "abstained": self.battery_all_abstained,
            "scope": self.scope, "cipher": self.cipher,
            "key_destroyed": self.key_destroyed, "key_destroyed_at": self.key_destroyed_at,
        })

    def signed(self, signing_key: bytes) -> "Certificate":
        sig = hmac.new(signing_key, self._signable(), hashlib.sha256).hexdigest()
        object.__setattr__(self, "signature", "hmac-sha256:" + sig)
        return self

    def verify_signature(self, signing_key: bytes) -> bool:
        expected = "hmac-sha256:" + hmac.new(signing_key, self._signable(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature, expected)


def _state_hash(graph) -> str:
    from .ledger import graph_fingerprint
    return "sha256:" + hashlib.sha256(graph_fingerprint(graph)).hexdigest()


def backfill_derivation_edges(graph) -> dict:
    """One-time migration: give pre-existing derived nodes the `derived_from` edges erasure needs.

    Derived nodes created before those edges existed are unreachable by compute_closure (they carry
    entities=[]), so erasure on an existing substrate now fails loudly with ErasureIncomplete
    instead of silently leaking. This recovers the derivation from the edges the operators were
    ALREADY writing:

        REFLECT   `supports`   evidence -> reflection
        COMPRESS  `supersedes` summary  -> source

    Measured on the 212-commit ops graph: 231/231 derived nodes recoverable, none orphaned.

    `supersedes` is only read as derivation when the edge's SOURCE is a `summary` node, because
    pipeline._mark_superseded writes the same relation type for ordinary revision. On a substrate
    older than the summary-labelling fix, COMPRESS output is typed `fact` and is therefore
    indistinguishable from a revision successor -- those summaries are reported as unrecoverable
    rather than guessed at, since a wrong guess would delete an unrelated node's successor during
    someone's erasure.

    Idempotent: nodes that already have a `derived_from` edge are skipped.
    """
    from .graph import Relation, RelationEvidence
    import uuid as _uuid

    nodes = {n.node_id: n for n in graph.all_nodes()}
    supports_to: dict[str, list[str]] = {}
    supersedes_from: dict[str, list[str]] = {}
    already: set[str] = set()
    for rel in graph._relations.values():
        if rel.relation_type == "derived_from":
            already.add(rel.source_id)
        elif rel.relation_type == "supports":
            supports_to.setdefault(rel.target_id, []).append(rel.source_id)
        elif rel.relation_type == "supersedes":
            supersedes_from.setdefault(rel.source_id, []).append(rel.target_id)

    linked, edges, unrecoverable = 0, 0, []
    for node in nodes.values():
        if node.node_type not in ("reflection", "summary") or node.node_id in already:
            continue
        if node.node_type == "reflection":
            sources = supports_to.get(node.node_id, [])
        else:
            sources = supersedes_from.get(node.node_id, [])
        sources = [sid for sid in dict.fromkeys(sources) if sid in nodes]
        if not sources:
            unrecoverable.append(node.node_id)
            continue
        for sid in sources:
            graph.add_relation(Relation(
                relation_id="rel_" + _uuid.uuid4().hex[:12],
                source_id=node.node_id, target_id=sid,
                relation_type="derived_from", polarity="positive",
                claim=f"{node.node_id} derived from {sid}", confidence=1.0,
                evidence=RelationEvidence([node.node_id, sid], []), status="validated",
            ))
            edges += 1
        linked += 1
    return {"linked": linked, "edges": edges,
            "unrecoverable": sorted(unrecoverable), "already_linked": len(already)}


class ErasureIncomplete(RuntimeError):
    """Raised when the probe battery still answers about a subject after erasure.

    A signed document titled an erasure certificate, issued while probes about the subject return
    evidence, is worse than no certificate: it is an attestation that the holder can be shown to
    be false. Previously erase() recorded battery_all_abstained=False and signed anyway, leaving
    the caller free to ignore it.

    The erasure itself is NOT rolled back -- it is an append-only ledger event and the content key
    is already destroyed. What is withheld is the attestation. The residue is reported so the
    caller can find out why (an entity-linking miss that left a bare-name mention, a derived node
    with no derivation edge) rather than guessing.
    """

    def __init__(self, subject_entity_id: str, answering: list[str], closure_manifest: dict):
        self.subject_entity_id = subject_entity_id
        self.answering_probes = answering
        self.closure_manifest = closure_manifest
        super().__init__(
            f"erasure of {subject_entity_id} is incomplete: "
            f"{len(answering)} probe(s) still answer ({', '.join(map(repr, answering[:5]))}). "
            f"Closure removed {len(closure_manifest.get('node_ids', []))} nodes "
            f"({len(closure_manifest.get('derived_invalidated', []))} derived). "
            f"No certificate issued."
        )


def erase(graph, ledger, tenant_id: str, subject_entity_id: str,
          probes, query_fn, signing_key: bytes, keyring=None) -> Certificate:
    """Erase a subject and return a signed certificate.

    1. hash state before; 2. compute the exact closure; 3. record an ERASE event (append-only) and
    apply it (removes the closure → the derived graph no longer carries the subject); 4. CRYPTO-SHRED —
    delete the subject's content key so the historical ledger payloads become permanently
    unrecoverable (even a from-zero replay yields tombstones); 5. run the probe battery — every probe
    must structurally abstain; 6. hash state after; 7. sign the certificate.
    """
    from .ledger import apply_event

    before = _state_hash(graph)
    closure = compute_closure(graph, subject_entity_id)

    seq = getattr(ledger, "graph_version", lambda t: 0)(tenant_id)
    ev = ledger.record(tenant_id, "ERASE", ts=f"erase:{seq}", event_id=f"{tenant_id}:erase:{seq}",
                       payload={"subject": subject_entity_id, "closure": closure.manifest()})
    apply_event(ev, graph)

    # crypto-shred (the irreversible act) — only when a keyring backs this substrate
    key_destroyed, key_destroyed_at = False, None
    if keyring is not None:
        from datetime import datetime, timezone
        from . import crypto
        key_destroyed = keyring.shred(subject_entity_id)
        key_destroyed_at = datetime.now(timezone.utc).isoformat()
        cipher = crypto.CIPHER
        scope = ("Certifies erasure of the subject's closure from the active substrate AND the ledger, "
                 "with the content encryption key DESTROYED — historical ciphertext is unrecoverable "
                 "(from-zero replay tombstones the subject). Structure + hash-chain integrity retained.")
    else:
        cipher = "none"
        scope = ("Certifies erasure of the subject's closure from the ACTIVE SUBSTRATE and ledger only. "
                 "No content key was destroyed (this substrate is not crypto-shredding), so residual "
                 "plaintext may persist in snapshots/history — governed by the store's retention policy, "
                 "NOT by this certificate.")

    answering = [p for p in probes if not getattr(query_fn(graph, p), "abstain", False)]
    all_abstained = not answering
    if answering:
        # Refuse to attest. See ErasureIncomplete: the erasure stands, the certificate does not.
        raise ErasureIncomplete(subject_entity_id, answering, closure.manifest())
    after = _state_hash(graph)
    return Certificate(
        subject_entity_id=subject_entity_id, state_hash_before=before, state_hash_after=after,
        closure_manifest=closure.manifest(), battery_all_abstained=all_abstained,
        scope=scope, cipher=cipher, key_destroyed=key_destroyed, key_destroyed_at=key_destroyed_at,
    ).signed(signing_key)
