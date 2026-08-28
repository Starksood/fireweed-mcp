"""Graph state: Node types and in-memory store for the resolver."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace. Used for exact-match deduplication."""
    return " ".join(text.lower().split())


# ── Nested Node types ─────────────────────────────────────────────────────────

@dataclass
class EntityRef:
    entity_id: str   # e.g., "ent_maya". Phase 3: string stub; Phase 4: resolved FK.
    role: str        # Semantic role: "actor", "location", "object", "subject", etc.


@dataclass
class Predicate:
    lemma: str                                       # Verb infinitive form, e.g., "buy", "live".
    polarity: Literal["positive", "negative", "neutral"]
    object: str | None                               # Direct object, if present.

    # The typed slot, from the authored vocabulary in `predicate_vocabulary.py`. Attached by the
    # one-directional gate in `predicate_extraction.py` AFTER grounding has already admitted the
    # claim, so all six fields being None is an ordinary, expected state: it means the claim is
    # indexed by its literal surface form only, exactly as every claim was before this existed.
    # Never a refusal, never a force-fit. See docs/FINDING_predicate_representation.md.
    #
    # Flat rather than nested so `Predicate(**d)` still round-trips a snapshot written by an older
    # version — a missing key simply takes its default.
    slot: str | None = None                          # e.g. "employer"; None = untyped
    slot_span: str | None = None                     # the evidence text that justified the slot
    slot_start: int | None = None                    # offsets into the ADMITTED EVIDENCE SPAN,
    slot_end: int | None = None                      #   i.e. into provenance.source_span
    slot_vocabulary: str | None = None               # vocabulary version that produced the label
    slot_proposer: str | None = None                 # "bootstrap" | "model"


@dataclass
class Motivation:
    # Phase 4: stub. Node.motivation is always None in Phase 3.
    rationale: str | None
    confidence: float


@dataclass
class MemoryContext:
    # Phase 4: stub. Node.context is always None in Phase 3.
    conditions: list[str]
    constraints: list[str]
    environment: list[str]


@dataclass
class Temporal:
    asserted_at: str           # ISO-8601 UTC. When the LLM proposed the claim. Set on CREATE.
    stored_at: str             # ISO-8601 UTC. When the node was written. Set on CREATE.
    event_time: str | None     # Phase 4: when the described event occurred.
    valid_from: str | None     # Phase 4: start of validity interval.
    valid_to: str | None       # Phase 4: end of validity interval (None = still current).
    superseded_by: str | None  # node_id that replaces this one. Set on MODIFY of the old node.


@dataclass
class Provenance:
    source_turn_id: str     # From Claim.source_turn_id.
    source_span: str        # From Claim.evidence_span.
    extraction_method: str  # Always "llm_candidate_plus_firewall" in Phase 3.
    confidence: float       # From Claim.confidence.
    # Sprint 3 — document receipt: byte-range binding into a hash-signed source document.
    # Optional (default None) → turn-based sources and pre-Sprint-3 snapshots are unchanged.
    # When set, source_span is a verified contiguous slice doc[byte_start:byte_end] of the doc
    # whose content hash is doc_hash (see receipts.py).
    doc_hash: str | None = None
    byte_start: int | None = None
    byte_end: int | None = None
    # Which gate admitted this claim: grounding.GROUNDED_VERBATIM (subject named in the cited span)
    # or GROUNDED_RESOLVED (subject resolved from outside it). Optional so pre-existing snapshots
    # load unchanged; the re-audit sweep fills it in retroactively.
    grounding_class: str | None = None


@dataclass
class Reinforcement:
    local_frequency: float           # Normalized occurrence count within current session.
    cross_session_recurrence: float  # Fraction of sessions this node has appeared in.
    overall: float                   # r = 0.3 * local_frequency + 0.7 * cross_session_recurrence.


@dataclass
class NodeStatus:
    memory_state: Literal["active", "quarantined", "disputed", "superseded", "frozen"]
    firewall_decision: Literal["ACCEPT", "RESCUE"]
    validation_state: Literal["validated", "provisional"]
    # validation_state transitions at VALIDATION_SESSION_THRESHOLD sessions.


@dataclass
class RelationRef:
    # Phase 3: Node.relations is always []. Field exists to prevent Phase 4 schema retrofit.
    relation_id: str
    relation_type: str
    target_node_id: str


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass
class Node:
    node_id: str
    # `summary` is COMPRESS output, `reflection` is REFLECT output — both derived, and they must be
    # distinguishable from extracted `fact`s. COMPRESS wrote node_type="fact" for its summaries,
    # so every report counted `summaries: 0` (unfalsifiable — the value was unreachable) and every
    # "a fact was superseded by a fact" was really "a fact was folded into a summary". Two opposite
    # events, one counter: revision means a belief was WRONG, compression means it was ABSORBED.
    node_type: Literal["fact", "event", "state", "preference", "constraint", "inference",
                       "reflection", "summary"]
    claim: str
    normalized_claim: str
    entities: list[EntityRef]
    domains: set[str]
    facets: list[str]
    predicate: Predicate
    temporal: Temporal
    provenance: Provenance
    reinforcement: Reinforcement
    status: NodeStatus
    # Optional fields — Phase 4 stubs. Default here because Python requires fields with
    # defaults to trail fields without. The resolver explicitly passes None / []; these
    # defaults only exist to satisfy the dataclass ordering rule.
    motivation: Motivation | None = None
    context: MemoryContext | None = None
    relations: list[RelationRef] = field(default_factory=list)


# ── Phase 4: Entity types ─────────────────────────────────────────────────────

@dataclass
class EntityProvenance:
    source_turn_id: str
    source_span: str


@dataclass
class Entity:
    entity_id: str
    canonical_name: str
    entity_type: Literal["person", "place", "organization", "object", "concept", "event"]
    aliases: list[str]
    scopes: list[str]
    attributes: dict[str, str]
    confidence: float
    provenance: list[EntityProvenance]


# ── Phase 4: Relation types ───────────────────────────────────────────────────

@dataclass
class RelationEvidence:
    source_node_ids: list[str]
    source_spans: list[str]


@dataclass
class Relation:
    """source_id / target_id semantics depend on relation_type:
      - supersedes:  node IDs (one node supersedes another)
      - co_occurs:   entity IDs (two entities co-occur in the source node)
      - causes:      node IDs (cause node -> effect node; Stage 3 W2, from grounded causes)
      - motivates:   node IDs (motivator node -> motivated node; Stage 3 W2, from grounded rationale)
      - before:      node IDs (earlier node -> later node; Stage 3 W2, from event_time)
      - contradicts: node IDs (two nodes held in standing tension; Stage 3 W3, DISPUTE)
    Future relation types must declare their referent type here.
    """
    relation_id: str
    source_id: str
    target_id: str
    relation_type: Literal[
        "causes", "constrains", "motivates", "co_occurs",
        "contradicts", "supersedes", "supports",
        "located_in", "before", "after",
    ]
    polarity: Literal["positive", "negative", "neutral"]
    claim: str
    confidence: float
    evidence: RelationEvidence
    status: Literal["candidate", "validated", "rejected"]


# ── GraphState ────────────────────────────────────────────────────────────────

class GraphState:
    """In-memory graph store backed by a dict[str, Node] keyed by node_id."""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        # Phase 4: entity store
        self._entities: dict[str, Entity] = {}
        self._entity_name_index: dict[str, str] = {}
        # Phase 4: relation store
        self._relations: dict[str, Relation] = {}
        self._relation_index: dict[str, list[str]] = {}
        # Sprint 2: entity_id -> node_ids inverted index (derived; rebuildable from _nodes).
        # Powers bounded ego-graph retrieval so an entity-anchored query touches O(Σ deg(eid))
        # nodes instead of scanning the whole graph. Never a source of truth — _nodes is.
        self._entity_to_nodes: dict[str, set[str]] = {}
        # Sprint 2b (write-path scale): derived indexes over ACTIVE nodes that turn the per-claim
        # ingest scans from O(N) into O(1)/O(domain). Maintained on add/update; rebuilt on restore.
        self._active_claim_index: dict[str, str] = {}       # normalized_claim -> active node_id
        self._domain_to_active: dict[str, set[str]] = {}    # domain -> active node_ids
        # Sprint 5: optional event-ledger capture. When a ledger is attached, every raw write emits a
        # write-grain event (the concrete object). `_replaying` guards against re-emitting during fold.
        self._ledger = None
        self._ledger_tenant = "local"
        self._ledger_seq = 0
        self._replaying = False
        self._sealed = False
        self._keyring = None     # Sprint 5 4b: when set, node CONTENT is crypto-shredded in payloads
        # Per-install secret that makes entity ids opaque instead of name-derived. Lives with the
        # keys, not the store, so a copy of the substrate cannot be used to confirm a guessed name.
        self._id_salt = ""
        self._write_semantic: tuple[str, dict | None] | None = None   # (semantic_kind, trace)

    def seal(self) -> None:
        """Enforce the chokepoint invariant: after seal(), a graph mutation with NO ledger attached
        raises — nothing writes state except as a captured event. Production/backend seals; dev/tests
        that don't care about the ledger stay unsealed and unchanged. (Replay is exempt.)"""
        self._sealed = True

    def attach_ledger(self, log, tenant_id: str = "local", keyring=None) -> None:
        """Capture every subsequent raw write into `log` as a write-grain event (Sprint 5). Opt-in;
        unattached graphs behave exactly as before. With a `keyring`, node CONTENT is encrypted in the
        persisted payload (crypto-shredding) so erasure = key deletion makes history unrecoverable."""
        self._ledger = log
        self._ledger_tenant = tenant_id
        self._keyring = keyring
        # Seed the counter from the ledger's own tail. `_ledger_seq` is only used to build `ts` and
        # `event_id`, and it starts at 0 in __init__ -- so a process that restores state from a
        # snapshot and re-attaches an existing ledger would restart the ids at zero and emit
        # "mcp:0", "mcp:1", ... a second time. The chain itself stays valid (record() takes seq and
        # prev_hash from its own tail, and the primary key is (tenant_id, seq)), so this is not
        # corruption -- but event_id is meant to identify an event, and duplicates make it useless
        # for dedup or for referring to one from outside. Measured on a two-run restart before this
        # was added: ids went 0,1,2 then 0,1,2,3.
        try:
            self._ledger_seq = len(log.events(tenant_id))
        except Exception:
            pass          # a ledger without a queryable tail keeps the previous behaviour

    @contextmanager
    def write_context(self, semantic_kind: str, trace: dict | None = None):
        """Label the writes inside this block with a semantic kind (CREATE/REINFORCE/MODIFY/…) and an
        optional DecisionTrace, carried as event AUDIT metadata (payload.semantic / payload.trace).
        Replay ignores it — it dispatches on the write-grain `kind` — so labeling can never change
        reconstructed state; it exists for erasure/retention/audit."""
        prev = self._write_semantic
        self._write_semantic = (semantic_kind, trace)
        try:
            yield
        finally:
            self._write_semantic = prev

    def _emit(self, kind: str, obj) -> None:
        from dataclasses import asdict
        from .ledger import _jsonable
        if self._replaying:
            return
        if self._ledger is None:
            if self._sealed:
                raise RuntimeError(f"sealed graph: {kind} write without an attached ledger — "
                                   "every mutation must be a captured event")
            return
        seq = self._ledger_seq
        self._ledger_seq += 1
        obj_dict = _jsonable(asdict(obj))
        if self._keyring is not None and kind in ("ADD_NODE", "UPDATE_NODE"):
            from .crypto import encrypt_node_content
            obj_dict = encrypt_node_content(obj_dict, self._keyring)
        elif self._keyring is not None and kind in ("ADD_ENTITY", "UPDATE_ENTITY"):
            # Entity payloads carry the canonical name and the verbatim span the entity was learned
            # from. Left in the clear they preserved an erased subject's name and sentence in an
            # append-only log that erasure cannot reach -- measured with grep after a completed
            # erasure. Entity dicts are shaped differently from nodes, so this needs its own
            # function rather than a wider `kind` tuple.
            from .crypto import encrypt_entity_content
            obj_dict = encrypt_entity_content(obj_dict, self._keyring)
        # Stamp the resolver version (Q1): in the payload so it is hashed (tamper-evident) + persisted,
        # enabling the offline audit query "would today's resolver have decided this differently?"
        from . import __version__ as _resolver_version
        payload = {"obj": obj_dict, "resolver_version": _resolver_version}
        if self._write_semantic is not None:
            payload["semantic"], payload["trace"] = self._write_semantic[0], self._write_semantic[1]
        self._ledger.record(self._ledger_tenant, kind, ts=f"w:{seq}",
                            event_id=f"{self._ledger_tenant}:{seq}", payload=payload)

    def _index_node(self, node: Node) -> None:
        for e in node.entities:
            self._entity_to_nodes.setdefault(e.entity_id, set()).add(node.node_id)
        if node.status.memory_state == "active":
            # first-writer-wins mirrors find_exact_match's "first active match" (dedup keeps it unique)
            self._active_claim_index.setdefault(node.normalized_claim, node.node_id)
            for d in node.domains:
                self._domain_to_active.setdefault(d, set()).add(node.node_id)

    def _deindex_node(self, node: Node) -> None:
        for e in node.entities:
            bucket = self._entity_to_nodes.get(e.entity_id)
            if bucket:
                bucket.discard(node.node_id)
                if not bucket:
                    del self._entity_to_nodes[e.entity_id]
        if self._active_claim_index.get(node.normalized_claim) == node.node_id:
            del self._active_claim_index[node.normalized_claim]
        for d in node.domains:
            b = self._domain_to_active.get(d)
            if b:
                b.discard(node.node_id)
                if not b:
                    del self._domain_to_active[d]

    def add_node(self, node: Node) -> None:
        """Store a new node. Raises ValueError if node_id already exists."""
        if node.node_id in self._nodes:
            raise ValueError(f"Node '{node.node_id}' already exists in graph.")
        self._nodes[node.node_id] = node
        self._index_node(node)
        self._emit("ADD_NODE", node)

    def update_node(self, node: Node) -> None:
        """Replace an existing node. Raises KeyError if node_id not found."""
        if node.node_id not in self._nodes:
            raise KeyError(f"Node '{node.node_id}' not found in graph.")
        self._deindex_node(self._nodes[node.node_id])   # drop old entity refs (entities may change)
        self._nodes[node.node_id] = node
        self._index_node(node)
        self._emit("UPDATE_NODE", node)

    # ── Sprint 5: erasure removals (driven by the ERASE event; not independently emitted) ───────
    def remove_node(self, node_id: str) -> None:
        node = self._nodes.pop(node_id, None)
        if node is not None:
            self._deindex_node(node)

    def remove_entity(self, entity_id: str) -> None:
        ent = self._entities.pop(entity_id, None)
        if ent is not None and ent.canonical_name.lower() in self._entity_name_index:
            del self._entity_name_index[ent.canonical_name.lower()]
        self._entity_to_nodes.pop(entity_id, None)

    def remove_relation(self, relation_id: str) -> None:
        rel = self._relations.pop(relation_id, None)
        if rel is None:
            return
        for side in (rel.source_id, rel.target_id):
            lst = self._relation_index.get(side)
            if lst and relation_id in lst:
                lst.remove(relation_id)
                if not lst:
                    del self._relation_index[side]

    def nodes_touching(self, entity_ids: set[str], timestamp: str | None = None) -> list[Node]:
        """Valid nodes whose entity set intersects `entity_ids` — the bounded ego-graph seed set.

        Uses the inverted index (O(Σ deg(eid)) not O(N)). Returns the same nodes a full valid-node
        scan filtered to these entities would return; determinism preserved by node_id ordering.
        Empty `entity_ids` returns [] (callers fall back to the full scan for entity-less queries).
        """
        if not entity_ids:
            return []
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()
        node_ids: set[str] = set()
        for eid in entity_ids:
            node_ids |= self._entity_to_nodes.get(eid, set())
        out: list[Node] = []
        for nid in sorted(node_ids):                     # sorted → deterministic order
            node = self._nodes.get(nid)
            if node is None:
                continue
            vt = node.temporal.valid_to
            if vt is None:
                out.append(node)
                continue
            try:
                if vt > timestamp:
                    out.append(node)
            except (ValueError, TypeError):
                out.append(node)
        return out

    def get_node(self, node_id: str) -> Node | None:
        """Return the node with this ID, or None."""
        return self._nodes.get(node_id)

    def find_exact_match(self, claim_text: str) -> Node | None:
        """Return the first active node whose normalized_claim equals normalize_text(claim_text).

        Only searches nodes with memory_state == 'active'.
        """
        target = normalize_text(claim_text)
        nid = self._active_claim_index.get(target)          # O(1) via the active-claim index
        return self._nodes.get(nid) if nid else None

    def find_related(
        self, claim_text: str, candidate_domains: set[str], limit: int = 5
    ) -> list[Node]:
        """Return up to `limit` active nodes ranked by Jaccard similarity to claim_text.

        Filtered to nodes with at least one domain in common with candidate_domains.
        Only nodes with Jaccard > 0 are returned (at least one word in common).
        Only active nodes (memory_state == 'active') are searched.
        Returned list is sorted descending by Jaccard score.
        """
        query_words = set(normalize_text(claim_text).split())
        # Candidates restricted to active nodes in the requested domains (via the domain index)
        # rather than scanning the whole graph. Equivalent to the old domain filter.
        cand_ids: set[str] = set()
        for d in candidate_domains:
            cand_ids |= self._domain_to_active.get(d, set())

        scored: list[tuple[float, Node]] = []
        for nid in cand_ids:
            node = self._nodes[nid]
            node_words = set(node.normalized_claim.split())
            union = query_words | node_words
            if not union:
                continue
            jaccard = len(query_words & node_words) / len(union)
            if jaccard > 0:
                scored.append((jaccard, node))

        # cand_ids is a set (unordered) → sort deterministically: score desc, then node_id.
        scored.sort(key=lambda x: (-x[0], x[1].node_id))
        return [node for _, node in scored[:limit]]

    def all_nodes(self) -> list[Node]:
        """Return all nodes (all memory states) as a list. Used for snapshots and tests."""
        return list(self._nodes.values())

    def get_valid_nodes(self, timestamp: str | None = None) -> list[Node]:
        """Return nodes valid at the given timestamp (excluding superseded nodes).

        A node is valid if its valid_to field is None or > timestamp.
        Used by retrieval to exclude superseded/invalid nodes from query results.
        """
        if timestamp is None:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()

        result = []
        for node in self._nodes.values():
            if node.temporal.valid_to is None:
                result.append(node)
            else:
                try:
                    valid_to = node.temporal.valid_to
                    if valid_to > timestamp:
                        result.append(node)
                except (ValueError, TypeError):
                    result.append(node)
        return result

    # ── Phase 4: Entity store methods ─────────────────────────────────────────

    def add_entity(self, entity: Entity) -> None:
        """Store a new entity. Raises ValueError if entity_id already exists."""
        if entity.entity_id in self._entities:
            raise ValueError(f"Entity '{entity.entity_id}' already exists.")
        self._entities[entity.entity_id] = entity
        self._entity_name_index[entity.canonical_name.lower()] = entity.entity_id
        self._emit("ADD_ENTITY", entity)

    def update_entity(self, entity: Entity) -> None:
        """Replace an existing entity (e.g., to add aliases or provenance).
        Raises KeyError if entity_id not found. Rebuilds name index entry."""
        if entity.entity_id not in self._entities:
            raise KeyError(f"Entity '{entity.entity_id}' not found.")
        old = self._entities[entity.entity_id]
        if old.canonical_name.lower() in self._entity_name_index:
            del self._entity_name_index[old.canonical_name.lower()]
        self._entities[entity.entity_id] = entity
        self._entity_name_index[entity.canonical_name.lower()] = entity.entity_id
        self._emit("UPDATE_ENTITY", entity)

    def get_entity(self, entity_id: str) -> Entity | None:
        """Return the entity with this ID, or None."""
        return self._entities.get(entity_id)

    def find_entity_by_name(self, name: str) -> Entity | None:
        """Case-insensitive lookup by canonical_name. O(1). Returns None if not found.
        Does NOT search aliases — alias lookup is the linker's responsibility."""
        entity_id = self._entity_name_index.get(name.lower())
        return self._entities.get(entity_id) if entity_id else None

    def all_entities(self) -> list[Entity]:
        """Return all entities as a list. Used for snapshots and tests."""
        return list(self._entities.values())

    # ── Phase 4: Relation store methods ───────────────────────────────────────

    def add_relation(self, relation: Relation) -> None:
        """Store a new relation. Raises ValueError if relation_id already exists.
        Updates the node-level index for both source_id and target_id."""
        if relation.relation_id in self._relations:
            raise ValueError(f"Relation '{relation.relation_id}' already exists.")
        self._relations[relation.relation_id] = relation
        self._relation_index.setdefault(relation.source_id, []).append(relation.relation_id)
        self._relation_index.setdefault(relation.target_id, []).append(relation.relation_id)
        self._emit("ADD_RELATION", relation)

    def get_relation(self, relation_id: str) -> Relation | None:
        """Return the relation with this ID, or None."""
        return self._relations.get(relation_id)

    def get_relations_for_node(self, node_id: str) -> list[Relation]:
        """Return all relations where this node is either source or target."""
        ids = self._relation_index.get(node_id, [])
        return [self._relations[rid] for rid in ids if rid in self._relations]

    def all_relations(self) -> list[Relation]:
        """Return all relations. Used for snapshots and tests."""
        return list(self._relations.values())

    def find_relations(self, node_id: str,
                       relation_type: str | None = None) -> list[Relation]:
        """Return Relations from this node's RelationRef list, optionally filtered by type.
        Uses node's own refs (not source/target index) — correct for co_occurs. Never raises.
        """
        try:
            node = self.get_node(node_id)
            if node is None:
                return []
            result = []
            for ref in node.relations:
                rel = self._relations.get(ref.relation_id)
                if rel is None:
                    continue
                if relation_type is None or rel.relation_type == relation_type:
                    result.append(rel)
            return result
        except Exception:
            return []

    # ── Phase 5: Synthesis candidate methods ──────────────────────────────────

    def get_synthesis_candidates(self, min_r: float = 0.50) -> list[Node]:
        """Return active, non-inference HOT/CORE nodes sorted by r descending.

        Filters: memory_state == 'active', node_type != 'inference',
        reinforcement.overall >= min_r.
        """
        candidates = [
            n for n in self._nodes.values()
            if n.status.memory_state == "active"
            and n.node_type != "inference"
            and n.reinforcement.overall >= min_r
        ]
        candidates.sort(key=lambda n: n.reinforcement.overall, reverse=True)
        return candidates

    def get_validated_nodes(self) -> list[Node]:
        """Return active nodes with validation_state == 'validated'."""
        return [
            n for n in self._nodes.values()
            if n.status.memory_state == "active"
            and n.status.validation_state == "validated"
        ]

    # ── Phase 7: External retrieval ───────────────────────────────────────────

    def retrieve(self, query: str, domains: set[str] | None = None,
                 entity_ids: list[str] | None = None,
                 min_r: float = 0.0, limit: int = 10) -> list[Node]:
        """Return the most relevant active nodes for an external query.
        Score = 0.4 * jaccard + 0.3 * domain_score + 0.3 * r. Never raises.
        """
        from .scoring import jaccard_score
        try:
            qd = domains or set()
            es = set(entity_ids) if entity_ids else None
            scored: list[tuple[float, Node]] = []
            for node in self._nodes.values():
                if node.status.memory_state != "active": continue
                if node.reinforcement.overall < min_r: continue
                if qd and not (qd & node.domains): continue
                if es and not (es & {e.entity_id for e in node.entities}): continue
                j = jaccard_score(query, node.normalized_claim)
                d = len(qd & node.domains) / max(len(qd | node.domains), 1) if qd else 0.0
                score = 0.4 * j + 0.3 * d + 0.3 * node.reinforcement.overall
                if score > 0.0:
                    scored.append((score, node))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [n for _, n in scored[:limit]]
        except Exception:
            return []
