"""End-to-end ingestion pipeline: claim text → firewall → resolver → graph + reinforcement.

Stage order:
  1. Build Claim (harness tags source_turn_id before this call)
  2. Firewall: ACCEPT / RESCUE / REJECT / QUARANTINE
  3. Entity linker: resolve or create Entity records, return list[EntityRef]
  4. Resolve mutation: CREATE / DEDUP / REINFORCE / MODIFY / NOOP
  5. Apply mutation + reinforcement update
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import combinations

from .claim import Claim, FirewallDecision
from .firewall import evaluate
from .graph import GraphState, Node, Reinforcement, Temporal, NodeStatus, Relation, RelationEvidence, RelationRef
from .entity_linker import link_entities
from .reinforcement import compute_reinforcement, get_layer
from .resolver import (
    resolve_mutation,
    CreateMutation, DedupMutation, ReinforceMutation, ModifyMutation, DisputeMutation, NoopMutation,
    Mutation,
)
from .constants import NOVEL_DOMAIN_CROSS_SESSION_FLOOR, VALIDATION_SESSION_THRESHOLD
from .coreference import resolve_pronouns


@dataclass
class IngestResult:
    mutation: Mutation
    firewall_decision: str      # "ACCEPT" | "RESCUE" | "REJECT" | "QUARANTINE"
    layer: str                  # "COLD" | "WARM" | "HOT" | "CORE" — for the affected node
    reinforcement_applied: bool # True when reinforcement was updated, even on a DEDUP path.
                                # Phase 6 background ops ask "why was this node promoted?" —
                                # this flag makes the answer auditable without log archaeology.


class IngestContext:
    """Mutable ingestion state: graph + per-node session tracking."""

    def __init__(self) -> None:
        self.graph = GraphState()
        self._sessions_seen: dict[str, set[str]] = {}  # node_id → set of session_ids
        self._total_sessions: int = 0
        self._seen_domains: set[str] = set()  # domains already in graph (for novel-domain detection)
        self._session_timestamp: str = "1970-01-01T00:00:00Z"  # Phase 4: current session anchor
        self._session_anchor: str | None = None  # entity_id of first person-role actor seen

    def begin_session(self, session_id: str, session_timestamp: str = "1970-01-01T00:00:00Z") -> None:
        """Increment the session counter. Call once before ingesting any claim in that session."""
        self._total_sessions += 1
        self._session_timestamp = session_timestamp

    def ingest(
        self,
        claim_text: str,
        evidence_span: str,
        confidence: float,
        source_turn_id: str,
        session_id: str,
    ) -> IngestResult:
        """Ingest one claim through the full pipeline."""
        claim = Claim(
            claim=claim_text,
            evidence_span=evidence_span,
            candidate_domains=set(),  # firewall resolves this
            confidence=confidence,
            source_turn_id=source_turn_id,
        )

        fw = evaluate(claim)

        if fw.decision in (FirewallDecision.REJECT, FirewallDecision.QUARANTINE):
            noop = NoopMutation(reason=fw.reason, trace=None)  # type: ignore[arg-type]
            return IngestResult(mutation=noop, firewall_decision=fw.decision.value,
                                layer="COLD", reinforcement_applied=False)

        # Phase 4: entity linker runs before resolver
        resolved_entities = link_entities(
            claim_text, source_turn_id, evidence_span, self.graph
        )

        # Update session anchor — first actor entity becomes the persistent pronoun referent.
        if self._session_anchor is None:
            for ref in resolved_entities:
                if ref.role == "actor":
                    self._session_anchor = ref.entity_id
                    break

        # Coreference: add anchor entity if claim contains pronouns.
        resolved_entities = resolve_pronouns(claim_text, self._session_anchor, resolved_entities)

        mutation = resolve_mutation(
            claim, fw.resolved_domains, resolved_entities, self._session_timestamp, self.graph
        )

        layer, r_applied = self._apply(mutation, fw.decision.value, session_id)
        return IngestResult(mutation=mutation, firewall_decision=fw.decision.value,
                            layer=layer, reinforcement_applied=r_applied)

    # ── Co-occurrence relation creation ───────────────────────────────────────

    def _add_co_occurs(self, node: Node) -> Node:
        """Auto-create co_occurs relations for each unordered pair of actor-class entities.
        Stores Relation objects in the graph and returns node with updated RelationRef list.
        Called before add_node — the node must not yet be in the graph. Never raises.
        """
        try:
            actor_refs = [r for r in node.entities if r.role in ("actor", "co_actor")]
            if len(actor_refs) < 2:
                return node
            new_rels = list(node.relations)
            for a, b in combinations(actor_refs, 2):
                src, tgt = sorted([a.entity_id, b.entity_id])
                relation_id = "rel_" + uuid.uuid4().hex[:12]
                relation = Relation(
                    relation_id=relation_id,
                    source_id=src,
                    target_id=tgt,
                    relation_type="co_occurs",
                    polarity="positive",
                    claim=f"{src} co_occurs with {tgt}",
                    confidence=0.90,
                    evidence=RelationEvidence(
                        source_node_ids=[node.node_id],
                        source_spans=[],
                    ),
                    status="validated",
                )
                self.graph.add_relation(relation)
                new_rels.append(RelationRef(
                    relation_id=relation_id,
                    relation_type="co_occurs",
                    target_node_id=tgt,
                ))
            return replace(node, relations=new_rels)
        except Exception:
            return node

    # ── Mutation application ───────────────────────────────────────────────────

    def _apply(self, mutation: Mutation, firewall_decision: str, session_id: str) -> tuple[str, bool]:
        """Returns (layer, reinforcement_applied). Labels the resulting write-grain events with the
        semantic mutation kind + DecisionTrace (Sprint 5 audit metadata; does not affect replay)."""
        kind = type(mutation).__name__.replace("Mutation", "").upper() or "NOOP"
        trace = getattr(getattr(mutation, "trace", None), "__dict__", None)
        with self.graph.write_context(kind, trace):
            return self._apply_dispatch(mutation, firewall_decision, session_id)

    def _apply_dispatch(self, mutation: Mutation, firewall_decision: str, session_id: str) -> tuple[str, bool]:
        if isinstance(mutation, CreateMutation):
            return self._apply_create(mutation.node, firewall_decision, session_id), True
        if isinstance(mutation, DedupMutation):
            # Exact match in a new session → also apply reinforcement (spacing effect).
            # Exact match in the same session → just report current layer, no update.
            node_id = mutation.existing_node_id
            if session_id not in self._sessions_seen.get(node_id, set()):
                return self._apply_reinforce(node_id, session_id), True
            node = self.graph.get_node(node_id)
            return (get_layer(node.reinforcement.overall) if node else "COLD"), False
        if isinstance(mutation, ReinforceMutation):
            # Mirror the DEDUP session check: semantic near-duplicate in a new session
            # → apply reinforcement. Same session → no-op (spacing effect already captured
            # by DEDUP; same-session REINFORCE would make r_applied lie and call update_node
            # with values equal to the old ones — wasted write, asymmetric with DEDUP).
            node_id = mutation.target_node_id
            if session_id not in self._sessions_seen.get(node_id, set()):
                return self._apply_reinforce(node_id, session_id), True
            node = self.graph.get_node(node_id)
            return (get_layer(node.reinforcement.overall) if node else "COLD"), False
        if isinstance(mutation, ModifyMutation):
            rel_id = self._mark_superseded(mutation.old_node_id, mutation.new_node.node_id)
            new_node = mutation.new_node
            if rel_id:
                ref = RelationRef(relation_id=rel_id, relation_type="supersedes",
                                  target_node_id=mutation.old_node_id)
                new_node = replace(new_node, relations=[ref])
            return self._apply_create(new_node, firewall_decision, session_id), True
        if isinstance(mutation, DisputeMutation):
            return self._apply_dispute(mutation.existing_node_id, mutation.new_node,
                                       firewall_decision, session_id), True
        return "COLD", False  # NoopMutation

    def _apply_create(self, node: Node, firewall_decision: str, session_id: str) -> str:
        # Deterministic-node-id collision → reinforce, don't raise.
        # The resolver's CREATE/MODIFY path can hand us a node whose content-hashed id
        # (normalized_claim | sorted(entity_ids) | session_timestamp; see resolver.
        # _generate_node_id) is already in the graph. This happens on restore + re-perceive
        # ("healing by re-exposure"): a surviving node's id is regenerated by re-perception,
        # but the active-only find_exact_match path didn't catch it (the survivor is
        # non-active, or its normalized_claim differs from the fresh claim's), so resolution
        # fell through to CREATE. Because the id is deterministic, a collision means the
        # SAME claim/entities/anchor — a genuine re-encounter of an existing fact — so we
        # reinforce the existing node instead of letting graph.add_node raise. Scoped to the
        # id-collision case: a non-deterministic id clash would be a real bug and is left to
        # surface elsewhere; this only ever fires on an exact content match.
        if self.graph.get_node(node.node_id) is not None:
            return self._apply_reinforce(node.node_id, session_id)
        # Novel-domain boost: first claim in a domain the graph has never seen before.
        # RESCUE claims also get the boost (they're definitionally novel-domain).
        # Semantics: the first time the system learns about a new area of Maya's life,
        # that fact is high-salience — cross_session_recurrence starts at the floor.
        is_novel_domain = bool(node.domains - self._seen_domains)
        apply_boost = firewall_decision == "RESCUE" or is_novel_domain
        r_init = Reinforcement(
            local_frequency=0.0,
            cross_session_recurrence=NOVEL_DOMAIN_CROSS_SESSION_FLOOR if apply_boost else 0.0,
            overall=0.0,
        )
        r_init = replace(r_init, overall=compute_reinforcement(r_init.local_frequency, r_init.cross_session_recurrence))
        node = replace(node, reinforcement=r_init)
        # Phase 9: auto-create co_occurs relations for multi-entity nodes.
        if len([r for r in node.entities if r.role in ("actor", "co_actor")]) >= 2:
            node = self._add_co_occurs(node)
        self.graph.add_node(node)
        self._sessions_seen[node.node_id] = {session_id}
        self._seen_domains |= node.domains  # mark these domains as seen
        return get_layer(node.reinforcement.overall)

    def _apply_reinforce(self, node_id: str, session_id: str) -> str:
        node = self.graph.get_node(node_id)
        if node is None:
            return "COLD"
        self._sessions_seen.setdefault(node_id, set()).add(session_id)
        denom = max(1, self._total_sessions)
        new_csr = len(self._sessions_seen[node_id]) / denom
        new_r = replace(
            node.reinforcement,
            cross_session_recurrence=new_csr,
            overall=compute_reinforcement(node.reinforcement.local_frequency, new_csr),
        )
        # Validation promotion: provisional → validated after VALIDATION_SESSION_THRESHOLD sessions.
        session_count = len(self._sessions_seen[node_id])
        new_status = node.status
        if (node.status.validation_state == "provisional"
                and session_count >= VALIDATION_SESSION_THRESHOLD):
            new_status = replace(node.status, validation_state="validated")
        self.graph.update_node(replace(node, reinforcement=new_r, status=new_status))
        return get_layer(new_r.overall)

    def _mark_superseded(self, old_node_id: str, new_node_id: str) -> str | None:
        """Mark old_node as superseded, create a supersedes Relation. Returns relation_id or None."""
        node = self.graph.get_node(old_node_id)
        if node is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        new_temporal = replace(node.temporal, superseded_by=new_node_id, valid_to=now)
        new_status = replace(node.status, memory_state="superseded")
        self.graph.update_node(replace(node, temporal=new_temporal, status=new_status))
        # Phase 4: auto-create supersedes relation (source = new, target = old)
        relation_id = "rel_" + uuid.uuid4().hex[:12]
        relation = Relation(
            relation_id=relation_id,
            source_id=new_node_id,
            target_id=old_node_id,
            relation_type="supersedes",
            polarity="positive",
            claim=f"{new_node_id} supersedes {old_node_id}",
            confidence=1.0,
            evidence=RelationEvidence(
                source_node_ids=[new_node_id, old_node_id],
                source_spans=[],
            ),
            status="validated",
        )
        self.graph.add_relation(relation)
        return relation_id

    # ── Dispute (W3: hold standing tension instead of collapsing) ─────────────

    def _apply_dispute(self, existing_id: str, new_node: Node,
                       firewall_decision: str, session_id: str) -> str:
        """Hold a contradiction as standing tension: add the opposing node, mark BOTH the
        existing and new nodes `disputed` (still valid and retrievable — NOT superseded), and
        link them with a symmetric `contradicts` relation. Returns the new node's layer."""
        layer = self._apply_create(new_node, firewall_decision, session_id)
        relation_id = self._add_contradicts(new_node.node_id, existing_id)
        for nid in (existing_id, new_node.node_id):
            node = self.graph.get_node(nid)
            if node is None:
                continue
            rels = list(node.relations)
            if relation_id:
                other = existing_id if nid == new_node.node_id else new_node.node_id
                rels.append(RelationRef(relation_id=relation_id, relation_type="contradicts",
                                        target_node_id=other))
            self.graph.update_node(replace(
                node, status=replace(node.status, memory_state="disputed"), relations=rels))
        return layer

    def _add_contradicts(self, a_id: str, b_id: str) -> str | None:
        """Create a `contradicts` relation between two node ids (tension link). Returns id or None."""
        if self.graph.get_node(a_id) is None or self.graph.get_node(b_id) is None:
            return None
        relation_id = "rel_" + uuid.uuid4().hex[:12]
        self.graph.add_relation(Relation(
            relation_id=relation_id, source_id=a_id, target_id=b_id,
            relation_type="contradicts", polarity="negative",
            claim=f"{a_id} contradicts {b_id}", confidence=1.0,
            evidence=RelationEvidence(source_node_ids=[a_id, b_id], source_spans=[]),
            status="validated",
        ))
        return relation_id
