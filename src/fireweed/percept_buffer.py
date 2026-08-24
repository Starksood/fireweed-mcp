"""Percept buffer — the bridge between Clock 1 (perception) and Clock 2 (consolidation).

Architectural role
------------------
Fireweed's two-clock design separates *perception* from *consolidation*:

  Clock 1 (perception):   raw signal -> tiny LLM -> Percept   (fast, must never block)
  [ PerceptBuffer ]       staging / working memory            (this module)
  Clock 2 (consolidation): batch -> entity-link, contradiction, decay, write to graph

The buffer exists so the two clocks can run at different cadences. Perception
acknowledges input the instant it produces a percept; consolidation drains the
buffer in batches and may lag real time without the system feeling unresponsive.
This is what gives "continuous situational awareness with persistent identity":
responsiveness is decoupled from consolidation.

Contract (load-bearing invariants)
----------------------------------
1. Admission NEVER blocks. A sensor that stalls is a blind spot. If the buffer is
   full, it EVICTS (lowest-salience first) rather than applying back-pressure.
2. Importance survives load. Eviction removes the lowest-salience percept, so a
   flood of low-salience noise can never push out a high-salience percept.
3. Two metabolisms. This buffer forgets what was never worth consolidating;
   graph decay (memory_loop.decay, to be ported) forgets what was consolidated
   but stopped mattering. Different stages, different timescales.
4. Drain is batched and salience-descending. One drain == one consolidation
   cycle == one `turn` (the unit graph decay counts in). The most important
   percepts consolidate first; under load, the least important fall behind by
   construction.
5. Everything is observable. Admissions, evictions, drains, and batch sizes are
   counted so sensory overload can be measured, not guessed.

Nothing a percept asserts is load-bearing: domains/polarity/entity are HINTS that
Clock 2's deterministic components (firewall, resolver, entity_linker) re-verify.
Perception proposes; consolidation disposes. That is why the perceptual model can
be small and dumb without costing identity coherence.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .constants import (
    PERCEPT_BUFFER_CAPACITY,
    PERCEPT_BUFFER_BATCH_SIZE,
    PERCEPT_BUFFER_MAX_IDLE_SECONDS,
)


@dataclass(frozen=True)
class Percept:
    """A single perception output, staged for consolidation.

    The three core fields (claim/evidence/confidence) mirror extractor.py's
    output and the Claim type — a percept is "what perception proposes". The hint
    fields (candidate_domains, polarity, entity_hint) accelerate Clock 2 but are
    re-verified there; the buffer never trusts them as authoritative. salience is
    the signal that governs eviction and drain ordering.

    frozen=True: a percept is an immutable value object that flows through the
    pipeline read-only, matching the Claim convention in claim.py.
    """
    claim: str                              # Single declarative statement (-> Turn.text)
    evidence: str                           # Verbatim source substring (-> Turn.evidence_span)
    confidence: float                       # Perception confidence 0.0-1.0
    salience: float                         # Importance 0.0-1.0 — drives eviction & drain order
    candidate_domains: tuple[str, ...] = () # Hint; firewall re-classifies
    polarity: int = 0                       # Hint: -1 / 0 / +1; resolver re-checks
    entity_hint: str | None = None          # Hint: surface name; entity_linker resolves
    source_id: str = ""                     # Provenance of the originating signal
    received_at: str = ""                   # ISO-8601 wall-clock admission time
    seq: int = 0                            # Monotonic admission order (tie-break + temporal)
    # Stage-3 significance (optional; populated by the perceiver after a code-side grounding
    # check, never trusted blind). rationale = grounded "why this matters"; cause = a verbatim
    # source clause this claim is caused-by (extracted only when a causal connective is present);
    # rationale_grounding = fraction of the rationale's content tokens found in the source (the
    # M-axis weight carries this through to significance.significance_weight).
    rationale: str | None = None
    cause: str | None = None
    rationale_grounding: float = 0.0

    def to_turn(self, turn_id: str):
        """Convert to a fabric.Turn for consolidation by the existing ingest path.

        Lazy import: fabric does not depend on percept_buffer, and we keep it
        that way (the buffer is upstream of the write path, not part of it).
        """
        from .fabric import Turn
        return Turn(
            turn_id=turn_id,
            text=self.claim,
            evidence_span=self.evidence,
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class AdmitResult:
    """Outcome of an admit() call. Admission always succeeds; this reports whether
    something had to be evicted to make room, and which percept was admitted."""
    percept: Percept
    evicted: Percept | None  # The percept dropped to make room, or None


@dataclass
class BufferStats:
    admitted: int = 0
    evicted: int = 0
    drained: int = 0
    drain_count: int = 0
    max_occupancy: int = 0

    def as_dict(self) -> dict:
        return {
            "admitted": self.admitted,
            "evicted": self.evicted,
            "drained": self.drained,
            "drain_count": self.drain_count,
            "max_occupancy": self.max_occupancy,
        }


class PerceptBuffer:
    """Bounded, salience-ordered staging area between perception and consolidation.

    Implementation: a min-heap keyed by (salience, seq) so the lowest-salience
    (and, on ties, oldest) percept is evicted in O(log n). Admission is O(log n)
    amortized and never blocks. Drain pulls the highest-salience batch and is the
    less frequent operation, so its O(k log k) sort is cheap.

    Time is injectable (clock) so drain-timer behaviour is deterministically
    testable without real sleeps.
    """

    def __init__(
        self,
        capacity: int = PERCEPT_BUFFER_CAPACITY,
        batch_size: int = PERCEPT_BUFFER_BATCH_SIZE,
        max_idle_seconds: float = PERCEPT_BUFFER_MAX_IDLE_SECONDS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.capacity = capacity
        self.batch_size = batch_size
        self.max_idle_seconds = max_idle_seconds
        self._clock = clock or time.time
        # Heap entries: (salience, seq, Percept). seq breaks ties (oldest first)
        # and keeps the tuple comparison from ever reaching the Percept (which is
        # not orderable).
        self._heap: list[tuple[float, int, Percept]] = []
        self._seq = 0
        self._last_drain_time = self._clock()
        self.stats = BufferStats()

    def __len__(self) -> int:
        return len(self._heap)

    def _now_iso(self) -> str:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc).isoformat()

    def admit(
        self,
        claim: str,
        evidence: str,
        confidence: float,
        salience: float,
        candidate_domains: tuple[str, ...] = (),
        polarity: int = 0,
        entity_hint: str | None = None,
        source_id: str = "",
        rationale: str | None = None,
        cause: str | None = None,
        rationale_grounding: float = 0.0,
    ) -> AdmitResult:
        """Stage a percept. ALWAYS succeeds; never blocks.

        If the buffer is at capacity, the lowest-salience resident percept is
        evicted to make room. If the incoming percept is itself the lowest-salience
        item (i.e. it would be the one evicted), it is dropped immediately rather
        than displacing something more important — admission still "succeeds" in
        the sense that it never blocks, and the dropped item is reported as the
        eviction.
        """
        percept = Percept(
            claim=claim,
            evidence=evidence,
            confidence=confidence,
            salience=salience,
            candidate_domains=tuple(candidate_domains),
            polarity=polarity,
            entity_hint=entity_hint,
            source_id=source_id,
            received_at=self._now_iso(),
            seq=self._seq,
            rationale=rationale,
            cause=cause,
            rationale_grounding=rationale_grounding,
        )
        self._seq += 1
        self.stats.admitted += 1

        evicted: Percept | None = None
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, (salience, percept.seq, percept))
        else:
            # Full: push then pop the global minimum. heapq.heappushpop is atomic
            # and O(log n). If `percept` is the new minimum it comes straight back
            # out (self-eviction), protecting the more-important residents.
            _, _, evicted = heapq.heappushpop(
                self._heap, (salience, percept.seq, percept)
            )
            self.stats.evicted += 1

        self.stats.max_occupancy = max(self.stats.max_occupancy, len(self._heap))
        return AdmitResult(percept=percept, evicted=evicted)

    def should_drain(self) -> bool:
        """True when consolidation should run: the buffer has filled to batch_size,
        OR it is non-empty and has been idle past max_idle_seconds (liveness, so a
        quiet trickle of percepts still gets consolidated)."""
        if not self._heap:
            return False
        if len(self._heap) >= self.batch_size:
            return True
        return (self._clock() - self._last_drain_time) >= self.max_idle_seconds

    def drain(self, max_items: int | None = None) -> list[Percept]:
        """Remove and return up to `max_items` (default batch_size) highest-salience
        percepts, ordered salience-descending (ties: oldest first). Resets the idle
        timer. One drain == one consolidation cycle == one graph `turn`.
        """
        if max_items is None:
            max_items = self.batch_size
        n = min(max_items, len(self._heap))
        # nlargest over (salience, seq) — but we want ties oldest-first within a
        # salience tier, so sort the taken items by (-salience, seq) explicitly.
        taken = heapq.nlargest(n, self._heap, key=lambda e: (e[0], -e[1]))
        taken_ids = {id(e) for e in taken}
        self._heap = [e for e in self._heap if id(e) not in taken_ids]
        heapq.heapify(self._heap)

        taken.sort(key=lambda e: (-e[0], e[1]))
        percepts = [e[2] for e in taken]

        self._last_drain_time = self._clock()
        self.stats.drained += len(percepts)
        self.stats.drain_count += 1
        return percepts

    def peek_salience(self) -> list[float]:
        """Current salience values (unordered) — for tests/introspection."""
        return [s for s, _, _ in self._heap]
