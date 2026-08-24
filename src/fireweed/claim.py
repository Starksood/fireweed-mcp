"""Claim and firewall result types for Fireweed Fabric v16.

Claim: what the LLM proposes (Stage 1 output). Consumed by the firewall; never stored.
FirewallResult: what the firewall returns after evaluating a claim.
FirewallDecision: the four-state gate (ACCEPT / RESCUE / REJECT / QUARANTINE).

Architectural rule: Claim and Node are separate types. Claim fields must not leak
into Node except through the explicit transfer points documented in docs/SCHEMAS.md.
"""

from dataclasses import dataclass
from enum import Enum


class FirewallDecision(str, Enum):
    ACCEPT = "ACCEPT"
    RESCUE = "RESCUE"
    REJECT = "REJECT"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True)
class Claim:
    """Immutable LLM proposal. Consumed by the firewall; never stored in the graph.

    frozen=True: attribute reassignment is forbidden by the dataclass machinery,
    making the intent machine-checked: Claim is a value object that flows through
    the pipeline read-only.

    Note on candidate_domains: the field is typed set[str] for caller convenience;
    callers pass set() since the pipeline always sets it to set() (the firewall
    re-classifies from the claim text). If Claim ever needs to be hashable,
    this field must be changed to frozenset[str] — set is unhashable so hash(claim)
    raises at runtime. As of Phase 11, no code hashes Claim instances.
    """
    claim: str                    # Normalized factual statement from LLM
    evidence_span: str            # Verbatim source text supporting the claim
    candidate_domains: set[str]   # Multi-domain classification (always set[str], never str)
    confidence: float             # LLM extraction confidence, 0.0–1.0
    source_turn_id: str           # Turn ID tagged by harness before LLM call


@dataclass
class FirewallResult:
    decision: FirewallDecision
    reason: str                   # Machine-readable reason code
    resolved_domains: set[str]    # Final domains after RESCUE correction
