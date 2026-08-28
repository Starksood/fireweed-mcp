"""Read auditing — who asked what, and what the gate said.

WHY THIS IS OPT-IN AND HASHED BY DEFAULT
----------------------------------------
Two reviewers disagreed about this feature and both were partly right. One called it trivial and
clearly worth adding: writes are already recorded immutably, queries are not, so "what did this
agent ask, and what did it get" is unanswerable. The other noted that recording queries stores
sensitive text and creates an attack surface that did not previously exist.

The second objection is sharper here than in an ordinary database, because this system's pitch is
"trust neither the agent nor the server." A server that quietly starts recording every question
asked of it is a worse fit for that pitch, not a better one.

So the default records what an auditor needs and not the text itself:

    always      timestamp, verdict, refusal reason, demand head, subject scope, a query FINGERPRINT
    never       the query text -- unless the operator explicitly turns it on

The fingerprint is a salted hash: identical queries are recognisable as repeats without the log
disclosing what was asked. The salt is the per-install id salt already used for entity identifiers,
so an attacker holding the log alone cannot dictionary-attack short queries back to plaintext.

WHAT THIS IS NOT
----------------
Not a ledger entry. Reads change no state, so recording them in the append-only mutation chain
would put non-events in a structure whose whole guarantee is that it replays to the live state.
This is a separate, ordinary append-only file, and it makes no cryptographic claim.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

# Off unless an operator turns it on. A privacy-relevant surface should never appear because
# somebody upgraded.
ENABLED = False
# Even when auditing is on, the query text stays out of the log unless this is also set.
RECORD_QUERY_TEXT = False


@dataclass(frozen=True)
class ReadEvent:
    at: str
    fingerprint: str          # salted hash of the normalised query
    answered: bool
    reason: str | None        # the typed abstention code, when it abstained
    demand_head: str | None
    subject: str | None
    mode: str                 # "semantic" | "lexical_only" -- a refusal means less in lexical mode
    query: str | None = None  # only when RECORD_QUERY_TEXT


def fingerprint(query: str, salt: str = "") -> str:
    """Salted, so identical queries are linkable but the log alone does not reveal them.

    Without a salt, a log of short queries is trivially reversible by dictionary attack -- the same
    reasoning that made entity identifiers salted rather than derived from names.
    """
    norm = " ".join((query or "").lower().split())
    return hashlib.sha256((salt + "\x00" + norm).encode("utf-8")).hexdigest()[:32]


def build_event(query: str, verdict, salt: str = "", subject: str | None = None) -> ReadEvent:
    """Shape a gate verdict into an audit row. Pure -- writing is the caller's decision."""
    demand = getattr(verdict, "demand", None)
    return ReadEvent(
        at=datetime.now(timezone.utc).isoformat(),
        fingerprint=fingerprint(query, salt),
        answered=not getattr(verdict, "abstain", True),
        reason=getattr(verdict, "reason", None),
        demand_head=getattr(demand, "head", None),
        subject=subject,
        mode=getattr(verdict, "mode", "unknown"),
        query=query if RECORD_QUERY_TEXT else None,
    )


def record(path: Path, event: ReadEvent) -> bool:
    """Append one row. Returns whether it was written.

    An unwritable audit log must never turn a successful read into a failure: the query already
    succeeded, and losing its audit row is the lesser harm. A deployment that needs the opposite
    guarantee needs a real audit sink, not a file.
    """
    if not ENABLED:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps({k: v for k, v in asdict(event).items() if v is not None}) + "\n")
        return True
    except OSError:
        return False


def summarise(path: Path, limit: int = 20) -> str:
    """A readable digest, for the operator who turned this on to actually use it."""
    if not path.exists():
        return ("Read auditing is off, or nothing has been asked yet.\n"
                "Enable with FIREWEED_MCP_READ_AUDIT=1. Query text is NOT recorded unless "
                "FIREWEED_MCP_READ_AUDIT_TEXT=1 is also set.")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return "Read auditing is on; no reads recorded yet."
    answered = sum(1 for r in rows if r.get("answered"))
    out = [f"{len(rows)} reads recorded — {answered} answered, {len(rows) - answered} abstained"]
    from collections import Counter
    reasons = Counter(r.get("reason") for r in rows if not r.get("answered") and r.get("reason"))
    if reasons:
        out.append("  abstention reasons: " +
                   ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    heads = Counter(r.get("demand_head") for r in rows if r.get("demand_head"))
    if heads:
        out.append("  most asked-for: " + ", ".join(f"{k} ({v})" for k, v in heads.most_common(8)))
    out.append("")
    out.append("Recent:")
    for r in rows[-limit:]:
        mark = "ANSWERED " if r.get("answered") else f"ABSTAINED({r.get('reason')})"
        q = r.get("query") or f"fp:{r['fingerprint'][:12]}"
        out.append(f"  {r['at'][:19]}  {mark:<28}{q}")
    if not any(r.get("query") for r in rows):
        out.append("")
        out.append("Query text is not recorded. Fingerprints are salted, so repeats are visible "
                   "but the log does not disclose what was asked.")
    return "\n".join(out)
