"""Phase 10 — Rule-based query understanding. Pure functions, never raises."""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from .graph import GraphState

_QUERY_SKIP_WORDS = {
    "I", "The", "A", "An", "This", "That", "These", "Those",
    "It", "He", "She", "They", "We", "You", "My", "His", "Her",
    "Their", "Our", "Its", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
}

_NEGATION_RE = re.compile(
    r"\b(no longer|doesn'?t|don'?t|won'?t|isn'?t|aren'?t|wasn'?t|weren'?t"
    r"|not|never|stopped|quit|left)\b",
    re.IGNORECASE,
)

_INTENT_FIRST_WORD = {
    "what": "what", "which": "what",
    "when": "when",
    "where": "where",
    "who": "who", "whose": "who",
    "why": "why",
    "how": "how",
    "did": "bool", "does": "bool", "do": "bool",
    "is": "bool", "are": "bool", "was": "bool", "were": "bool",
    "has": "bool", "have": "bool",
}

_MONTH_MAP = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
_WEEKDAY_OFFSETS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


@dataclass
class QueryParsed:
    entity_ids: list[str]          # entity_ids in graph that matched query tokens
    entity_mentions: list[str]     # all candidate name tokens (incl. unresolved)
    temporal_window: tuple[str, str] | None  # (from_iso, to_iso), or None
    polarity: Literal["positive", "negative", "any"]
    intent: Literal["what", "when", "where", "who", "why", "how", "bool", "general"]


def parse_query(question: str, graph: GraphState,
                current_timestamp: str | None = None) -> QueryParsed:
    """Rule-based query understanding. Read-only: never creates graph entities."""
    try:
        return _parse(question, graph, current_timestamp)
    except Exception:
        return QueryParsed([], [], None, "any", "general")


def _is_name_token(clean: str) -> bool:
    return (len(clean) > 1 and clean[0].isupper()
            and clean[1:].replace("'", "").isalpha() and clean not in _QUERY_SKIP_WORDS)


def _parse(question: str, graph: GraphState, ts: str | None) -> QueryParsed:
    entity_ids: list[str] = []
    entity_mentions: list[str] = []
    tokens = [t.rstrip(".,!?\"'") for t in question.split()]
    n = len(tokens)
    i = 0
    while i < n:
        if not _is_name_token(tokens[i]):
            i += 1
            continue
        # Greedy LONGEST-match over the run of consecutive name tokens: "Theodore Roosevelt Jr" must
        # resolve to the full-name entity, not the bare-surname "Roosevelt" fragment (which cross-wires
        # distinct same-surname people). Try the longest span first; fall back to shorter, then single.
        j = i
        while j < n and _is_name_token(tokens[j]):
            j += 1
        matched_to = None
        for end in range(j, i, -1):                       # full run down to a single token
            span = " ".join(tokens[i:end])
            lookup = span[:-2] if span.endswith("'s") else span
            ent = graph.find_entity_by_name(lookup)
            if ent is not None:
                entity_ids.append(ent.entity_id)
                entity_mentions.append(span)
                matched_to = end
                break
        if matched_to is not None:
            i = matched_to                                # consume the matched span (no surname re-match)
        else:
            entity_mentions.append(tokens[i])             # keep the unresolved mention
            i += 1

    temporal_window = _detect_temporal(question, ts)
    polarity: Literal["positive", "negative", "any"] = (
        "negative" if _NEGATION_RE.search(question) else "any"
    )
    first = question.strip().split()[0].lower().rstrip("?") if question.strip() else ""
    intent: Literal["what", "when", "where", "who", "why", "how", "bool", "general"] = (
        _INTENT_FIRST_WORD.get(first, "general")
    )
    return QueryParsed(entity_ids, entity_mentions, temporal_window, polarity, intent)


def _detect_temporal(question: str, ts: str | None) -> tuple[str, str] | None:
    """Return (from_iso, to_iso) window if a temporal cue is found, else None."""
    if ts is None:
        return None
    try:
        anchor = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    lower = question.lower()
    if "yesterday" in lower:
        d = (anchor - timedelta(days=1)).date().isoformat()
        return d, d
    if "last week" in lower:
        end = (anchor - timedelta(days=1)).date().isoformat()
        start = (anchor - timedelta(days=7)).date().isoformat()
        return start, end
    if "last month" in lower:
        end = (anchor - timedelta(days=1)).date().isoformat()
        start = (anchor - timedelta(days=30)).date().isoformat()
        return start, end
    m = re.search(r"\bin\s+(" + "|".join(_MONTH_MAP.keys()) + r")\s+(\d{4})\b", lower)
    if m:
        d = f"{m.group(2)}-{_MONTH_MAP[m.group(1)]}"
        return d, d
    m = re.search(r"\bon\s+(" + "|".join(_WEEKDAY_OFFSETS.keys()) + r")\b", lower)
    if m:
        days_back = (anchor.weekday() - _WEEKDAY_OFFSETS[m.group(1)]) % 7 or 7
        d = (anchor - timedelta(days=days_back)).date().isoformat()
        return d, d
    return None
