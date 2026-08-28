"""Proposing claim/evidence pairs from conversation, and letting the gate decide.

The proposer is swappable and none of them are trusted. `LMStudioProposer` runs a local model;
`EchoProposer` is a deterministic sentence-splitter used as a control in the evaluation and as the
fallback when no model is reachable, so nothing here depends on a server being up.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable

# Note the import direction: this package depends on `fireweed`, never the reverse. A test asserts
# no module under `fireweed.*` imports `fireweed_extractor`, because the moment the engine depends
# on the extractor, the extractor is inside the trust boundary and the whole argument collapses.
from fireweed.grounding import claim_faithful, classify


@dataclass(frozen=True)
class Proposal:
    """One (claim, evidence) pair a proposer suggests. Not yet admitted to anything."""
    claim: str
    evidence: str
    proposer: str


@dataclass
class ExtractionResult:
    """What a proposer produced, and what the deterministic gate did with it.

    `rejected` is not an error list. It is the record of the extractor being wrong and being
    caught, which is the behaviour this design exists to produce -- so it is returned rather than
    logged and discarded.
    """
    admitted: list[Proposal] = field(default_factory=list)
    rejected: list[tuple[Proposal, str]] = field(default_factory=list)

    @property
    def proposed(self) -> int:
        return len(self.admitted) + len(self.rejected)

    @property
    def rejection_rate(self) -> float:
        return len(self.rejected) / self.proposed if self.proposed else 0.0

    def render(self) -> str:
        out = [f"proposed {self.proposed}, admitted {len(self.admitted)}, "
               f"refused {len(self.rejected)}"]
        for p, why in self.rejected:
            out.append(f"  REFUSED ({why}) {p.claim!r}")
        return "\n".join(out)


# ── the gate ──────────────────────────────────────────────────────────────────

def _why_refused(claim: str, evidence: str) -> str | None:
    """The specific deterministic check that failed, or None if the pair is admissible.

    THESE ARE THE SERVER'S FOUR CHECKS, IN THE SERVER'S ORDER, and that is not a stylistic choice.
    The first version of this function used `claim_faithful`, whose `document_mode` argument
    defaults to False and therefore omits `predicate_grounded`. The adversarial control caught it
    immediately: `overreach` ("...joined Acme in 2019 UNDER DURESS") and `unsupported_inference`
    ("...IS SENIOR at Acme") were both admitted, because the check that rejects a claim asserting
    more than its span was never run.

    An extractor held to a weaker standard than a hand-written caller is not an untrusted client --
    it is a privileged one, and the whole safety argument evaporates. `test_extractor.py` asserts
    this list matches the server's, so the two cannot drift apart silently.
    """
    from fireweed.grounding import (subject_grounded, order_preserved, numerals_grounded,
                                    predicate_grounded)
    if not claim.strip() or not evidence.strip():
        return "empty"
    if evidence not in _SOURCE_HINT.get("text", evidence):
        return "span_not_in_source"
    if not subject_grounded(claim, evidence):
        return "unknown_subject"
    if not order_preserved(claim, evidence):
        return "relation_transposed"
    if not numerals_grounded(claim, evidence):
        return "numeral_invented"
    if not predicate_grounded(claim, evidence):
        return "asserts_more_than_evidence"
    return None


# Set only for the duration of one `extract` call, so span containment can be checked against the
# actual source text rather than trusted. A proposer that quotes text the conversation never
# contained is the most basic hallucination there is, and it must not survive.
_SOURCE_HINT: dict[str, str] = {}


# ── proposers ─────────────────────────────────────────────────────────────────

_SENT = re.compile(r"(?<=[.!?])\s+")


class EchoProposer:
    """Deterministic control: every sentence becomes a claim quoting itself.

    Trivially admissible by construction, which makes it the ceiling for admission rate and the
    floor for interestingness -- it never abstracts, resolves a pronoun, or combines two sentences.
    Its role in the evaluation is to separate "the gate is lenient" from "the model is good".
    """

    name = "echo"

    def propose(self, text: str) -> list[Proposal]:
        out = []
        for s in _SENT.split(text.strip()):
            s = s.strip()
            if len(s) >= 10:
                out.append(Proposal(claim=s, evidence=s, proposer=self.name))
        return out


_PROMPT = """Extract standalone facts from the conversation below.

Rules:
- Each fact must be a complete sentence that stands alone without the conversation.
- For each fact, quote the EXACT text from the conversation that states it. Copy it character for
  character. Do not paraphrase the quote.
- Only extract facts the text actually states. Do not infer, summarise, or add detail.
- If the text states no durable facts, return an empty list.

Return ONLY a JSON array, no other text:
[{"claim": "...", "evidence": "..."}]

Conversation:
---
%s
---
JSON:"""


class LMStudioProposer:
    """A local model behind LM Studio's OpenAI-compatible endpoint. Untrusted by design.

    Deliberately given no examples of the gate's rules beyond the prompt above, and no retry loop
    on rejection. If the model cannot produce an admissible pair, the correct outcome is a missing
    memory, not a coaxed one -- a retry loop that reshapes proposals until they pass would be
    optimising against the verifier, which is precisely how a gate stops meaning anything.
    """

    def __init__(self, model: str = "qwen/qwen3-4b-2507",
                 endpoint: str = "http://localhost:1234/v1/chat/completions",
                 timeout: float = 120.0, temperature: float = 0.0):
        self.model, self.endpoint = model, endpoint
        self.timeout, self.temperature = timeout, temperature
        self.name = f"lmstudio:{model}"

    def available(self) -> bool:
        try:
            base = self.endpoint.rsplit("/chat/", 1)[0]
            with urllib.request.urlopen(base + "/models", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def propose(self, text: str) -> list[Proposal]:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": _PROMPT % text}],
            "temperature": self.temperature,
            "max_tokens": 1200,
        }).encode()
        req = urllib.request.Request(self.endpoint, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                content = json.load(r)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
            return []
        return [Proposal(claim=str(d.get("claim", "")), evidence=str(d.get("evidence", "")),
                         proposer=self.name)
                for d in _parse_json_array(content)
                if isinstance(d, dict)]


def _parse_json_array(content: str) -> list:
    """Recover a JSON array from model output that may be wrapped in prose or a code fence.

    A malformed response yields an empty list, never an exception: the proposer failing to produce
    parseable output is an ordinary outcome with the same consequence as proposing nothing.
    """
    content = content.strip()
    if "```" in content:
        parts = content.split("```")
        content = max(parts, key=len).lstrip("json").strip()
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(content[start:end + 1])
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


# ── the entry point ───────────────────────────────────────────────────────────

def extract(text: str, proposer=None) -> ExtractionResult:
    """Propose claim/evidence pairs from `text`, then submit each to the deterministic gate.

    Returns both halves. The caller decides what to do with the admitted proposals -- this function
    writes nothing, because a package that both proposes and stores would be back inside the trust
    boundary it exists to stay outside of.
    """
    proposer = proposer or EchoProposer()
    result = ExtractionResult()
    _SOURCE_HINT["text"] = text
    try:
        for p in proposer.propose(text):
            why = _why_refused(p.claim, p.evidence)
            if why is None:
                result.admitted.append(p)
            else:
                result.rejected.append((p, why))
    finally:
        _SOURCE_HINT.clear()
    return result


def grounding_class(claim: str, evidence: str) -> str | None:
    """The provenance class the engine would record for this pair. Read-only convenience."""
    return classify(claim, evidence)
