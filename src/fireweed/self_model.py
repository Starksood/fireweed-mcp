"""Recursive self-modeling (Stage 3, W5) — predict, observe, calibrate.

A field-like self does not just hold facts; it forms EXPECTATIONS and learns how reliable they
are. W5 closes that loop:

  1. PREDICT   — a forward-looking claim ("I'll probably finish the report Friday") becomes a
                 prediction, with a probability read from its hedge word (probably -> 0.65).
  2. OBSERVE   — a later claim about the same subject ("I finished the report") RESOLVES the
                 prediction: confirmed (actual=1.0) or contradicted (actual=0.0).
  3. CALIBRATE — a Brier-style score over resolved predictions measures whether the self's
                 confidence matched reality. A well-calibrated self can be trusted about itself.

This is the recursive turn: the system models not just the world but its own predictive
accuracy. Like significance/decay it lives in a side-table (dict[node_id -> Prediction]); no
Node schema change. It is deterministic — the LLM proposes the claim, but CODE decides what is
a prediction (hedge/future markers) and whether it came true (subject + content + polarity).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .significance import _content_tokens
from .constants import PREDICTION_RESOLVE_MIN_OVERLAP

# Hedge / intention markers -> a probability the SELF is asserting. Strongest match wins.
_STRONG = ("will ", "i'll ", "definitely", "certainly", "for sure", "going to ", "gonna ", "is going to")
_MEDIUM = ("probably", "likely", "expect", "expecting", "plan to", "planning to", "intend", "should ")
_WEAK = ("might ", "maybe", "possibly", "may ", "hope ", "hoping", "could ", "thinking of", "thinking about")
_STRONG_P, _MEDIUM_P, _WEAK_P = 0.85, 0.65, 0.4

_NEGATION = {"not", "no", "never", "didn't", "don't", "won't", "can't", "cannot",
             "isn't", "aren't", "wasn't", "weren't", "failed", "couldn't", "wouldn't"}


def predicted_probability(claim: str) -> float | None:
    """The probability a claim asserts about a future outcome, or None if it is not a prediction.
    Read from the hedge/intention marker (strongest wins). CODE decides what counts — a bare
    past/present fact ('Maya is a nurse') is not a prediction."""
    # Most-hedged marker wins: "will probably" is a 0.65, not a 0.85 — the hedge qualifies the
    # certainty. So check weak -> medium -> strong, not the reverse.
    c = " " + claim.lower() + " "
    if any(m in c for m in _WEAK):
        return _WEAK_P
    if any(m in c for m in _MEDIUM):
        return _MEDIUM_P
    if any(m in c for m in _STRONG):
        return _STRONG_P
    return None


def _is_negated(claim: str) -> bool:
    return bool(_content_tokens_with_neg(claim) & _NEGATION)


def _content_tokens_with_neg(text: str) -> set[str]:
    # negation words are short (<=3) so the len>2 tokenizer would drop some; tokenize plainly here
    return set(text.lower().replace(".", " ").replace(",", " ").split())


@dataclass
class Prediction:
    """A forward-looking claim and how it turned out. Kept in a side-table (not on the Node)."""
    node_id: str
    claim: str
    predicted_prob: float
    subject: str                       # primary entity_id the prediction is about ("" if none)
    made_turn: int
    resolved: bool = False
    actual: float | None = None        # 1.0 confirmed / 0.0 contradicted
    resolved_turn: int | None = None
    resolved_by: str | None = None     # node_id of the observation that resolved it

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id, "claim": self.claim,
            "predicted_prob": self.predicted_prob, "subject": self.subject,
            "made_turn": self.made_turn, "resolved": self.resolved,
            "actual": self.actual, "resolved_turn": self.resolved_turn,
            "resolved_by": self.resolved_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Prediction":
        return cls(node_id=d["node_id"], claim=d["claim"], predicted_prob=float(d["predicted_prob"]),
                   subject=d.get("subject", ""), made_turn=int(d.get("made_turn", 0)),
                   resolved=bool(d.get("resolved", False)),
                   actual=(None if d.get("actual") is None else float(d["actual"])),
                   resolved_turn=d.get("resolved_turn"), resolved_by=d.get("resolved_by"))


def _overlap(a: str, b: str) -> float:
    """Content-token overlap of b's tokens covered by a (asymmetric toward the observation):
    how much of the observation's content the prediction already contained."""
    ta, tb = _content_tokens(a), _content_tokens(b)
    return 0.0 if not tb else len(ta & tb) / len(tb)


def record_prediction(store: dict[str, "Prediction"], node_id: str, claim: str,
                      subject: str, turn: int) -> Prediction | None:
    """If `claim` is a prediction, add it to the store keyed by node_id. Returns it or None."""
    prob = predicted_probability(claim)
    if prob is None:
        return None
    p = Prediction(node_id=node_id, claim=claim, predicted_prob=prob, subject=subject, made_turn=turn)
    store[node_id] = p
    return p


def resolve_with(store: dict[str, "Prediction"], observation_claim: str, subject: str,
                 turn: int, observation_node_id: str = "") -> list[str]:
    """Resolve open predictions that this observation bears on. A prediction resolves when the
    observation shares its subject and enough content (>= PREDICTION_RESOLVE_MIN_OVERLAP), is
    NOT itself a prediction, and is not the prediction's own node. actual = 0.0 if the
    observation's negation polarity flips the prediction's, else 1.0. Returns resolved ids."""
    if predicted_probability(observation_claim) is not None:
        return []  # an observation, not another forecast
    resolved: list[str] = []
    obs_neg = _is_negated(observation_claim)
    for pid, p in store.items():
        if p.resolved or pid == observation_node_id:
            continue
        if p.subject and subject and p.subject != subject:
            continue
        if _overlap(p.claim, observation_claim) < PREDICTION_RESOLVE_MIN_OVERLAP:
            continue
        p.resolved = True
        p.actual = 0.0 if (obs_neg != _is_negated(p.claim)) else 1.0
        p.resolved_turn = turn
        p.resolved_by = observation_node_id
        resolved.append(pid)
    return resolved


def calibration_report(store: dict[str, "Prediction"]) -> dict:
    """Brier-style calibration over resolved predictions. brier = mean((prob - actual)^2),
    lower is better-calibrated; accuracy = mean(actual). None when nothing has resolved yet."""
    preds = list(store.values())
    resolved = [p for p in preds if p.resolved and p.actual is not None]
    n_res = len(resolved)
    brier = (sum((p.predicted_prob - p.actual) ** 2 for p in resolved) / n_res) if n_res else None
    accuracy = (sum(p.actual for p in resolved) / n_res) if n_res else None
    mean_conf = (sum(p.predicted_prob for p in resolved) / n_res) if n_res else None
    return {
        "n_predictions": len(preds),
        "n_resolved": n_res,
        "brier": (round(brier, 4) if brier is not None else None),
        "accuracy": (round(accuracy, 4) if accuracy is not None else None),
        "mean_confidence": (round(mean_conf, 4) if mean_conf is not None else None),
        "well_calibrated": (brier is not None and brier <= 0.25),  # 0.25 = an uninformed 0.5 guess
    }


def store_as_dict(store: dict[str, "Prediction"]) -> dict:
    return {nid: p.as_dict() for nid, p in store.items()}


def store_from_dict(d: dict) -> dict[str, "Prediction"]:
    return {nid: Prediction.from_dict(pd) for nid, pd in (d or {}).items()}
