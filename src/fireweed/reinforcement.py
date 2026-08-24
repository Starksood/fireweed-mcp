"""Reinforcement formula and layer classification.

Ported from v15: memory_loop.py, novel_domain_boost.py
Extracted as pure functions with no side effects.
"""
from __future__ import annotations
from .constants import (
    LOCAL_FREQUENCY_WEIGHT,
    CROSS_SESSION_WEIGHT,
    COLD_THRESHOLD,
    WARM_THRESHOLD,
    HOT_THRESHOLD,
    CORE_THRESHOLD,
)


def compute_reinforcement(local_frequency: float, cross_session_recurrence: float) -> float:
    """Compute overall reinforcement score.
    
    Formula: r = 0.3 * local_frequency + 0.7 * cross_session_recurrence
    
    The 0.3/0.7 weighting was tuned empirically and validated across 40+ experiments.
    The direction (cross-session > within-session) is supported by spacing effect
    literature in cognitive science.
    
    Args:
        local_frequency: Within-session frequency (0.0 to 1.0)
        cross_session_recurrence: Cross-session recurrence (0.0 to 1.0)
    
    Returns:
        Overall reinforcement score (0.0 to 1.0)
    """
    return LOCAL_FREQUENCY_WEIGHT * local_frequency + CROSS_SESSION_WEIGHT * cross_session_recurrence


def get_layer(r: float) -> str:
    """Classify a reinforcement score into a memory layer.
    
    Layers:
        COLD: r < 0.40 (candidate for pruning)
        WARM: 0.40 ≤ r < 0.50 (new facts, recently created)
        HOT:  0.50 ≤ r < 0.87 (well-reinforced, synthesis-eligible)
        CORE: r ≥ 0.87 (permanent identity facts)
    
    Args:
        r: Reinforcement score
    
    Returns:
        Layer name: "COLD", "WARM", "HOT", or "CORE"
    """
    if r >= CORE_THRESHOLD:
        return "CORE"
    elif r >= HOT_THRESHOLD:
        return "HOT"
    elif r >= WARM_THRESHOLD:
        return "WARM"
    else:
        return "COLD"


def is_synthesis_eligible(r: float) -> bool:
    """Check if a node is eligible for synthesis (HOT or CORE).
    
    Synthesis only runs on nodes with r >= 0.50 to avoid operating on
    weak, unconfirmed facts.
    
    Args:
        r: Reinforcement score
    
    Returns:
        True if r >= HOT_THRESHOLD (0.50)
    """
    return r >= HOT_THRESHOLD
