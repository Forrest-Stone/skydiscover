"""State containers for CostAda control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FrontierState:
    """Persistent per-frontier control state used by CostAda."""

    frontier_id: int
    local_best: float = 0.0
    H: float = 0.0
    recent_improvements: List[float] = field(default_factory=list)
    stagnation_steps: int = 0
    routing_reward: float = 0.0
    selection_count: int = 0
    last_update_iteration: int = -1


@dataclass
class CompactControlState:
    """Compact pre-step control state used by CostAda.

    This aligns to the method tuple:
      (remaining_budget_ratio, recent_improvement_avg, stagnation_steps, frontier_signal)
    """

    remaining_budget_ratio: float
    recent_improvement_avg: float
    stagnation_steps: int
    frontier_signal: float
