"""Cost-aware frontier routing for CostAda."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable


class CostAwareFrontierRouter:
    """UCB frontier router driven by cost-calibrated global reward."""

    def __init__(self, c_ucb: float = 0.5, gamma: float = 0.9, beta: float | None = None):
        self.c_ucb = float(c_ucb if beta is None else beta)
        self.gamma = float(gamma)
        self.rewards: Dict[int, float] = defaultdict(float)
        self.visits: Dict[int, int] = defaultdict(int)
        self.total_steps: int = 0

    def select(self, frontier_ids: Iterable[int]) -> int:
        """Select a frontier with optimism bonus (UCB)."""
        frontier_list = list(frontier_ids)
        if not frontier_list:
            raise ValueError("frontier_ids must not be empty")

        self.total_steps += 1
        log_n = math.log(max(self.total_steps, 1) + 1.0)
        best_frontier = frontier_list[0]
        best_score = float("-inf")
        for fid in frontier_list:
            r = float(self.rewards[fid])
            v = int(self.visits[fid])
            bonus = self.c_ucb * math.sqrt(log_n / (v + 1.0))
            score = r + bonus
            if score > best_score:
                best_score = score
                best_frontier = fid

        self.visits[best_frontier] += 1
        return best_frontier

    def update(
        self,
        frontier_id: int,
        routing_reward_value: float,
    ) -> float:
        """Update frontier reward with the realized cost-calibrated reward."""
        reward = max(float(routing_reward_value), 0.0)
        old = float(self.rewards[frontier_id])
        self.rewards[frontier_id] = self.gamma * old + (1.0 - self.gamma) * reward
        return reward

    def get_reward(self, frontier_id: int) -> float:
        """Return the current decayed reward estimate for a frontier."""
        return float(self.rewards[frontier_id])
