"""Tier scheduler for step-level spending in CostAda."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Tuple

from skydiscover.search.costada.state import CompactControlState


class TierScheduler:
    """Contextual-UCB scheduler over {cheap, standard, rich} tiers."""

    TIERS = ("cheap", "standard", "rich")

    def __init__(self, beta: float = 0.5):
        self.beta = float(beta)
        self._counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self._totals: Dict[Tuple[str, str], float] = defaultdict(float)
        self._state_counts: Dict[str, int] = defaultdict(int)

    def bucketize_state(self, state: CompactControlState) -> str:
        """Bucketize compact state for contextual bandit lookup."""
        rho = state.remaining_budget_ratio
        if rho >= 0.66:
            rb = "hi"
        elif rho >= 0.33:
            rb = "mid"
        else:
            rb = "lo"

        if state.recent_improvement_avg > 0.02:
            ib = "gain_hi"
        elif state.recent_improvement_avg > 0.0:
            ib = "gain_mid"
        else:
            ib = "gain_lo"

        if state.stagnation_steps >= 8:
            sb = "stag_hi"
        elif state.stagnation_steps >= 3:
            sb = "stag_mid"
        else:
            sb = "stag_lo"

        hb = "H_hi" if state.frontier_signal > 0.01 else "H_lo"
        return f"{rb}|{ib}|{sb}|{hb}"

    def select(self, state: CompactControlState, feasible_tiers: Iterable[str] | None = None) -> str:
        """Select a tier via contextual UCB."""
        bucket = self.bucketize_state(state)
        tiers = [t for t in (feasible_tiers or self.TIERS) if t in self.TIERS]
        if not tiers:
            tiers = ["cheap"]

        self._state_counts[bucket] += 1
        n_state = self._state_counts[bucket]
        logn = math.log(n_state + 1.0)

        best_tier = tiers[0]
        best_score = float("-inf")
        for tier in tiers:
            key = (bucket, tier)
            n = self._counts[key]
            mean = self._totals[key] / n if n > 0 else 0.0
            bonus = self.beta * math.sqrt(logn / (n + 1.0))
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_tier = tier
        return best_tier

    def update(self, state: CompactControlState, chosen_tier: str, realized_utility: float) -> None:
        """Update contextual statistics with realized utility."""
        bucket = self.bucketize_state(state)
        key = (bucket, chosen_tier)
        self._counts[key] += 1
        self._totals[key] += float(realized_utility)
