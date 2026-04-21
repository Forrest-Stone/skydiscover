"""BudgetEvolve adaptation helpers.

Keeps AdaEvolve's original adaptive core untouched and adds only budget-aware
state/action utilities.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Reuse original AdaEvolve adaptive core (G signal + UCB island routing)
from skydiscover.search.adaevolve.adaptation import AdaptiveState, MultiDimensionalAdapter


@dataclass
class UsageRecord:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    raw_usage: Optional[Dict[str, Any]] = None


class BudgetLedger:
    """Tracks spent token budget and spent monetary budget."""

    def __init__(
        self,
        total_budget: int,
        strict_stop: bool = True,
        cost_budget_total: float = 0.0,
        input_token_cost: float = 0.0,
        output_token_cost: float = 0.0,
    ):
        self.total_budget = max(0, int(total_budget))
        self.strict_stop = bool(strict_stop)
        self.cost_budget_total = max(0.0, float(cost_budget_total))
        self.input_token_cost = max(0.0, float(input_token_cost))
        self.output_token_cost = max(0.0, float(output_token_cost))
        self.spent_tokens = 0
        self.spent_cost = 0.0

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.total_budget - self.spent_tokens)

    @property
    def remaining_ratio(self) -> float:
        if self.total_budget <= 0:
            return 0.0
        return max(0.0, 1.0 - self.spent_tokens / self.total_budget)

    @property
    def remaining_cost(self) -> float:
        if self.cost_budget_total <= 0:
            return float("inf")
        return max(0.0, self.cost_budget_total - self.spent_cost)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.input_token_cost + output_tokens * self.output_token_cost

    def feasible(self, est_input_tokens: int, reserve_output_tokens: int) -> bool:
        token_ok = self.spent_tokens + est_input_tokens + \
            reserve_output_tokens <= self.total_budget
        if self.cost_budget_total <= 0:
            return token_ok
        return self.spent_cost + self.estimate_cost(est_input_tokens, reserve_output_tokens) <= self.cost_budget_total

    def update(self, usage: UsageRecord) -> None:
        total = usage.total_tokens or (
            usage.input_tokens + usage.output_tokens)
        self.spent_tokens += max(0, int(total))
        self.spent_cost += self.estimate_cost(
            usage.input_tokens, usage.output_tokens)


@dataclass
class BudgetState:
    island_id: int
    intensity: float
    remaining_ratio: float
    spent_tokens: int
    spent_cost: float
    recent_frontier_gain_ma: float
    no_improve_steps: int
    progress_regime: str
    budget_bin: str
    burn_rate: float


@dataclass(frozen=True)
class BudgetAction:
    family: str
    tier: str
    max_output_tokens: int


class BudgetStateBuilder:
    def __init__(self, config):
        self.config = config

    def _budget_bin(self, remaining_ratio: float) -> str:
        bins = list(getattr(self.config.search.database,
                    "budget_bins", [0.2, 0.5, 0.8]))
        while len(bins) < 3:
            bins.append(1.0)
        b1, b2, b3 = bins[:3]
        if remaining_ratio <= b1:
            return "low"
        if remaining_ratio <= b2:
            return "mid_low"
        if remaining_ratio <= b3:
            return "mid_high"
        return "high"

    def _progress_regime(self, recent_gain_ma: float, no_improve_steps: int) -> str:
        eps = getattr(self.config.search.database,
                      "budget_significant_gain_eps", 1e-6)
        if recent_gain_ma > 10 * eps:
            return "breakthrough"
        if no_improve_steps > 0 and recent_gain_ma <= eps:
            return "stagnant"
        return "slow"

    def build(
        self,
        island_id: int,
        intensity: float,
        ledger: BudgetLedger,
        recent_gain_ma: float,
        no_improve_steps: int,
        burn_rate: float,
    ) -> BudgetState:
        rr = ledger.remaining_ratio
        return BudgetState(
            island_id=island_id,
            intensity=float(intensity),
            remaining_ratio=rr,
            spent_tokens=ledger.spent_tokens,
            spent_cost=ledger.spent_cost,
            recent_frontier_gain_ma=recent_gain_ma,
            no_improve_steps=no_improve_steps,
            progress_regime=self._progress_regime(
                recent_gain_ma, no_improve_steps),
            budget_bin=self._budget_bin(rr),
            burn_rate=burn_rate,
        )


class BudgetActionScheduler:
    """Simple contextual bandit over (family, tier)."""

    def __init__(self, config):
        self.config = config
        self.stats = defaultdict(
            lambda: {"n": 0, "gain_sum": 0.0, "cost_sum": 0.0})
        db = config.search.database
        self.actions = [
            BudgetAction("refine", "cheap", db.cheap_max_output_tokens),
            BudgetAction("refine", "standard", db.standard_max_output_tokens),
            BudgetAction("refine", "rich", db.rich_max_output_tokens),
            BudgetAction("structural", "cheap", db.cheap_max_output_tokens),
            BudgetAction("structural", "standard",
                         db.standard_max_output_tokens),
            BudgetAction("structural", "rich", db.rich_max_output_tokens),
            BudgetAction("tactic_guided", "cheap", db.cheap_max_output_tokens),
            BudgetAction("tactic_guided", "standard",
                         db.standard_max_output_tokens),
            BudgetAction("tactic_guided", "rich", db.rich_max_output_tokens),
        ]

    def _bucket_key(self, state: BudgetState) -> Tuple[str, str]:
        return (state.budget_bin, state.progress_regime)

    def select(self, state: BudgetState) -> BudgetAction:
        key = self._bucket_key(state)
        total_n = sum(self.stats[(key, a)]["n"] for a in self.actions) + 1
        lam = float(getattr(self.config.search.database, "budget_lambda", 1e-4))
        beta = float(
            getattr(self.config.search.database, "budget_ucb_beta", 0.5))
        best_a, best_u = self.actions[0], float("-inf")

        for action in self.actions:
            s = self.stats[(key, action)]
            n = s["n"]
            gain = s["gain_sum"] / max(1, n)
            cost = s["cost_sum"] / max(1, n)
            ucb = beta * math.sqrt(math.log(total_n) / (n + 1))
            utility = gain - lam * cost + ucb
            if utility > best_u:
                best_u, best_a = utility, action
        return best_a

    def update(self, state: BudgetState, action: BudgetAction, frontier_gain: float, cost: int) -> None:
        key = self._bucket_key(state)
        s = self.stats[(key, action)]
        s["n"] += 1
        s["gain_sum"] += float(frontier_gain)
        s["cost_sum"] += float(max(0, cost))
