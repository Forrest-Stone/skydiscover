from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class CallRole(str, Enum):
    """Semantic role of a search-side LLM call."""

    GENERATION = "generation"
    RETRY = "retry"
    GUIDE = "guide"


@dataclass
class BudgetConfig:
    """Centralized budget config for search-side instrumentation."""

    nominal_budget: float
    budget_protocol: str = "soft"
    strict_budget_enforcement: bool = False
    eps: float = 1e-8


@dataclass
class CallCostRecord:
    """One search-side LLM call with realized usage and cost."""

    role: CallRole
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_cost: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationBudgetRecord:
    """Aggregated budget accounting for one iteration."""

    iteration: int
    calls: List[CallCostRecord] = field(default_factory=list)

    generation_cost: float = 0.0
    retry_cost: float = 0.0
    guide_cost: float = 0.0
    iteration_cost: float = 0.0

    cumulative_cost: float = 0.0
    remaining_budget_ratio: float = 1.0

    meta: Dict[str, Any] = field(default_factory=dict)


class BudgetLedger:
    """Single source of truth for search-side budget accounting."""

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.cumulative_cost: float = 0.0
        self.records: List[IterationBudgetRecord] = []

    def start_iteration(self, iteration: int) -> IterationBudgetRecord:
        return IterationBudgetRecord(iteration=iteration)

    def add_call(self, record: IterationBudgetRecord, call: CallCostRecord) -> None:
        record.calls.append(call)

        if call.role == CallRole.GENERATION:
            record.generation_cost += call.raw_cost
        elif call.role == CallRole.RETRY:
            record.retry_cost += call.raw_cost
        elif call.role == CallRole.GUIDE:
            record.guide_cost += call.raw_cost
        else:
            raise ValueError(f"Unknown call role: {call.role}")

    def finalize_iteration(self, record: IterationBudgetRecord) -> None:
        record.iteration_cost = record.generation_cost + record.retry_cost + record.guide_cost
        self.cumulative_cost += record.iteration_cost
        record.cumulative_cost = self.cumulative_cost
        record.remaining_budget_ratio = self.remaining_ratio()
        self.records.append(record)

    def remaining_ratio(self) -> float:
        return max(
            0.0,
            1.0 - self.cumulative_cost / max(self.config.nominal_budget, self.config.eps),
        )

    def is_oob(self) -> bool:
        return self.cumulative_cost > self.config.nominal_budget

    def overshoot(self) -> float:
        return max(self.cumulative_cost - self.config.nominal_budget, 0.0)

    def summary(self) -> Dict[str, Any]:
        num_generation_calls = 0
        num_retry_calls = 0
        num_guide_calls = 0

        for record in self.records:
            for call in record.calls:
                if call.role == CallRole.GENERATION:
                    num_generation_calls += 1
                elif call.role == CallRole.RETRY:
                    num_retry_calls += 1
                elif call.role == CallRole.GUIDE:
                    num_guide_calls += 1

        return {
            "nominal_budget": self.config.nominal_budget,
            "total_cost": self.cumulative_cost,
            "oob": self.is_oob(),
            "overshoot": self.overshoot(),
            "overshoot_ratio": (
                self.overshoot() / self.config.nominal_budget
                if self.config.nominal_budget > self.config.eps
                else 0.0
            ),
            "num_iterations": len(self.records),
            "num_generation_calls": num_generation_calls,
            "num_retry_calls": num_retry_calls,
            "num_guide_calls": num_guide_calls,
        }
