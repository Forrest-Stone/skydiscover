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
        total_generation_cost = 0.0
        total_retry_cost = 0.0
        total_guide_cost = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for record in self.records:
            total_generation_cost += float(record.generation_cost or 0.0)
            total_retry_cost += float(record.retry_cost or 0.0)
            total_guide_cost += float(record.guide_cost or 0.0)
            for call in record.calls:
                total_prompt_tokens += int(call.prompt_tokens or 0)
                total_completion_tokens += int(call.completion_tokens or 0)
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
            "generation_cost_total": total_generation_cost,
            "retry_cost_total": total_retry_cost,
            "guide_cost_total": total_guide_cost,
            "component_cost_total": (total_generation_cost + total_retry_cost + total_guide_cost),
            "prompt_tokens_total": total_prompt_tokens,
            "completion_tokens_total": total_completion_tokens,
            "input_tokens_total": total_prompt_tokens,
            "output_tokens_total": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "avg_iteration_cost": (
                self.cumulative_cost / len(self.records) if self.records else 0.0
            ),
            "generation_cost_fraction": (
                total_generation_cost / self.cumulative_cost if self.cumulative_cost > self.config.eps else 0.0
            ),
            "retry_cost_fraction": (
                total_retry_cost / self.cumulative_cost if self.cumulative_cost > self.config.eps else 0.0
            ),
            "guide_cost_fraction": (
                total_guide_cost / self.cumulative_cost if self.cumulative_cost > self.config.eps else 0.0
            ),
        }
