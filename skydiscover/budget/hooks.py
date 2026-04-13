from __future__ import annotations

from typing import Any

from skydiscover.budget.core import CallCostRecord, CallRole, IterationBudgetRecord


def call_record_from_response(llm_response: Any, role: CallRole, **meta) -> CallCostRecord:
    """Convert a provider response object to a call-cost record."""
    prompt_tokens = int(getattr(llm_response, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(llm_response, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(llm_response, "total_tokens", 0) or (prompt_tokens + completion_tokens))
    raw_cost = float(getattr(llm_response, "estimated_cost", 0.0) or 0.0)
    model_name = str(getattr(llm_response, "model_name", None) or "unknown")
    return CallCostRecord(
        role=role,
        model_name=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        raw_cost=raw_cost,
        meta=meta,
    )


def aggregate_tokens(record: IterationBudgetRecord) -> tuple[int, int]:
    prompt_tokens = sum(c.prompt_tokens for c in record.calls)
    completion_tokens = sum(c.completion_tokens for c in record.calls)
    return prompt_tokens, completion_tokens
