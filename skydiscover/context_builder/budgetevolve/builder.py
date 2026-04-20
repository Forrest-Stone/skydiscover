"""BudgetEvolve context builder.

Small extension over AdaEvolve context builder: add budget status and tier-aware
context truncation.
"""

from __future__ import annotations

from typing import Any, Dict, Union

from skydiscover.context_builder.adaevolve.builder import AdaEvolveContextBuilder
from skydiscover.search.base_database import Program


class BudgetEvolveContextBuilder(AdaEvolveContextBuilder):
    def _format_budget_guidance(self, context: Dict[str, Any]) -> str:
        if not context.get("budget_enabled"):
            return ""
        return (
            "## BUDGET STATUS\n"
            f"Used tokens: {context.get('spent_tokens', 0)}\n"
            f"Spent cost: {float(context.get('spent_cost', 0.0)):.6f}\n"
            f"Remaining budget ratio: {float(context.get('remaining_budget_ratio', 0.0)):.3f}\n"
            f"Budget stage: {context.get('budget_bin', 'unknown')}\n"
            f"Current search family: {context.get('budget_action_family', 'unknown')}\n"
            f"Current compute tier: {context.get('budget_action_tier', 'unknown')}\n"
            "Use remaining budget carefully; maximize expected gain per cost."
        )

    def _truncate_context_by_tier(self, context: Dict[str, Any]) -> Dict[str, Any]:
        tier = context.get("budget_action_tier", "standard")
        other_context = context.get("other_context_programs")
        if isinstance(other_context, dict) and "" in other_context:
            values = list(other_context.get("", []))
            if tier == "cheap":
                other_context[""] = values[:1]
            elif tier == "standard":
                other_context[""] = values[:2]
            else:
                other_context[""] = values[:3]
            context["other_context_programs"] = other_context
        return context

    def _build_search_guidance(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any],
    ) -> str:
        context = self._truncate_context_by_tier(context)
        base_guidance = super()._build_search_guidance(current_program, context)
        budget_guidance = self._format_budget_guidance(context)
        if not budget_guidance:
            return base_guidance
        if not base_guidance:
            return budget_guidance
        return f"{base_guidance}\n\n{budget_guidance}"
