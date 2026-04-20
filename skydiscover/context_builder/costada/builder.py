"""CostAda context builder.

Extends AdaEvolve guidance with a budget-aware block and tier-conditioned
verbosity control for guide/tactic content.
"""

from __future__ import annotations

from typing import Any, Dict, Union

from skydiscover.config import Config
from skydiscover.context_builder.adaevolve.builder import AdaEvolveContextBuilder
from skydiscover.search.base_database import Program


class CostAdaContextBuilder(AdaEvolveContextBuilder):
    """Context builder with BCHD-specific budget/tier guidance."""

    def __init__(self, config: Config):
        super().__init__(config)

    def _build_search_guidance(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any],
    ) -> str:
        base_guidance = super()._build_search_guidance(current_program, context)

        remaining_ratio = float(context.get("remaining_budget_ratio", 1.0) or 1.0)
        tier = str(context.get("costada_tier", "standard") or "standard")
        preference = (
            "Prefer compact, high-yield revisions; avoid broad rewrites unless strongly justified."
            if remaining_ratio < 0.5
            else "Budget is relatively available; balanced local refinement and broader alternatives are allowed."
        )

        budget_block = (
            "## BUDGET STATUS\n"
            f"Remaining budget ratio: {remaining_ratio:.3f}\n"
            f"Current spending tier: {tier}\n"
            f"{preference}"
        )

        trimmed = self._apply_tier_guidance_policy(base_guidance, tier)
        if trimmed:
            return f"{budget_block}\n\n{trimmed}"
        return budget_block

    @staticmethod
    def _apply_tier_guidance_policy(guidance: str, tier: str) -> str:
        """Apply tier-conditioned guidance verbosity policy.

        cheap: keep at most one short guidance line
        standard: keep first two sections (compressed)
        rich: keep full guidance
        """
        text = (guidance or "").strip()
        if not text:
            return ""
        if tier == "rich":
            return text
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if tier == "cheap":
            return lines[0] if lines else ""
        return "\n".join(lines[: min(8, len(lines))])
