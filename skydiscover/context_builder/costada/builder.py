"""CostAda context builder.

Extends the shared evolutionary guidance with budget-aware optional prompt gating.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from skydiscover.config import Config
from skydiscover.context_builder.adaevolve.builder import AdaEvolveContextBuilder
from skydiscover.search.base_database import Program


class CostAdaContextBuilder(AdaEvolveContextBuilder):
    """Context builder with CostAda budget guidance."""

    def __init__(self, config: Config):
        super().__init__(config)
        self._active_prompt_budget_mode = "standard"

    def _build_search_guidance(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any],
    ) -> str:
        prompt_mode = str(
            context.get("prompt_budget_mode", context.get("costada_tier", "standard"))
            or "standard"
        ).strip().lower()
        if prompt_mode == "cheap":
            prompt_mode = "lean"
        if prompt_mode not in {"lean", "standard", "rich"}:
            prompt_mode = "standard"
        self._active_prompt_budget_mode = prompt_mode
        base_guidance = super()._build_search_guidance(current_program, context)

        remaining_ratio = float(context.get("remaining_budget_ratio", 1.0) or 1.0)
        local_mode = str(context.get("costada_local_mode") or "").strip().lower()
        if not local_mode:
            local_mode = "exploration" if context.get("costada_explore") else "exploitation"
        preference = (
            "Prefer compact, high-yield revisions; avoid broad rewrites unless strongly justified."
            if prompt_mode == "lean"
            else "Use the richer context to consider a structurally different move, then make the smallest complete implementation."
            if prompt_mode == "rich"
            else "Budget is relatively available; balanced local refinement and broader alternatives are allowed."
        )

        budget_block = (
            "## BUDGET STATUS\n"
            f"Remaining budget ratio: {remaining_ratio:.3f}\n"
            f"Prompt budget mode: {prompt_mode}\n"
            f"Local search mode: {local_mode}\n"
            f"{preference}"
        )

        trimmed = self._apply_budget_guidance_policy(base_guidance, prompt_mode)
        if trimmed:
            return f"{budget_block}\n\n{trimmed}"
        return budget_block

    def _format_evaluator_feedback(self, parent_program: Program) -> Optional[str]:
        """Format evaluator feedback with CostAda prompt-mode truncation."""
        artifacts = getattr(parent_program, "artifacts", None)
        if not artifacts:
            return None

        feedback = artifacts.get("feedback")
        if not feedback or not isinstance(feedback, str):
            return None

        db_cfg = self.config.search.database
        mode = self._active_prompt_budget_mode
        if mode == "lean":
            max_len = int(
                getattr(
                    db_cfg,
                    "costada_lean_feedback_chars",
                    getattr(db_cfg, "costada_cheap_feedback_chars", 300),
                )
            )
        elif mode == "rich":
            max_len = int(getattr(db_cfg, "costada_rich_feedback_chars", 1500))
        else:
            max_len = int(getattr(db_cfg, "costada_standard_feedback_chars", 800))

        if len(feedback) > max_len:
            feedback = feedback[:max_len] + "\n... (truncated)"

        return (
            "## EVALUATOR FEEDBACK ON CURRENT PROGRAM\n"
            "Use this diagnostic feedback for targeted improvements:\n\n"
            f"{feedback}"
        )

    @staticmethod
    def _apply_budget_guidance_policy(guidance: str, prompt_mode: str) -> str:
        """Apply budget-conditioned guidance verbosity policy.

        lean: keep a short compressed summary
        standard: keep a moderate amount of guidance
        rich: keep full guidance
        """
        text = (guidance or "").strip()
        if not text:
            return ""
        if prompt_mode == "rich":
            return text
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if prompt_mode == "lean":
            return "\n".join(lines[: min(5, len(lines))])
        return "\n".join(lines[: min(14, len(lines))])
