"""BudgetEvolve paradigm generator wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from skydiscover.search.adaevolve.paradigm.generator import ParadigmGenerator as BaseParadigmGenerator


class ParadigmGenerator(BaseParadigmGenerator):
    """Adds lightweight `mode` control for cheaper summary paradigms."""

    async def generate(
        self,
        current_program_solution: str,
        current_best_score: float,
        previously_tried_ideas: Optional[List[str]] = None,
        evaluator_feedback: Optional[str] = None,
        mode: str = "full",
    ) -> List[Dict[str, Any]]:
        paradigms = await super().generate(
            current_program_solution=current_program_solution,
            current_best_score=current_best_score,
            previously_tried_ideas=previously_tried_ideas,
            evaluator_feedback=evaluator_feedback,
        )
        if mode != "summary":
            return paradigms

        return [
            {
                "idea": p.get("idea", "")[:180],
                "description": p.get("description", "")[:320],
                "what_to_optimize": p.get("what_to_optimize", ""),
                "cautions": p.get("cautions", "")[:200],
                "approach_type": p.get("approach_type", "summary"),
            }
            for p in paradigms[:3]
        ]
