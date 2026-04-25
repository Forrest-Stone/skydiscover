from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

_EXCLUDE_KEYS = {
    "combined_score",
    "score",
    "validity",
    "eval_time",
    "timeout",
    "error",
    "error_message",
    "target",
    "ratio",
}


@dataclass
class ObjectiveSnapshot:
    objective_key: Optional[str] = None
    objective_value: Optional[float] = None
    target_value: Optional[float] = None
    target_ratio: Optional[float] = None
    combined_score: Optional[float] = None
    validity: Optional[float] = None


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def resolve_objective_from_metrics(metrics: Dict[str, Any] | None) -> ObjectiveSnapshot:
    metrics = metrics or {}
    combined = _to_float(metrics.get("combined_score", metrics.get("score")))
    target_value = _to_float(metrics.get("target"))
    target_ratio = _to_float(metrics.get("ratio"))
    validity = _to_float(metrics.get("validity"))

    objective_key = None
    objective_value = None
    for key, value in metrics.items():
        if key in _EXCLUDE_KEYS:
            continue
        parsed = _to_float(value)
        if parsed is not None:
            objective_key = str(key)
            objective_value = parsed
            break

    if objective_key is None:
        objective_key = "combined_score" if combined is not None else None
        objective_value = combined

    return ObjectiveSnapshot(
        objective_key=objective_key,
        objective_value=objective_value,
        target_value=target_value,
        target_ratio=target_ratio,
        combined_score=combined,
        validity=validity,
    )
