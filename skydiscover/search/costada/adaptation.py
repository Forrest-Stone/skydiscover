"""Budget-calibrated utility helpers for CostAda / BCHD."""

from __future__ import annotations


def local_gain(score_new: float, local_best_prev: float, eps: float = 1e-8) -> float:
    """Compute local normalized gain delta_t^(k)."""
    denom = max(abs(float(local_best_prev)), eps)
    return max((float(score_new) - float(local_best_prev)) / denom, 0.0)


def global_gain(score_new: float, global_best_prev: float, eps: float = 1e-8) -> float:
    """Compute global normalized gain g_t."""
    denom = max(abs(float(global_best_prev)), eps)
    return max((float(score_new) - float(global_best_prev)) / denom, 0.0)


def budget_mix(remaining_ratio: float) -> float:
    """Compute lambda_t = 1 - rho_t."""
    return 1.0 - float(remaining_ratio)


def utility(
    local_gain_value: float,
    global_gain_value: float,
    raw_iteration_cost: float,
    remaining_ratio: float,
) -> float:
    """Compute budget-calibrated utility.

    u_t = (lambda_t * g_t + (1 - lambda_t) * delta_t) / (1 + c_t)
    """
    lambda_t = budget_mix(remaining_ratio)
    numerator = (lambda_t * float(global_gain_value)) + ((1.0 - lambda_t) * float(local_gain_value))
    return numerator / (1.0 + max(float(raw_iteration_cost), 0.0))


def update_signal(prev_H: float, utility_value: float, alpha: float = 0.9) -> float:
    """EMA update: H_t = alpha * H_(t-1) + (1-alpha) * u_t."""
    return float(alpha) * float(prev_H) + (1.0 - float(alpha)) * float(utility_value)
