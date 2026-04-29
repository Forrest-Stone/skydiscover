"""Budget-calibrated utility helpers for CostAda."""

from __future__ import annotations

import math


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
    return max(0.0, min(1.0, 1.0 - float(remaining_ratio)))


def normalized_cost(step_cost: float, ref_cost: float, eps_c: float = 1e-8) -> float:
    """Compute ctilde_t = c_t / (ref_cost + eps_c)."""
    return max(float(step_cost), 0.0) / (max(float(ref_cost), 0.0) + float(eps_c))


def cost_transform(ctilde: float) -> float:
    """Compute phi_t = log(1 + ctilde_t)."""
    return math.log1p(max(float(ctilde), 0.0))


def cost_denominator(remaining_ratio: float, ctilde: float) -> float:
    """Compute d_t = 1 + lambda_t * phi_t."""
    lambda_t = budget_mix(remaining_ratio)
    return 1.0 + lambda_t * cost_transform(ctilde)


def utility(
    local_gain_value: float,
    global_gain_value: float,
    step_cost: float,
    remaining_ratio: float,
    ref_cost: float,
    eps_c: float = 1e-8,
) -> float:
    """Compute budget-calibrated utility.

    u_t = (lambda_t * g_t + (1 - lambda_t) * delta_t) / d_t
    d_t = 1 + lambda_t * log(1 + c_t / (ref_cost + eps_c))
    """
    lambda_t = budget_mix(remaining_ratio)
    numerator = (lambda_t * float(global_gain_value)) + ((1.0 - lambda_t) * float(local_gain_value))
    ctilde = normalized_cost(step_cost, ref_cost, eps_c=eps_c)
    return numerator / cost_denominator(remaining_ratio, ctilde)


def routing_reward(global_gain_value: float, denominator: float) -> float:
    """Compute the cost-calibrated frontier routing reward."""
    return max(float(global_gain_value), 0.0) / max(float(denominator), 1e-12)


def update_signal(prev_H: float, utility_value: float, alpha: float = 0.9) -> float:
    """EMA update: H_t = alpha * H_(t-1) + (1-alpha) * u_t."""
    return float(alpha) * float(prev_H) + (1.0 - float(alpha)) * float(utility_value)
