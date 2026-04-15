# CostAda / BCHD Search

This directory implements **Budget-Calibrated Hierarchical Discovery (BCHD)** as a first-class search method (`search.type: costada`) parallel to methods such as `adaevolve` and `evox`.

## 1) Goal

CostAda keeps the existing frontier-based discovery scaffold, but replaces progress-only adaptation with a **budget-calibrated utility signal** that coordinates:

1. **Step-level spending** (`cheap` / `standard` / `rich` tiers)
2. **Frontier-level allocation** (cost-aware routing)
3. **Regime-level intervention** (meta trigger under stagnation + low utility)

---

## 2) Core equations (implemented)

Let `score_new = f'_t`, frontier local best `b_{t-1}^{(k)}`, global best `y_{t-1}`, remaining budget ratio `rho_t`, and realized raw iteration cost `c_t`.

- Local gain

  \[
  \delta_t^{(k)} = \max\left(\frac{f'_t - b_{t-1}^{(k)}}{\max(|b_{t-1}^{(k)}|,\epsilon)},0\right)
  \]

- Global gain

  \[
  g_t = \max\left(\frac{f'_t - y_{t-1}}{\max(|y_{t-1}|,\epsilon)},0\right)
  \]

- Budget mix

  \[
  \lambda_t = 1 - \rho_t
  \]

- Budget-calibrated utility

  \[
  u_t^{(k)} = \frac{\lambda_t g_t + (1-\lambda_t)\delta_t^{(k)}}{1+c_t}
  \]

- Frontier signal (EMA)

  \[
  H_t^{(k)} = \alpha H_{t-1}^{(k)} + (1-\alpha)u_t^{(k)}
  \]

---

## 3) File map

- `adaptation.py`
  - Formula helpers (`local_gain`, `global_gain`, `utility`, `update_signal`).
- `state.py`
  - `FrontierState`, `CompactControlState`.
- `router.py`
  - UCB frontier routing with cost-aware reward.
- `tier_scheduler.py`
  - Contextual-UCB tier selection (`cheap/standard/rich`).
- `database.py`
  - Minimal tier-aware extension over `AdaEvolveDatabase` (context depth/feedback budget knobs).
- `controller.py`
  - Main orchestrator; reuses AdaEvolve scaffold and shared budget logging.

---

## 4) Controller flow (per iteration)

1. Select frontier with cost-aware router.
2. Build compact state and select tier.
3. Sample parent/context from selected frontier with tier-conditioned settings.
4. Build prompt via `CostAdaContextBuilder` and run LLM call(s).
5. Evaluate candidate.
6. Read realized iteration cost from shared budget record.
7. Update gains, utility, frontier signal `H`.
8. Update router reward and tier scheduler stats.
9. Compute regime trigger (`stagnant && rho > eta_min && recent_H < tau_H`).
10. Persist iteration trace and summary through the shared budget subsystem.

---

## 5) Consistency notes vs method text

- ✅ Utility equations match BCHD equations.
- ✅ Frontier reward is cost-aware: `max(score_new - global_best_prev, 0)/(1+cost)`.
- ✅ Tier scheduler is contextual-UCB (`cheap/standard/rich`).
- ✅ Prompt side supports tier-conditioned verbosity (via context builder).
- ✅ Frontier intensity now depends on frontier signal `H`:

  \[
  I_t = I_{min} + \frac{I_{max}-I_{min}}{1+\sqrt{H_t+\epsilon}}
  \]

- ⚠️ Hard-budget tier masking is not enforced by default (soft-budget setting).

---

## 6) How to run

Typical command:

```bash
uv run skydiscover-run \
  benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config.yaml \
  --search costada \
  --model openrouter/z-ai/glm-5
```

You can also start from `configs/costada.yaml` and override fields per benchmark.

---

## 7) Outputs (phase-1 compatible + costada fields)

Each run still writes:

- `iterations.jsonl`
- `summary.json`

CostAda records additional iteration-level fields used by analysis:

- `frontier_id`
- `tier`
- `remaining_budget_ratio`
- `local_gain`
- `global_gain`
- `utility`
- `frontier_signal`
- `router_reward`
- `meta_triggered`

These are consumed by `scripts/plot_budget_curves.py` for metrics/plots.
