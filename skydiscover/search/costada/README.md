# CostAda / BCHD (`skydiscover.search.costada`)

This directory contains the implementation of **Budget-Calibrated Hierarchical Discovery (BCHD)** as `search.type: costada`.

CostAda is parallel to other search methods (`adaevolve`, `evox`) and reuses the shared search scaffold + shared budget subsystem.

---

## 1) What CostAda changes vs AdaEvolve

CostAda keeps:

- frontier/island evolutionary scaffold
- generation + retry + evaluation pipeline
- paradigm/guide mechanics
- shared budget ledger (`skydiscover.budget`)

CostAda replaces the adaptation signal with budget-calibrated control and uses it to drive:

1. **Step-level spending** (`cheap`, `standard`, `rich`)
2. **Frontier-level allocation** (cost-aware reward routing)
3. **Regime-level intervention trigger**

---

## 2) Core equations

Given candidate score `score_new = f'_t`, local best `b_{t-1}^{(k)}`, global best `y_{t-1}`, remaining budget ratio `rho_t`, and realized iteration cost `c_t`:

\[
\delta_t^{(k)} = \max\left(\frac{f'_t - b_{t-1}^{(k)}}{\max(|b_{t-1}^{(k)}|,\epsilon)},0\right)
\]

\[
g_t = \max\left(\frac{f'_t - y_{t-1}}{\max(|y_{t-1}|,\epsilon)},0\right)
\]

\[
\lambda_t = 1 - \rho_t
\]

\[
u_t^{(k)} = \frac{\lambda_t g_t + (1-\lambda_t)\delta_t^{(k)}}{1+c_t}
\]

\[
H_t^{(k)} = \alpha H_{t-1}^{(k)} + (1-\alpha)u_t^{(k)}
\]

Frontier intensity mapping used for sampling:

\[
I_t^{(k)} = I_{\min} + \frac{I_{\max}-I_{\min}}{1+\sqrt{H_t^{(k)}+\epsilon}}
\]

Cost-aware frontier reward:

\[
r_t^{(k)} = \frac{\max(f'_t-y_{t-1},0)}{1+c_t}
\]

---

## 3) File-by-file guide

### `controller.py`

Main orchestrator.

Responsibilities:

- initializes CostAda control components
- selects frontier via `CostAwareFrontierRouter`
- builds compact control state and selects tier via `TierScheduler`
- samples with frontier/tier context
- computes local/global gains and utility
- updates `H`, router reward, scheduler stats
- writes method metadata into budget iteration rows

### `adaptation.py`

Pure formula helpers:

- `local_gain`
- `global_gain`
- `budget_mix`
- `utility`
- `update_signal`

No ledger logic here by design.

### `router.py`

Cost-aware UCB routing across frontiers.

- decayed reward state
- visit counts
- optimism bonus

### `tier_scheduler.py`

Deterministic step-level adaptation policy over tiers.

- `compute_intensity`
- `base_tier_from_intensity`
- `apply_budget_override`
- `select`
- `update` (compatibility no-op)

### `state.py`

State containers:

- `FrontierState`
- `CompactControlState`

### `database.py`

Minimal extension over `AdaEvolveDatabase` for tier-aware sampling knobs:

- context program count caps/floors by tier
- feedback budget hints by tier

### `__init__.py`

Public exports (`CostAdaController`, `CostAdaDatabase`).

---

## 4) Runtime integration points

- `search/route.py` registers `costada` database and controller.
- `cli.py` accepts `--search costada`.
- `default_discovery_controller.py` supports `template == "costada"` context builder path.

---

## 5) Budget artifacts emitted by CostAda

CostAda writes the same base artifacts as other methods:

- `iterations.jsonl`
- `summary.json`

and fills CostAda-specific iteration fields (when available):

- `frontier_id`
- `tier`
- `remaining_budget_ratio`
- `local_gain`
- `global_gain`
- `utility`
- `frontier_signal`
- `routing_reward` (and backward-compatible alias `router_reward`)
- `meta_triggered`

These can be aggregated by `scripts/plot_budget_curves.py`.

---

## 6) Typical command

```bash
uv run skydiscover-run \
  benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config.yaml \
  --search costada \
  --model openrouter/z-ai/glm-5
```

---

## 7) Validation checklist for a finished run

In the output directory, verify:

1. `iterations.jsonl` exists and has one row per iteration
2. `summary.json` exists and has `total_cost`, `oob`, `overshoot`
3. if matplotlib available: `best_score_vs_cost.png`, `budget_report.png`
4. if plot missing: check `budget_plot_status.txt`
