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

1. **Step-level local control** (intensity-conditioned local sampling)
2. **Frontier-level allocation** (cost-aware reward routing)
3. **Regime-level intervention trigger**
4. **Budget-gated optional prompt components**

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
\tilde{c}_t = c_t / (\bar{c} + \epsilon_c)
\quad
\phi_t = \log(1+\tilde{c}_t)
\quad
d_t = 1+\lambda_t\phi_t
\]

\[
u_t^{(k)} = \frac{\lambda_t g_t + (1-\lambda_t)\delta_t^{(k)}}{d_t}
\]

\[
H_t^{(k)} = \alpha H_{t-1}^{(k)} + (1-\alpha)u_t^{(k)}
\]

Frontier intensity mapping used for sampling:

\[
I_t^{(k)} = I_{\min} + \frac{I_{\max}-I_{\min}}{1+\sqrt{H_{t-1}^{(k)}+\epsilon}}
\]

Local mode:

\[
\Pr(a_t=\mathrm{exploration}) = I_t^{(k)},\quad
\Pr(a_t=\mathrm{exploitation}) = 0.7(1-I_t^{(k)}),\quad
\Pr(a_t=\mathrm{balanced}) = 0.3(1-I_t^{(k)}).
\]

Cost-aware frontier reward:

\[
r_t^{(k)} = \frac{g_t}{d_t}
\]

---

## 3) File-by-file guide

### `controller.py`

Main orchestrator.

Responsibilities:

- initializes CostAda control components
- selects frontier via `CostAwareFrontierRouter`
- builds compact control state and samples the local search mode
- samples with frontier and prompt-budget context
- computes local/global gains and utility
- updates `H` and router reward
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

### `state.py`

State containers:

- `FrontierState`
- `CompactControlState`

### `database.py`

Minimal extension over `AdaEvolveDatabase` for CostAda sampling knobs:

- explicit exploration vs exploitation sampling
- context program count caps/floors by prompt budget mode
- feedback budget hints by prompt budget mode

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
- `prompt_budget_mode`
- `local_search_mode`
- `remaining_budget_ratio`
- `normalized_cost`
- `cost_denominator`
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
