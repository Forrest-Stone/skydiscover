# Budget Subsystem (`skydiscover.budget`)

This package is the shared search-side budget instrumentation layer used by multiple search methods (`topk`, `adaevolve`, `evox`, `costada`, ...).

It is intentionally method-agnostic: methods **read from and write to** this layer, but do not redefine budget schemas.

---

## 1) Design goals

1. **Single source of truth** for per-call and per-iteration search-side cost.
2. **Stable run artifacts** that every method can emit:
   - `iterations.jsonl`
   - `summary.json`
3. **Cross-method compatibility** for aggregation scripts.
4. **Low coupling**: methods only pass `CallRole`, response usage fields, and iteration metadata.

---

## 2) File-by-file reference

### `core.py`

Core datatypes and ledger logic:

- `BudgetConfig`
- `CallRole` (`generation` / `retry` / `guide`)
- `CallCostRecord`
- `IterationBudgetRecord`
- `BudgetLedger`

`BudgetLedger` responsibilities:

- start a new iteration record (`start_iteration`)
- add per-call cost entries (`add_call`)
- finalize an iteration (`finalize_iteration`)
- expose summary fields (`summary`, `is_oob`, `overshoot`, `remaining_ratio`)

### `hooks.py`

Bridges provider responses to budget records:

- `call_record_from_response(...)`
- `attach_call_to_iteration(...)`
- `aggregate_tokens(...)`

This is where `LLMResponse` usage/cost fields become typed budget call records.

### `io.py`

Artifact and plotting utilities:

- `write_iteration_record(...)`
- `write_summary(...)`
- `load_iterations(...)`
- `load_summary(...)`
- `plot_run_best_score_vs_cost(...)`
- `plot_run_budget_panels(...)`

`write_iteration_record` stores call-level arrays and method metadata fields when present
(e.g., `tier`, `utility`, `frontier_signal`, `router_reward`, `meta_triggered`).

### `metrics.py`

Small reusable helpers for budget metrics:

- OOB / overshoot
- best-score-at-budget
- success-at-target
- cost-to-target

### `__init__.py`

Public exports for use in controllers and method packages.

---

## 3) Standard run outputs

Each run directory should contain:

- `iterations.jsonl` (one row per iteration)
- `summary.json` (run-level totals)

When plotting dependencies are available:

- `best_score_vs_cost.png`
- `budget_report.png`

If plots are not generated, controller may write a status note (e.g., missing matplotlib).

---

## 4) Controller integration pattern

Typical usage pattern in a controller:

1. `record = budget_ledger.start_iteration(iter_idx)`
2. For each search-side LLM call:
   - build `CallCostRecord` from response
   - `budget_ledger.add_call(record, call_record)`
3. Add method metadata into `record.meta`
4. `budget_ledger.finalize_iteration(record)`
5. `write_iteration_record(...)` and `write_summary(...)`

---

## 5) Required `LLMResponse` fields

For accurate accounting, responses should provide:

- `model_name`
- `prompt_tokens`
- `completion_tokens`
- `estimated_cost`

If a provider does not expose usage, fields can fall back to zero; accounting remains structurally valid.

---

## 6) Troubleshooting

### Missing `summary.json`

- Ensure controller calls budget finalize/write path.
- Check run interruption; latest controller path refreshes summary every iteration.

### Missing plots

- Usually matplotlib unavailable in runtime.
- Check output note file indicating skip reason.

### Empty aggregation results

- Verify each run directory has both `summary.json` and `iterations.jsonl`.

