# CostAda Context Builder (`skydiscover.context_builder.costada`)

This package adapts prompt construction for CostAda while reusing the shared evolutionary prompt templates.

---

## 1) Purpose

`CostAdaContextBuilder` extends the shared prompt builder and adds budget-aware guidance without rewriting the whole prompt stack.

It allows the prompt budget gate and local search mode to be reflected in prompts.

---

## 2) Added behavior

### A) Budget status block

Injected block includes:

- remaining budget ratio
- prompt budget mode
- local search mode
- concise spending preference text

### B) Budget-conditioned verbosity

- `lean`: minimal guidance
- `standard`: compressed guidance
- `rich`: full guidance

This keeps optional prompt components aligned with the remaining budget.

---

## 3) Expected controller context keys

Controller should pass:

- `remaining_budget_ratio`
- `prompt_budget_mode`
- `costada_local_mode`
- `costada_explore`

plus the standard evolutionary context keys:

- `paradigm`
- `siblings`
- `error_context`
- `other_context_programs`

---

## 4) Design constraints

1. Reuse existing scaffold and templates.
2. Keep CostAda-specific additions small and explicit.
3. Avoid duplicating the default template pipelines.

---

## 5) Debug tips

If budget text is not visible in prompts:

1. verify `template: costada` in config
2. verify controller passes `remaining_budget_ratio` and `prompt_budget_mode`
3. confirm `default_discovery_controller` selects `CostAdaContextBuilder`
