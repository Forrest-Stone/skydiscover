# CostAda Context Builder (`skydiscover.context_builder.costada`)

This package adapts prompt construction for BCHD/CostAda while reusing AdaEvolve templates.

---

## 1) Purpose

`CostAdaContextBuilder` extends `AdaEvolveContextBuilder` and adds budget/tier-aware guidance without rewriting the whole prompt stack.

It allows step-level spending decisions to be reflected in prompts.

---

## 2) Added behavior

### A) Budget status block

Injected block includes:

- remaining budget ratio
- current spending tier
- concise spending preference text

### B) Tier-conditioned verbosity

- `cheap`: minimal guidance (one-line style)
- `standard`: compressed guidance
- `rich`: full guidance

This keeps prompt verbosity aligned with compute tier.

---

## 3) Expected controller context keys

Controller should pass:

- `remaining_budget_ratio`
- `costada_tier`

plus standard AdaEvolve context keys:

- `paradigm`
- `siblings`
- `error_context`
- `other_context_programs`

---

## 4) Design constraints

1. Reuse existing AdaEvolve scaffold and templates.
2. Keep CostAda-specific additions small and explicit.
3. Avoid duplicating default/adaevolve template pipelines.

---

## 5) Debug tips

If budget/tier text is not visible in prompts:

1. verify `template: costada` in config
2. verify controller passes `remaining_budget_ratio` and `costada_tier`
3. confirm `default_discovery_controller` selects `CostAdaContextBuilder`

