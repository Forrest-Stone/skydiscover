# CostAda Context Builder

`CostAdaContextBuilder` extends `AdaEvolveContextBuilder` and adds BCHD-specific prompt controls.

## What it adds

1. **Budget status block**
   - Remaining budget ratio
   - Current spending tier
   - Spending preference text

2. **Tier-conditioned verbosity**
   - `cheap`: minimal guidance (one-line style)
   - `standard`: compressed guidance
   - `rich`: full guidance

## Why

This keeps prompt construction aligned with BCHD step-level spending:
- cheaper tiers use compact prompts,
- richer tiers permit fuller tactic detail.

## Inputs expected from controller

The controller should pass in context keys:
- `remaining_budget_ratio`
- `costada_tier`
- plus regular AdaEvolve guidance keys (`paradigm`, `siblings`, `error_context`, ...)

## Design principle

- Reuse AdaEvolve prompt scaffold.
- Add minimal CostAda-specific block.
- Avoid duplicating entire template pipelines.
