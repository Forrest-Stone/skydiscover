# BudgetEvolve (Standalone Budget-Aware AdaEvolve Variant)

BudgetEvolve keeps original `adaevolve` untouched and implements budget-aware
control as a separate search method.

This implementation preserves the pre-`31d1d28` BudgetEvolve method logic and
adds shared budget-trace compatibility so runs emit the same `iterations.jsonl`
/ `summary.json` cost records used by `costada`, `adaevolve`, and `evox`.

## Run

```bash
uv run skydiscover-run initial_program.py evaluator.py \
  --config configs/budgetevolve.yaml \
  --search budgetevolve \
  --iterations 100
```

## Core loop

1. Select island (same AdaEvolve global routing)
2. Build budget state
3. Select `(family, tier)` action
4. Family/tier-conditioned sampling
5. Tier-aware prompt building
6. Feasibility check (token/cost)
7. LLM call + usage accounting
8. Evaluate and update scheduler statistics
9. Budget-aware paradigm trigger
10. Stop on iteration cap or budget exhaustion
