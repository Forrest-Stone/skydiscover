# CostAda Context Builder (`skydiscover.context_builder.costada`)

This package keeps CostAda prompt construction aligned with AdaEvolve while reusing the shared evolutionary prompt templates.

---

## 1) Purpose

`CostAdaContextBuilder` currently aliases the AdaEvolve prompt builder.

This keeps budget experiments from changing prompt verbosity, evaluator feedback length, or context shape.

---

## 2) Added behavior

None. The builder intentionally inherits AdaEvolve behavior as-is.

---

## 3) Expected controller context keys

Controller may pass CostAda metadata for tracing, but prompt rendering consumes only the standard evolutionary context keys:

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

If prompt output differs from AdaEvolve:

1. verify `CostAdaContextBuilder` still subclasses `AdaEvolveContextBuilder` without overriding rendering hooks
2. verify no CostAda-specific templates are configured through `template_dir`
3. compare rendered prompts with `template: adaevolve`
