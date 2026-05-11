# Model efficiency benchmark — stakeholder summary

## Headline

Quantising the compact model preserves **94%** of the full model's accuracy while using only **31%** of its energy and **15%** of its memory.

Accuracy-per-Wh improves from **300** to **914** — a **3.0×** efficiency gain.

## What this means

For inference workloads where a few percentage points of accuracy can be traded against
operating cost and footprint, the quantised compact variant is a clear win. For workloads
where the absolute accuracy ceiling matters (e.g. fraud scoring near a regulatory threshold),
the full model remains justified — and the harness gives a quantified basis for that decision.

## Full results

| variant   |   accuracy |   latency_ms_p50 |   memory_mb |   flops_per_inference |   energy_per_1000_wh |   co2_per_1000_g |   accuracy_per_wh |
|:----------|-----------:|-----------------:|------------:|----------------------:|---------------------:|-----------------:|------------------:|
| full      |      0.918 |              142 |     419.617 |           22000000000 |             0.003056 |           0.001  |            300.44 |
| compact   |      0.879 |               78 |     251.77  |           17000000000 |             0.002361 |           0.0008 |            372.28 |
| quantised |      0.863 |               41 |      62.943 |           17000000000 |             0.000944 |           0.0003 |            913.76 |

## Caveats

- Energy figures are FLOP-based estimates (see METHODOLOGY.md). Validate with `nvidia-smi`
  or CodeCarbon before committing to a procurement decision.
- Accuracy on a synthetic benchmark task. Repeat on production-representative data before deploying.
- Quantisation accuracy loss is task-dependent.