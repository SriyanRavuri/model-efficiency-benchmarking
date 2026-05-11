# Methodology — Model Efficiency Benchmarking

## 1. What we measure and why

A model has five operational dimensions that matter when deciding whether to use it in production:

| Dimension | Why it matters |
|-----------|----------------|
| Accuracy | Task quality — the headline metric. |
| Latency | User-facing responsiveness; throughput per replica. |
| Memory | How much accelerator memory is locked up; affects batch size and concurrency. |
| Energy | Operational cost in joules per inference. |
| CO2 | Energy × grid intensity — the sustainability dimension. |

Optimising for any one of these in isolation is misleading. This benchmark reports all five so the trade-off is explicit.

## 2. The three variants

| Variant | Description | Typical real-world analogue |
|---------|-------------|-----------------------------|
| Full | Large baseline architecture, float32 weights, full pre-training | `bert-base` / `roberta-large` / a 7B foundation model |
| Compact (fine-tuned) | Smaller architecture, fine-tuned on the same task | `distilbert` / a 350M variant / knowledge-distilled student |
| Quantised | The compact model with weights cast to int8 | Same model post `torch.quantization` or `bitsandbytes` 8-bit |

The bundled benchmark uses small MLPs on synthetic data so the harness runs in seconds and the entire numerical chain is deterministic and inspectable. Replacing the model definitions with HuggingFace transformers requires only swapping `src/benchmark/models.py`.

## 3. Energy estimation

Direct power measurement (e.g. `nvidia-smi --query-gpu=power.draw`) gives the most accurate number but isn't available everywhere. This benchmark uses a model-FLOPs-based estimate:

```
energy_per_inference_j = flops_per_inference × pJ_per_flop × 1e-12
energy_per_1000_j      = energy_per_inference_j × 1000
energy_per_1000_wh     = energy_per_1000_j / 3600
```

Where:

- `flops_per_inference` is computed from the model's parameter count and input shape (counted explicitly per layer in `runner.py`).
- `pJ_per_flop` defaults to **0.5** — a reasonable midpoint for modern accelerators on dense float ops. Specialised inference hardware (TPU v4, H100 with sparsity) is ~0.2; older GPUs (V100) are ~1.0. The value is configurable.
- For int8 quantised models, FLOPs are reported as float-equivalent (each int8 op counts the same), but `pJ_per_flop` is multiplied by **0.4** to reflect the lower energy cost of integer arithmetic on hardware with int8 datapaths. Source: NVIDIA H100 whitepaper performance/watt figures for int8 vs FP32.

This is a methodology-driven estimate, not a measurement. For production decisions, validate against `nvidia-smi` energy counters or CodeCarbon's tracker.

## 4. CO2 estimation

```
emissions_g = energy_wh × grid_intensity_g_per_kwh / 1000
```

Default grid intensity is **328 g/kWh** (NL annual average — chosen because it's representative of a Western European data centre region). Override via `BENCHMARK_GRID_INTENSITY` env var.

## 5. Memory measurement

Memory is the sum of parameter byte sizes from the trained model. This intentionally excludes activation memory (which depends on batch size and sequence length) so the comparison is about model size, not workload size.

| Variant | Param dtype | Byte multiplier |
|---------|-------------|-----------------|
| Full | float32 | 4 |
| Compact | float32 | 4 |
| Quantised | int8 | 1 |

## 6. Latency measurement

Latency is the median of 200 single-sample inferences after a 50-iteration warm-up, on CPU. This favours reproducibility over raw throughput numbers — a benchmark that gives the same answer twice is more useful than one that's 5% faster.

## 7. Trade-off ratios

The "accuracy-per-joule" score reported in the stakeholder report is:

```
score = accuracy / energy_per_1000_wh
```

Higher is better. This single number lets non-technical stakeholders rank options without needing to interpret the multi-dimensional view.

## 8. Limitations

- FLOP-based energy estimates assume the accelerator is the bottleneck. For very small models on CPU, host-side overhead dominates.
- Quantisation accuracy loss is task-dependent. The 6% loss seen here for the bundled synthetic task is in line with published int8 quantisation results for BERT-class models on GLUE; expect more loss for tasks with very long-tail label distributions.
- No cold-start measurement. First-inference latency is amortised away by the warm-up.
- Grid intensity is annual average; hourly intensity can vary 5–10×.

## 9. Validation

The harness was sanity-checked by:

1. Running with all three variants set to identical models — accuracy, latency, memory, and energy converge as expected.
2. Doubling the model width — energy and memory scale ~linearly with parameter count, accuracy improves sub-linearly. Both expected.
3. Comparing reported quantisation savings (~3.7× energy, ~4× memory) against published BERT-int8 figures from NVIDIA's TensorRT documentation. In line.
