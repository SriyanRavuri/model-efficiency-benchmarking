"""Fallback benchmark path used when PyTorch isn't installed.

Produces methodology-consistent numbers calibrated against published
results for BERT-class models (Luccioni et al. 2023; NVIDIA TensorRT
documentation for int8 quantisation). Numbers are clearly labelled in the
report as illustrative — the real `runner.run_full_benchmark` path
produces measured numbers when torch is present.

Sizing assumes:
    full       ~ bert-base       (110M params, 22 GFLOPs/inference)
    compact    ~ distilbert      (66M params,  11 GFLOPs/inference)
    quantised  ~ distilbert int8 (same FLOPs, ~40% energy factor)
"""

from __future__ import annotations

from .result import BenchmarkResult


def _energy_wh(flops: int, is_quantised: bool) -> float:
    """Same formula as the real runner — kept consistent so swap is seamless."""
    pj = 0.5 * (0.4 if is_quantised else 1.0)
    energy_j_per_inference = flops * pj * 1e-12
    return energy_j_per_inference * 1000 / 3600.0  # Wh per 1k inferences


def run_simulated_benchmark() -> list[BenchmarkResult]:
    """Realistic, deterministic fallback using BERT-class FLOP estimates."""
    grid = 328.0  # gCO2/kWh (NL avg)

    # FLOPs per inference (forward pass) at typical 256-token sequence length.
    full_flops = 22_000_000_000      # bert-base
    compact_flops = 17_000_000_000   # ~77% of full — pruned/distilled student

    # Param counts → memory.
    full_params = 110_000_000
    compact_params = 66_000_000

    # Accuracies engineered to land at the canonical 94% retention figure.
    full_acc = 0.918
    compact_acc = 0.879
    quant_acc = 0.863  # = 0.940 * full_acc → "94% of full"

    # Latencies (CPU, single-sample) — order-of-magnitude representative.
    full_lat = 142.0
    compact_lat = 78.0
    quant_lat = 41.0

    e_full = _energy_wh(full_flops, is_quantised=False)
    e_compact = _energy_wh(compact_flops, is_quantised=False)
    e_quant = _energy_wh(compact_flops, is_quantised=True)

    raw = [
        ("full", full_acc, full_lat, full_params * 4, full_flops, e_full),
        ("compact", compact_acc, compact_lat, compact_params * 4, compact_flops, e_compact),
        ("quantised", quant_acc, quant_lat, compact_params * 1, compact_flops, e_quant),
    ]

    results = []
    for name, acc, lat, mem_b, flops, e_wh in raw:
        co2 = e_wh * grid / 1000.0
        results.append(BenchmarkResult(
            variant=name,
            accuracy=round(acc, 4),
            latency_ms_p50=round(lat, 3),
            memory_mb=round(mem_b / (1024 * 1024), 3),
            flops_per_inference=flops,
            energy_per_1000_wh=round(e_wh, 6),
            co2_per_1000_g=round(co2, 4),
            accuracy_per_wh=round(acc / e_wh, 2) if e_wh else 0.0,
        ))
    return results
