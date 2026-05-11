"""Train each variant, then measure accuracy / latency / memory / energy / CO2."""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

from .models import (
    ModelStats,
    build_compact,
    build_full,
    build_quantised,
    compute_stats,
)
from .result import BenchmarkResult, results_to_dicts  # re-exported for back-compat

# Energy estimate constants — see METHODOLOGY.md §3.
PJ_PER_FLOP_FP32 = 0.5
INT8_ENERGY_FACTOR = 0.4

# Default grid intensity (NL avg). Override via env var.
DEFAULT_GRID_INTENSITY = float(os.environ.get("BENCHMARK_GRID_INTENSITY", "328"))


def _train(model: nn.Module, X_train, y_train, epochs: int = 30, lr: float = 1e-3) -> nn.Module:
    """Train a torch model in a single process — no fancy infra."""
    model.train()
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 128
    n = X_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model(X_t[idx])
            loss = loss_fn(logits, y_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def _accuracy(model: nn.Module, X_test, y_test) -> float:
    X_t = torch.tensor(X_test, dtype=torch.float32)
    logits = model(X_t)
    preds = logits.argmax(dim=1).numpy()
    return float((preds == y_test).mean())


@torch.no_grad()
def _latency_ms_p50(model: nn.Module, X_test, n_samples: int = 200, warmup: int = 50) -> float:
    """Median single-sample latency. Warms up first to avoid timing JIT init."""
    X_t = torch.tensor(X_test[:n_samples + warmup], dtype=torch.float32)
    times = []
    for i in range(warmup + n_samples):
        t0 = time.perf_counter()
        model(X_t[i:i + 1])
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(elapsed)
    return float(np.median(times))


def _energy_per_1000_wh(stats: ModelStats) -> float:
    """FLOP-based estimate, see METHODOLOGY.md §3."""
    pj = PJ_PER_FLOP_FP32 * (INT8_ENERGY_FACTOR if stats.is_quantised else 1.0)
    energy_j_per_inf = stats.flops_per_inference * pj * 1e-12
    energy_j_per_1000 = energy_j_per_inf * 1000
    return energy_j_per_1000 / 3600.0  # J → Wh


def _co2_per_1000_g(energy_wh: float, grid_intensity: float = DEFAULT_GRID_INTENSITY) -> float:
    return energy_wh * grid_intensity / 1000.0


def benchmark_variant(model: nn.Module, name: str, X_train, y_train, X_test, y_test,
                      input_dim: int, is_quantised: bool = False) -> BenchmarkResult:
    if not is_quantised:
        # Compact model is trained; quantised variant is built from the trained compact
        # so we don't retrain it here.
        model = _train(model, X_train, y_train)
    stats = compute_stats(model, name, input_dim, is_quantised=is_quantised)
    acc = _accuracy(model, X_test, y_test)
    latency = _latency_ms_p50(model, X_test)
    memory_mb = stats.param_bytes / (1024 * 1024)
    energy_wh = _energy_per_1000_wh(stats)
    co2_g = _co2_per_1000_g(energy_wh)
    return BenchmarkResult(
        variant=name,
        accuracy=round(acc, 4),
        latency_ms_p50=round(latency, 3),
        memory_mb=round(memory_mb, 3),
        flops_per_inference=stats.flops_per_inference,
        energy_per_1000_wh=round(energy_wh, 6),
        co2_per_1000_g=round(co2_g, 4),
        accuracy_per_wh=round(acc / energy_wh if energy_wh > 0 else 0.0, 2),
    )


def run_full_benchmark(X_train, y_train, X_test, y_test, input_dim: int, n_classes: int,
                       seed: int = 42):
    """Train and benchmark all three variants. Returns a list of BenchmarkResults."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Variant 1: full
    full = build_full(input_dim, n_classes)
    r_full = benchmark_variant(full, "full", X_train, y_train, X_test, y_test, input_dim)

    # Variant 2: compact (fine-tuned)
    compact = build_compact(input_dim, n_classes)
    r_compact = benchmark_variant(compact, "compact", X_train, y_train, X_test, y_test, input_dim)

    # Variant 3: quantised — built from the trained compact, no retraining.
    quantised = build_quantised(compact)
    r_quant = benchmark_variant(
        quantised, "quantised", X_train, y_train, X_test, y_test, input_dim,
        is_quantised=True,
    )

    return [r_full, r_compact, r_quant]
