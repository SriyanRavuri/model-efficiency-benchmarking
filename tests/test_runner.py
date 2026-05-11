"""Smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmark.data import feature_dim, make_dataset, n_classes  # noqa: E402
from benchmark.runner import run_full_benchmark  # noqa: E402


def test_benchmark_returns_three_variants():
    X_train, X_test, y_train, y_test = make_dataset(n_samples=400)
    results = run_full_benchmark(
        X_train, y_train, X_test, y_test,
        input_dim=feature_dim(), n_classes=n_classes(),
    )
    names = [r.variant for r in results]
    assert names == ["full", "compact", "quantised"]


def test_quantised_uses_less_memory():
    X_train, X_test, y_train, y_test = make_dataset(n_samples=400)
    results = run_full_benchmark(
        X_train, y_train, X_test, y_test,
        input_dim=feature_dim(), n_classes=n_classes(),
    )
    by_name = {r.variant: r for r in results}
    assert by_name["quantised"].memory_mb < by_name["compact"].memory_mb
    assert by_name["compact"].memory_mb < by_name["full"].memory_mb
