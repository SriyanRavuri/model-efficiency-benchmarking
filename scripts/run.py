"""End-to-end benchmark run.

    python scripts/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from benchmark.reporting import (  # noqa: E402
    chart_accuracy_vs_energy,
    chart_efficiency_radar,
    write_results_csv,
    write_stakeholder_report,
)

OUTPUTS = ROOT / "outputs"


def main() -> None:
    # Try the real torch-based benchmark first; fall back to the simulated
    # path if torch isn't installed. The simulated path is calibrated against
    # published quantisation results — see src/benchmark/fallback.py.
    from benchmark.result import results_to_dicts
    try:
        import torch  # noqa: F401
        from benchmark.data import feature_dim, make_dataset, n_classes
        from benchmark.runner import run_full_benchmark

        print("Generating dataset...")
        X_train, X_test, y_train, y_test = make_dataset()
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print("Running benchmark (PyTorch path)...")
        results = run_full_benchmark(
            X_train, y_train, X_test, y_test,
            input_dim=feature_dim(),
            n_classes=n_classes(),
        )
    except ImportError:
        from benchmark.fallback import run_simulated_benchmark
        print("PyTorch not installed — using simulated benchmark "
              "(install torch for the measured path).")
        results = run_simulated_benchmark()
    df = pd.DataFrame(results_to_dicts(results))
    print(df.to_string(index=False))

    write_results_csv(df, OUTPUTS / "benchmark_results.csv")
    chart_accuracy_vs_energy(df, OUTPUTS / "accuracy_vs_energy.png")
    chart_efficiency_radar(df, OUTPUTS / "efficiency_radar.png")
    write_stakeholder_report(df, OUTPUTS / "report.md")
    print(f"Wrote outputs to {OUTPUTS}")


if __name__ == "__main__":
    main()
