"""Pure-data result type — kept torch-free so reporting/fallback can use it
without forcing a torch dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BenchmarkResult:
    variant: str
    accuracy: float
    latency_ms_p50: float
    memory_mb: float
    flops_per_inference: int
    energy_per_1000_wh: float
    co2_per_1000_g: float
    accuracy_per_wh: float


def results_to_dicts(results) -> list[dict]:
    return [asdict(r) for r in results]
