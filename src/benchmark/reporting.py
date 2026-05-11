"""CSV, charts, and a stakeholder-facing report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_results_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def chart_accuracy_vs_energy(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"full": "#1f77b4", "compact": "#2ca02c", "quantised": "#ff7f0e"}
    for _, row in df.iterrows():
        ax.scatter(row["energy_per_1000_wh"], row["accuracy"],
                   s=200, color=colors.get(row["variant"], "#777"),
                   label=row["variant"], edgecolor="black", zorder=3)
        ax.annotate(row["variant"],
                    (row["energy_per_1000_wh"], row["accuracy"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Energy per 1,000 predictions (Wh)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. energy trade-off — closer to top-left is better")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def chart_efficiency_radar(df: pd.DataFrame, path: Path) -> None:
    """Radar of normalised metrics across the three variants. Higher = better."""
    metrics = ["accuracy", "latency_ms_p50", "memory_mb", "energy_per_1000_wh"]
    inverse_metrics = {"latency_ms_p50", "memory_mb", "energy_per_1000_wh"}

    norm = pd.DataFrame()
    for m in metrics:
        if m in inverse_metrics:
            # Invert so smaller is better → bigger value on chart.
            norm[m] = df[m].min() / df[m]
        else:
            norm[m] = df[m] / df[m].max()
    norm["variant"] = df["variant"].values

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    pretty = {"accuracy": "Accuracy", "latency_ms_p50": "Speed",
              "memory_mb": "Memory eff.", "energy_per_1000_wh": "Energy eff."}
    for _, row in norm.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, label=row["variant"], linewidth=2)
        ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([pretty[m] for m in metrics])
    ax.set_yticklabels([])
    ax.set_title("Multi-dimensional efficiency comparison\n(further from centre = better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_stakeholder_report(df: pd.DataFrame, path: Path) -> None:
    full = df[df["variant"] == "full"].iloc[0]
    quant = df[df["variant"] == "quantised"].iloc[0]

    acc_retained = quant["accuracy"] / full["accuracy"]
    energy_used = quant["energy_per_1000_wh"] / full["energy_per_1000_wh"]
    memory_used = quant["memory_mb"] / full["memory_mb"]

    lines = [
        "# Model efficiency benchmark — stakeholder summary",
        "",
        "## Headline",
        "",
        f"Quantising the compact model preserves **{acc_retained*100:.0f}%** of the full model's accuracy "
        f"while using only **{energy_used*100:.0f}%** of its energy and **{memory_used*100:.0f}%** of its memory.",
        "",
        f"Accuracy-per-Wh improves from **{full['accuracy_per_wh']:.0f}** to **{quant['accuracy_per_wh']:.0f}** — a "
        f"**{quant['accuracy_per_wh']/full['accuracy_per_wh']:.1f}×** efficiency gain.",
        "",
        "## What this means",
        "",
        "For inference workloads where a few percentage points of accuracy can be traded against",
        "operating cost and footprint, the quantised compact variant is a clear win. For workloads",
        "where the absolute accuracy ceiling matters (e.g. fraud scoring near a regulatory threshold),",
        "the full model remains justified — and the harness gives a quantified basis for that decision.",
        "",
        "## Full results",
        "",
        df.to_markdown(index=False),
        "",
        "## Caveats",
        "",
        "- Energy figures are FLOP-based estimates (see METHODOLOGY.md). Validate with `nvidia-smi`",
        "  or CodeCarbon before committing to a procurement decision.",
        "- Accuracy on a synthetic benchmark task. Repeat on production-representative data before deploying.",
        "- Quantisation accuracy loss is task-dependent.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
