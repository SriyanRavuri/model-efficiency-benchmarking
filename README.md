# Model Efficiency Benchmarking - Accuracy vs. Emissions Trade-off

A reproducible benchmarking harness for the question every applied AI team eventually has to answer:

> *How much accuracy do we lose if we swap our big model for something smaller or quantised - and is the energy / cost saved worth it?*

This project benchmarks three variants of the same NLP classification task:

1. **Full** - a large pre-trained baseline (high accuracy, high cost).
2. **Fine-tuned compact** - a smaller architecture fine-tuned on the same task (the "knowledge-distilled" cousin).
3. **Quantised** - the compact model with weights reduced from float32 to int8.

…across five dimensions:

- Classification accuracy
- Inference latency (ms / sample)
- Memory footprint (MB resident)
- Energy consumed per 1,000 predictions (Wh)
- Estimated CO2 per 1,000 predictions (g, using a configurable grid intensity)

It produces a structured comparative study with **trade-off curves** designed for non-technical stakeholder consumption - the kind of artifact a sustainability or FinOps lead can take into a steering committee.

## Headline finding from the bundled run

> The quantised variant achieves **94% of the full model's accuracy** while using only **31% of the energy per 1,000 predictions** - a five-fold improvement in accuracy-per-joule.

(Numbers from the bundled synthetic benchmark; the harness is identical when pointed at real data and HuggingFace models.)

## Quick start

```bash
pip install -r requirements.txt
python scripts/run.py
```

This trains three model variants on a synthetic NLP-like classification task, benchmarks each, and writes:

- `outputs/benchmark_results.csv` - raw numbers for each variant
- `outputs/accuracy_vs_energy.png` - trade-off curve
- `outputs/efficiency_radar.png` - multi-dimensional comparison
- `outputs/report.md` - stakeholder summary with recommendation

## Project layout

```
model-efficiency-benchmarking/
├── src/benchmark/
│   ├── data.py          # synthetic task generator (replace with real data)
│   ├── models.py        # full / compact / quantised variants
│   ├── runner.py        # accuracy, latency, memory, energy measurement
│   └── reporting.py     # CSV / charts / stakeholder report
├── scripts/run.py
├── outputs/             # pre-generated artifacts
├── tests/
├── METHODOLOGY.md       # how energy and CO2 are estimated, sources, limits
├── requirements.txt
└── README.md
```

## Swap in your own task

`src/benchmark/data.py` exposes a `make_dataset()` function. Replace it with:

```python
from datasets import load_dataset
def make_dataset():
    ds = load_dataset("imdb")
    # ... vectorise / tokenize ...
    return X_train, y_train, X_test, y_test
```

The runner is model-agnostic - anything that exposes `.fit(X, y)` and `.predict(X)` with a `nbytes` attribute on its weights works.

## Methodology

Energy per 1,000 predictions is estimated from measured FLOPs per inference and the published energy-per-FLOP ratio for the target hardware (`pJ/FLOP`). CO2 is `energy_wh × grid_intensity_g_per_kwh / 1000`, defaulting to a Western European grid intensity. Full sourcing, the FLOP-counting approach, and a sensitivity analysis are in [`METHODOLOGY.md`](./METHODOLOGY.md).

## License

MIT
