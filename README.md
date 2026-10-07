# Spotter AI — Freight Rate Prediction

Predicts `posted_rate` (USD) for 12,000 freight loads in November–December 2025 from
48,000 labelled loads in January–October 2025.

- **Predictions:** [`validation_predictions.csv`](validation_predictions.csv)
- **December chart:** [`scorer_results/candidate_december.png`](scorer_results/candidate_december.png) (the filled inputs are in `data/december_chart_inputs.csv`)
- **Report (DOCX):** [`reports/Freight_Rate_Report.docx`](reports/Freight_Rate_Report.docx), covering the validation and split approach, the model, and the December chart

## Approach

1. **Clean.**
   - 677 training labels (1.4%) are corrupted by a multiplier: about ×3.3 or ×0.29 of the expected price.
   - A two-pass robust residual fit finds them, and they are dropped from training only. The detector is refitted inside every fold.
   - Weight sign flips, cap, floor and gaps get the absolute value, flags and median imputation. Missing `market_index` gets that day's mean.
   - No validation row is dropped.
2. **Model: a two-stage hybrid.**
   - Stage 1 is least squares on log price: distance, weight, equipment, `market_index`, a smooth surface in coordinates, a linear time trend, and an equipment-specific quarter-end ramp.
   - Stage 2 is LightGBM on the stage-1 residuals, with no trend input, so it cannot flatten the forward projection.
   - Rows that touch a city absent from training use a stage-2 model without city features.
3. **Validate forward in time.**
   - Expanding folds test Aug, Sep and Oct, plus **h2** (train Jan–Aug, test Sep–Oct) as the stand-in for Nov–Dec.
   - Selection is by raw MAE averaged over the three folds.
4. **`quote_signal` is dropped.** It carries no signal once lane and equipment are controlled for, and adding it hurts.

## Results

MAE in dollars on raw labels, corrupted rows included. Clean MAPE excludes the corrupted rows.

| Model | Aug | Sep | Oct | h2 | **Fold avg MAE** | Clean MAPE |
|---|---:|---:|---:|---:|---:|---:|
| **Hybrid (submitted)** | 79.9 | 80.4 | 95.4 | 88.5 | **85.2** | **1.33%** |
| Hybrid, plan defaults (untuned) | 81.9 | 82.4 | 97.1 | 90.3 | 87.1 | 1.42% |
| Stage 1 only (linear) | 92.8 | 95.2 | 106.2 | 101.3 | 98.0 | 1.93% |
| LightGBM-L1, no `quote_signal` | 85.3 | 116.2 | 107.2 | 117.6 | 102.9 | 2.16% |
| XGB/LightGBM blend (0.75/0.25) | 118.2 | 115.6 | 107.2 | 118.9 | 113.7 | 2.63% |
| Hybrid without trend | 120.8 | 136.1 | 163.7 | 154.3 | 140.2 | 3.68% |
| Hybrid without `market_index` | 197.3 | 205.8 | 101.1 | 169.2 | 168.1 | 4.85% |

| Split (MAE) | Hybrid | LightGBM |
|---|---:|---:|
| Random 10% holdout | 91.5 | 95.3 |
| Forward: train Jan–Sep, test Oct | 95.4 | 110.6 |
| 8 cities withheld, test Sep–Oct | 88.2 | 110.6 |

The split comparison makes two points:
- A random split would have shown the two models as nearly equal. The difference appears only when predicting forward in time.
- RMSE is about $620–640 for every model, because the corrupted labels dominate it. That's why selection uses MAE.

The full tables are in `artifacts/fold_summary.csv`, `artifacts/contrasts.csv` and `artifacts/tuning_results.csv`.

## Setup

Python 3.12.

```bash
python -m venv .venv            # or: uv venv .venv --python 3.12
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`pandas` is pinned to 2.x because Spotter's `score.py` requires `pandas>=2.0,<3`.

## Run

```bash
python -m src.run_all                    # full pipeline, about 10 minutes on 16 cores
python -m src.run_all --skip-validation  # reuse the committed fold results (about 2 minutes)
python -m pytest -q tests                # data-quality counts and output-format checks
```

`run_all` runs these steps in order:
1. Schema checks.
2. The data-quality table.
3. The forward-fold benchmark and the contrast experiments.
4. The final fit on January–October.
5. Writing `validation_predictions.csv` and filling `data/december_chart_inputs.csv`.
6. `score.py`.
7. Figures.
8. The DOCX report.

Each step also runs on its own, for example `python -m src.validate --contrasts` or `python -m src.predict`.

The tuning grid and its selection rule were committed before the tuning ran (see `src/config.py`). To rerun it:

```bash
python -m src.tune               # writes artifacts/chosen_config.json
```

Spotter's scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Layout

```
data/          Spotter's input files (README naming); the pipeline fills december_chart_inputs.csv in place
src/
  config.py    paths, constants, the pre-registered tuning grid
  data.py      loading and schema checks
  clean.py     input repairs and the corrupted-label detector
  features.py  feature builders and the stage-1 design matrix
  models.py    HybridModel, StructuralModel and the baselines
  validate.py  forward folds, metrics, random-split and unseen-city contrasts
  tune.py      the tuning run
  train.py     final fit on Jan-Oct (artifacts/final_model/)
  predict.py   validation predictions, December chart, sanity checks
  figures.py   report figures (reports/figures/)
  report.py    builds reports/Freight_Rate_Report.docx
  run_all.py   the whole pipeline
tests/         data-quality counts and output-format checks mirroring score.py
artifacts/     fold results, tuning results, fitted model (coefficients JSON + LightGBM text)
reports/       DOCX report, figures, Loom script
score.py       Spotter's scorer, unchanged
```

Seeds are fixed at 0. Results were produced with the exact versions in `requirements.txt`.
