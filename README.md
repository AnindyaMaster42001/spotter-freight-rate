# Spotter AI — Freight Rate Prediction

Predicts `posted_rate` (USD) for freight loads in November–December 2025 from labeled
January–October 2025 history.

> Work in progress. Results, the model description and the report link are added as the
> pipeline is built.

## Layout

```
data/        Spotter's input files (README naming)
src/         pipeline code (config, data, clean, features, models, validate, train, predict, report)
tests/       output-format checks that mirror score.py
artifacts/   fold results, fitted model files, figures
reports/     DOCX report
score.py     Spotter's scorer, unchanged
```

## Setup

Python 3.12.

```bash
python -m venv .venv            # or: uv venv .venv --python 3.12
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python -m src.data              # load all four input files and check their schema
```

The scorer (from Spotter's instructions):

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
