"""Forward-in-time validation harness.

Folds (expanding window):
    F1  train Jan-Jul  -> test Aug
    F2  train Jan-Aug  -> test Sep
    F3  train Jan-Sep  -> test Oct
    h2  train Jan-Aug  -> test Sep-Oct  (two months ahead, the stand-in for Nov-Dec)

Metrics are computed on raw labels (corrupted rows included, as Spotter's validation
labels will be) and on clean labels. The clean mask used for *reporting* is the
detector fitted on all of Jan-Oct; models never see it.

``python -m src.validate`` runs the benchmark set and writes artifacts/fold_results.csv
(long: fold x model x metric) and artifacts/fold_summary.csv.
``python -m src.validate --contrasts`` adds the random-split and unseen-city contrasts.
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from . import clean, config, data, models

FOLDS = [  # (name, test start, test end exclusive); training = everything before test start
    ("F1 Aug", "2025-08-01", "2025-09-01"),
    ("F2 Sep", "2025-09-01", "2025-10-01"),
    ("F3 Oct", "2025-10-01", "2025-11-01"),
    ("h2 Sep-Oct", "2025-09-01", "2025-11-01"),
]
CV_FOLDS = ["F1 Aug", "F2 Sep", "F3 Oct"]
METRICS = ["MAE", "RMSE", "MAPE", "MedAE", "MedAPE", "cMAE", "cMAPE", "cBias"]

Factory = Callable[[], object]


def metrics(pred: np.ndarray, actual: np.ndarray, clean_mask: np.ndarray) -> dict:
    pred = np.clip(pred, 1.0, None)
    e = np.abs(pred - actual)
    ape = e / actual * 100
    return dict(MAE=e.mean(), RMSE=np.sqrt(((pred - actual) ** 2).mean()), MAPE=ape.mean(),
                MedAE=np.median(e), MedAPE=np.median(ape), cMAE=e[clean_mask].mean(),
                cMAPE=ape[clean_mask].mean(),
                # mean signed log error on clean rows, in %: >0 means over-prediction
                cBias=np.mean(np.log(pred[clean_mask] / actual[clean_mask])) * 100, n=len(e))


def load_labeled() -> pd.DataFrame:
    """Raw training data plus the reporting-only clean mask (detector on all Jan-Oct)."""
    raw = data.load_train()
    full = clean.prepare(raw, clean.weight_fill_value(raw))
    raw = raw.copy()
    raw["report_clean"] = ~clean.corrupted_mask(full)
    return raw


def split(raw: pd.DataFrame, train_mask, test_mask):
    """Prepare a train/test pair; the weight fill comes from the training side only."""
    tr, te = raw[train_mask], raw[test_mask]
    fill = clean.weight_fill_value(tr)
    return clean.prepare(tr, fill), clean.prepare(te, fill)


def evaluate(raw: pd.DataFrame, factories: dict[str, Factory],
             blends: dict[str, dict[str, float]] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Fit every model once per training window and score it on every fold."""
    rows = []
    for start in sorted({f[1] for f in FOLDS}):
        horizon = raw["date"] >= start
        tr, te = split(raw, raw["date"] < start, horizon)
        preds = {}
        for name, make in factories.items():
            t0 = time.time()
            preds[name] = make().fit(tr).predict(te)
            if verbose:
                print(f"  train < {start}  {name:<40} {time.time() - t0:5.1f}s", flush=True)
        for name, w in (blends or {}).items():
            preds[name] = sum(wt * preds[m] for m, wt in w.items())
        for fold, s, e in FOLDS:
            if s != start:
                continue
            k = (te["date"] < e).to_numpy()
            actual = te["posted_rate"].to_numpy()[k]
            cm = te["report_clean"].to_numpy()[k]
            for name, p in preds.items():
                rows.append(dict(fold=fold, model=name, **metrics(p[k], actual, cm)))
    wide = pd.DataFrame(rows)
    return wide.melt(id_vars=["fold", "model", "n"], var_name="metric")


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    """One row per model: per-fold MAE, h2 MAE, 3-fold averages. Sorted by the selection score."""
    w = long.pivot_table(index=["model", "fold"], columns="metric", values="value")
    mae = w["MAE"].unstack("fold")
    cv = w.loc[(slice(None), CV_FOLDS), :].groupby("model").mean()
    out = pd.DataFrame({
        "Aug MAE": mae["F1 Aug"], "Sep MAE": mae["F2 Sep"], "Oct MAE": mae["F3 Oct"],
        "h2 MAE": mae["h2 Sep-Oct"], "avg MAE": cv["MAE"], "avg RMSE": cv["RMSE"],
        "avg MAPE": cv["MAPE"], "avg MedAPE": cv["MedAPE"], "avg cMAE": cv["cMAE"],
        "avg cMAPE": cv["cMAPE"], "h2 cMAPE": w["cMAPE"].unstack("fold")["h2 Sep-Oct"],
        "Oct cBias%": w["cBias"].unstack("fold")["F3 Oct"],
        "h2 cBias%": w["cBias"].unstack("fold")["h2 Sep-Oct"],
    })
    return out.sort_values(["avg MAE", "h2 MAE"])


def benchmark_set(hybrid_kwargs: dict | None = None):
    """The models reported in the comparison table (plan sections 4 and 9.3)."""
    hk = dict(hybrid_kwargs or {})
    no_trend = {**hk, "trend": "none"}
    factories = {
        "Hybrid (chosen)": lambda: models.HybridModel(**hk),
        "Hybrid, untuned defaults": models.HybridModel,
        "Hybrid, no city categoricals": lambda: models.HybridModel(**{**hk, "city_cats": False}),
        "Hybrid, city cats routed": lambda: models.HybridModel(**{**hk, "city_cats": True,
                                                                   "route_unseen": True}),
        "Hybrid + quote_signal": lambda: models.HybridModel(**{**hk, "quote_signal": True}),
        "Hybrid without trend": lambda: models.HybridModel(**no_trend),
        "Hybrid without quarter-end ramp": lambda: models.HybridModel(**{**hk, "ramp": False}),
        "Hybrid without trend or ramp": lambda: models.HybridModel(**{**no_trend, "ramp": False}),
        "Structural only (stage 1)": lambda: models.HybridModel(**{**hk, "stage2": False}),
        "Linear, city indicators, no time terms": models.NoTimeLinear,
        "Codex LGBM-L1": lambda: models.CodexTree("lgbm"),
        "Codex LGBM-L1 without quote_signal": lambda: models.CodexTree("lgbm", quote_signal=False),
        "Codex XGB-L1": lambda: models.CodexTree("xgb"),
    }
    blends = {
        "Codex blend 0.75 XGB / 0.25 LGBM": {"Codex XGB-L1": 0.75, "Codex LGBM-L1": 0.25},
        "Hybrid 50/50 with Codex XGB": {"Hybrid (chosen)": 0.5, "Codex XGB-L1": 0.5},
    }
    return factories, blends


def contrasts(raw: pd.DataFrame, hybrid_kwargs: dict | None = None) -> pd.DataFrame:
    """Random 10% split vs forward split, and the unseen-city simulation."""
    hk = dict(hybrid_kwargs or {})
    makers = {"Hybrid (chosen)": lambda: models.HybridModel(**hk),
              "Hybrid, no city categoricals": lambda: models.HybridModel(**{**hk, "city_cats": False}),
              "Hybrid, city cats routed": lambda: models.HybridModel(**{**hk, "city_cats": True,
                                                                         "route_unseen": True}),
              "Codex LGBM-L1": lambda: models.CodexTree("lgbm")}
    rows = []

    def score(setting, tr, te, which=makers):
        for name, make in which.items():
            p = make().fit(tr).predict(te)
            m = metrics(p, te["posted_rate"].to_numpy(), te["report_clean"].to_numpy())
            rows.append(dict(setting=setting, model=name, **m))
            print(f"  {setting:<44} {name:<30} MAE {m['MAE']:7.2f}  cMAPE {m['cMAPE']:5.2f}", flush=True)

    rnd = np.random.RandomState(config.SEED).rand(len(raw)) < 0.10
    score("Random 10% holdout (all months)", *split(raw, ~rnd, rnd))
    score("Forward: train Jan-Sep, test Oct", *split(raw, raw["date"] < "2025-10-01",
                                                     raw["date"] >= "2025-10-01"))

    rng = np.random.RandomState(1)
    held = list(rng.choice(sorted(set(raw["pickup"])), 8, replace=False))
    touch = raw["pickup"].isin(held) | raw["delivery"].isin(held)
    tr, te = split(raw, (raw["date"] < "2025-09-01") & ~touch, (raw["date"] >= "2025-09-01") & touch)
    score("Unseen cities: 8 withheld, test Sep-Oct", tr, te)
    out = pd.DataFrame(rows)
    out.attrs["held_cities"] = held
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrasts", action="store_true", help="also run the contrast experiments")
    args = ap.parse_args()
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    raw = load_labeled()
    hk = models.load_chosen_config()

    factories, blends = benchmark_set(hk)
    long = evaluate(raw, factories, blends)
    long.to_csv(config.ARTIFACTS_DIR / "fold_results.csv", index=False)
    summary = summarize(long)
    summary.round(3).to_csv(config.ARTIFACTS_DIR / "fold_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(summary.round(2).to_string())

    if args.contrasts:
        c = contrasts(raw, hk)
        c.to_csv(config.ARTIFACTS_DIR / "contrasts.csv", index=False)
        print("held-out cities:", ", ".join(c.attrs["held_cities"]))
        print(c.pivot_table(index="setting", columns="model", values=["MAE", "cMAPE"]).round(2).to_string())


if __name__ == "__main__":
    main()
