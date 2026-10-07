"""Fit the chosen hybrid on all of January-October and save it.

Writes artifacts/final_model.pkl (used by predict.py; not committed) and a readable copy
in artifacts/final_model/: stage-1 coefficients (JSON), the stage-2 LightGBM models
(text) and metadata.
"""
from __future__ import annotations

import json
import pickle

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from . import clean, config, data, models

MODEL_PKL = config.ARTIFACTS_DIR / "final_model.pkl"
MODEL_DIR = config.ARTIFACTS_DIR / "final_model"


def fit_final() -> dict:
    raw = data.load_train()
    fill = clean.weight_fill_value(raw)
    t = clean.prepare(raw, fill)
    params = models.load_chosen_config()
    model = models.HybridModel(**params).fit(t)
    return {"model": model, "weight_fill": fill, "params": params, "n_train": len(t),
            "n_corrupt": int(model.corrupt_.sum())}


def save(bundle: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(bundle, f)
    m = bundle["model"]
    (MODEL_DIR / "stage1_coefficients.json").write_text(json.dumps(m.stage1.coefficients(), indent=2) + "\n")
    m.lgbm_.booster_.save_model(MODEL_DIR / "stage2_lgbm.txt")
    if m.route_unseen:
        m.lgbm_nocity_.booster_.save_model(MODEL_DIR / "stage2_lgbm_no_city.txt")
    meta = {k: bundle[k] for k in ("weight_fill", "params", "n_train", "n_corrupt")}
    meta.update(training_cities=m.cities_, trend_end_tm=m.stage1.tm_end_,
                versions={"pandas": pd.__version__, "numpy": np.__version__,
                          "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__})
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def load_or_fit() -> dict:
    if MODEL_PKL.is_file():
        with open(MODEL_PKL, "rb") as f:
            return pickle.load(f)
    bundle = fit_final()
    save(bundle)
    return bundle


def main() -> None:
    bundle = fit_final()
    save(bundle)
    print(f"fitted on {bundle['n_train']:,} rows ({bundle['n_corrupt']} corrupted labels dropped); "
          f"settings {bundle['params']}")
    print(f"saved {MODEL_PKL.relative_to(config.ROOT)} and {MODEL_DIR.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
