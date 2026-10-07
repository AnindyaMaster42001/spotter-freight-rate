"""Write validation_predictions.csv and fill data/december_chart_inputs.csv.

December chart rows carry no market_index; each day gets the mean market_index of that
day's December loads in validation.csv (an input feature, not a label). A Codex-style
LightGBM without market_index is fitted as a cross-check for the report appendix.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import clean, config, data, models
from .train import load_or_fit


def city_coords(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    out = {}
    for side in ("pickup", "delivery"):
        g = train.groupby(side)[[f"{side}_lat", f"{side}_lon"]].first()
        for city, (lat, lon) in g.iterrows():
            out.setdefault(city, (lat, lon))
    return out


def december_frame(dec: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    coords = city_coords(train)
    (plat, plon), (dlat, dlon) = coords["Lexington"], coords["Fort Wayne"]
    daily_mi = val.groupby("date")["market_index"].mean()
    d = dec.drop(columns="predicted_rate").copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.assign(load_id=[f"DEC-{i:02d}" for i in range(1, len(d) + 1)],
                 pickup_lat=plat, pickup_lon=plon, delivery_lat=dlat, delivery_lon=dlon,
                 market_index=d["date"].map(daily_mi).to_numpy(), quote_signal=np.nan)
    assert d["market_index"].notna().all(), "a December day has no validation market_index"
    return d


def main() -> None:
    bundle = load_or_fit()
    model, fill = bundle["model"], bundle["weight_fill"]
    train, val = data.load_train(), data.load_validation()

    # Validation predictions in the template's row order.
    v = clean.prepare(val, fill)
    pred = np.clip(model.predict(v), 1.0, None).round(2)
    template = data.load_template()
    out = template[["load_id"]].merge(pd.DataFrame({"load_id": v["load_id"], "predicted_rate": pred}),
                                      on="load_id", how="left", validate="one_to_one")
    assert out["predicted_rate"].notna().all() and len(out) == 12_000
    out.to_csv(config.PREDICTIONS_PATH, index=False)

    # December chart: fill predicted_rate in place, keeping the raw text of the other columns.
    dec_raw = pd.read_csv(config.DECEMBER_PATH, dtype=str, keep_default_na=False)
    dfr = december_frame(data.load_december(), train, val)
    d = clean.prepare(dfr, fill)
    dec_pred = model.predict(d).round(2)
    dec_raw["predicted_rate"] = [f"{x:.2f}" for x in dec_pred]
    dec_raw.to_csv(config.DECEMBER_PATH, index=False)

    # Cross-check: Codex-style LightGBM with no market_index / quote_signal, fitted on Jan-Oct.
    t = clean.prepare(train, fill)
    codex = models.CodexTree("lgbm", quote_signal=False, market_index=False).fit(t)
    pd.DataFrame({"date": d["date"].dt.date, "weekday": d["date"].dt.day_name().str[:3],
                  "market_index": d["market_index"].round(5), "hybrid": dec_pred,
                  "codex_lgbm_no_market_index": codex.predict(d).round(2)}
                 ).to_csv(config.ARTIFACTS_DIR / "december_predictions.csv", index=False)

    # Sanity checks.
    rpm = pd.Series(pred / v["distance"].to_numpy(), index=v.index)
    tc = t[~model.corrupt_]
    unseen = model.unseen_city(v)
    checks = {
        "validation": {"n": int(len(pred)), "min": float(pred.min()), "max": float(pred.max()),
                       "mean": float(pred.mean()), "median": float(np.median(pred))},
        "median_rate_per_mile": {e: {"predicted": round(float(rpm[v["equipment"] == e].median()), 4),
                                     "train_clean": round(float((tc["posted_rate"] / tc["distance"])
                                                                [tc["equipment"] == e].median()), 4)}
                                 for e in config.EQUIPMENT},
        "unseen_city_rows": {"n": int(unseen.sum()),
                             "mean_rate_per_mile_unseen": round(float(rpm[unseen].mean()), 4),
                             "mean_rate_per_mile_seen": round(float(rpm[~unseen].mean()), 4)},
        "december": {"min": float(dec_pred.min()), "max": float(dec_pred.max()),
                     "mean": round(float(dec_pred.mean()), 2),
                     "first": float(dec_pred[0]), "last": float(dec_pred[-1])},
    }
    (config.ARTIFACTS_DIR / "prediction_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
