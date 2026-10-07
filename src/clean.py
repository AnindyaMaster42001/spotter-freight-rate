"""Input repairs and the corrupted-label detector.

Input repairs (``fix_inputs``) touch features only and are applied to every frame,
validation included. Validation rows are never dropped or altered beyond that.

The detector (``corrupted_mask``) flags training labels whose price is a multiple
(~x3.5 or ~x0.29) of what the structural model expects. It must be fitted on the
training part of each fold only; flagged rows are dropped from training, never from
evaluation.

Run ``python -m src.clean`` to write ``artifacts/data_quality.csv`` and
``artifacts/label_flags.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config, data
from .features import add_features, struct_X


def weight_fill_value(train: pd.DataFrame) -> float:
    """Median |weight| of the fit frame, used to fill missing weights."""
    return float(train["weight"].abs().median())


def fix_inputs(df: pd.DataFrame, weight_fill: float) -> pd.DataFrame:
    t = df.copy()
    w = t["weight"]
    t["w_miss"] = w.isna().astype(int)
    t["w_neg"] = (w < 0).astype(int)                                   # sign flip
    t["w_cap"] = (w.abs() >= config.WEIGHT_CAP).astype(int)            # censored at the cap
    t["w_floor"] = (w.abs() == config.WEIGHT_FLOOR).astype(int)        # stuck at the floor
    t["w"] = w.abs()
    t["wf"] = t["w"].fillna(weight_fill)

    # market_index is a daily signal: fill gaps with that date's mean. This needs no
    # labels, so each frame is filled from its own daily means.
    mi = t["market_index"]
    t["mi_miss"] = mi.isna().astype(int)
    day_mean = mi.groupby(t["date"]).transform("mean")
    t["mi"] = mi.fillna(day_mean).fillna(mi.mean())
    return t


def prepare(df: pd.DataFrame, weight_fill: float) -> pd.DataFrame:
    """Input repairs followed by feature building."""
    return add_features(fix_inputs(df, weight_fill))


def label_residuals(t: pd.DataFrame, refit_cut: float = config.RESID_REFIT_CUT) -> np.ndarray:
    """Log-price residuals from a two-pass robust fit of the structural design.

    Pass 1 fits OLS on every row; pass 2 refits on rows with |residual| < refit_cut,
    so the corrupted labels no longer pull the fit.
    """
    X, y = struct_X(t), t["y"].to_numpy()
    r = y - LinearRegression().fit(X, y).predict(X)
    keep = np.abs(r) < refit_cut
    return y - LinearRegression().fit(X[keep], y[keep]).predict(X)


def corrupted_mask(t: pd.DataFrame, refit_cut: float = config.RESID_REFIT_CUT,
                   flag_cut: float = config.RESID_FLAG_CUT) -> np.ndarray:
    """True where the label looks corrupted (|log residual| >= flag_cut)."""
    return np.abs(label_residuals(t, refit_cut)) >= flag_cut


def data_quality_table(train: pd.DataFrame, val: pd.DataFrame, corrupt: np.ndarray,
                       resid: np.ndarray) -> pd.DataFrame:
    rules = [
        ("Weight missing", lambda d: d["weight"].isna(), "Fill with training median |weight|; w_miss flag"),
        ("Weight negative (sign flip)", lambda d: d["weight"] < 0, "Absolute value; w_neg flag"),
        ("Weight at cap |w| = 47,500", lambda d: d["weight"].abs() >= config.WEIGHT_CAP, "Keep (censored); w_cap flag"),
        ("  of which -47,500", lambda d: d["weight"] <= -config.WEIGHT_CAP, "Absolute value"),
        ("Weight at floor |w| = 5,000", lambda d: d["weight"].abs() == config.WEIGHT_FLOOR, "Keep; w_floor flag"),
        ("market_index missing", lambda d: d["market_index"].isna(), "Fill with that date's mean"),
        ("Distance at 70 mi floor", lambda d: d["distance"] <= 70, "Keep (clipped, not an error)"),
    ]
    rows = [dict(issue=name, train=int(f(train).sum()), validation=int(f(val).sum()), handling=how)
            for name, f, how in rules]
    hi, lo = corrupt & (resid > 0), corrupt & (resid < 0)
    ratio = np.exp(resid)
    rows += [
        dict(issue="Corrupted label (|log resid| >= 0.25)", train=int(corrupt.sum()), validation=None,
             handling="Dropped from training only (detector refitted inside each fold)"),
        dict(issue=f"  high cluster (median x{np.median(ratio[hi]):.2f}, range x{ratio[hi].min():.2f}-{ratio[hi].max():.2f})",
             train=int(hi.sum()), validation=None, handling=""),
        dict(issue=f"  low cluster (median x{np.median(ratio[lo]):.2f}, range x{ratio[lo].min():.2f}-{ratio[lo].max():.2f})",
             train=int(lo.sum()), validation=None, handling=""),
    ]
    out = pd.DataFrame(rows)
    out["train_pct"] = (out["train"] / len(train) * 100).round(2)
    out["validation_pct"] = (out["validation"] / len(val) * 100).round(2)
    return out[["issue", "train", "train_pct", "validation", "validation_pct", "handling"]]


def main() -> None:
    train, val = data.load_train(), data.load_validation()
    fill = weight_fill_value(train)
    t = prepare(train, fill)
    resid = label_residuals(t)
    corrupt = np.abs(resid) >= config.RESID_FLAG_CUT

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    dq = data_quality_table(train, val, corrupt, resid)
    dq.to_csv(config.ARTIFACTS_DIR / "data_quality.csv", index=False)
    pd.DataFrame({"load_id": t["load_id"], "date": t["date"].dt.date, "equipment": t["equipment"],
                  "log_resid": resid.round(5), "corrupt": corrupt.astype(int)}
                 ).to_csv(config.ARTIFACTS_DIR / "label_flags.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(f"weight fill (training median |weight|): {fill:,.0f}")
        print(dq.to_string(index=False))
    by_month = pd.Series(corrupt).groupby(t["month"].to_numpy()).mean() * 100
    print("corrupted share by month (%):", " ".join(f"{m}:{v:.2f}" for m, v in by_month.items()))


if __name__ == "__main__":
    main()
