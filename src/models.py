"""Models: the chosen two-stage hybrid, its ablations, and the baselines.

Every model exposes ``fit(t)`` and ``predict(t)`` on frames from ``clean.prepare``;
``predict`` returns dollars. Models that learn from cleaned labels run the
corrupted-label detector on the frame they are fitted on (so it is refitted per fold).
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from . import clean, config
from .features import struct_X

# Stage-2 inputs: no trend or calendar-position feature beyond the within-month
# position ``frac``, so stage 2 cannot bend the forward projection of stage 1.
RESIDUAL_FEATURES = ["ld", "wf", "w_miss", "lmi", "eq", "pickup_lat", "pickup_lon",
                     "delivery_lat", "delivery_lon", "dlat", "dlon", "dow", "frac"]

CODEX_FEATURES = ["pickup", "delivery", "lane", "pickup_lat", "pickup_lon", "delivery_lat",
                  "delivery_lon", "dlat", "dlon", "hav", "bearing", "distance", "equipment", "w",
                  "w_neg", "w_miss", "market_index", "quote_signal", "tdays", "dow", "dom", "month",
                  "wsin", "wcos", "asin", "acos"]
CODEX_CATEGORICAL = ["pickup", "delivery", "lane", "equipment"]

LGBM_OBJECTIVE = {"huber": "huber", "l2": "regression", "l1": "regression_l1"}


def load_chosen_config() -> dict:
    """HybridModel settings picked by src.tune (defaults if tuning has not run)."""
    path = config.CHOSEN_CONFIG_PATH
    return json.loads(path.read_text())["hybrid"] if path.is_file() else {}


def _quarter_index(t: pd.DataFrame) -> pd.Series:
    return (t["date"].dt.year - 2025) * 4 + t["date"].dt.quarter


class StructuralModel:
    """Stage 1: least squares on log price.

    trend: "linear" (months since 2025-01-01), "step" (one level per quarter; quarters
    after the training window carry the last fitted level), "damped" (linear inside the
    training window, slope times ``damp`` beyond it) or "none".
    """

    def __init__(self, trend="linear", ramp=True, damp=0.5, quote_signal=False):
        self.trend, self.ramp, self.damp, self.quote_signal = trend, ramp, damp, quote_signal

    def design(self, t: pd.DataFrame) -> pd.DataFrame:
        X = struct_X(t, trend=False, ramp=self.ramp)
        if self.trend == "linear":
            X["tm"] = t["tm"]
        elif self.trend == "damped":
            over = (t["tm"] - self.tm_end_).clip(lower=0)
            X["tm"] = t["tm"].clip(upper=self.tm_end_) + self.damp * over
        elif self.trend == "step":
            q = _quarter_index(t).clip(upper=self.q_last_)
            for k in self.quarters_[1:]:
                X[f"q{k}"] = (q == k).astype(float)
        elif self.trend != "none":
            raise ValueError(self.trend)
        if self.quote_signal:
            X["lqs"] = np.log(t["quote_signal"])
        return X

    def fit(self, t: pd.DataFrame) -> "StructuralModel":
        self.tm_end_ = float(t["tm"].max())
        self.quarters_ = sorted(_quarter_index(t).unique())
        self.q_last_ = self.quarters_[-1]
        X = self.design(t)
        self.columns_ = list(X.columns)
        self.ols_ = LinearRegression().fit(X, t["y"])
        return self

    def predict_log(self, t: pd.DataFrame) -> np.ndarray:
        return self.ols_.predict(self.design(t)[self.columns_])

    def coefficients(self) -> dict:
        return {"intercept": float(self.ols_.intercept_),
                **{c: float(b) for c, b in zip(self.columns_, self.ols_.coef_)}}


class HybridModel:
    """Stage 1 structural model, then LightGBM on its log residuals (cleaned rows only)."""

    def __init__(self, trend="linear", ramp=True, damp=0.5, stage2=True, n_estimators=600,
                 num_leaves=31, loss="huber", city_cats=False, route_unseen=False,
                 quote_signal=False, seed=config.SEED):
        """route_unseen (with city_cats): rows whose pickup or delivery city was not in the
        training data get a second stage-2 model fitted without the city categoricals."""
        self.stage1 = StructuralModel(trend, ramp, damp, quote_signal)
        self.stage2, self.city_cats, self.quote_signal = stage2, city_cats, quote_signal
        self.route_unseen = route_unseen and city_cats
        self.lgbm_params = dict(
            objective=LGBM_OBJECTIVE[loss], n_estimators=n_estimators, learning_rate=0.03,
            num_leaves=num_leaves, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, verbose=-1)
        if loss == "huber":
            self.lgbm_params["alpha"] = 0.05

    def _residual_X(self, t: pd.DataFrame, city_cats: bool | None = None) -> pd.DataFrame:
        X = t[RESIDUAL_FEATURES].copy()
        if self.quote_signal:
            X["quote_signal"] = t["quote_signal"]
        if self.city_cats if city_cats is None else city_cats:
            for c in ("pickup", "delivery"):
                X[c] = pd.Categorical(t[c], categories=self.cities_)   # unseen city -> missing
        return X

    def fit(self, t: pd.DataFrame) -> "HybridModel":
        self.corrupt_ = clean.corrupted_mask(t)
        c = t[~self.corrupt_]
        self.stage1.fit(c)
        if self.stage2:
            self.cities_ = sorted(set(c["pickup"]) | set(c["delivery"]))
            resid = c["y"].to_numpy() - self.stage1.predict_log(c)
            self.lgbm_ = lgb.LGBMRegressor(**self.lgbm_params).fit(self._residual_X(c), resid)
            if self.route_unseen:
                self.lgbm_nocity_ = lgb.LGBMRegressor(**self.lgbm_params).fit(
                    self._residual_X(c, city_cats=False), resid)
        return self

    def unseen_city(self, t: pd.DataFrame) -> np.ndarray:
        known = set(self.cities_)
        return (~t["pickup"].isin(known) | ~t["delivery"].isin(known)).to_numpy()

    def predict_log(self, t: pd.DataFrame) -> np.ndarray:
        out = self.stage1.predict_log(t)
        if self.stage2:
            r = self.lgbm_.predict(self._residual_X(t))
            if self.route_unseen:
                u = self.unseen_city(t)
                if u.any():
                    r[u] = self.lgbm_nocity_.predict(self._residual_X(t[u], city_cats=False))
            out = out + r
        return out

    def predict(self, t: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(t))


class NoTimeLinear:
    """Baseline: log-price OLS with pickup/delivery city indicators and no time terms."""

    def fit(self, t):
        self.corrupt_ = clean.corrupted_mask(t)
        c = t[~self.corrupt_]
        self.pickups_, self.deliveries_ = sorted(c["pickup"].unique()), sorted(c["delivery"].unique())
        X = self._X(c)
        self.columns_ = list(X.columns)
        self.ols_ = LinearRegression().fit(X, c["y"])
        return self

    def _X(self, t):
        X = t[["ld", "lw", "wsmall", "w_miss", "lmi", "is_Flatbed", "is_Reefer"]].astype(float)
        cities = {f"p_{p}": (t["pickup"] == p) for p in self.pickups_[1:]}
        cities |= {f"d_{d}": (t["delivery"] == d) for d in self.deliveries_[1:]}
        return pd.concat([X, pd.DataFrame(cities, index=t.index).astype(float)], axis=1)

    def predict(self, t):
        return np.exp(self.ols_.predict(self._X(t)[self.columns_]))


class CodexTree:
    """Codex's tree baselines: raw dollars, L1 loss, all rows (no label cleaning)."""

    def __init__(self, kind="lgbm", quote_signal=True, seed=config.SEED):
        self.kind = kind
        self.features = [f for f in CODEX_FEATURES if quote_signal or f != "quote_signal"]
        self.seed = seed

    def _X(self, t):
        X = t[self.features].copy()
        for c in CODEX_CATEGORICAL:
            X[c] = pd.Categorical(t[c], categories=self.cats_[c])
        return X

    def fit(self, t):
        self.cats_ = {c: sorted(t[c].unique()) for c in CODEX_CATEGORICAL}
        if self.kind == "lgbm":
            self.m_ = lgb.LGBMRegressor(
                objective="regression_l1", n_estimators=1500, learning_rate=0.03, num_leaves=63,
                min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=self.seed, verbose=-1)
        else:
            self.m_ = xgb.XGBRegressor(
                objective="reg:absoluteerror", tree_method="hist", enable_categorical=True,
                n_estimators=1500, learning_rate=0.03, max_depth=8, subsample=0.8,
                colsample_bytree=0.8, max_cat_to_onehot=1, random_state=self.seed)
        self.m_.fit(self._X(t), t["posted_rate"])
        return self

    def predict(self, t):
        return self.m_.predict(self._X(t))
