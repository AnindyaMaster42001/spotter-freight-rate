"""Feature builders.

``add_features`` expects a frame that has been through ``clean.fix_inputs`` (it needs
``wf``, ``w_miss`` and ``mi``). ``struct_X`` is the design matrix of the stage-1 linear
model on log price; the corrupted-label detector in ``clean.py`` reuses it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

GEO = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df.copy()
    t["ld"] = np.log(t["distance"])
    t["lw"] = np.log(t["wf"].clip(lower=config.WEIGHT_CLIP_MIN))
    t["wsmall"] = (t["wf"] < config.SMALL_WEIGHT_KNOT) * np.log(
        config.SMALL_WEIGHT_KNOT / t["wf"].clip(lower=config.WEIGHT_CLIP_MIN))
    t["lmi"] = np.log(t["mi"])

    d = t["date"]
    t["tdays"] = (d - pd.Timestamp(config.TIME_ORIGIN)).dt.days
    t["tm"] = t["tdays"] / config.DAYS_PER_MONTH
    t["month"] = d.dt.month
    t["quarter"] = d.dt.quarter
    t["dow"] = d.dt.dayofweek
    t["dom"] = d.dt.day
    # Position within the month, 0 on the 1st and 1 on the last day.
    t["frac"] = (t["dom"] - 1) / (d.dt.days_in_month - 1)
    t["qe"] = np.where(t["month"] % 3 == 0, t["frac"], 0.0)   # quarter-end month ramp

    t["eq"] = t["equipment"].map({e: i for i, e in enumerate(config.EQUIPMENT)})
    for e in config.EQUIPMENT[1:]:
        t[f"is_{e}"] = (t["equipment"] == e).astype(float)
    for e in config.EQUIPMENT:
        t[f"qe_{e}"] = t["qe"] * (t["equipment"] == e)

    t["dlat"] = t["delivery_lat"] - t["pickup_lat"]
    t["dlon"] = t["delivery_lon"] - t["pickup_lon"]
    t["lane"] = t["pickup"] + "__" + t["delivery"]
    if "posted_rate" in t:
        t["y"] = np.log(t["posted_rate"])
    return t


def struct_X(t: pd.DataFrame, trend: bool = True, ramp: bool = True) -> pd.DataFrame:
    """Stage-1 design: log distance/weight/market index, equipment, smooth geography, time."""
    X = t[["ld", "lw", "wsmall", "w_miss", "lmi", "is_Flatbed", "is_Reefer"] + GEO].astype(float)
    for s in ("pickup", "delivery"):
        X[f"{s}_lat2"] = t[f"{s}_lat"] ** 2
        X[f"{s}_lon2"] = t[f"{s}_lon"] ** 2
        X[f"{s}_ll"] = t[f"{s}_lat"] * t[f"{s}_lon"]
    if trend:
        X["tm"] = t["tm"]
    if ramp:
        for e in config.EQUIPMENT:
            X[f"qe_{e}"] = t[f"qe_{e}"]
    return X
