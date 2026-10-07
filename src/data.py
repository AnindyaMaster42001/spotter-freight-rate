"""Load the four Spotter files and check their schema.

Run ``python -m src.data`` to load everything and print a short summary.
"""
from __future__ import annotations

import pandas as pd

from . import config

FEATURE_COLUMNS = [
    "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat",
    "delivery_lon", "distance", "equipment", "weight", "date", "market_index", "quote_signal",
]
TRAIN_COLUMNS = FEATURE_COLUMNS + ["posted_rate"]
TEMPLATE_COLUMNS = ["load_id", "predicted_rate"]
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]

NUMERIC = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "weight",
           "market_index", "quote_signal"]
# Columns that may legitimately be missing in the raw files (handled in clean.py).
NULLABLE = {"weight", "market_index"}


class SchemaError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _check_loads(df: pd.DataFrame, name: str, columns: list[str], id_prefix: str,
                 start: str, end: str) -> None:
    _check(list(df.columns) == columns, f"{name}: unexpected columns {list(df.columns)}")
    _check(df["load_id"].is_unique, f"{name}: duplicate load_id")
    _check(df["load_id"].str.match(rf"^{id_prefix}-\d{{6}}$").all(), f"{name}: malformed load_id")
    for col in NUMERIC + (["posted_rate"] if "posted_rate" in columns else []):
        _check(pd.api.types.is_numeric_dtype(df[col]), f"{name}: {col} is not numeric")
        if col not in NULLABLE:
            _check(df[col].notna().all(), f"{name}: {col} has missing values")
    _check(df["date"].notna().all(), f"{name}: unparseable dates")
    _check(df["date"].min() >= pd.Timestamp(start) and df["date"].max() <= pd.Timestamp(end),
           f"{name}: dates outside {start}..{end}")
    _check(set(df["equipment"]) <= set(config.EQUIPMENT), f"{name}: unknown equipment")
    _check((df["distance"] > 0).all(), f"{name}: non-positive distance")
    _check((df["market_index"].dropna() > 0).all(), f"{name}: non-positive market_index")
    # Each city must map to a single coordinate pair (coordinates are shifted but consistent).
    for side in ("pickup", "delivery"):
        n = df.groupby(side)[[f"{side}_lat", f"{side}_lon"]].nunique().max().max()
        _check(n == 1, f"{name}: a {side} city has more than one coordinate pair")


def load_train(path=config.TRAIN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    _check_loads(df, "train", TRAIN_COLUMNS, "TR", "2025-01-01", "2025-10-31")
    _check((df["posted_rate"] > 0).all(), "train: non-positive posted_rate")
    return df


def load_validation(path=config.VALIDATION_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    _check_loads(df, "validation", FEATURE_COLUMNS, "TE", "2025-11-01", "2025-12-31")
    return df


def load_template(path=config.TEMPLATE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    _check(list(df.columns) == TEMPLATE_COLUMNS, f"template: unexpected columns {list(df.columns)}")
    _check(len(df) == 12_000, "template: expected 12,000 rows")
    expected = [f"TE-{i:06d}" for i in range(1, 12_001)]
    _check(sorted(df["load_id"]) == expected, "template: IDs are not TE-000001..TE-012000")
    return df


def load_december(path=config.DECEMBER_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    _check(list(df.columns) == DECEMBER_COLUMNS, f"december: unexpected columns {list(df.columns)}")
    days = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    _check(len(df) == 31 and set(df["date"]) == set(days), "december: need one row per day of Dec 2025")
    _check(df["pickup"].eq("Lexington").all() and df["delivery"].eq("Fort Wayne").all(),
           "december: lane must be Lexington -> Fort Wayne")
    _check(df["equipment"].eq("Dry Van").all(), "december: equipment must be Dry Van")
    return df


def load_all() -> dict[str, pd.DataFrame]:
    train, val = load_train(), load_validation()
    template, december = load_template(), load_december()
    _check(set(template["load_id"]) == set(val["load_id"]), "template IDs do not match validation IDs")
    for city in ("Lexington", "Fort Wayne"):
        _check(city in set(train["pickup"]) | set(train["delivery"]), f"{city} not in training data")
    return {"train": train, "validation": val, "template": template, "december": december}


def main() -> None:
    frames = load_all()
    print(f"pandas {pd.__version__}: schema checks passed")
    for name, df in frames.items():
        span = ""
        if "date" in df and pd.api.types.is_datetime64_any_dtype(df["date"]):
            span = f"  {df['date'].min():%Y-%m-%d} -> {df['date'].max():%Y-%m-%d}"
        print(f"  {name:<11}{len(df):>7,} rows x {df.shape[1]} cols{span}")


if __name__ == "__main__":
    main()
