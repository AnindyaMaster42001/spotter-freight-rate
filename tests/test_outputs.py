"""Output-format checks mirroring score.py, plus template order. Run after src.predict."""
import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture(scope="module")
def preds():
    if not config.PREDICTIONS_PATH.is_file():
        pytest.skip("run `python -m src.predict` first")
    return pd.read_csv(config.PREDICTIONS_PATH)


def test_prediction_file(preds):
    assert list(preds.columns) == ["load_id", "predicted_rate"]
    assert len(preds) == 12_000
    assert not preds["load_id"].isna().any() and not preds["load_id"].duplicated().any()
    assert set(preds["load_id"]) == {f"TE-{i:06d}" for i in range(1, 12_001)}
    v = pd.to_numeric(preds["predicted_rate"], errors="coerce")
    assert v.notna().all() and np.isfinite(v).all() and (v > 0).all()


def test_template_order(preds):
    template = pd.read_csv(config.TEMPLATE_PATH)
    assert (preds["load_id"].to_numpy() == template["load_id"].to_numpy()).all()


def test_december_file():
    d = pd.read_csv(config.DECEMBER_PATH)
    assert list(d.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    dates = pd.to_datetime(d["date"])
    assert len(d) == 31 and not dates.duplicated().any()
    assert set(dates) == set(pd.date_range("2025-12-01", "2025-12-31", freq="D"))
    assert d["pickup"].eq("Lexington").all() and d["delivery"].eq("Fort Wayne").all()
    assert np.isclose(d["distance"], 360).all() and np.isclose(d["weight"], 32_000).all()
    assert d["equipment"].eq("Dry Van").all()
    p = pd.to_numeric(d["predicted_rate"], errors="coerce")
    if p.isna().all():
        pytest.skip("December predictions not filled yet")
    assert p.notna().all() and np.isfinite(p).all() and (p > 0).all()
