"""Data-quality counts from the plan (section 2) must reproduce exactly."""
import numpy as np
import pytest

from src import clean, data


@pytest.fixture(scope="module")
def frames():
    tr, va = data.load_train(), data.load_validation()
    return tr, va, clean.prepare(tr, clean.weight_fill_value(tr))


def test_weight_and_market_index_counts(frames):
    tr, va, t = frames
    assert t["w_miss"].sum() == 300 and t["w_neg"].sum() == 292
    assert (tr["weight"] >= 47_500).sum() == 1_191 and t["w_cap"].sum() == 1_204
    assert (tr["weight"] == 5_000).sum() == 31
    assert t["mi_miss"].sum() == 374
    assert t["wf"].notna().all() and (t["wf"] > 0).all() and t["mi"].notna().all()


def test_validation_rows_untouched(frames):
    _, va, _ = frames
    v = clean.prepare(va, 31_496.0)
    assert len(v) == len(va) and (v["load_id"].to_numpy() == va["load_id"].to_numpy()).all()
    assert v["mi"].notna().all() and v["wf"].notna().all()


def test_corrupted_labels_separate_cleanly(frames):
    _, _, t = frames
    a = np.abs(clean.label_residuals(t))
    assert (a >= 0.25).sum() == 677
    assert not ((a > 0.15) & (a < 0.75)).any()   # empty gap between clean rows and both clusters


def test_detector_inside_fold(frames):
    tr, _, _ = frames
    s = tr[tr["date"] <= "2025-08-31"]
    assert clean.corrupted_mask(clean.prepare(s, clean.weight_fill_value(s))).sum() == 533
