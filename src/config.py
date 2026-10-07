"""Paths, seeds and fixed constants shared by the whole pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
SCORER_DIR = ROOT / "scorer_results"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
CHOSEN_CONFIG_PATH = ARTIFACTS_DIR / "chosen_config.json"

SEED = 0

EQUIPMENT = ("Dry Van", "Flatbed", "Reefer")
TIME_ORIGIN = "2025-01-01"
DAYS_PER_MONTH = 30.4

# Weight data-quality rules
WEIGHT_CAP = 47_500.0       # |weight| at or above this is censored
WEIGHT_FLOOR = 5_000.0      # |weight| exactly at this value is a stuck reading
WEIGHT_FILL = 31_400.0      # fallback fill (training median); clean.py recomputes it from the fit frame
WEIGHT_CLIP_MIN = 1_000.0   # lower clip before taking logs
SMALL_WEIGHT_KNOT = 12_000.0

# Corrupted-label detector (robust two-pass residual cut on log price)
RESID_REFIT_CUT = 0.30
RESID_FLAG_CUT = 0.25

# ---------------------------------------------------------------------------
# Pre-registered tuning grid (committed before any tuning run).
# Coordinate-wise search in this order; each step starts from the winner so far.
# Score = raw MAE averaged over folds F1-F3. Candidates within TIE_TOLERANCE dollars
# of the best score are tied, and the tie goes to the lowest h2 (Sep-Oct) MAE.
# The corrupted-label cut (0.20/0.25/0.30) is not tuned: all three flag the same
# 677 rows (empty residual gap 0.118-0.795, see Phase B).
HYBRID_DEFAULTS = dict(trend="linear", ramp=True, damp=0.5, n_estimators=600, num_leaves=31,
                       loss="huber", city_cats=False)
TUNING_STEPS = [
    ("trend form", [dict(trend="linear"), dict(trend="step"), dict(trend="damped", damp=0.5)]),
    ("stage-2 size", [dict(n_estimators=n, num_leaves=l) for n in (400, 600, 1000) for l in (15, 31)]),
    ("stage-2 loss", [dict(loss=x) for x in ("huber", "l2", "l1")]),
    ("city categoricals in stage 2", [dict(city_cats=False), dict(city_cats=True)]),
]
TIE_TOLERANCE = 0.25
STABILITY_SEEDS = (0, 1, 2)
