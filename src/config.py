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
