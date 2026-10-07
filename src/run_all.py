"""Run the whole pipeline end to end.

    python -m src.run_all                    # everything (about 10 minutes on 16 cores)
    python -m src.run_all --skip-validation  # reuse artifacts/fold_*.csv and contrasts.csv

Steps: schema check -> data-quality table -> forward-fold benchmark + contrasts ->
final fit on Jan-Oct -> predictions + December chart -> score.py -> figures -> DOCX report.
Tuning (python -m src.tune) is not rerun; its result is committed in
artifacts/chosen_config.json.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from . import clean, config, data, figures, predict, report, train, validate


def step(name, fn, *args):
    t0 = time.time()
    print(f"\n== {name}", flush=True)
    out = fn(*args)
    print(f"   ({time.time() - t0:.0f}s)", flush=True)
    return out


def score():
    subprocess.run([sys.executable, "score.py", "--predictions", str(config.PREDICTIONS_PATH.name),
                    "--december-predictions", "data/december_chart_inputs.csv"],
                   cwd=config.ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()
    step("schema checks", data.main)
    step("data quality", clean.main)
    if not args.skip_validation:
        sys.argv = [sys.argv[0], "--contrasts"]
        step("forward-fold benchmark + contrasts", validate.main)
    step("final fit (Jan-Oct)", train.main)
    step("predictions + December chart", predict.main)
    step("score.py", score)
    step("figures", figures.main)
    step("DOCX report", report.main)


if __name__ == "__main__":
    main()
