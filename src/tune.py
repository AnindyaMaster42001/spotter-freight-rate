"""Pre-registered coordinate-wise tuning of the hybrid (grid lives in config.py).

``python -m src.tune`` writes artifacts/tuning_results.csv (every candidate) and
artifacts/chosen_config.json (the winner, plus its seed-stability check).
"""
from __future__ import annotations

import json

import pandas as pd

from . import config, models, validate


def score(raw, kwargs, seed=config.SEED) -> dict:
    long = validate.evaluate(raw, {"c": lambda: models.HybridModel(**kwargs, seed=seed)}, verbose=False)
    s = validate.summarize(long).iloc[0]
    return {k: float(s[k]) for k in ["Aug MAE", "Sep MAE", "Oct MAE", "h2 MAE", "avg MAE",
                                      "avg cMAPE", "h2 cMAPE", "Oct cBias%", "h2 cBias%"]}


def pick(cands: list[dict]) -> dict:
    best = min(c["avg MAE"] for c in cands)
    tied = [c for c in cands if c["avg MAE"] <= best + config.TIE_TOLERANCE]
    return min(tied, key=lambda c: c["h2 MAE"])


def main() -> None:
    raw = validate.load_labeled()
    current, rows, cache = dict(config.HYBRID_DEFAULTS), [], {}
    for step, options in config.TUNING_STEPS:
        cands = []
        for opt in options:
            kw = {**current, **opt}
            key = json.dumps(kw, sort_keys=True)
            if key not in cache:
                cache[key] = score(raw, kw)
            c = {"step": step, "option": json.dumps(opt), **cache[key], "config": kw}
            cands.append(c)
            print(f"{step:<30} {c['option']:<45} avg {c['avg MAE']:7.2f}  h2 {c['h2 MAE']:7.2f}  "
                  f"Oct bias {c['Oct cBias%']:+5.2f}%  h2 bias {c['h2 cBias%']:+5.2f}%", flush=True)
        win = pick(cands)
        for c in cands:
            rows.append({**{k: v for k, v in c.items() if k != "config"}, "chosen": c is win})
        current = win["config"]
        print(f"  -> {win['option']}\n", flush=True)

    stability = {s: score(raw, current, seed=s)["h2 MAE"] for s in config.STABILITY_SEEDS}
    print("seed stability (h2 MAE):", {s: round(v, 2) for s, v in stability.items()})

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).round(3).to_csv(config.ARTIFACTS_DIR / "tuning_results.csv", index=False)
    config.CHOSEN_CONFIG_PATH.write_text(json.dumps(
        {"hybrid": current, "score": cache[json.dumps(current, sort_keys=True)],
         "seed_stability_h2_mae": stability}, indent=2) + "\n")
    print("chosen:", current)


if __name__ == "__main__":
    main()
