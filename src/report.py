"""Build reports/Freight_Rate_Report.pdf (LaTeX).

The prose lives in reports/Freight_Rate_Report.tex. Every number and table in it comes
from files this module writes to reports/generated/ (``numbers.tex`` holds one macro per
number, ``tab_*.tex`` hold the tables), all read from artifacts/. The PDF is then compiled
with latexmk. Run after src.validate, src.predict, score.py (src.run_all does this).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from . import config, figures

TEX = config.REPORTS_DIR / "Freight_Rate_Report.tex"
PDF = TEX.with_suffix(".pdf")
GEN = config.REPORTS_DIR / "generated"
A = config.ARTIFACTS_DIR


# ---------------------------------------------------------------- formatting
def esc(s: str) -> str:
    """Escape plain text for LaTeX."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    return s.replace("|w|", r"$|w|$").replace(">=", r"$\geq$").replace("<=", r"$\leq$")


def num(x, fmt=".1f") -> str:
    return format(x, fmt).replace("-", r"\textminus{}")


def money(x, fmt=",.1f") -> str:
    return r"\$" + num(x, fmt)


def pct(x, fmt=".2f") -> str:
    return num(x, fmt) + r"\%"


def write_tabular(path, df: pd.DataFrame, colspec: str, bold_first=False):
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             " & ".join(rf"\textbf{{{c}}}" for c in df.columns) + r" \\", r"\midrule"]
    for i, row in enumerate(df.itertuples(index=False)):
        cells = [str(v) for v in row]
        if bold_first and i == 0:
            cells = [rf"\textbf{{{c}}}" for c in cells]
            lines.append(r"\rowcolor{chosen}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- generated inputs
def generate(stats: dict) -> None:
    s = pd.read_csv(A / "fold_summary.csv", index_col=0)
    con = pd.read_csv(A / "contrasts.csv")
    dq = pd.read_csv(A / "data_quality.csv")
    chosen = json.loads((A / "chosen_config.json").read_text())
    checks = json.loads((A / "prediction_checks.json").read_text())
    dec = pd.read_csv(A / "december_predictions.csv")
    coef = json.loads((A / "final_model" / "stage1_coefficients.json").read_text())
    flags = pd.read_csv(A / "label_flags.csv")
    GEN.mkdir(parents=True, exist_ok=True)

    H = s.loc["Hybrid (chosen)"]
    cv = lambda m, k: con[(con.setting.str.startswith(m)) & (con.model == k)].iloc[0]
    months = (pd.Timestamp("2025-10-31") - pd.Timestamp(config.TIME_ORIGIN)).days / config.DAYS_PER_MONTH
    corrupt = flags[flags.corrupt == 1]
    hi, lo = corrupt[corrupt.log_resid > 0], corrupt[corrupt.log_resid < 0]
    gap_lo = flags.loc[flags.corrupt == 0, "log_resid"].abs().max()
    gap_hi = corrupt["log_resid"].abs().min()
    rpm = checks["median_rate_per_mile"]
    ramp = {e: (np.exp(coef[f"qe_{e}"]) - 1) * 100 for e in config.EQUIPMENT}
    seeds = list(chosen["seed_stability_h2_mae"].values())

    n = {
        # headline
        "HybAvgMAE": money(H["avg MAE"]), "HybHtwoMAE": money(H["h2 MAE"]),
        "HybAvgCMAPE": pct(H["avg cMAPE"]), "HybHtwoCMAPE": pct(H["h2 cMAPE"]),
        "HybAvgMAPE": pct(H["avg MAPE"]), "HybOctBias": pct(abs(H["Oct cBias%"]), ".1f"),
        "BlendAvgMAE": money(s.loc["Codex blend 0.75 XGB / 0.25 LGBM", "avg MAE"]),
        "LgbmAvgMAE": money(s.loc["Codex LGBM-L1", "avg MAE"]),
        "LgbmNoQsAvgMAE": money(s.loc["Codex LGBM-L1 without quote_signal", "avg MAE"]),
        "HybQsAvgMAE": money(s.loc["Hybrid + quote_signal", "avg MAE"]),
        "HybUntunedAvgMAE": money(s.loc["Hybrid, untuned defaults", "avg MAE"]),
        "HybNoMiHtwoCMAPE": pct(s.loc["Hybrid without market_index", "h2 cMAPE"]),
        "RmseMin": money(s["avg RMSE"].min(), ",.0f"), "RmseMax": money(s["avg RMSE"].max(), ",.0f"),
        "SeedsHtwo": " / ".join(money(v) for v in seeds),
        # splits
        "RndHyb": money(cv("Random", "Hybrid (chosen)").MAE), "RndLgbm": money(cv("Random", "Codex LGBM-L1").MAE),
        "FwdHyb": money(cv("Forward", "Hybrid (chosen)").MAE), "FwdLgbm": money(cv("Forward", "Codex LGBM-L1").MAE),
        "UcHyb": money(cv("Unseen", "Hybrid (chosen)").MAE), "UcLgbm": money(cv("Unseen", "Codex LGBM-L1").MAE),
        "UcNotRouted": money(cv("Unseen", "Hybrid, city cats not routed").MAE),
        # data
        "NCorrupt": f"{len(corrupt):,}", "PctCorrupt": pct(len(corrupt) / len(flags) * 100, ".1f"),
        "NHigh": f"{len(hi)}", "NLow": f"{len(lo)}",
        "MultHigh": num(np.exp(hi.log_resid.median()), ".1f"), "MultLow": num(np.exp(lo.log_resid.median()), ".2f"),
        "GapLo": num(gap_lo, ".2f"), "GapHi": num(gap_hi, ".2f"),
        "CorrClean": num(stats["corr_distance_clean"], ".3f"), "CorrRaw": num(stats["corr_distance_raw"], ".3f"),
        "RpmDV": money(rpm["Dry Van"]["train_clean"], ".2f"), "RpmFB": money(rpm["Flatbed"]["train_clean"], ".2f"),
        "RpmRF": money(rpm["Reefer"]["train_clean"], ".2f"),
        "PredRpmDV": money(rpm["Dry Van"]["predicted"], ".2f"), "PredRpmFB": money(rpm["Flatbed"]["predicted"], ".2f"),
        "PredRpmRF": money(rpm["Reefer"]["predicted"], ".2f"),
        "DistSlope": num(coef["ld"], ".2f"), "MiElast": num(coef["lmi"], ".2f"),
        "MiTrain": num(stats["mi_mean_train"], ".2f"), "MiVal": num(stats["mi_mean_val"], ".2f"),
        "Drift": pct((np.exp(coef["tm"] * months) - 1) * 100, ".0f"),
        "RampDV": pct(ramp["Dry Van"], ".1f"), "RampFB": pct(ramp["Flatbed"], ".1f"), "RampRF": pct(ramp["Reefer"], ".1f"),
        "CityLo": pct(stats["range_pct"][0], "+.1f"), "CityHi": pct(stats["range_pct"][1], "+.1f"),
        "RLat": num(stats["r_lat"], ".2f"), "RPd": num(stats["r_pickup_delivery"], ".2f"),
        "QsWithout": num(stats["quote_signal"]["without"], ".4f"), "QsWith": num(stats["quote_signal"]["with"], ".4f"),
        "NUnseen": f"{checks['unseen_city_rows']['n']:,}",
        "UnseenRpm": money(checks["unseen_city_rows"]["mean_rate_per_mile_unseen"], ".2f"),
        "SeenRpm": money(checks["unseen_city_rows"]["mean_rate_per_mile_seen"], ".2f"),
        # December
        "DecFirst": money(checks["december"]["first"], ",.0f"), "DecLast": money(checks["december"]["last"], ",.0f"),
        "DecMean": money(dec["hybrid"].mean(), ",.0f"),
        "DecNoMiMin": money(dec["codex_lgbm_no_market_index"].min(), ",.0f"),
        "DecNoMiMax": money(dec["codex_lgbm_no_market_index"].max(), ",.0f"),
        # environment
        "Versions": esc(f"Python 3.12, pandas {pd.__version__}, numpy {np.__version__}, scikit-learn "
                        f"{sklearn.__version__}, LightGBM {lightgbm.__version__}; seeds fixed at {config.SEED}"),
    }
    (GEN / "numbers.tex").write_text(
        "% Generated by src/report.py from artifacts/. Do not edit.\n"
        + "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in n.items()))

    # data-quality table
    d = dq.copy()
    d["validation"] = d["validation"].map(lambda v: "---" if pd.isna(v) else f"{int(v):,}")
    d["train"] = d["train"].map(lambda v: f"{int(v):,}")
    d["issue"] = d["issue"].map(lambda x: (r"\quad " + esc(x.strip())) if x.startswith("  ") else esc(x))
    d["issue"] = d["issue"].str.replace(r"x(\d)", r"$\\times$\1", regex=True)
    d["handling"] = d["handling"].fillna("").map(esc)
    d = d[["issue", "train", "validation", "handling"]]
    d.columns = ["Issue", "Train", "Validation", "Handling"]
    write_tabular(GEN / "tab_quality.tex", d, r"@{}>{\raggedright\arraybackslash}p{0.44\linewidth}rr>{\raggedright\arraybackslash}p{0.33\linewidth}@{}")

    # split contrast table
    c = pd.DataFrame([["Random 10\\% holdout (all months)", n["RndHyb"], n["RndLgbm"]],
                      ["Forward: train Jan--Sep, test Oct", n["FwdHyb"], n["FwdLgbm"]],
                      ["Unseen cities: 8 withheld, test Sep--Oct", n["UcHyb"], n["UcLgbm"]]],
                     columns=["Split (MAE)", "Hybrid", "LightGBM-L1"])
    write_tabular(GEN / "tab_splits.tex", c, "@{}lrr@{}")

    # model comparison table: chosen first, then by fold-average MAE
    rows = ["Hybrid (chosen)", "Hybrid, untuned defaults", "Hybrid, no city categoricals",
            "Hybrid + quote_signal", "Hybrid without market_index", "Hybrid 50/50 with Codex XGB",
            "Hybrid without quarter-end ramp", "Structural only (stage 1)",
            "Codex LGBM-L1 without quote_signal", "Codex blend 0.75 XGB / 0.25 LGBM", "Codex LGBM-L1",
            "Codex XGB-L1", "Linear, city indicators, no time terms", "Hybrid without trend",
            "Hybrid without trend or ramp"]
    m = s.loc[rows]
    m = pd.concat([m.iloc[:1], m.iloc[1:].sort_values("avg MAE")])
    f = lambda x: num(x, ".1f")
    mt = pd.DataFrame({"Model": [esc(k) for k in m.index], "Aug": m["Aug MAE"].map(f),
                       "Sep": m["Sep MAE"].map(f), "Oct": m["Oct MAE"].map(f), "h2": m["h2 MAE"].map(f),
                       "Fold avg": m["avg MAE"].map(f), "MAPE": m["avg MAPE"].map(lambda x: num(x, ".2f")),
                       "Clean MAPE": m["avg cMAPE"].map(lambda x: num(x, ".2f"))})
    write_tabular(GEN / "tab_models.tex", mt, "@{}lrrrrrrr@{}", bold_first=True)


def compile_pdf() -> None:
    if not shutil.which("latexmk"):
        print("latexmk not found: wrote", GEN.relative_to(config.ROOT), "but did not compile the PDF")
        return
    # Fixed timestamps and ID so rebuilding an unchanged report gives an identical PDF.
    env = {**os.environ, "SOURCE_DATE_EPOCH": "1791331200", "FORCE_SOURCE_DATE": "1"}
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-quiet", TEX.name],
                   cwd=config.REPORTS_DIR, check=True, stdout=subprocess.DEVNULL, env=env)
    subprocess.run(["latexmk", "-c", TEX.name], cwd=config.REPORTS_DIR, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("report ->", PDF.relative_to(config.ROOT))


def main() -> None:
    generate(figures.main())
    compile_pdf()


if __name__ == "__main__":
    main()
