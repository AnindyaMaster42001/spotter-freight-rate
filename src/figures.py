"""Report figures, drawn with matplotlib from data/ and artifacts/ (reproducible).

``python -m src.figures`` writes PNGs to reports/figures/.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import clean, config, data, models

FIG_DIR = config.REPORTS_DIR / "figures"
# Categorical slots 1-3 of the validated reference palette (all-pairs CVD safe).
EQ_COLOR = {"Dry Van": "#2a78d6", "Flatbed": "#eb6834", "Reefer": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
NEUTRAL = "#b9b8b2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 200, "lines.linewidth": 1.6,
})


def _save(fig, name):
    """Vector PDF for the LaTeX report."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _clean_train():
    raw = data.load_train()
    t = clean.prepare(raw, clean.weight_fill_value(raw))
    return t[~clean.corrupted_mask(t)].copy()


def time_structure(c):
    """Daily mean residual by equipment from a model with no time terms."""
    m = models.StructuralModel(trend="none", ramp=False).fit(c)
    c = c.assign(r=(c["y"] - m.predict_log(c)) * 100)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for e, col in EQ_COLOR.items():
        s = c[c["equipment"] == e].groupby("date")["r"].mean().rolling(3, center=True, min_periods=1).mean()
        ax.plot(s.index, s.to_numpy(), color=col, lw=1.3, label=e)
    for q in ("2025-03-31", "2025-06-30", "2025-09-30"):
        ax.axvline(pd.Timestamp(q), color=MUTED, lw=0.8, ls=":")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("Residual vs. no-time model (%)")
    ax.set_title("Price level drifts up and ramps through each quarter-end month")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.legend(loc="upper left", ncol=3)
    _save(fig, "time_structure.png")


def market_index(train, val):
    both = pd.concat([train.assign(src="Train (Jan-Oct)"), val.assign(src="Validation (Nov-Dec)")])
    daily = both.groupby(["date", "src"])["market_index"].mean().reset_index()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [2.3, 1]})
    for src, col in (("Train (Jan-Oct)", EQ_COLOR["Dry Van"]), ("Validation (Nov-Dec)", EQ_COLOR["Flatbed"])):
        d = daily[daily["src"] == src]
        a.plot(d["date"], d["market_index"], color=col, lw=1.0, label=src)
    a.text(pd.Timestamp("2025-05-20"), 1.43, "Train (Jan-Oct)", color=INK2, fontsize=8, ha="center")
    a.text(pd.Timestamp("2025-12-01"), 1.10, "Validation\n(Nov-Dec)", color=INK2, fontsize=8, ha="center")
    a.set_ylim(None, 1.5)
    a.set_title("Daily mean market_index")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = np.arange(7)
    for i, (src, col) in enumerate((("Train (Jan-Oct)", EQ_COLOR["Dry Van"]),
                                     ("Validation (Nov-Dec)", EQ_COLOR["Flatbed"]))):
        d = both[both["src"] == src]
        prof = d.groupby(d["date"].dt.dayofweek)["market_index"].mean()
        b.plot(x, prof / prof.mean(), color=col, marker="o", ms=4, lw=1.3, label=src.split(" ")[0])
    b.set_xticks(x, days)
    b.legend(loc="lower center", fontsize=7.5)
    b.set_title("Weekday profile (÷ mean)")
    fig.tight_layout()
    _save(fig, "market_index.png")


def city_effects(c):
    """City price effect (avg of pickup and delivery indicator) vs latitude."""
    base = c[["ld", "lw", "wsmall", "w_miss", "lmi", "is_Flatbed", "is_Reefer", "tm"] +
             [f"qe_{e}" for e in config.EQUIPMENT]].astype(float)
    P = pd.get_dummies(c["pickup"], prefix="p", dtype=float)
    D = pd.get_dummies(c["delivery"], prefix="d", dtype=float)
    X = pd.concat([base, P.iloc[:, 1:], D.iloc[:, 1:]], axis=1)
    m = LinearRegression().fit(X, c["y"])
    coef = pd.Series(m.coef_, index=X.columns)
    cities = sorted(set(c["pickup"]) & set(c["delivery"]))
    pe = pd.Series({k: coef.get(f"p_{k}", 0.0) for k in cities})
    de = pd.Series({k: coef.get(f"d_{k}", 0.0) for k in cities})
    pe, de = pe - pe.mean(), de - de.mean()
    eff = (pe + de) / 2 * 100
    lat = c.groupby("pickup")["pickup_lat"].first().reindex(cities)
    r_pd = np.corrcoef(pe, de)[0, 1]
    r_lat = np.corrcoef(eff, lat)[0, 1]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.scatter(lat, eff, s=22, color=EQ_COLOR["Dry Van"], edgecolor="white", linewidth=0.6, zorder=3)
    b1, b0 = np.polyfit(lat, eff, 1)
    xs = np.linspace(lat.min(), lat.max(), 10)
    ax.plot(xs, b0 + b1 * xs, color=MUTED, lw=1.0, ls="--")
    for k in (eff.idxmax(), eff.idxmin()):
        ax.annotate(k, (lat[k], eff[k]), xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("City latitude (as given)")
    ax.set_ylabel("City price effect (%)")
    ax.set_title("Northern cities are cheaper")
    ax.text(0.98, 0.95, f"r(effect, latitude) = {r_lat:.2f}\nr(pickup, delivery effect) = {r_pd:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    _save(fig, "city_effects.png")
    return {"r_lat": r_lat, "r_pickup_delivery": r_pd, "range_pct": (eff.min(), eff.max())}


def corrupted_labels():
    f = pd.read_csv(config.ARTIFACTS_DIR / "label_flags.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    bins = np.linspace(-2.0, 2.0, 161)
    ax.hist(f.loc[f["corrupt"] == 0, "log_resid"], bins=bins, color=EQ_COLOR["Dry Van"], label="Kept")
    ax.hist(f.loc[f["corrupt"] == 1, "log_resid"], bins=bins, color=EQ_COLOR["Flatbed"], label="Flagged as corrupted")
    ax.set_yscale("log")
    for x in (-config.RESID_FLAG_CUT, config.RESID_FLAG_CUT):
        ax.axvline(x, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("log(actual / expected price)")
    ax.set_ylabel("Loads (log scale)")
    ax.set_title("Two clusters of corrupted labels, far from the clean rows")
    hi, lo = f[(f["corrupt"] == 1) & (f["log_resid"] > 0)], f[(f["corrupt"] == 1) & (f["log_resid"] < 0)]
    ax.annotate(f"×{np.exp(hi['log_resid'].median()):.1f}  (n={len(hi)})", (hi["log_resid"].median(), 30),
                ha="center", fontsize=8, color=INK2)
    ax.annotate(f"×{np.exp(lo['log_resid'].median()):.2f}  (n={len(lo)})", (lo["log_resid"].median(), 30),
                ha="center", fontsize=8, color=INK2)
    ax.legend(loc="upper left")
    _save(fig, "corrupted_labels.png")


def folds_diagram():
    rows = [("F1", "2025-01-01", "2025-08-01", "2025-09-01"),
            ("F2", "2025-01-01", "2025-09-01", "2025-10-01"),
            ("F3", "2025-01-01", "2025-10-01", "2025-11-01"),
            ("h2", "2025-01-01", "2025-09-01", "2025-11-01"),
            ("Final", "2025-01-01", "2025-11-01", "2026-01-01")]
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    for i, (name, s, m, e) in enumerate(rows[::-1]):
        s, m, e = map(pd.Timestamp, (s, m, e))
        ax.barh(i, (m - s).days, left=s, height=0.62, color=NEUTRAL)
        ax.barh(i, (e - m).days - 2, left=m + pd.Timedelta(days=2), height=0.62,
                color=EQ_COLOR["Flatbed"] if name == "Final" else EQ_COLOR["Dry Van"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Expanding forward folds (grey = train, blue = test, orange = Nov-Dec to predict)")
    _save(fig, "folds.png")


def backtest_bias():
    h = pd.read_csv(config.ARTIFACTS_DIR / "h2_predictions.csv", parse_dates=["date"])
    h = h[h["report_clean"]]
    series = [("Hybrid (chosen)", EQ_COLOR["Dry Van"]), ("Codex LGBM-L1", EQ_COLOR["Flatbed"]),
              ("Hybrid without trend", EQ_COLOR["Reefer"])]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for name, col in series:
        err = np.log(h[name] / h["posted_rate"]) * 100
        s = err.groupby(h["date"]).mean()
        ax.plot(s.index, s.to_numpy(), color=col, lw=1.4, label=name)
        ax.annotate(f"{s.iloc[-5:].mean():+.1f}%", (s.index[-1], s.iloc[-5:].mean()), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(pd.Timestamp("2025-10-01"), color=MUTED, lw=0.8, ls=":")
    ax.set_ylabel("Daily mean error (%)")
    ax.set_title("h2 backtest (train Jan-Aug): daily mean signed error, clean labels")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
    _save(fig, "backtest_bias.png")


def quote_signal_check(c):
    """Residual SD of log price within lane x equipment groups, with and without quote_signal."""
    g = c["lane"] + "|" + c["equipment"]
    cols = ["lw", "wsmall", "w_miss", "lmi", "tm"] + [f"qe_{e}" for e in config.EQUIPMENT]
    X = c[cols].astype(float)
    Xq = X.assign(lqs=np.log(c["quote_signal"]))
    y = c["y"] - c["y"].groupby(g).transform("mean")
    out = {}
    for name, M in (("without", X), ("with", Xq)):
        Md = M - M.groupby(g).transform("mean")
        out[name] = float(np.std(y - LinearRegression().fit(Md, y).predict(Md)))
    return out


def december_crosscheck():
    d = pd.read_csv(config.ARTIFACTS_DIR / "december_predictions.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for name, col, lab in (("hybrid", EQ_COLOR["Dry Van"], "Hybrid (submitted)"),
                           ("codex_lgbm_no_market_index", EQ_COLOR["Flatbed"], "Codex-style LGBM, no market_index")):
        ax.plot(d["date"], d[name], color=col, marker="o", ms=3, lw=1.4, label=lab)
    ax.set_ylabel("Predicted rate ($)")
    ax.set_title("December chart lane: submitted model vs. a model without market_index")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left")
    _save(fig, "december_crosscheck.png")


def main() -> dict:
    train, val = data.load_train(), data.load_validation()
    c = _clean_train()
    time_structure(c)
    market_index(train, val)
    stats = city_effects(c)
    corrupted_labels()
    folds_diagram()
    backtest_bias()
    december_crosscheck()
    stats["quote_signal"] = quote_signal_check(c)
    stats["corr_distance_raw"] = float(np.corrcoef(train["distance"], train["posted_rate"])[0, 1])
    stats["corr_distance_clean"] = float(np.corrcoef(c["distance"], c["posted_rate"])[0, 1])
    stats["mi_mean_train"] = float(train["market_index"].mean())
    stats["mi_mean_val"] = float(val["market_index"].mean())
    print("figures ->", FIG_DIR.relative_to(config.ROOT), stats)
    return stats


if __name__ == "__main__":
    main()
