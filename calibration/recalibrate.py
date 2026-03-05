"""
Post-hoc Calibration: Platt Scaling & Isotonic Regression
===========================================================
For each cohort:
  - Fit Platt (logistic) calibrator on OOF predicted probs vs MACE
  - Fit Isotonic Regression calibrator on same data
  - Recompute E/O ratio, calibration slope/intercept, H-L chi2
  - Recompute Brier Score and Brier Skill Score (BSS)
  - Plot before vs after calibration curves (2 calibrators x 2 cohorts)
  - Save calibrated probabilities to a new CSV

Outputs: outputs/ensemble_model/
  recalibration_curves.pdf/png
  recalibration_metrics.csv
  real_model_predictions_recalibrated.csv        <- Platt-scaled probs
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import logit, expit
from scipy import stats

ROOT = Path(r"D:\cardio-oncology")
OUT  = ROOT / "outputs" / "ensemble_model"

# ── Load OOF predictions ──────────────────────────────────────────────────────
pred = pd.read_csv(OUT / "real_model_predictions_v2.csv", encoding="utf-8-sig")
pred.columns = pred.columns.str.strip()
pred = pred.copy()

COLORS = {"SCH": "#2166AC", "WHU": "#D6604D"}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def calibration_stats(y_true, y_prob, label=""):
    """Compute calibration slope/intercept, E/O, H-L, Brier, BSS."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    prev   = y_true.mean()

    # Calibration slope & intercept  (logit scale)
    lp = logit(y_prob).reshape(-1, 1)
    cal_lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    cal_lr.fit(lp, y_true)
    slope     = float(cal_lr.coef_[0][0])
    intercept = float(cal_lr.intercept_[0])

    # E/O
    eo = y_prob.mean() / max(prev, 1e-9)

    # Hosmer-Lemeshow (10 decile bins)
    df_hl = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p").reset_index(drop=True)
    df_hl["bin"] = pd.qcut(df_hl.index, 10, labels=False)
    hl_stat = 0.0
    for b in range(10):
        g = df_hl[df_hl["bin"] == b]
        o1 = g["y"].sum(); n_b = len(g)
        e1 = g["p"].sum(); e0 = n_b - e1; o0 = n_b - o1
        if e1 > 0: hl_stat += (o1 - e1)**2 / e1
        if e0 > 0: hl_stat += (o0 - e0)**2 / e0
    hl_p = 1 - stats.chi2.cdf(hl_stat, df=8)

    # Brier Score
    brier = np.mean((y_prob - y_true) ** 2)

    # Brier Skill Score (vs naive reference = always predict prevalence)
    brier_ref = np.mean((np.full_like(y_prob, prev) - y_true) ** 2)
    bss = 1.0 - brier / max(brier_ref, 1e-9)

    if label:
        print(f"  [{label}] slope={slope:.3f}, intercept={intercept:.3f}, "
              f"E/O={eo:.3f}, H-L p={hl_p:.4f}, Brier={brier:.5f}, BSS={bss:.4f}")
    return dict(slope=slope, intercept=intercept, eo=eo,
                hl_stat=hl_stat, hl_p=hl_p, brier=brier, bss=bss)


def hl_bins_for_plot(y_true, y_prob, n_bins=10):
    """Return (mean_pred_per_bin, frac_obs_per_bin) for scatter plot."""
    frac_obs, mean_pred = calibration_curve(y_true, y_prob,
                                             n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_obs


# ─────────────────────────────────────────────────────────────────────────────
# Fit calibrators and collect metrics
# ─────────────────────────────────────────────────────────────────────────────
all_metrics = []
cal_probs   = {}   # center -> {"raw": ..., "platt": ..., "isotonic": ...}

for center in ["SCH", "WHU"]:
    df = pred[pred["Center"] == center].copy().reset_index(drop=True)
    y_true  = df["MACE"].values.astype(int)
    y_raw   = df["predicted_prob_within_cv"].values
    n, n_ev = len(y_true), y_true.sum()
    print(f"\n{'='*60}")
    print(f"{center}  (n={n:,}, events={n_ev}, prev={n_ev/n:.4f})")
    print("="*60)

    # ── Before calibration ───────────────────────────────────────────────────
    print("Before calibration:")
    stats_raw = calibration_stats(y_true, y_raw, label="raw")

    # ── Platt scaling (logistic recalibration on logit scale) ────────────────
    # Fit:  logit(p_cal) = a + b * logit(p_raw)
    lp_raw = logit(np.clip(y_raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    platt  = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    platt.fit(lp_raw, y_true)
    y_platt = platt.predict_proba(lp_raw)[:, 1]

    print("After Platt scaling:")
    stats_platt = calibration_stats(y_true, y_platt, label="Platt")

    # ── Isotonic regression calibration ─────────────────────────────────────
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_raw, y_true)
    y_iso = iso.predict(y_raw)

    print("After Isotonic Regression:")
    stats_iso = calibration_stats(y_true, y_iso, label="Isotonic")

    cal_probs[center] = {"raw": y_raw, "platt": y_platt, "isotonic": y_iso,
                          "y_true": y_true, "n": n, "n_ev": n_ev}

    for method, s in [("Raw", stats_raw), ("Platt", stats_platt), ("Isotonic", stats_iso)]:
        all_metrics.append({
            "Cohort": center, "Method": method, "N": n, "N_events": n_ev,
            "Cal_slope":     round(s["slope"],     4),
            "Cal_intercept": round(s["intercept"], 4),
            "E/O_ratio":     round(s["eo"],        4),
            "HL_chi2":       round(s["hl_stat"],   3),
            "HL_p":          round(s["hl_p"],      4),
            "Brier_score":   round(s["brier"],     6),
            "Brier_skill_score": round(s["bss"],   4),
        })

# Save metrics CSV
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(OUT / "recalibration_metrics.csv", index=False, encoding="utf-8-sig")
print("\nSaved: recalibration_metrics.csv")
print(metrics_df[["Cohort","Method","E/O_ratio","Cal_slope","HL_p","Brier_score","Brier_skill_score"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Figure: 2 rows (SCH, WHU) × 3 cols (raw / Platt / Isotonic)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.subplots_adjust(hspace=0.45, wspace=0.38)

method_titles = {
    "raw":      ("Raw (uncalibrated)", "#888888"),
    "platt":    ("Platt Scaling",      "#1A6FBF"),
    "isotonic": ("Isotonic Regression","#CC4400"),
}

for row_i, center in enumerate(["SCH", "WHU"]):
    cp    = cal_probs[center]
    y_true = cp["y_true"]
    color  = COLORS[center]
    n, n_ev = cp["n"], cp["n_ev"]

    for col_j, method_key in enumerate(["raw", "platt", "isotonic"]):
        ax     = axes[row_i, col_j]
        y_prob = cp[method_key]
        title, m_color = method_titles[method_key]

        # Isotonic smoothed line through calibration data
        iso_sm = IsotonicRegression(out_of_bounds="clip")
        iso_sm.fit(y_prob, y_true)
        x_plot = np.linspace(float(y_prob.min()), float(y_prob.max()), 300)
        y_sm   = iso_sm.predict(x_plot)

        # Decile H-L bins for scatter
        mp, fo = hl_bins_for_plot(y_true, y_prob, n_bins=10)

        # Stats
        s = calibration_stats(y_true, y_prob)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="Perfect", zorder=1)

        # Smoothed line
        ax.plot(x_plot, y_sm, color=m_color, lw=2.0,
                label="Smooth (isotonic)", zorder=3)

        # H-L decile dots
        ax.scatter(mp, fo, s=65, color=color, edgecolors="black",
                   linewidths=0.7, zorder=5, label="Decile bins")

        # Rug
        ax.plot(y_prob[y_true == 0], np.full((y_true == 0).sum(), -0.025),
                "|", color="#BBBBBB", alpha=0.15, ms=3, markeredgewidth=0.4)
        ax.plot(y_prob[y_true == 1], np.full(y_true.sum(), -0.045),
                "|", color=color, alpha=0.4, ms=3, markeredgewidth=0.4)

        # Stats box
        txt = (f"E/O = {s['eo']:.3f}\n"
               f"Slope = {s['slope']:.3f}  Int = {s['intercept']:.3f}\n"
               f"H-L p = {s['hl_p']:.3f}\n"
               f"Brier = {s['brier']:.5f}\n"
               f"BSS = {s['bss']:.4f}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                fontsize=7.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          alpha=0.88, ec="gray", lw=0.6))

        # After-calibration E/O badge
        eo_color = "green" if 0.80 <= s["eo"] <= 1.20 else \
                   "orange" if 0.60 <= s["eo"] <= 1.50 else "red"
        ax.text(0.97, 0.03, f"E/O={s['eo']:.3f}",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="bottom", ha="right", color=eo_color)

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.07, 1.08)
        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Observed event fraction", fontsize=9)
        ax.set_title(f"{center} — {title}", fontsize=9.5, fontweight="bold")
        if col_j == 0:
            ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(labelsize=8)

fig.suptitle(
    "Ensemble Model Recalibration: Raw vs Platt Scaling vs Isotonic Regression\n"
    "(E/O ratio target ≈ 1.0;  H-L p > 0.05 = well-calibrated)",
    fontsize=11, fontweight="bold", y=1.01
)

for ext in ["pdf", "png"]:
    fig.savefig(OUT / f"recalibration_curves.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: recalibration_curves.pdf/png")

# ─────────────────────────────────────────────────────────────────────────────
# Brier Score summary bar chart
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
fig2.subplots_adjust(wspace=0.38)
methods_ord = ["Raw", "Platt", "Isotonic"]
bar_colors  = ["#888888", "#1A6FBF", "#CC4400"]

for ax2, center in zip(axes2, ["SCH", "WHU"]):
    sub = metrics_df[metrics_df["Cohort"] == center].set_index("Method")
    briers = [sub.loc[m, "Brier_score"] for m in methods_ord]
    bsss   = [sub.loc[m, "Brier_skill_score"] for m in methods_ord]
    x = np.arange(3)
    bars = ax2.bar(x, briers, color=bar_colors, edgecolor="black", lw=0.8, width=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(methods_ord, fontsize=9)
    ax2.set_ylabel("Brier Score", fontsize=9)
    ax2.set_title(f"{center} Cohort — Brier Score by Method", fontsize=9.5, fontweight="bold")
    for i, (b, bss_val) in enumerate(zip(briers, bsss)):
        ax2.text(i, b + 0.0002, f"{b:.5f}\nBSS={bss_val:.3f}",
                 ha="center", va="bottom", fontsize=7.5)
    ax2.set_ylim(0, max(briers) * 1.25)

fig2.suptitle("Brier Score Before and After Calibration", fontsize=11, fontweight="bold")
for ext in ["pdf", "png"]:
    fig2.savefig(OUT / f"recalibration_brier.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved: recalibration_brier.pdf/png")

# ─────────────────────────────────────────────────────────────────────────────
# Save calibrated predictions
# ─────────────────────────────────────────────────────────────────────────────
pred_out = pred.copy()
pred_out["prob_platt"]    = np.nan
pred_out["prob_isotonic"] = np.nan

for center in ["SCH", "WHU"]:
    idx = pred_out["Center"] == center
    pred_out.loc[idx, "prob_platt"]    = cal_probs[center]["platt"]
    pred_out.loc[idx, "prob_isotonic"] = cal_probs[center]["isotonic"]

pred_out.to_csv(OUT / "real_model_predictions_recalibrated.csv",
                index=False, encoding="utf-8-sig")
print("Saved: real_model_predictions_recalibrated.csv")
print("\nDone.")
