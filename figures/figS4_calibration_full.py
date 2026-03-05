#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Appendix Figure S4 — Full Calibration Comparison
=================================================
Calibration before and after recalibration: raw, Platt scaling,
and isotonic regression for both SCH and WHU cohorts.

Layout: 2 rows × 3 cols (SCH top row, WHU bottom row;
        cols = Raw | Platt | Isotonic)

Output: outputs/final/FigS4_Calibration_Full.{pdf,png}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR  = WORKSPACE / "data"
OUT_DIR   = WORKSPACE / "outputs" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECAL_FILE = DATA_DIR / "real_model_predictions_recalibrated.csv"
if not RECAL_FILE.exists():
    RECAL_FILE = Path(r"D:\cardio-oncology\outputs\ensemble_model") / "real_model_predictions_recalibrated.csv"

METRICS_FILE = DATA_DIR / "recalibration_metrics.csv"
if not METRICS_FILE.exists():
    METRICS_FILE = Path(r"D:\cardio-oncology\outputs\ensemble_model") / "recalibration_metrics.csv"

# ── Lancet style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "legend.framealpha": 0.95,
})

C_RAW  = "#969696"
C_PLTT = "#FB8C00"
C_ISO  = "#2166AC"
C_WRF  = "#D6604D"    # WHU raw
COLORS = {"SCH": {"Raw": C_RAW, "Platt": C_PLTT, "Isotonic": C_ISO},
          "WHU": {"Raw": C_WRF, "Platt": C_PLTT, "Isotonic": C_ISO}}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(RECAL_FILE, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

metrics = pd.read_csv(METRICS_FILE, encoding="utf-8-sig")
metrics.columns = metrics.columns.str.strip()

print(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")

# Column mapping: predicted_prob = raw, prob_platt = Platt, prob_isotonic = Isotonic
COL_MAP = {
    "Raw":      "predicted_prob",
    "Platt":    "prob_platt",
    "Isotonic": "prob_isotonic",
}

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(170 / 25.4, 128 / 25.4))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35,
                        left=0.08, right=0.97, top=0.86, bottom=0.08)

panel_labels = [
    ["(A) SCH — Raw",     "(B) SCH — Platt",     "(C) SCH — Isotonic"],
    ["(D) WHU — Raw",     "(E) WHU — Platt",     "(F) WHU — Isotonic"],
]

for row_i, cohort in enumerate(["SCH", "WHU"]):
    sub = df[df["Center"] == cohort].copy()
    y   = sub["MACE"].values
    brier_null = brier_score_loss(y, np.full(len(y), y.mean()))

    for col_i, method in enumerate(["Raw", "Platt", "Isotonic"]):
        ax = fig.add_subplot(gs[row_i, col_i])
        col = COL_MAP[method]

        if col not in sub.columns:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)
            continue

        p   = sub[col].values
        color = COLORS[cohort][method]

        # Calibration curve (10 quantile bins)
        frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, "o-", color=color, lw=2.0, ms=4, label=method, zorder=4)
        ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.5, label="Perfect")

        # Histogram of predicted probabilities (background)
        ax2 = ax.twinx()
        ax2.hist(p, bins=30, color=color, alpha=0.12, density=True, zorder=1)
        ax2.set_yticks([])
        ax2.set_ylabel("")

        # Stats from metrics table
        m_row = metrics[(metrics["Cohort"] == cohort) & (metrics["Method"] == method)]
        if len(m_row):
            HL_p  = float(m_row["HL_p"].values[0])
            brier = float(m_row["Brier_score"].values[0])
            bss   = float(m_row["Brier_skill_score"].values[0])
            slope = float(m_row["Cal_slope"].values[0])
            eo    = float(m_row["E/O_ratio"].values[0])
            stats_txt = (f"Cal slope={slope:.3f}\nE/O={eo:.2f}\n"
                         f"HL p={HL_p:.3f}\nBrier={brier:.4f}\nBSS={bss:.4f}")
        else:
            brier = brier_score_loss(y, p)
            bss   = 1 - brier / brier_null
            stats_txt = f"Brier={brier:.4f}\nBSS={bss:.4f}"

        ax.text(0.97, 0.03, stats_txt,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.0, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.7))

        ax.set_xlim(-0.02, min(1.02, p.max() * 1.12))
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability", fontsize=7.5)
        if col_i == 0:
            ax.set_ylabel("Observed MACE fraction", fontsize=7.5)
        ax.set_title(panel_labels[row_i][col_i], fontsize=8, fontweight="bold", pad=4)
        ax.legend(loc="upper left", fontsize=6.5)

fig.suptitle(
    "Appendix Figure S4. Calibration before and after recalibration — full comparison\n"
    "Raw model output, Platt scaling, and isotonic regression for both cohorts",
    fontsize=8.5, fontweight="bold", y=0.975)

for ext in (".pdf", ".png"):
    fp = OUT_DIR / f"FigS4_Calibration_Full{ext}"
    plt.savefig(fp, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved: {fp}")
plt.close()
print("Done — Figure S4")
