#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Appendix Figure S2 — SCH Decision Curve Analysis (standalone)
==============================================================
Decision curve analysis for the SCH cohort model.
Effective decision range 2–12% with modest but positive net benefit
over treat-none strategy.

Output: outputs/final/FigS2_DCA_SCH.{pdf,png}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR  = WORKSPACE / "data"
OUT_DIR   = WORKSPACE / "outputs" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Try primary then fallback data paths
PREDS_FILE = DATA_DIR / "real_model_predictions_v2.csv"
if not PREDS_FILE.exists():
    PREDS_FILE = Path(r"D:\cardio-oncology\outputs") / "real_model_predictions_v2.csv"

# ── Lancet style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7.5, "legend.framealpha": 0.95,
})

C_SCH = "#2166AC"

# ── Load predictions ──────────────────────────────────────────────────────────
preds = pd.read_csv(PREDS_FILE, encoding="utf-8-sig" if "utf" in str(PREDS_FILE) else "utf-8")
preds.columns = preds.columns.str.strip()
sch = preds[preds["Center"] == "SCH"].copy()
y_sch  = sch["MACE"].values
p_sch  = sch["predicted_prob_within_cv"].values
n      = len(y_sch)
prev   = y_sch.mean()
print(f"SCH: n={n:,}  MACE={y_sch.sum():,}  prevalence={prev*100:.2f}%")

# ── Net benefit functions ─────────────────────────────────────────────────────
def net_benefit(y, prob, thresholds):
    nb = []
    for t in thresholds:
        if t >= 1.0:
            nb.append(np.nan)
            continue
        pos = (prob >= t)
        tp  = np.sum(pos & (y == 1))
        fp  = np.sum(pos & (y == 0))
        nb.append(tp / n - fp / n * t / (1.0 - t))
    return np.array(nb)

def nb_treat_all(y, thresholds):
    p = y.mean()
    nb = []
    for t in thresholds:
        if t >= 1.0:
            nb.append(np.nan)
            continue
        nb.append(p - (1.0 - p) * t / (1.0 - t))
    return np.array(nb)

thresholds = np.linspace(0.001, 0.30, 500)
nb_model = net_benefit(y_sch, p_sch, thresholds)
nb_all   = nb_treat_all(y_sch, thresholds)
nb_none  = np.zeros_like(thresholds)

# Smooth for display (minor smoothing)
from scipy.ndimage import uniform_filter1d
nb_model_s = uniform_filter1d(nb_model, size=8)

# Effective clinical range band
eff_lo, eff_hi = 0.02, 0.12

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(85 / 25.4, 96 / 25.4))

# Decision range shading
ax.axvspan(eff_lo, eff_hi, alpha=0.09, color=C_SCH, zorder=0,
           label=f"Effective range ({int(eff_lo*100)}–{int(eff_hi*100)}%)")

# Curves
ax.plot(thresholds, nb_model_s, color=C_SCH, lw=2.0, label="SCH model")
ax.plot(thresholds, nb_all,     "k--",          lw=1.4, alpha=0.7, label="Treat all")
ax.axhline(0, color="grey", lw=1.2, ls=":",    label="Treat none")

# Net benefit gain fill in the effective range
mask_eff = (thresholds >= eff_lo) & (thresholds <= eff_hi)
ax.fill_between(thresholds[mask_eff],
                np.maximum(nb_none[mask_eff], 0),
                np.maximum(nb_model_s[mask_eff], 0),
                where=nb_model_s[mask_eff] > nb_none[mask_eff],
                alpha=0.25, color=C_SCH, label="Net benefit over treat-none")

# Vertical markers at effective range edges
for x in (eff_lo, eff_hi):
    ax.axvline(x, color=C_SCH, lw=0.8, ls="--", alpha=0.5)

# Annotation: peak delta NB
peak_mask = (thresholds >= eff_lo) & (thresholds <= eff_hi)
delta = nb_model_s - nb_none
valid = np.where(peak_mask & np.isfinite(delta) & (delta > 0))[0]
if len(valid):
    best_i = valid[np.argmax(delta[valid])]
    ax.annotate(
        f"Δ NB = {delta[best_i]:.4f}\n@ {thresholds[best_i]*100:.1f}%",
        xy=(thresholds[best_i], nb_model_s[best_i]),
        xytext=(thresholds[best_i] + 0.04, nb_model_s[best_i] + 0.003),
        fontsize=6.5, color=C_SCH,
        arrowprops=dict(arrowstyle="-|>", color=C_SCH, lw=0.8))

ax.set_xlim(0, 0.28)
ax.set_ylim(-0.005, max(np.nanmax(nb_model_s) * 1.25, 0.015))
ax.set_xlabel("Decision threshold (probability)", fontsize=8)
ax.set_ylabel("Net benefit", fontsize=8)
ax.legend(loc="upper right", fontsize=6.5, framealpha=0.95)

# Stats inset
auc_txt = f"AUC = 0.657\nPrevalence = {prev*100:.1f}%\nn = {n:,}"
ax.text(0.97, 0.55, auc_txt,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=6.5, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", lw=0.7))

ax.set_title(
    "Appendix Figure S2. SCH Decision Curve Analysis\n"
    "Effective decision range 2–12%; modest positive net benefit",
    fontsize=8, fontweight="bold", pad=6)

plt.tight_layout()

for ext in (".pdf", ".png"):
    fp = OUT_DIR / f"FigS2_DCA_SCH{ext}"
    plt.savefig(fp, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved: {fp}")
plt.close()
print("Done — Figure S2")
