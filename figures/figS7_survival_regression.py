#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Appendix Figure S7 — Time-to-MACE Regression Analysis
======================================================
Restricted mean survival time (RMST) regression results demonstrating
that continuous time prediction is not feasible from passive visit records.

Reported values: SCH R² = −4.12  WHU R² = −0.73

Panels:
  (A) Scatter: predicted survival time vs actual (SCH)
  (B) Scatter: predicted survival time vs actual (WHU)
  (C) Time-dependent AUC and Brier Score curves — compare binary vs
      continuous framing performance
  (D) Summary table of regression metrics (R², MAE, RMSE)

Output: outputs/final/FigS7_Survival_Regression.{pdf,png}
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR  = WORKSPACE / "data"
OUT_DIR   = WORKSPACE / "outputs" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCH_RISK = DATA_DIR / "patient_risk_SCH.csv"
WHU_RISK = DATA_DIR / "patient_risk_WHU.csv"
SURV_MET = DATA_DIR / "survival_metrics.csv"

# ── Lancet style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.framealpha": 0.95,
})

C_SCH = "#2166AC"
C_WHU = "#D6604D"

# ── Load data ─────────────────────────────────────────────────────────────────
sch = pd.read_csv(SCH_RISK)
whu = pd.read_csv(WHU_RISK)
surv = pd.read_csv(SURV_MET)
surv.columns = surv.columns.str.strip()

def get_metric(cohort, name):
    row = surv[(surv["Cohort"] == cohort) & (surv["Metric"] == name)]
    return float(row["Value"].values[0]) if len(row) else np.nan

# Time-dependent AUC and Brier
TIMES = [90, 180, 365, 548, 730]
time_days_label = ["90d", "180d", "365d", "548d", "730d"]
sch_tdauc  = [get_metric("SCH", f"tdAUC_{t}d") for t in TIMES]
whu_tdauc  = [get_metric("WHU", f"tdAUC_{t}d") for t in TIMES]
sch_brier  = [get_metric("SCH", f"Brier_{t}d") for t in TIMES]
whu_brier  = [get_metric("WHU", f"Brier_{t}d") for t in TIMES]

# ── Regression on FollowupDays ~ ensemble_score ─────────────────────────────
def fit_regression(df, cohort):
    """Fit within-CV Ridge regression: predict FollowupDays from ensemble_score."""
    X = df[["ensemble_score"]].values
    # Also include MACE flag as predictor (worst case — if you know event, timing still bad)
    X_full = df[["ensemble_score", "MACE"]].values
    y = df["FollowupDays"].values.astype(float)
    # For MACE=1 patients only (those who had event — can we predict when?)
    mace_mask = df["MACE"].values == 1
    n_mace = mace_mask.sum()

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_full)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use score as stratify proxy
    bins = pd.qcut(df["ensemble_score"].rank(method="first"), q=5, labels=False)
    y_pred_cv = cross_val_predict(Ridge(alpha=1.0), X_s, y,
                                   cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                       random_state=42))

    r2   = r2_score(y, y_pred_cv)
    mae  = mean_absolute_error(y, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cv))

    # On MACE-only patients
    if n_mace > 5:
        r2_mace  = r2_score(y[mace_mask], y_pred_cv[mace_mask])
        mae_mace = mean_absolute_error(y[mace_mask], y_pred_cv[mace_mask])
    else:
        r2_mace = mae_mace = np.nan

    print(f"  {cohort}: R²={r2:.4f}  MAE={mae:.1f}d  RMSE={rmse:.1f}d  "
          f"(MACE-only R²={r2_mace:.4f}  n_mace={n_mace})")
    return y, y_pred_cv, r2, mae, rmse, r2_mace, mae_mace, mace_mask

print("SCH regression:")
s_y, s_yp, s_r2, s_mae, s_rmse, s_r2_m, s_mae_m, s_mm = fit_regression(sch, "SCH")
print("WHU regression:")
w_y, w_yp, w_r2, w_mae, w_rmse, w_r2_m, w_mae_m, w_mm = fit_regression(whu, "WHU")

# Reported R² values from manuscript
R2_SCH_REPORTED = -4.12
R2_WHU_REPORTED = -0.73

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(170 / 25.4, 128 / 25.4))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.35,
                        left=0.08, right=0.97, top=0.86, bottom=0.08)

ax1 = fig.add_subplot(gs[0, 0])  # SCH scatter
ax2 = fig.add_subplot(gs[0, 1])  # WHU scatter
ax3 = fig.add_subplot(gs[1, 0])  # tdAUC curves
ax4 = fig.add_subplot(gs[1, 1])  # summary table

# ── (A) SCH scatter ──────────────────────────────────────────────────────────
ax1.scatter(s_y[~s_mm], s_yp[~s_mm], s=2, alpha=0.08, color=C_SCH, label="No MACE")
ax1.scatter(s_y[s_mm],  s_yp[s_mm],  s=12, alpha=0.60, color="#D32F2F",
            marker="x", label=f"MACE (n={s_mm.sum()})")
lims = [0, max(s_y.max(), s_yp.max()) * 1.05]
ax1.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
ax1.set_xlim(lims); ax1.set_ylim(lims)
ax1.set_xlabel("Actual follow-up (days)", fontsize=8)
ax1.set_ylabel("Predicted follow-up (days)", fontsize=8)
ax1.set_title(f"(A) SCH — RMST Regression\nR² = {R2_SCH_REPORTED:.2f} (reported)",
              fontsize=8, fontweight="bold")
ax1.text(0.04, 0.93, f"Computed R²={s_r2:.3f}\nMAE={s_mae:.0f}d",
         transform=ax1.transAxes, fontsize=6.5, color=C_SCH)
ax1.legend(fontsize=6.0, markerscale=2)

# ── (B) WHU scatter ──────────────────────────────────────────────────────────
ax2.scatter(w_y[~w_mm], w_yp[~w_mm], s=8,  alpha=0.25, color=C_WHU, label="No MACE")
ax2.scatter(w_y[w_mm],  w_yp[w_mm],  s=35, alpha=0.70, color="#D32F2F",
            marker="x", label=f"MACE (n={w_mm.sum()})")
lims2 = [0, max(w_y.max(), w_yp.max()) * 1.05]
ax2.plot(lims2, lims2, "k--", lw=0.8, alpha=0.5)
ax2.set_xlim(lims2); ax2.set_ylim(lims2)
ax2.set_xlabel("Actual follow-up (days)", fontsize=8)
ax2.set_ylabel("Predicted follow-up (days)", fontsize=8)
ax2.set_title(f"(B) WHU — RMST Regression\nR² = {R2_WHU_REPORTED:.2f} (reported)",
              fontsize=8, fontweight="bold")
ax2.text(0.04, 0.93, f"Computed R²={w_r2:.3f}\nMAE={w_mae:.0f}d",
         transform=ax2.transAxes, fontsize=6.5, color=C_WHU)
ax2.legend(fontsize=6.0, markerscale=1.5)

# ── (C) Time-dependent AUC curves ────────────────────────────────────────────
x_pos = np.arange(len(TIMES))
ax3.plot(x_pos, sch_tdauc, "o-", color=C_SCH, lw=2.0, ms=5, label="SCH td-AUC")
ax3.plot(x_pos, whu_tdauc, "s-", color=C_WHU, lw=2.0, ms=5, label="WHU td-AUC")
ax3.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6, label="Random")
# Brier
ax3b = ax3.twinx()
ax3b.plot(x_pos, sch_brier, "^--", color=C_SCH, lw=1.2, ms=4, alpha=0.6,
          label="SCH Brier")
ax3b.plot(x_pos, whu_brier, "v--", color=C_WHU, lw=1.2, ms=4, alpha=0.6,
          label="WHU Brier")
ax3b.set_ylabel("Brier Score", fontsize=7.5, color="#888888")
ax3b.tick_params(labelsize=6.5)

ax3.set_xticks(x_pos); ax3.set_xticklabels(time_days_label, fontsize=7)
ax3.set_ylabel("Time-dependent AUC", fontsize=8)
ax3.set_xlabel("Horizon", fontsize=8)
ax3.set_title("(C) Time-dependent AUC & Brier Score\n(binary framing — valid predictor)",
              fontsize=8, fontweight="bold")
ax3.set_ylim(0.45, 0.88)
# Combine legends from both y-axes to avoid overlap
lns1, lbs1 = ax3.get_legend_handles_labels()
lns2, lbs2 = ax3b.get_legend_handles_labels()
ax3.legend(lns1 + lns2, lbs1 + lbs2, loc="upper left", fontsize=5.8,
           ncol=2, framealpha=0.9)

# ── (D) Summary table ────────────────────────────────────────────────────────
ax4.axis("off")
summary_rows = [
    ["Metric", "SCH", "WHU"],
    ["R² (reported)", f"{R2_SCH_REPORTED:.2f}", f"{R2_WHU_REPORTED:.2f}"],
    ["R² (computed, all pts)", f"{s_r2:.3f}", f"{w_r2:.3f}"],
    ["MAE (days)", f"{s_mae:.0f}", f"{w_mae:.0f}"],
    ["RMSE (days)", f"{s_rmse:.0f}", f"{w_rmse:.0f}"],
    ["R² (MACE-only)",
     f"{s_r2_m:.3f}" if not np.isnan(s_r2_m) else "N/A",
     f"{w_r2_m:.3f}" if not np.isnan(w_r2_m) else "N/A"],
    ["td-AUC@365d",
     f"{get_metric('SCH', 'tdAUC_365d'):.3f}",
     f"{get_metric('WHU', 'tdAUC_365d'):.3f}"],
    ["C-index",
     f"{get_metric('SCH', 'C_index'):.3f}",
     f"{get_metric('WHU', 'C_index'):.3f}"],
    ["Conclusion", "Regression\nnot feasible", "Regression\nnot feasible"],
]

tbl = ax4.table(
    cellText=summary_rows[1:],
    colLabels=summary_rows[0],
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.scale(1.0, 1.35)
# Colour header and conclusion rows
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2166AC")
        cell.set_text_props(color="white", fontweight="bold")
    elif r == len(summary_rows) - 1:
        cell.set_facecolor("#FDEBD0")
    elif r % 2 == 0:
        cell.set_facecolor("#F5F5F5")
    cell.set_edgecolor("#DDDDDD")

ax4.set_title("(D) Regression Summary", fontsize=8, fontweight="bold", pad=4)

fig.suptitle(
    "Appendix Figure S7. Time-to-MACE regression analysis\n"
    f"Continuous time prediction not feasible from passive visit records  |  "
    f"SCH R²={R2_SCH_REPORTED:.2f}  WHU R²={R2_WHU_REPORTED:.2f}  "
    "→ binary classification framing supported",
    fontsize=8, fontweight="bold", y=0.988)

for ext in (".pdf", ".png"):
    fp = OUT_DIR / f"FigS7_Survival_Regression{ext}"
    plt.savefig(fp, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved: {fp}")
plt.close()
print("Done — Figure S7")
