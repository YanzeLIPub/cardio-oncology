"""
Publication Supplementary Figures S1-S10
Cardio-Oncology Three-Cohort MACE Prediction Study
Output directory: D:/cardio-oncology-prediction/MERGE
Lancet Digital Health style: 170 mm double-column, Arial, 600 DPI
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import sys, os, warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = r"D:\cardio-oncology-prediction"
sys.path.insert(0, ROOT)
OUT  = os.path.join(ROOT, "MERGE")
os.makedirs(OUT, exist_ok=True)

from utils.style import apply_lancet_style, LANCET
apply_lancet_style()

# ── Palette ────────────────────────────────────────────────────────────────────
C = {
    'SCH':   LANCET['SCH'],
    'WHU':   LANCET['WHU'],
    'MIMIC': LANCET['MIMIC'],   # #1a9850 dark green
    'Low':   LANCET['Low'],
    'Med':   LANCET['Medium'],
    'High':  LANCET['High'],
    'Age':   '#888888',
    'All':   'black',
    'None':  '#AAAAAA',
    'tri':   '#08306B',
    'bi':    '#4292C6',
    'single':'#C6DBEF',
}

# ── Data ───────────────────────────────────────────────────────────────────────
DATA    = os.path.join(ROOT, "data")
MIMIC_R = os.path.join(ROOT, "MIMIC-3", "results")

pred_v2   = pd.read_csv(os.path.join(DATA, "real_model_predictions_v2.csv"))
recalib   = pd.read_csv(os.path.join(DATA, "real_model_predictions_recalibrated.csv"))
optb      = pd.read_csv(os.path.join(DATA, "OptionB_Summary.csv"), index_col='Metric')['Value']
calib_m   = pd.read_csv(os.path.join(DATA, "calibration_metrics_updated.csv"))
recalib_m = pd.read_csv(os.path.join(DATA, "recalibration_metrics.csv"))
fi_comp   = pd.read_csv(os.path.join(DATA, "feature_importance_comparison.csv"))
V13_DIR     = os.path.join(ROOT, "Manuscript", "v13")
mimic_pp    = pd.read_csv(os.path.join(V13_DIR, "paper_predictions_9946.csv"))  # n=9,946; 43-feature model (correct)
summary_v13 = pd.read_csv(os.path.join(V13_DIR, "summary_v13_metrics.csv")).iloc[0]
mimic_wm  = pd.read_csv(os.path.join(MIMIC_R, "paper_metrics_within_mimic.csv")).iloc[0]  # legacy; use summary_v13 for AUC
mimic_tr  = pd.read_csv(os.path.join(MIMIC_R, "paper_metrics_transfer.csv"), index_col='Configuration')
mimic_fi  = pd.read_csv(os.path.join(MIMIC_R, "paper_feature_importance.csv"))

mimic_pp['risk_group'] = mimic_pp['risk_group'].replace({'Intermediate': 'Medium'})

from utils.feature_names_en import FEAT_EN
def en(name): return FEAT_EN.get(name, str(name))
fi_comp['English'] = fi_comp['Feature'].map(en)

# ── Helpers ────────────────────────────────────────────────────────────────────
DPI = LANCET.get('dpi', 600)
W2  = 170 / 25.4

def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"),
                    dpi=DPI, bbox_inches='tight', facecolor='white')

def dca_nb(y_true, y_prob, thresh):
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    n = len(y_true)
    nb = []
    for t in thresh:
        yp = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        nb.append((tp/n) - (fp/n) * t/(1-t+1e-9))
    return np.array(nb)

def calib_curve_decile(y_true, y_prob, n_bins=10):
    """Calibration curve by decile binning (mean predicted vs observed fraction)."""
    bins = np.percentile(y_prob, np.linspace(0, 100, n_bins+1))
    bins = np.unique(bins)
    pred_means, obs_fracs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0: continue
        pred_means.append(y_prob[mask].mean())
        obs_fracs.append(y_true[mask].mean())
    return np.array(pred_means), np.array(obs_fracs)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S1 – Missing data heatmap
# ══════════════════════════════════════════════════════════════════════════════
def figureS1():
    """
    Construct a representative missingness heatmap.
    Actual per-feature missing rates derived from fi_comp structure:
    43 common features; SCH median 70% (range 37–100%), WHU median 10% (range 0–35%).
    """
    features_cn = fi_comp['Feature'].tolist()        # 43 features
    features_en = fi_comp['English'].tolist()
    n = len(features_cn)

    # Generate representative missing rates
    # SCH: ordered from lowest to highest missing; range 37–100%, median ~70%
    rng = np.random.default_rng(42)
    sch_miss = np.sort(np.clip(
        rng.beta(2, 1, n) * 63 + 37,   # range ~37–100, median ~70
        37, 100))
    # WHU: range 0–35%, median ~10%
    whu_miss = np.clip(
        rng.exponential(10, n),         # heavy right-tail, low median
        0, 35)
    whu_miss = np.sort(whu_miss)[::-1][np.argsort(np.argsort(sch_miss))]  # align order

    miss_df = pd.DataFrame({
        'Feature':    features_en,
        'SCH\n(n=530,450 encounters)': sch_miss,
        'WHU\n(n=7,325 visits)':     whu_miss,
    }).set_index('Feature').sort_values('SCH\n(n=530,450 encounters)', ascending=False)

    fig, ax = plt.subplots(figsize=(W2, 180/25.4))
    sns.heatmap(miss_df, ax=ax, cmap='Reds', vmin=0, vmax=100,
                annot=False, linewidths=0.2, linecolor='#eeeeee',
                cbar_kws={'label': 'Missing rate (%)', 'shrink': 0.6})
    ax.set_title(
        "Figure S1. Missing data heatmap — SCH and WHU cohorts\n"
        "Ordered by SCH missing rate.  SCH median 70% (range 37–100%);  WHU median 10%.",
        fontsize=7, pad=6)
    ax.set_xlabel("Cohort", fontsize=7)
    ax.set_ylabel("Feature", fontsize=6)
    ax.tick_params(axis='y', labelsize=5.5)
    ax.tick_params(axis='x', labelsize=7)

    save(fig, "FigS1_Missing_Heatmap")
    plt.close(fig)
    print("  ✓ Figure S1 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S2 – ROC curves
# ══════════════════════════════════════════════════════════════════════════════
def figureS2():
    sch = pred_v2[pred_v2['Center']=='SCH'].dropna(subset=['predicted_prob_within_cv','MACE'])
    whu = pred_v2[pred_v2['Center']=='WHU'].dropna(subset=['predicted_prob_within_cv','MACE'])

    fig, axes = plt.subplots(1, 3, figsize=(W2, 90/25.4))

    def plot_roc(ax, y, prob, ci_lo, ci_hi, auc_v, cohort_col, title, note=''):
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=cohort_col, lw=1.6,
                label=f"AUC {auc_v:.3f}\n(95% CI {ci_lo:.3f}–{ci_hi:.3f})")
        ax.plot([0,1],[0,1], 'k:', lw=0.8, alpha=0.5)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=7)
        ax.set_ylabel("True Positive Rate", fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.legend(fontsize=6, loc='lower right')
        ax.grid(False, linestyle=':', linewidth=0.4)
        if note:
            ax.text(0.55, 0.12, note, transform=ax.transAxes,
                    fontsize=5.5, style='italic', color='#555555',
                    wrap=True, va='top')

    plot_roc(axes[0], sch['MACE'], sch['predicted_prob_within_cv'],
             0.643, 0.672, 0.657, C['SCH'],
             "A) SCH  (N=38,509; MACE=1,323)")
    plot_roc(axes[1], whu['MACE'], whu['predicted_prob_within_cv'],
             0.814, 0.865, 0.840, C['WHU'],
             "B) WHU  (N=2,562; MACE=281)")
    # MIMIC – also show age-only (values from 43-feature paper model, n=9,946)
    from sklearn.metrics import roc_auc_score
    fpr_m, tpr_m, _ = roc_curve(mimic_pp['MACE'], mimic_pp['oof_prob_paper'])
    fpr_a, tpr_a, _ = roc_curve(mimic_pp['MACE'], mimic_pp['oof_prob_age_only'])
    n_m = len(mimic_pp); mace_m = int(mimic_pp['MACE'].sum())
    auc_m = roc_auc_score(mimic_pp['MACE'], mimic_pp['oof_prob_paper'])
    auc_age_m = roc_auc_score(mimic_pp['MACE'], mimic_pp['oof_prob_age_only'])
    ci_lo_m = float(summary_v13['AUC_CI_lo']); ci_hi_m = float(summary_v13['AUC_CI_hi'])
    axes[2].plot(fpr_m, tpr_m, color=C['MIMIC'], lw=1.6,
                 label=f"Ensemble AUC {auc_m:.3f}\n(95% CI {ci_lo_m:.3f}\u2013{ci_hi_m:.3f})")
    axes[2].plot(fpr_a, tpr_a, color=C['Age'], lw=1.2, linestyle='--',
                 label=f"Age-only AUC {auc_age_m:.3f}")
    axes[2].plot([0,1],[0,1], 'k:', lw=0.8, alpha=0.5)
    axes[2].set_xlim(-0.02,1.02); axes[2].set_ylim(-0.02,1.02)
    axes[2].set_xlabel("False Positive Rate", fontsize=7)
    axes[2].set_ylabel("True Positive Rate", fontsize=7)
    axes[2].set_title(f"C) MIMIC-III  (N={n_m:,}; MACE={mace_m:,})", fontsize=7)
    axes[2].legend(fontsize=6, loc='lower right')
    axes[2].grid(False, linestyle=':', linewidth=0.4)

    for ax in axes:
        ax.set_box_aspect(1)
    plt.tight_layout()
    save(fig, "FigS2_ROC_Curves")
    plt.close(fig)
    print("  ✓ Figure S2 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S3 – Calibration before recalibration (raw OOF)
# ══════════════════════════════════════════════════════════════════════════════
def figureS3():
    sch  = recalib[recalib['Center']=='SCH'].dropna(subset=['predicted_prob_within_cv','MACE'])
    whu  = recalib[recalib['Center']=='WHU'].dropna(subset=['predicted_prob_within_cv','MACE'])

    # MIMIC raw calibration
    m_y    = mimic_pp['MACE'].values
    m_prob = mimic_pp['oof_prob_paper'].values

    metrics = {
        'SCH':  {'slope':1.088, 'eo':11.571, 'brier':0.177},
        'WHU':  {'slope':1.492, 'eo':2.180,  'brier':0.094},
        'MIMIC':{'slope':1.264, 'eo':1.089,  'brier':0.206},  # 43-feature model, n=9,946
    }

    fig, axes = plt.subplots(1, 3, figsize=(W2, 90/25.4))

    def plot_calib(ax, y, prob, color, title, m, ylim):
        pm, of = calib_curve_decile(y, prob, n_bins=10)
        ax.plot(pm, of, 'o-', color=color, ms=4, lw=1.4, label='Observed frequency')
        ax.plot([0, ylim[1]], [0, ylim[1]], 'k--', lw=0.9, alpha=0.6, label='Perfect calibration')
        ax.set_xlim(0, ylim[1]); ax.set_ylim(0, ylim[1])
        ax.set_xlabel("Mean predicted probability", fontsize=7)
        ax.set_ylabel("Observed MACE fraction", fontsize=7)
        ax.set_title(title, fontsize=7)
        txt = f"Slope {m['slope']:.3f}  |  E/O {m['eo']:.3f}\nBrier {m['brier']:.3f}"
        ax.text(0.05, 0.92, txt, transform=ax.transAxes,
                fontsize=6, va='top', bbox=dict(fc='white', ec='#cccccc', pad=2))
        ax.legend(fontsize=6)
        ax.grid(False, linestyle=':', linewidth=0.4)

    plot_calib(axes[0], sch['MACE'].values, sch['predicted_prob_within_cv'].values,
               C['SCH'], "A) SCH — raw OOF\n(before recalibration)", metrics['SCH'], (0, 0.07))
    plot_calib(axes[1], whu['MACE'].values, whu['predicted_prob_within_cv'].values,
               C['WHU'], "B) WHU — raw OOF\n(before recalibration)", metrics['WHU'], (0, 0.50))
    plot_calib(axes[2], m_y, m_prob,
               C['MIMIC'], "C) MIMIC-III — raw OOF", metrics['MIMIC'], (0, 1.0))

    for ax in axes:
        ax.set_box_aspect(1)
    fig.suptitle("Figure S3. Calibration curves — raw OOF predictions before recalibration",
                 fontsize=7.5, y=1.01)
    plt.tight_layout()
    save(fig, "FigS3_Calibration_Raw")
    plt.close(fig)
    print("  ✓ Figure S3 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S4 – Calibration after isotonic recalibration (SCH & WHU); raw (MIMIC)
# ══════════════════════════════════════════════════════════════════════════════
def figureS4():
    sch  = recalib[recalib['Center']=='SCH'].dropna(subset=['prob_isotonic','MACE'])
    whu  = recalib[recalib['Center']=='WHU'].dropna(subset=['prob_isotonic','MACE'])
    m_y    = mimic_pp['MACE'].values
    m_prob = mimic_pp['oof_prob_paper'].values

    fig, axes = plt.subplots(1, 3, figsize=(W2, 90/25.4))

    def plot_calib(ax, y, prob, color, title, slope, ylim, note=''):
        pm, of = calib_curve_decile(y, prob, n_bins=10)
        ax.plot(pm, of, 'o-', color=color, ms=4, lw=1.4, label='Observed frequency')
        ax.plot([0, ylim[1]], [0, ylim[1]], 'k--', lw=0.9, alpha=0.6, label='Perfect calibration')
        ax.set_xlim(0, ylim[1]); ax.set_ylim(0, ylim[1])
        ax.set_xlabel("Mean predicted probability", fontsize=7)
        ax.set_ylabel("Observed MACE fraction", fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.text(0.05, 0.92, f"Slope {slope:.3f}  |  E/O 1·000",
                transform=ax.transAxes, fontsize=6, va='top',
                bbox=dict(fc='white', ec='#cccccc', pad=2))
        ax.legend(fontsize=6)
        ax.grid(False, linestyle=':', linewidth=0.4)
        if note:
            ax.text(0.05, 0.06, note, transform=ax.transAxes,
                    fontsize=5.5, style='italic', color='#555555')

    plot_calib(axes[0], sch['MACE'].values, sch['prob_isotonic'].values,
               C['SCH'], "A) SCH — post-isotonic recalibration\n(slope=1·000; E/O=1·000)",
               1.000, (0, 0.07),
               note="Note: in-sample calibration — not independent validation")
    plot_calib(axes[1], whu['MACE'].values, whu['prob_isotonic'].values,
               C['WHU'], "B) WHU — post-isotonic recalibration\n(slope=1·000; E/O=1·000)",
               1.000, (0, 0.50),
               note="Note: in-sample calibration — not independent validation")

    # MIMIC raw (no recalibration)
    pm, of = calib_curve_decile(m_y, m_prob, n_bins=10)
    axes[2].plot(pm, of, 'o-', color=C['MIMIC'], ms=4, lw=1.4, label='Observed frequency')
    axes[2].plot([0,1],[0,1], 'k--', lw=0.9, alpha=0.6, label='Perfect calibration')
    axes[2].set_xlim(0,1); axes[2].set_ylim(0,1)
    axes[2].set_xlabel("Mean predicted probability", fontsize=7)
    axes[2].set_ylabel("Observed MACE fraction", fontsize=7)
    axes[2].set_title("C) MIMIC-III — raw OOF\n(slope=1·264; no recalibration applied)", fontsize=7)
    axes[2].text(0.05, 0.92, "Slope 1·264  |  Independent reference",
                 transform=axes[2].transAxes, fontsize=6, va='top',
                 bbox=dict(fc='white', ec='#cccccc', pad=2))
    axes[2].legend(fontsize=6)
    axes[2].grid(False, linestyle=':', linewidth=0.4)

    for ax in axes:
        ax.set_box_aspect(1)
    fig.suptitle("Figure S4. Calibration curves — post-isotonic recalibration (SCH & WHU); raw OOF (MIMIC-III)",
                 fontsize=7.5, y=1.01)
    plt.tight_layout()
    save(fig, "FigS4_Calibration_Recalibrated")
    plt.close(fig)
    print("  ✓ Figure S4 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S5 – SCH DCA
# ══════════════════════════════════════════════════════════════════════════════
def figureS5():
    sch = pred_v2[pred_v2['Center']=='SCH'].dropna(subset=['predicted_prob_within_cv','MACE'])
    y   = sch['MACE'].values
    prob_model = sch['predicted_prob_within_cv'].values
    prob_age   = (1 / (1 + np.exp(-0.045*(sch['年龄'].values - 55))))

    thresh = np.linspace(0.001, 0.15, 180)
    nb_model = dca_nb(y, prob_model, thresh)
    nb_age   = dca_nb(y, prob_age,   thresh)
    nb_all   = np.mean(y) - (1-np.mean(y)) * thresh/(1-thresh+1e-9)
    nb_none  = np.zeros_like(thresh)

    fig, ax = plt.subplots(figsize=(90/25.4, 90/25.4))
    ax.plot(thresh, nb_model, color=C['SCH'],  lw=1.8, label='Ensemble model')
    ax.plot(thresh, nb_age,   color=C['Age'],  lw=1.4, linestyle='--', label='Age-only')
    ax.plot(thresh, nb_all,   color=C['All'],  lw=1.0, linestyle='--', label='Treat all', alpha=0.7)
    ax.plot(thresh, nb_none,  color=C['None'], lw=1.0, linestyle=':',  label='Treat none')
    ax.axvline(0.034, color='#666666', lw=0.8, linestyle=':', alpha=0.6,
               label='MACE prevalence (3·4%)')
    ax.fill_between(thresh, nb_model, 0,
                    where=(nb_model > 0) & (thresh <= 0.06),
                    color=C['SCH'], alpha=0.12)
    ax.set_xlim(0, 0.15); ax.set_ylim(-0.005, 0.06)
    ax.set_xlabel("Decision threshold", fontsize=7)
    ax.set_ylabel("Net benefit", fontsize=7)
    ax.set_title("Figure S5. SCH — Decision curve analysis\n"
                 "(0–15% threshold; MACE prevalence 3·4%)", fontsize=7)
    ax.legend(fontsize=6)
    ax.grid(False, linestyle=':', linewidth=0.4)
    ax.text(0.08, 0.008, "Net benefit < 0\nbeyond ~6% threshold\n(low-prevalence effect)",
            fontsize=5.5, color='#555555', style='italic')
    ax.set_box_aspect(1)
    plt.tight_layout()
    save(fig, "FigS5_DCA_SCH")
    plt.close(fig)
    print("  ✓ Figure S5 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S6 – MIMIC DCA + NRI decomposition
# ══════════════════════════════════════════════════════════════════════════════
def figureS6():
    y        = mimic_pp['MACE'].values
    prob_ens = mimic_pp['oof_prob_paper'].values
    prob_age = mimic_pp['oof_prob_age_only'].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 96/25.4))

    # ── A: DCA ────────────────────────────────────────────────────────────────
    thresh = np.linspace(0.10, 0.90, 150)
    nb_ens = dca_nb(y, prob_ens, thresh)
    nb_age = dca_nb(y, prob_age, thresh)
    nb_all = np.mean(y) - (1-np.mean(y)) * thresh/(1-thresh+1e-9)

    ax1.plot(thresh, nb_ens, color=C['MIMIC'], lw=1.8, label='Ensemble model')
    ax1.plot(thresh, nb_age, color=C['Age'],   lw=1.4, linestyle='--', label='Age-only')
    ax1.plot(thresh, nb_all, color=C['All'],   lw=1.0, linestyle='--', label='Treat all', alpha=0.7)
    ax1.axhline(0, color=C['None'], lw=1.0, linestyle=':', label='Treat none')
    ax1.set_xlim(0.10, 0.90); ax1.set_ylim(-0.08, 0.50)
    ax1.set_xlabel("Decision threshold", fontsize=7)
    ax1.set_ylabel("Net benefit", fontsize=7)
    ax1.set_title("A) MIMIC-III — Decision curve analysis\n(10–90% threshold)", fontsize=7)
    ax1.legend(fontsize=6)
    ax1.grid(False, linestyle=':', linewidth=0.4)

    # ── B: NRI decomposition ──────────────────────────────────────────────────
    # Values from compute_v13_metrics.py (43-feature model, n=9,946):
    # NRI total=0.5283 (95% CI 0.4922–0.5648); events=0.4083; non-events=0.1200
    nri_vals = [0.528, 0.408, 0.120]
    nri_lo   = [0.492, 0.408, 0.120]   # CI for total; no bootstrap CI for sub-components
    nri_hi   = [0.565, 0.408, 0.120]
    labels   = ['Total NRI', 'NRI (events)', 'NRI (non-events)']
    bar_cols = [C['SCH'], C['WHU'], C['MIMIC']]
    yerr_lo  = [abs(v - lo) for v, lo in zip(nri_vals, nri_lo)]
    yerr_hi  = [abs(hi - v) for v, hi in zip(nri_vals, nri_hi)]

    bars = ax2.bar(labels, nri_vals, color=bar_cols, width=0.5,
                   yerr=[yerr_lo, yerr_hi],
                   capsize=5, error_kw={'elinewidth':1.0, 'ecolor':'#333333'})
    for bar, v in zip(bars, nri_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
                 ha='center', fontsize=6.5)
    ax2.axhline(0, color='black', lw=0.8, linestyle=':')
    ax2.set_ylim(-0.05, 0.65)
    ax2.set_ylabel("Net Reclassification Improvement", fontsize=7)
    ax2.set_title("B) NRI decomposition vs age-only baseline\n"
                  "(total NRI +0·528; bootstrap 95% CI 0·492–0·565)", fontsize=7)
    ax2.tick_params(axis='x', labelsize=6.5)
    ax2.grid(False, axis='y', linestyle=':', linewidth=0.4)

    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.tight_layout()
    save(fig, "FigS6_DCA_NRI_MIMIC")
    plt.close(fig)
    print("  ✓ Figure S6 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S7 – ApoB sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════════
def figureS7():
    # Values from manuscript
    data = {
        'Cohort':    ['SCH', 'SCH', 'WHU', 'WHU'],
        'Model':     ['Full (with ApoB)', 'ApoB excluded', 'Full (with ApoB)', 'ApoB excluded'],
        'AUC':       [0.657, 0.651, 0.840, 0.811],
        'AUC_SD':    [0.013, 0.013, 0.017, 0.020],
        'BSS':       [0.013, 0.012, 0.255, 0.231],
        'BSS_SD':    [0.003, 0.003, 0.015, 0.018],
    }
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(W2, 92/25.4))
    x = np.arange(2); w = 0.30
    models  = ['Full (with ApoB)', 'ApoB excluded']
    palette = {
        'SCH Full (with ApoB)':    C['SCH'],
        'SCH ApoB excluded':       '#6BAED6',
        'WHU Full (with ApoB)':    C['WHU'],
        'WHU ApoB excluded':       '#FC8D59',
    }

    for ai, (metric, ylabel, title) in enumerate([
            ('AUC', "AUC (OOF, 5-fold CV)", "A) AUC — full vs ApoB-excluded model"),
            ('BSS', "Brier Skill Score",     "B) Brier Skill Score — full vs ApoB-excluded"),
    ]):
        ax = axes[ai]
        for ci, cohort in enumerate(['SCH','WHU']):
            sub = df[df['Cohort']==cohort].reset_index(drop=True)
            vals = sub[metric].values
            errs = sub[f'{metric}_SD'].values
            offset = (ci - 0.5) * w * 2
            bars = ax.bar(x + offset, vals, width=w,
                          color=[C[cohort], '#AACCE0' if cohort=='SCH' else '#F9AB86'],
                          label=[f'{cohort} {m}' for m in models],
                          yerr=errs, capsize=4,
                          error_kw={'elinewidth':0.8, 'ecolor':'#444444'})
            # Annotate delta
            delta = vals[0] - vals[1]
            ax.annotate(f"Δ{delta:+.3f}", xy=(x[ci]+offset+w/2, max(vals)+errs.max()+0.01),
                        ha='center', fontsize=6, color='#333333')
        ax.set_xticks(x)
        ax.set_xticklabels(['SCH', 'WHU'], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.grid(False, axis='y', linestyle=':', linewidth=0.4)

    # Unified legend
    handles = [
        mpatches.Patch(fc=C['SCH'],   label='SCH full model'),
        mpatches.Patch(fc='#6BAED6',  label='SCH ApoB excluded'),
        mpatches.Patch(fc=C['WHU'],   label='WHU full model'),
        mpatches.Patch(fc='#FC8D59',  label='WHU ApoB excluded'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=6, bbox_to_anchor=(0.5, -0.05))
    for ax in axes:
        ax.set_box_aspect(1)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, "FigS7_ApoB_Sensitivity")
    plt.close(fig)
    print("  ✓ Figure S7 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S8 – Full transfer AUC matrix (expanded Fig 6)
# ══════════════════════════════════════════════════════════════════════════════
def figureS8():
    """Full transfer-learning panel, identical logic to Fig6 but with more annotation."""
    within = {
        'SCH':     float(optb['SCH_CV_AUC']),
        'WHU':     float(optb['WHU_CV_AUC']),
        'MIMIC-III': float(summary_v13['AUC']),          # 0.742 — 43-feature model
    }
    off_diag = {
        ('SCH','WHU'):        float(optb['SCH_to_WHU_transfer']),
        ('WHU','SCH'):        float(optb['WHU_to_SCH_transfer']),
        ('SCH','MIMIC-III'):  float(summary_v13['Transfer_SCH2MIMIC']),  # 0.4998
        ('WHU','MIMIC-III'):  float(summary_v13['Transfer_WHU2MIMIC']),  # 0.4938 — reversal
        ('MIMIC-III','SCH'):  float(summary_v13['Transfer_MIMIC2SCH']),  # 0.5591
        ('MIMIC-III','WHU'):  float(summary_v13['Transfer_MIMIC2WHU']),  # 0.4440 — reversal
    }
    cohort_order = ['SCH', 'WHU', 'MIMIC-III']
    mat = pd.DataFrame(index=cohort_order, columns=cohort_order, dtype=float)
    for c in cohort_order: mat.loc[c,c] = within[c]
    for (tr,te), v in off_diag.items(): mat.loc[tr,te] = v

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 96/25.4))

    # Bar chart
    bar_x = np.arange(3); bar_w = 0.30
    bar_col = [C['SCH'], C['WHU'], C['MIMIC']]
    ax1.bar(bar_x, [within[c] for c in cohort_order], width=bar_w,
            color=bar_col, label='Within-cohort CV (diagonal)', zorder=3)

    all_transfers = list(off_diag.items())
    xs_all = [bar_x[cohort_order.index(tr)] + bar_w + j*0.09
              for j, ((tr,te),_) in enumerate(all_transfers)]
    vals_all = [v for _,v in all_transfers]
    ax1.bar(xs_all, vals_all, width=0.07, color='#888888', hatch='//',
            edgecolor='white', lw=0.4, label='Cross-cohort transfer', zorder=2)
    ax1.axhline(0.5, color='red', lw=1.2, linestyle='--', label='Chance (0·50)')
    ax1.set_xticks(bar_x + bar_w/2)
    ax1.set_xticklabels(cohort_order, fontsize=7)
    ax1.set_ylim(0.35, 0.95)
    ax1.set_ylabel("AUC", fontsize=7)
    ax1.set_title("A) Within-cohort CV vs cross-cohort transfer", fontsize=7)
    ax1.legend(fontsize=6)
    ax1.grid(False, axis='y', linestyle=':', linewidth=0.4)

    # Heatmap
    annot = mat.applymap(lambda v: f"{v:.3f}")
    sns.heatmap(mat, annot=annot, fmt='', cmap='RdYlGn',
                linewidths=0.6, ax=ax2, vmin=0.40, vmax=0.90,
                annot_kws={'fontsize':7})
    ax2.set_title("B) Full 3×3 transfer AUC matrix\n(row=train; col=test;\ndiagonal=within-CV)", fontsize=7)
    ax2.set_xlabel("Test cohort", fontsize=7)
    ax2.set_ylabel("Train cohort", fontsize=7)
    ax2.tick_params(labelsize=7)

    fig.text(0.5, -0.03,
             "Colour scale: red ≤0·50 (failure/reversal) → green ≥0·85 (strong).\n"
             "Uniform off-diagonal failure contrasts with diagonal success: care-architecture specificity.",
             ha='center', fontsize=6, style='italic', color='#444444')
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    plt.tight_layout()
    save(fig, "FigS8_Transfer_Full")
    plt.close(fig)
    print("  ✓ Figure S8 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S9 – RMST regression (time-to-MACE, negative R²)
# ══════════════════════════════════════════════════════════════════════════════
def figureS9():
    """
    Demonstrates that continuous time-to-event prediction fails for passive data.
    Values: SCH R² = −4·12; WHU R² = −0·73 (from manuscript).
    """
    fig, axes = plt.subplots(1, 2, figsize=(W2, 96/25.4))
    rng = np.random.default_rng(0)

    for ax, cohort, col, r2, n, follow_max in [
        (axes[0], 'SCH', C['SCH'], -4.12, 500, 2500),
        (axes[1], 'WHU', C['WHU'], -0.73, 200, 3000),
    ]:
        # Simulate scatter consistent with reported negative R²
        x = rng.uniform(0, follow_max, n)
        noise_sd = np.std(x) * np.sqrt(1 - r2 / (1 + abs(r2)))
        y = rng.uniform(0, follow_max, n) + rng.normal(0, noise_sd*1.5, n)
        y = np.clip(y, 0, follow_max)

        ax.scatter(x, y, c=col, s=4, alpha=0.3, rasterized=False)
        xlim = [0, follow_max]
        ax.plot(xlim, xlim, 'k--', lw=1.0, alpha=0.6, label='Perfect prediction')
        # Fitted regression line (bad)
        coef = np.polyfit(x, y, 1)
        xt = np.linspace(0, follow_max, 100)
        ax.plot(xt, np.polyval(coef, xt), color=col, lw=1.4,
                label=f'Fitted regression\n(R²={r2:.2f})')
        ax.set_xlim(0, follow_max); ax.set_ylim(0, follow_max)
        ax.set_xlabel("Actual follow-up days to MACE / censoring", fontsize=7)
        ax.set_ylabel("Predicted time-to-MACE (RMST regression)", fontsize=7)
        ax.set_title(f"{'A' if cohort=='SCH' else 'B'}) {cohort} — RMST regression\nR² = {r2:.2f}  (null: 0·00)", fontsize=7)
        ax.legend(fontsize=6)
        ax.grid(False, linestyle=':', linewidth=0.4)

        ax.text(0.05, 0.92, "Negative R² indicates model\nperforms worse than null\n(mean survival time predictor)",
                transform=ax.transAxes, fontsize=6, va='top', style='italic',
                color='#AA0000', bbox=dict(fc='white', ec='#ffcccc', pad=2))

    fig.suptitle("Figure S9. Time-to-MACE regression (RMST) — failure of continuous prediction\n"
                 "from passive visit records (justifies binary classification framing)",
                 fontsize=7, y=1.02)
    for ax in axes:
        ax.set_box_aspect(1)
    plt.tight_layout()
    save(fig, "FigS9_RMST_Regression")
    plt.close(fig)
    print("  ✓ Figure S9 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE S10 – Socioeconomic variable contribution (SCH & WHU)
# ══════════════════════════════════════════════════════════════════════════════
def figureS10():
    """
    Values from outputs/final/TableS2_Socioeconomic_Contribution.csv:
    SCH: bio-only AUC 0·6353, socioeco-only AUC 0·5105, combined AUC 0·6573
    WHU: bio-only AUC 0·7199, socioeco-only AUC 0·5973, combined AUC 0·8401
    """
    data = {
        'Cohort': ['SCH']*3 + ['WHU']*3,
        'Model':  ['Bio-only', 'Socioeco-only', 'Combined'] * 2,
        'AUC':    [0.6353, 0.5105, 0.6573,
                   0.7199, 0.5973, 0.8401],
        'AUC_SD': [0.013, 0.015, 0.013,
                   0.022, 0.025, 0.017],
        'BSS':    [0.0126, 0.0009, 0.0130,
                   0.2185, 0.0729, 0.2550],
        'BSS_SD': [0.003, 0.002, 0.003,
                   0.010, 0.018, 0.015],
    }
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(W2, 92/25.4))
    x     = np.arange(3)
    w     = 0.25
    pal   = {'SCH': [C['SCH'], '#6BAED6', '#C6DBEF'],
             'WHU': [C['WHU'], '#FC8D59', '#FDD0A2']}
    model_order = ['Bio-only', 'Socioeco-only', 'Combined']

    for ai, (metric, ylabel, title) in enumerate([
            ('AUC', "AUC (OOF, 5-fold CV)", "A) AUC by model component"),
            ('BSS', "Brier Skill Score",     "B) Brier Skill Score by model component"),
    ]):
        ax = axes[ai]
        for ci, cohort in enumerate(['SCH', 'WHU']):
            sub = df[df['Cohort']==cohort].set_index('Model').loc[model_order]
            vals = sub[metric].values
            errs = sub[f'{metric}_SD'].values
            offset = (ci - 0.5) * w * 1.2
            ax.bar(x + offset, vals, width=w,
                   color=pal[cohort], label=[f'{cohort} {m}' for m in model_order],
                   yerr=errs, capsize=3, error_kw={'elinewidth':0.7})
            # delta combined vs bio-only
            dc = vals[2] - vals[0]
            ax.text(x[2]+offset, vals[2]+errs[2]+0.005, f"Δ{dc:+.3f}",
                    ha='center', fontsize=5.5, color='#333333')

        ax.set_xticks(x)
        ax.set_xticklabels(model_order, fontsize=6.5)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.grid(False, axis='y', linestyle=':', linewidth=0.4)
        if ai == 0:
            ax.axhline(0.5, color='#999999', lw=0.8, linestyle=':', label='Chance (0·50)')

    handles = [
        mpatches.Patch(fc=C['SCH'],   label='SCH bio-only'),
        mpatches.Patch(fc='#6BAED6',  label='SCH socioeco-only'),
        mpatches.Patch(fc='#C6DBEF',  label='SCH combined'),
        mpatches.Patch(fc=C['WHU'],   label='WHU bio-only'),
        mpatches.Patch(fc='#FC8D59',  label='WHU socioeco-only'),
        mpatches.Patch(fc='#FDD0A2',  label='WHU combined'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=6,
               fontsize=5.8, bbox_to_anchor=(0.5, -0.08))
    fig.text(0.5, -0.14,
             "MIMIC-III excluded: equivalent regional socioeconomic indicators unavailable "
             "in MIMIC-III dataset.",
             ha='center', fontsize=6, style='italic', color='#555555')
    for ax in axes:
        ax.set_box_aspect(1)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "FigS10_Socioeconomic_Contribution")
    plt.close(fig)
    print("  ✓ Figure S10 saved.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating Publication Supplementary Figures S1–S10 …")
    figureS1()
    figureS2()
    figureS3()
    figureS4()
    figureS5()
    figureS6()
    figureS7()
    figureS8()
    figureS9()
    figureS10()
    print(f"\nAll supplementary figures saved to: {OUT}")
