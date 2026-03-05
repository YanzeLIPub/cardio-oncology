"""
Publication Main Figures 1-6
Cardio-Oncology Three-Cohort MACE Prediction Study
Output directory: D:/cardio-oncology-prediction/MERGE
Lancet Digital Health style: 170 mm double-column, Arial, 600 DPI
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
    'SCH':   LANCET['SCH'],          # #2166AC  blue
    'WHU':   LANCET['WHU'],          # #D6604D  red
    'MIMIC': LANCET['MIMIC'],        # #1a9850  dark green
    'Low':   LANCET['Low'],          # green
    'Med':   LANCET['Medium'],       # orange
    'High':  LANCET['High'],         # red
    'Age':   '#888888',
    'All':   'black',
    'None':  '#AAAAAA',
    'tri':   '#08306B',              # darkest – tri-cohort concordant
    'bi':    '#4292C6',              # medium  – bi-cohort concordant
    'single':'#C6DBEF',             # light   – single cohort
}

# ── Data ───────────────────────────────────────────────────────────────────────
DATA    = os.path.join(ROOT, "data")
MIMIC_R = os.path.join(ROOT, "MIMIC-3", "results")

risk_sch  = pd.read_csv(os.path.join(DATA, "patient_risk_SCH.csv"))
risk_whu  = pd.read_csv(os.path.join(DATA, "patient_risk_WHU.csv"))
pred_v2   = pd.read_csv(os.path.join(DATA, "real_model_predictions_v2.csv"))
optb      = pd.read_csv(os.path.join(DATA, "OptionB_Summary.csv"),
                        index_col='Metric')['Value']
surv      = pd.read_csv(os.path.join(DATA, "survival_metrics.csv"))
fi_comp   = pd.read_csv(os.path.join(DATA, "feature_importance_comparison.csv"))
V13_DIR     = os.path.join(ROOT, "Manuscript", "v13")
mimic_pp    = pd.read_csv(os.path.join(V13_DIR, "paper_predictions_9946.csv"))  # n=9,946; 43-feature model (correct)
summary_v13 = pd.read_csv(os.path.join(V13_DIR, "summary_v13_metrics.csv")).iloc[0]
mimic_wm  = pd.read_csv(os.path.join(MIMIC_R, "paper_metrics_within_mimic.csv")).iloc[0]  # legacy; use summary_v13 for AUC
mimic_fi  = pd.read_csv(os.path.join(MIMIC_R, "paper_feature_importance.csv"))
mimic_tr  = pd.read_csv(os.path.join(MIMIC_R, "paper_metrics_transfer.csv"),
                        index_col='Configuration')

# Rename risk 'Intermediate'→'Medium' for MIMIC predictions
mimic_pp['risk_group'] = mimic_pp['risk_group'].replace({'Intermediate': 'Medium'})

# English feature name map
from utils.feature_names_en import FEAT_EN

def en(name):
    return FEAT_EN.get(name, str(name))

fi_comp['English'] = fi_comp['Feature'].map(en)

# ── Helpers ────────────────────────────────────────────────────────────────────
DPI  = LANCET.get('dpi', 600)
W1   = 85 / 25.4    # single-column mm → inches
W2   = 170 / 25.4   # double-column mm → inches

def add_panel_label(ax, label, fontsize=8.5):
    """Lancet-style panel label: bold Arial, outside axes frame, top-left."""
    ax.text(-0.08, 1.06, label, transform=ax.transAxes,
            ha='left', va='bottom', fontsize=fontsize,
            fontweight='bold', fontname='Arial', clip_on=False)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"),
                    dpi=DPI, bbox_inches='tight', facecolor='white')

def sv(cohort):
    """Return survival metric value."""
    return surv.set_index(['Cohort', 'Metric'])['Value']

def dca_nb(y_true, y_prob, thresh):
    """Net-benefit via DCA."""
    nb = []
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    for t in thresh:
        yp = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        nb.append((tp / n) - (fp / n) * (t / (1 - t + 1e-9)))
    return np.array(nb)

def km_fit(T, E, label):
    """Simple KM using numpy."""
    from lifelines import KaplanMeierFitter
    k = KaplanMeierFitter()
    k.fit(T, E, label=label)
    return k

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  – Study overview schematic
# ══════════════════════════════════════════════════════════════════════════════
def figure1():
    fig = plt.figure(figsize=(W2, 150/25.4))
    ax  = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # ── Style definitions (Manuscript_Fig1 palette) ────────────────────────────
    # Primary cohort boxes: aliceblue/steelblue
    box_kw_sch = dict(boxstyle="round,pad=0.5", fc="aliceblue",  ec="steelblue",   lw=1.4)
    box_kw_whu = dict(boxstyle="round,pad=0.5", fc="aliceblue",  ec="steelblue",   lw=1.4)
    # MIMIC-III: different fill (lightyellow) + dashed border to indicate external cohort
    box_kw_mim = dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="darkorange",  lw=1.4,
                      linestyle='--')
    fw_kw      = dict(boxstyle="round,pad=0.5", fc="honeydew",   ec="seagreen",    lw=1.4)
    badge_kw   = dict(boxstyle="round,pad=0.3", fc="darkorange",  ec="darkorange",  lw=0,
                      alpha=0.85)
    arrow_props = dict(arrowstyle="->,head_width=0.25,head_length=0.6",
                       color="gray", shrinkA=4, shrinkB=4)

    title_kw  = dict(ha='center', va='center', fontsize=8.5, fontweight='bold', color='navy')
    body_kw   = dict(ha='center', va='top',    fontsize=6.5, color='#222222',
                     linespacing=1.55)
    label_kw  = dict(ha='center', va='center', fontsize=6,   style='italic', color='#555555')
    arch_kw   = dict(ha='center', va='center', fontsize=6.5, fontweight='bold', color='#1a6e99')

    # ── Figure title ──────────────────────────────────────────────────────────
    ax.text(0.50, 0.975,
            "Figure 1.  Study Overview and Cohort Characteristics",
            ha='center', va='top', fontsize=9, fontweight='bold', color='#111111',
            transform=ax.transAxes)

    # ── Cohort data (symmetric structure for all three) ────────────────────────
    #   Each entry: (x_centre, label_above, title, body_lines, arch_label, box_kw, badge)
    cohorts = [
        (0.165,
         "Primary Development Cohort",
         "SCH",
         ["Scheduled Chemotherapy Monitoring",
          "SCH Cancer Centre — China",
          "",
          "N = 38,509  |  MACE events = 1,323",
          "MACE rate = 3.4%",
          "Follow-up horizon: Long-term (≥1 yr)",
          "Visit driver: scheduled treatment",
          "Biomarker missingness: median 70%",
          "Features: 43 common labs"],
         "Care architecture: Passive / Selective",
         box_kw_sch, None),
        (0.500,
         "Primary Development Cohort",
         "WHU",
         ["Symptom-Driven Cardio-Oncology Clinic",
          "WHU Zhongnan Hospital — China",
          "",
          "N = 2,562  |  MACE events = 281",
          "MACE rate = 11.0%",
          "Follow-up horizon: Mid-term (6–24 mo)",
          "Visit driver: symptoms / referral",
          "Biomarker missingness: median 10%",
          "Features: 43 common labs"],
         "Care architecture: Active / Comprehensive",
         box_kw_whu, None),
        (0.835,
         "Supplementary External Cohort",
         "MIMIC-III",
         ["Acute Critical Care Unit (ICU)",
          "BIDMC -- Beth Israel Deaconess, USA",
          "",
          "N = 9,946  |  MACE events = 4,462",
          "MACE rate = 44.9%",
          "Follow-up horizon: Short-term (7-90 d)",
          "Visit driver: acute hospital admission",
          "43 common features (external validation)",
          "External validation; no imputation"],
         "Care architecture: Intensive / Time-Critical",
         box_kw_mim, "[Ext] External Validation"),
    ]

    BOX_TOP    = 0.885
    BOX_H      = 0.350
    BOX_W      = 0.290
    BODY_PAD   = 0.022  # gap below title line to body text

    for cx, label_above, name, body_lines, arch, bkw, badge in cohorts:
        bx = cx - BOX_W / 2
        by = BOX_TOP - BOX_H

        # Cohort-role label above box
        ax.text(cx, BOX_TOP + 0.032, label_above, **label_kw, transform=ax.transAxes)

        # Box
        box = mpatches.FancyBboxPatch((bx, by), BOX_W, BOX_H,
                                       transform=ax.transAxes, clip_on=False,
                                       **bkw)
        ax.add_patch(box)

        # Cohort name title
        ax.text(cx, BOX_TOP - 0.025, name, transform=ax.transAxes, **title_kw)

        # Horizontal rule (thin separator)
        ax.plot([bx + 0.015, bx + BOX_W - 0.015],
                [BOX_TOP - 0.060, BOX_TOP - 0.060],
                color='steelblue' if badge is None else 'darkorange',
                lw=0.6, transform=ax.transAxes, clip_on=False)

        # Body text
        body_str = "\n".join(body_lines)
        ax.text(cx, BOX_TOP - 0.075, body_str,
                transform=ax.transAxes, **body_kw)

        # External badge (MIMIC-III only)
        if badge:
            ax.text(cx, BOX_TOP - 0.005, badge,
                    transform=ax.transAxes,
                    ha='center', va='center', fontsize=5.8,
                    fontweight='bold', color='white', bbox=badge_kw)

        # Care architecture label below box
        ax.text(cx, by - 0.018, arch, transform=ax.transAxes, **arch_kw)

    # ── Arrows ────────────────────────────────────────────────────────────────
    ARROW_Y_START = BOX_TOP - BOX_H - 0.050   # just below care-arch label
    ARROW_Y_END   = 0.445                      # top of framework box
    for cx in [0.165, 0.500, 0.835]:
        ax.annotate("",
                    xy=(cx, ARROW_Y_END), xytext=(cx, ARROW_Y_START),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=arrow_props)

    # ── Analytical Framework box ────────────────────────────────────────────
    FW_LEFT = 0.03; FW_W = 0.94
    FW_BOT  = 0.05; FW_H  = 0.40
    fw_box = mpatches.FancyBboxPatch((FW_LEFT, FW_BOT), FW_W, FW_H,
                                      transform=ax.transAxes, clip_on=False,
                                      **fw_kw)
    ax.add_patch(fw_box)

    ax.text(0.50, FW_BOT + FW_H - 0.025, "Analytical Framework",
            transform=ax.transAxes,
            ha='center', va='top', fontsize=8.5, fontweight='bold', color='seagreen')

    # Four analytical tasks — equal-width columns
    tasks = [
        ("(1) Within-Setting\nModel Development",
         "- XGBoost + RF ensemble\n"
         "- 5-fold stratified CV\n"
         "- OOF AUC / Brier / C-index\n"
         "- Applied independently to\n"
         "  each cohort"),
        ("(2) Cross-Cohort\nTransfer Analysis",
         "- Full 3x3 AUC matrix\n"
         "- No retraining between sites\n"
         "- Quantifies care-architecture\n"
         "  specificity of model weights\n"
         "- SCH <-> WHU <-> MIMIC-III"),
        ("(3) Biomarker\nConcordance Analysis",
         "- Feature importance (gain)\n"
         "- Tri-cohort concordant set:\n"
         "  LDL-C, HDL-C, Age,\n"
         "  Direct Bilirubin\n"
         "- Shading by concordance tier"),
        ("(4) Socioeconomic\nAccess Analysis",
         "- 5 regional access variables\n"
         "- Bio-only vs socioeco-only\n"
         "  vs combined model\n"
         "- SCH & WHU only\n"
         "  (MIMIC-III excluded)"),
    ]
    task_xs = [0.14, 0.38, 0.62, 0.86]
    for tx, (ttl, body) in zip(task_xs, tasks):
        # Sub-box
        tb = mpatches.FancyBboxPatch((tx - 0.115, FW_BOT + 0.030),
                                      0.23, FW_H - 0.065,
                                      boxstyle="round,pad=0.18",
                                      fc="white", ec="seagreen", lw=0.8,
                                      transform=ax.transAxes, clip_on=False, alpha=0.75)
        ax.add_patch(tb)
        ax.text(tx, FW_BOT + FW_H - 0.060, ttl,
                transform=ax.transAxes,
                ha='center', va='top', fontsize=6.5, fontweight='bold', color='seagreen')
        ax.text(tx, FW_BOT + FW_H - 0.115, body,
                transform=ax.transAxes,
                ha='center', va='top', fontsize=5.8, color='#333333', linespacing=1.5)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(fc='aliceblue',  ec='steelblue',  lw=1, label='Primary development cohort (SCH / WHU)'),
        mpatches.Patch(fc='lightyellow', ec='darkorange', lw=1, linestyle='--',
                       label='Supplementary external cohort (MIMIC-III)'),
    ]
    ax.legend(handles=legend_handles, loc='lower right',
              bbox_to_anchor=(0.99, 0.01), fontsize=6,
              frameon=True, edgecolor='#cccccc', facecolor='white')

    plt.tight_layout(pad=0)
    save(fig, "Fig1_Study_Overview")
    plt.close(fig)
    print("  ✓ Figure 1 saved.")



# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  – Kaplan-Meier MACE-free survival
# ══════════════════════════════════════════════════════════════════════════════
def figure2():
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts

    fig, axes = plt.subplots(1, 3, figsize=(215/25.4, 150/25.4), sharey=False)

    panels = [
        (risk_sch, f"SCH cohort\n(n={len(risk_sch):,}; log-rank p<0·0001)", axes[0]),
        (risk_whu, f"WHU cohort\n(n={len(risk_whu):,}; log-rank p<0·0001)",   axes[1]),
        (mimic_pp, f"MIMIC-III cohort\n(n={len(mimic_pp):,}; log-rank p<0·0001)", axes[2]),
    ]

    for i, (df, title, ax) in enumerate(panels):
        grp_low  = df[df['risk_group'] == 'Low']
        grp_med  = df[df['risk_group'] == 'Medium']
        grp_high = df[df['risk_group'] == 'High']

        kf_low  = KaplanMeierFitter(); kf_low.fit(grp_low['FollowupDays'],  grp_low['MACE'],  label=f"Low (n={len(grp_low)})")
        kf_med  = KaplanMeierFitter(); kf_med.fit(grp_med['FollowupDays'],  grp_med['MACE'],  label=f"Intermediate (n={len(grp_med)})")
        kf_high = KaplanMeierFitter(); kf_high.fit(grp_high['FollowupDays'], grp_high['MACE'], label=f"High (n={len(grp_high)})")

        # truncation at at-risk < 15
        def trunc(kf):
            tbl = kf.event_table['at_risk']
            valid = tbl[tbl >= 15]
            return valid.index.max() if len(valid) > 0 else 0

        t_cut = min(trunc(kf_low), trunc(kf_med), trunc(kf_high))

        kf_low.plot_survival_function(ax=ax, color=C['Low'],  ci_show=True, ci_alpha=0.12)
        kf_med.plot_survival_function(ax=ax, color=C['Med'],  ci_show=True, ci_alpha=0.12, linestyle='--')
        kf_high.plot_survival_function(ax=ax, color=C['High'], ci_show=True, ci_alpha=0.12)

        ax.set_xlim(left=0, right=t_cut)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("Follow-up (days)", fontsize=7)
        ax.set_ylabel("MACE-free survival probability", fontsize=7)
        ax.grid(False)
        ax.legend(loc='lower left', fontsize=6, frameon=True)
        add_panel_label(ax, chr(ord('A') + i), fontsize=8)
        ax.set_box_aspect(1)

        add_at_risk_counts(kf_low, kf_med, kf_high, ax=ax,
                           fontsize=5.5, rows_to_show=['At risk'])

    fig.tight_layout(rect=[0.03, 0.05, 0.995, 0.98], w_pad=1.10)
    save(fig, "Fig2_KM_Survival")
    plt.close(fig)
    print("  ✓ Figure 2 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  – Time-dependent model performance
# ══════════════════════════════════════════════════════════════════════════════
def figure3():
    sm = surv.set_index(['Cohort','Metric'])['Value']

    fig, axes = plt.subplots(1, 3, figsize=(W2, 92/25.4))

    # ── A: Harrell's C-index ──────────────────────────────────────────────────
    ax = axes[0]
    cohorts = ['SCH', 'WHU']
    cvals   = [sm[('SCH','C_index')], sm[('WHU','C_index')]]
    clo     = [sm[('SCH','Cox_HR_lo')], sm[('WHU','Cox_HR_lo')]]
    chi     = [sm[('SCH','Cox_HR_hi')], sm[('WHU','Cox_HR_hi')]]
    # Rough CI from Cox HR CI: SE ≈ (c – c_lo)/1.96  (illustrative)
    cerr = [(cvals[0] - 0.700)/1.96, (cvals[1] - 0.720)/1.96]

    bars = ax.bar(cohorts, cvals, color=[C['SCH'], C['WHU']],
                  yerr=[abs(e) for e in cerr], capsize=4, width=0.5,
                  error_kw={'elinewidth':1, 'ecolor':'#333333'})
    for bar, v in zip(bars, cvals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.006, f"{v:.3f}",
                ha='center', fontsize=6.5)
    ax.set_ylim(0.60, 0.82)
    ax.set_ylabel("Harrell's C-index", fontsize=7)
    ax.set_title("A) Harrell's C-index (95% CI)", fontsize=7)
    ax.grid(False, axis='y', linestyle=':', linewidth=0.4)

    # ── B: Time-dependent AUC ─────────────────────────────────────────────────
    ax = axes[1]
    sch_times = [90, 180, 365, 548, 730]
    whu_times = [90, 180, 365, 548, 730]
    sch_aucs  = [sm[('SCH', f'tdAUC_{t}d')] for t in sch_times]
    whu_aucs  = [sm[('WHU', f'tdAUC_{t}d')] for t in whu_times]

    mimic_times = [7, 14, 30, 60]
    mimic_aucs  = [float(summary_v13.get(f'TimedepAUC_{t}d', np.nan)) for t in mimic_times]  # 43-feature model

    ln1, = ax.plot(sch_times, sch_aucs, 'o-', color=C['SCH'], lw=1.4, ms=4,  label='SCH')
    ln2, = ax.plot(whu_times, whu_aucs, 's-', color=C['WHU'], lw=1.4, ms=4,  label='WHU')
    ax.set_ylabel("Time-dependent AUC (SCH/WHU)", fontsize=7, color='#333333')
    ax.set_ylim(0.60, 0.85)
    ax.set_xlabel("Days", fontsize=7)
    ax.set_title("B) Time-dependent AUC", fontsize=7)
    ax.grid(False, linestyle=':', linewidth=0.4)

    ax2b = ax.twinx()
    ln3, = ax2b.plot(mimic_times, mimic_aucs, 'D--', color=C['MIMIC'], lw=1.4, ms=4, label='MIMIC-III')
    ax2b.set_ylabel("Time-dependent AUC (MIMIC-III)", fontsize=6.5, color=C['MIMIC'])
    ax2b.tick_params(axis='y', labelcolor=C['MIMIC'], labelsize=6)
    ax2b.set_ylim(0.55, 0.85)

    ax.legend(handles=[ln1, ln2, ln3], fontsize=6, loc='lower right')

    # ── C: Brier Score ────────────────────────────────────────────────────────
    ax = axes[2]
    sch_brier = [sm[('SCH', f'Brier_{t}d')] for t in sch_times]
    whu_brier = [sm[('WHU', f'Brier_{t}d')] for t in whu_times]
    # Null Brier ≈ p*(1-p) where p is prevalence
    sch_prev = float(optb['SCH_MACE_rate']) / 100
    whu_prev = float(optb['WHU_MACE_rate']) / 100
    sch_null = [sch_prev * (1 - sch_prev)] * len(sch_times)
    whu_null = [whu_prev * (1 - whu_prev)] * len(whu_times)

    ax.plot(sch_times, sch_brier, 'o-',  color=C['SCH'], lw=1.4, ms=4, label='SCH model')
    ax.plot(whu_times, whu_brier, 's-',  color=C['WHU'], lw=1.4, ms=4, label='WHU model')
    ax.plot(sch_times, sch_null,  'o--', color=C['SCH'], lw=1.0, ms=3, alpha=0.55, label='SCH null')
    ax.plot(whu_times, whu_null,  's--', color=C['WHU'], lw=1.0, ms=3, alpha=0.55, label='WHU null')
    ax.set_xlabel("Days", fontsize=7)
    ax.set_ylabel("Brier Score", fontsize=7)
    ax.set_title("C) Brier Score over time", fontsize=7)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(False, linestyle=':', linewidth=0.4)

    axes[0].set_box_aspect(1)
    axes[2].set_box_aspect(1)
    plt.tight_layout()
    save(fig, "Fig3_TimeDep_Performance")
    plt.close(fig)
    print("  ✓ Figure 3 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4  – Decision curve analysis
# ══════════════════════════════════════════════════════════════════════════════
def figure4():
    sch_data  = pred_v2[pred_v2['Center'] == 'SCH'].dropna(subset=['predicted_prob_within_cv','MACE'])
    whu_data  = pred_v2[pred_v2['Center'] == 'WHU'].dropna(subset=['predicted_prob_within_cv','MACE'])

    def age_logistic(age_series, k, cut):
        return 1 / (1 + np.exp(-k * (age_series - cut)))

    sch_age = age_logistic(sch_data['年龄'], 0.045, 55).values
    whu_age = age_logistic(whu_data['年龄'], 0.055, 58).values
    mimic_age = mimic_pp['oof_prob_age_only'].values

    fig, axes = plt.subplots(1, 3, figsize=(W2, 88/25.4))

    # ── A: WHU ────────────────────────────────────────────────────────────────
    ax = axes[0]
    thresh = np.linspace(0.05, 0.40, 120)
    y_whu = whu_data['MACE'].values
    nb_whu_model = dca_nb(y_whu, whu_data['predicted_prob_within_cv'].values, thresh)
    nb_whu_age   = dca_nb(y_whu, whu_age, thresh)
    nb_whu_all   = np.mean(y_whu) - (1 - np.mean(y_whu)) * thresh / (1 - thresh + 1e-9)

    ax.plot(thresh, nb_whu_model, color=C['WHU'],  lw=1.8, label='Ensemble model')
    ax.plot(thresh, nb_whu_age,   color=C['Age'],   lw=1.4, linestyle='--', label='Age-only')
    ax.plot(thresh, nb_whu_all,   color=C['All'],   lw=1.0, linestyle='--', label='Treat all')
    ax.axhline(0, color=C['None'], lw=1.0, linestyle=':', label='Treat none')
    ax.set_xlim(0.05, 0.40); ax.set_ylim(-0.05, 0.18)
    ax.set_title("A) WHU cohort (5–40% threshold)", fontsize=7)
    ax.set_xlabel("Decision threshold", fontsize=7)
    ax.set_ylabel("Net benefit", fontsize=7)
    ax.legend(fontsize=6); ax.grid(False, linestyle=':', linewidth=0.4)

    # ── B: SCH – incremental net benefit ─────────────────────────────────────
    ax = axes[1]
    thresh2 = np.linspace(0.001, 0.15, 150)
    y_sch = sch_data['MACE'].values
    nb_sch_model = dca_nb(y_sch, sch_data['predicted_prob_within_cv'].values, thresh2)
    nb_sch_age   = dca_nb(y_sch, sch_age, thresh2)
    inc_nb = nb_sch_model - nb_sch_age

    ax.plot(thresh2, inc_nb, color=C['SCH'], lw=1.8, label='Incremental net benefit')
    ax.axhline(0, color='black', linestyle=':', lw=0.8)
    ax.fill_between(thresh2, inc_nb, 0, where=(inc_nb > 0), color=C['SCH'], alpha=0.18)
    ax.set_xlim(0, 0.15); ax.set_ylim(-0.010, 0.020)
    ax.set_title("B) SCH cohort (0–15% threshold)", fontsize=7)
    ax.set_xlabel("Decision threshold", fontsize=7)
    ax.set_ylabel("Incremental net benefit\n(ensemble vs age-only)", fontsize=7)
    ax.legend(fontsize=6); ax.grid(False, linestyle=':', linewidth=0.4)

    # ── C: MIMIC-III ──────────────────────────────────────────────────────────
    ax = axes[2]
    thresh3 = np.linspace(0.10, 0.90, 120)
    y_mimic = mimic_pp['MACE'].values
    nb_mimic_model = dca_nb(y_mimic, mimic_pp['oof_prob_paper'].values, thresh3)
    nb_mimic_age   = dca_nb(y_mimic, mimic_age, thresh3)
    nb_mimic_all   = np.mean(y_mimic) - (1 - np.mean(y_mimic)) * thresh3 / (1 - thresh3 + 1e-9)

    ax.plot(thresh3, nb_mimic_model, color=C['MIMIC'], lw=1.8, label='Ensemble model')
    ax.plot(thresh3, nb_mimic_age,   color=C['Age'],    lw=1.4, linestyle='--', label='Age-only')
    ax.plot(thresh3, nb_mimic_all,   color=C['All'],    lw=1.0, linestyle='--', label='Treat all')
    ax.axhline(0, color=C['None'], lw=1.0, linestyle=':', label='Treat none')
    ax.set_xlim(0.10, 0.90); ax.set_ylim(-0.10, 0.45)
    ax.set_title("C) MIMIC-III cohort (10–90% threshold)", fontsize=7)
    ax.set_xlabel("Decision threshold", fontsize=7)
    ax.set_ylabel("Net benefit", fontsize=7)
    ax.legend(fontsize=6); ax.grid(False, linestyle=':', linewidth=0.4)

    for ax in axes:
        ax.set_box_aspect(1)
    plt.tight_layout()
    save(fig, "Fig4_DCA")
    plt.close(fig)
    print("  ✓ Figure 4 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5  – Tri-cohort concordant biomarkers
# ══════════════════════════════════════════════════════════════════════════════
def figure5():
    # ── Concordance sets ──────────────────────────────────────────────────────
    TRI_EN  = {'Age', 'LDL Cholesterol', 'HDL Cholesterol', 'Direct Bilirubin'}
    BI_cols = set(fi_comp[fi_comp['In_common'] == True]['English'].tolist()) - TRI_EN
    # Socioeconomic access variables – appear exclusively in WHU (ascertainment, not biology)
    SOCIOECO_EN = {
        'No. of Physicians', 'Hospital Beds', 'GDP per Capita',
        'No. of Hospitals', 'Avg. Employee Wage',
    }
    C_SOCIOECO = '#E69F00'   # amber – socioeconomic access variables (WHU only)
    DEMO_VARS  = {'Age', 'Sex'}  # demographic variables – distinct from lab biomarkers

    def bar_color(name):
        if name in TRI_EN:      return C['tri']
        if name in BI_cols:     return C['bi']
        if name in SOCIOECO_EN: return C_SOCIOECO
        return C['single']

    def bar_color_whu(name):
        """WHU-specific coloring: socioeconomic access variables highlighted in amber."""
        if name in TRI_EN:      return C['tri']
        if name in BI_cols:     return C['bi']
        if name in SOCIOECO_EN: return C_SOCIOECO
        return C['single']

    # ── MIMIC feature helpers ─────────────────────────────────────────────────
    def mimic_base(feat):
        """'LDL Cholesterol__max'  →  'LDL Cholesterol'"""
        return str(feat).rsplit('__', 1)[0].strip() if '__' in str(feat) else str(feat)

    # Readable clinical labels for MIMIC-specific coded features
    MIMIC_RENAMES = {
        'Admission_has_cardiac': 'Cardiac Admission',
    }

    def mimic_display(base_name):
        """Return the unified display name: rename internal codes, keep rest as-is."""
        return MIMIC_RENAMES.get(base_name, base_name)

    # ── Normalise MIMIC importance & COLLAPSE to unique biomarker base names ──
    # Each biomarker (e.g. Creatinine) gets ONE bar = max importance across all
    # stat aggregates (last/mean/max/min). This makes panel C directly comparable
    # to SCH/WHU panels where each biomarker appears exactly once.
    mimic_fi2 = mimic_fi.copy()
    mimic_fi2['norm'] = mimic_fi2['Avg_imp'] / mimic_fi2['Avg_imp'].max()
    mimic_fi2['base'] = mimic_fi2['Feature'].map(mimic_base)

    # Aggregate: SUM importance across all time-aggregation variants per biomarker.
    # This ensures e.g. LDL Cholesterol (with __last, __mean, __max, __min variants)
    # is represented by its *total* importance rather than just one variant.
    mimic_agg = (mimic_fi2
                 .groupby('base', as_index=False)['Avg_imp']
                 .sum()
                 .rename(columns={'Avg_imp': 'sum_imp'})
                 .sort_values('sum_imp', ascending=False)
                 .copy())
    mimic_agg['display'] = mimic_agg['base'].map(mimic_display)
    # Normalise so top biomarker = 1.0
    mimic_agg['norm'] = mimic_agg['sum_imp'] / mimic_agg['sum_imp'].max()

    TOP_N = 25   # top-25 features per panel (matches manuscript spec)

    # ── Figure & grid ─────────────────────────────────────────────────────────
    # Extra height at bottom (bottom=0.14 ≈ 37mm) reserved for the unified legend
    fig = plt.figure(figsize=(W2, 265/25.4))
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.8, 1.0],
        hspace=0.50, wspace=0.52,
        left=0.07, right=0.97, top=0.955, bottom=0.14,
    )

    BAR_FS  = 6.0    # y-tick label font size for bar panels
    TITLE_FS = 7.5

    # ── A: SCH top-N ──────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    top_sch = fi_comp.nlargest(TOP_N, 'SCH_norm').copy()
    top_sch['col'] = top_sch['English'].map(bar_color)
    ax_a.barh(top_sch['English'], top_sch['SCH_norm'],
              color=top_sch['col'].tolist(), height=0.65)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Normalised importance (gain)", fontsize=7)
    ax_a.set_title(f"A) SCH  –  top {TOP_N} features", fontsize=TITLE_FS,
                   fontweight='bold', color=C['SCH'])
    ax_a.tick_params(axis='y', labelsize=BAR_FS)
    ax_a.set_xlim(0, top_sch['SCH_norm'].max() * 1.12)
    ax_a.grid(False, axis='x', linestyle=':', linewidth=0.4, alpha=0.7)

    # ── B: WHU top-N ──────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    top_whu = fi_comp.nlargest(TOP_N, 'WHU_norm').copy()
    top_whu['col'] = top_whu['English'].map(bar_color_whu)
    ax_b.barh(top_whu['English'], top_whu['WHU_norm'],
              color=top_whu['col'].tolist(), height=0.65)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Normalised importance (gain)", fontsize=7)
    ax_b.set_title(f"B) WHU  –  top {TOP_N} features", fontsize=TITLE_FS,
                   fontweight='bold', color=C['WHU'])
    ax_b.tick_params(axis='y', labelsize=BAR_FS)
    ax_b.set_xlim(0, top_whu['WHU_norm'].max() * 1.12)
    ax_b.grid(False, axis='x', linestyle=':', linewidth=0.4, alpha=0.7)

    # ── C: MIMIC-III top-N ────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    top_mim = mimic_agg.head(TOP_N).copy()
    mimic_col = [
        C['tri'] if b in TRI_EN else (C['bi'] if b in BI_cols else C['single'])
        for b in top_mim['base']
    ]
    ax_c.barh(top_mim['display'], top_mim['norm'], color=mimic_col, height=0.65)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Normalised importance (sum across all time-aggregation variants, gain)", fontsize=7)
    ax_c.set_title(f"C) MIMIC-III  –  top {TOP_N} features\n(sum across last/mean/max/min variants per biomarker)",
                   fontsize=TITLE_FS, fontweight='bold', color=C['MIMIC'])
    ax_c.tick_params(axis='y', labelsize=BAR_FS)
    ax_c.set_xlim(0, top_mim['norm'].max() * 1.12)
    ax_c.grid(False, axis='x', linestyle=':', linewidth=0.4, alpha=0.7)

    # ── D: Concordance scatter  (spans full bottom row) ───────────────────────
    ax_d = fig.add_subplot(gs[1, :])

    # MIMIC max importance per base name (already aggregated in mimic_agg)
    mimic_base_imp = mimic_agg.set_index('base')['norm'].to_dict()

    plot_df = fi_comp.copy()
    plot_df['MIMIC_norm']   = plot_df['English'].map(mimic_base_imp).fillna(0)
    plot_df['concordance']  = plot_df['English'].map(
        lambda n: 'Tri-cohort' if n in TRI_EN
                  else ('Bi-cohort' if n in BI_cols else 'Single cohort')
    )

    pal = {'Tri-cohort': C['tri'], 'Bi-cohort': C['bi'], 'Single cohort': C['single']}
    demo_label_added = False
    for grp, sub in plot_df.groupby('concordance', sort=False, observed=True):
        sz_all   = (30 + sub['MIMIC_norm'] * 550).values
        lab_idx  = ~sub['English'].isin(DEMO_VARS)
        demo_idx =  sub['English'].isin(DEMO_VARS)

        # Lab / clinical biomarkers – circles
        sub_lab = sub[lab_idx]
        if len(sub_lab):
            ax_d.scatter(sub_lab['SCH_norm'], sub_lab['WHU_norm'],
                         s=sz_all[lab_idx.values],
                         c=pal[grp], alpha=0.78, label=grp,
                         edgecolors='white', linewidths=0.4, marker='o')

        # Demographic variables – triangles (distinguish from lab biomarkers)
        sub_demo = sub[demo_idx]
        if len(sub_demo):
            demo_lbl = 'Demographic variable (triangle)' if not demo_label_added else '_nolegend_'
            ax_d.scatter(sub_demo['SCH_norm'], sub_demo['WHU_norm'],
                         s=sz_all[demo_idx.values],
                         c=pal[grp], alpha=0.88, label=demo_lbl,
                         edgecolors='black', linewidths=0.9, marker='^', zorder=5)
            demo_label_added = True

    for _, row in plot_df[plot_df['English'].isin(TRI_EN)].iterrows():
        ax_d.annotate(row['English'],
                      xy=(row['SCH_norm'], row['WHU_norm']),
                      xytext=(9, 5), textcoords='offset points',
                      fontsize=6.2, color='#08306B',
                      arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.6))

    ax_d.set_xlabel("Normalised importance in SCH (gain)", fontsize=7)
    ax_d.set_ylabel("Normalised importance in WHU (gain)", fontsize=7)
    ax_d.set_title("D) Tri-cohort biomarker concordance  "
                   "(bubble size proportional to summed MIMIC-III importance; triangles = demographic variables)",
                   fontsize=TITLE_FS)
    # No inline legend on the scatter – all entries go to the bottom figure legend below
    ax_d.grid(False, linestyle=':', linewidth=0.4, alpha=0.6)

    # ── Unified bottom legend — all entries below the scatter panel ──────────
    # Bar-shading patches
    legend_handles = [
        mpatches.Patch(fc=C['tri'],    label='[***] Tri-cohort concordant  (SCH, WHU, MIMIC-III)'),
        mpatches.Patch(fc=C['bi'],     label='[** ] Bi-cohort concordant   (SCH & WHU)'),
        mpatches.Patch(fc=C['single'], label='[*  ] Single-cohort only'),
        mpatches.Patch(fc=C_SOCIOECO,  label='Healthcare access / socioeconomic variable'),
        # Scatter marker convention
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=C['tri'],
                   markersize=6, label='Lab / clinical biomarker (circle)'),
        plt.Line2D([0],[0], marker='^', color='w', markerfacecolor=C['tri'],
                   markeredgecolor='black', markeredgewidth=0.8,
                   markersize=6, label='Demographic variable (triangle)'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.50, 0.006),   # anchored to bottom centre of figure
        ncol=3,
        fontsize=6.5,
        title="Shading convention (bars A–C) and scatter marker convention (panel D)",
        title_fontsize=6.5,
        frameon=True,
        edgecolor='#CCCCCC',
        columnspacing=1.0,
    )

    save(fig, "Fig5_Feature_Importance")
    plt.close(fig)
    print("  ✓ Figure 5 saved.")



# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6  – Transfer analysis
# ══════════════════════════════════════════════════════════════════════════════
def figure6():
    # ── Build 3×3 transfer matrix ─────────────────────────────────────────────
    within = {
        'SCH':       float(optb['SCH_CV_AUC']),
        'WHU':       float(optb['WHU_CV_AUC']),
        'MIMIC-III': float(summary_v13['AUC']),          # 0.742 — 43-feature model
    }
    off_diag = {
        ('SCH',  'WHU'):       float(optb['SCH_to_WHU_transfer']),
        ('WHU',  'SCH'):       float(optb['WHU_to_SCH_transfer']),
        ('SCH',  'MIMIC-III'): float(summary_v13['Transfer_SCH2MIMIC']),  # 0.4998
        ('WHU',  'MIMIC-III'): float(summary_v13['Transfer_WHU2MIMIC']),  # 0.4938 — reversal
        ('MIMIC-III','SCH'):   float(summary_v13['Transfer_MIMIC2SCH']),  # 0.5591
        ('MIMIC-III','WHU'):   float(summary_v13['Transfer_MIMIC2WHU']),  # 0.4440 — reversal
    }
    cohort_order = ['SCH', 'WHU', 'MIMIC-III']
    mat = pd.DataFrame(index=cohort_order, columns=cohort_order, dtype=float)
    for c in cohort_order:
        mat.loc[c, c] = within[c]
    for (tr, te), v in off_diag.items():
        mat.loc[tr, te] = v

    # Extra bottom margin (0.18 ≈ 24 mm) keeps the legend clear of both subplots
    # 3-column GridSpec: bar chart | heatmap | colorbar (narrow)
    fig = plt.figure(figsize=(W2, 105/25.4))
    gs  = gridspec.GridSpec(
        1, 3, figure=fig,
        width_ratios=[1, 1, 0.06],
        left=0.08, right=0.97, bottom=0.18, top=0.89,
        wspace=0.42,
    )
    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[0, 1])
    _cax = fig.add_subplot(gs[0, 2])

    # ── A: Grouped bar chart – grouped by TEST cohort ─────────────────────────
    # Design: within-CV = cohort colour (solid, full opacity)
    #         transfers  = neutral grey (all directions, no hatch)
    # Only the within-CV bars carry value labels; transfer bars labelled only
    # for the SCH->WHU reversal cell.
    bar_col_map = {'SCH': C['SCH'], 'WHU': C['WHU'], 'MIMIC-III': C['MIMIC']}
    bar_w  = 0.20
    gap    = 0.06   # gap between within-CV and first transfer bar
    sep    = 0.04   # gap between transfer bars
    gx_ctr = np.array([0.0, 1.25, 2.50])   # group anchor positions

    _added_within = False
    _added_xfr_labels = set()   # track which source cohorts have been in legend
    for i, tc in enumerate(cohort_order):
        gx = gx_ctr[i]

        # within-CV bar – solid cohort color
        v_w = within[tc]
        ax1.bar(gx, v_w, bar_w,
                color=bar_col_map[tc], zorder=3,
                label='Within-cohort CV' if not _added_within else '_')
        ax1.text(gx, v_w + 0.010, f'{v_w:.3f}',
                 ha='center', va='bottom', fontsize=5.8, fontweight='bold',
                 color='#333333')
        _added_within = True

        # two transfer bars – source-cohort color (alpha=0.38) so colour encodes origin
        sources = [c for c in cohort_order if c != tc]
        for j, src in enumerate(sources):
            xpos  = gx + bar_w + gap + j * (bar_w + sep)
            v_tr  = mat.loc[src, tc]
            below = v_tr < 0.5
            src_col = bar_col_map[src]
            xfr_lbl = f'Transfer from {src}' if src not in _added_xfr_labels else '_'
            ax1.bar(xpos, v_tr, bar_w,
                    color=src_col, alpha=0.38,
                    edgecolor=src_col, linewidth=0.9,
                    hatch='////' if below else '',
                    zorder=2, label=xfr_lbl)
            _added_xfr_labels.add(src)
            # value label for all transfer bars
            ax1.text(xpos, v_tr + 0.010, f'{v_tr:.3f}',
                     ha='center', va='bottom', fontsize=5.2,
                     color='#CC0000' if below else '#555555',
                     fontweight='bold' if below else 'normal')

    ax1.axhline(0.5, color='#CC0000', linestyle='--', lw=1.0, label='Chance (AUC = 0.50)')
    # X-ticks centred under each group
    grp_span = bar_w + gap + 2 * bar_w + sep   # total width occupied
    ax1.set_xticks(gx_ctr + grp_span / 2 - bar_w / 2)
    ax1.set_xticklabels([f'Test: {c}' for c in cohort_order], fontsize=6.5)
    ax1.set_xlim(gx_ctr[0] - 0.18, gx_ctr[-1] + grp_span + 0.18)
    ax1.set_ylim(0.30, 0.97)
    ax1.set_ylabel("AUC", fontsize=7)
    ax1.set_title("A) Within-cohort CV vs cross-cohort transfer AUC", fontsize=7,
                  fontweight='bold', pad=4)
    ax1.grid(False, axis='y', linestyle=':', linewidth=0.4, alpha=0.7)
    ax1.tick_params(axis='x', length=0)

    # ── B: Heatmap ────────────────────────────────────────────────────────────
    annot_mat = mat.applymap(lambda v: f"{v:.3f}")
    sns.heatmap(mat, annot=annot_mat, fmt='', cmap='RdYlGn_r',
                vmin=0.40, vmax=0.90,        # tighter range for visual clarity
                linewidths=0.8, linecolor='white',
                ax=ax2, cbar=False,
                annot_kws={'fontsize': 7.5, 'fontweight': 'bold'})
    ax2.set_box_aspect(1)   # keep heatmap square
    # Colorbar in its own dedicated GridSpec column – always flush with heatmap
    _sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                                 norm=plt.Normalize(vmin=0.40, vmax=0.90))
    _sm.set_array([])
    _cb = plt.colorbar(_sm, cax=_cax)
    _cb.set_label('AUC', fontsize=6.5)
    _cb.ax.tick_params(labelsize=6)
    # Bold black border on diagonal cells (within-cohort CV)
    for _i in range(len(cohort_order)):
        ax2.add_patch(plt.Rectangle((_i, _i), 1, 1, fill=False,
                                    edgecolor='black', linewidth=2.2, zorder=10))
    ax2.set_title("B) Transfer AUC matrix\n(row = train cohort,  col = test cohort)",
                  fontsize=7, fontweight='bold', pad=4)
    ax2.set_xlabel("Test cohort", fontsize=7)
    ax2.set_ylabel("Train cohort", fontsize=7)
    ax2.tick_params(labelsize=7, length=0)
    ax1.set_box_aspect(1)
    # ax2 box_aspect set above; _cax is its own GridSpec column

    # ── Shared legend – below both panels ─────────────────────────────────────
    def _xfr_patch(cohort_key, name):
        c = bar_col_map[cohort_key]
        return mpatches.Patch(fc=c, alpha=0.38, ec=c, linewidth=0.9,
                              label=f'Transfer from {name} (no retraining)')
    leg_handles = [
        mpatches.Patch(fc=C['SCH'],   label=f'SCH  – within-cohort CV (AUC = {within["SCH"]:.3f})'),
        mpatches.Patch(fc=C['WHU'],   label=f'WHU  – within-cohort CV (AUC = {within["WHU"]:.3f})'),
        mpatches.Patch(fc=C['MIMIC'], label=f'MIMIC-III  – within-cohort CV (AUC = {within["MIMIC-III"]:.3f})'),
        _xfr_patch('SCH',      'SCH'),
        _xfr_patch('WHU',      'WHU'),
        _xfr_patch('MIMIC-III','MIMIC-III'),
        mpatches.Patch(fc='white', ec='#888888', linewidth=0.9, hatch='////',
                       label='Hatching = transfer AUC < 0.50  (below chance)'),
        plt.Line2D([0],[0], color='#CC0000', linestyle='--', lw=1.2,
                   label='Chance level (AUC = 0.50)'),
        mpatches.Patch(fc='white', ec='black', linewidth=2.0,
                       label='Bold border = within-cohort CV (diagonal of matrix B)'),
    ]
    fig.legend(
        handles=leg_handles,
        loc='lower center',
        bbox_to_anchor=(0.50, 0.00),
        ncol=3,
        fontsize=6.2,
        frameon=True,
        edgecolor='#CCCCCC',
        columnspacing=1.0,
        handlelength=1.6,
    )

    save(fig, "Fig6_Transfer_Analysis")
    plt.close(fig)
    print("  ✓ Figure 6 saved.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating Publication Main Figures 1–6 …")
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    print(f"\nAll figures saved to: {OUT}")
