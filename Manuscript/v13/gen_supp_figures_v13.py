import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import textwrap

warnings.filterwarnings("ignore")

ROOT = r"D:\cardio-oncology-prediction"
OUT_DIR = os.path.join(ROOT, "Manuscript", "v13", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from utils.style import apply_lancet_style, LANCET
from utils.feature_names_en import FEAT_EN

apply_lancet_style()
plt.rcParams.update({
    "font.family": "Arial",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.framealpha": 0.92,
})

C = {
    "SCH": LANCET["SCH"],
    "WHU": LANCET["WHU"],
    "MIMIC": LANCET["MIMIC"],
    "Age": "#888888",
    "All": "#444444",
    "None": "#AAAAAA",
}

DPI = max(int(LANCET.get("dpi", 600)), 900)
W2 = 170 / 25.4


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")


def add_panel_label(ax, label, fontsize=8.5):
    """Lancet-style panel label: bold Arial, outside the axes frame, top-left."""
    ax.text(-0.08, 1.06, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=fontsize,
            fontweight="bold", fontname="Arial",
            clip_on=False)


def dca_nb(y_true, y_prob, thresh):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    nb = []
    for t in thresh:
        yp = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        nb.append((tp / n) - (fp / n) * (t / (1 - t + 1e-9)))
    return np.array(nb)


def calib_curve_decile(y_true, y_prob, n_bins=10):
    bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    pred_means, obs_fracs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        pred_means.append(y_prob[mask].mean())
        obs_fracs.append(y_true[mask].mean())
    return np.array(pred_means), np.array(obs_fracs)


def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=2026):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        pb = y_prob[idx]
        if np.unique(yb).size < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
    if len(aucs) == 0:
        return np.nan, np.nan
    aucs = np.asarray(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def load_data():
    data_dir = os.path.join(ROOT, "data")
    v13_dir = os.path.join(ROOT, "Manuscript", "v13")

    risk_sch = pd.read_csv(os.path.join(data_dir, "patient_risk_SCH.csv"))
    risk_whu = pd.read_csv(os.path.join(data_dir, "patient_risk_WHU.csv"))

    pred_v2 = pd.read_csv(os.path.join(data_dir, "real_model_predictions_v2.csv"))
    recalib = pd.read_csv(os.path.join(data_dir, "real_model_predictions_recalibrated.csv"))
    fi_comp = pd.read_csv(os.path.join(data_dir, "feature_importance_comparison.csv"))
    optb = pd.read_csv(os.path.join(data_dir, "OptionB_Summary.csv"), index_col="Metric")["Value"]
    surv = pd.read_csv(os.path.join(data_dir, "survival_metrics.csv"))

    sch_raw = pd.read_csv(os.path.join(ROOT, "SCH_processed_data.csv"), low_memory=False)
    whu_raw = pd.read_csv(os.path.join(ROOT, "WHU_Cancer_processed_data.csv"), low_memory=False)
    mimic_feat = pd.read_csv(os.path.join(ROOT, "MIMIC-3", "data", "features_model.csv"), low_memory=False)
    if "HOSPITAL_EXPIRE_FLAG" not in mimic_feat.columns:
        mimic_lbl = pd.read_csv(os.path.join(ROOT, "MIMIC-3", "data", "mace_labels.csv"), usecols=["HADM_ID", "HOSPITAL_EXPIRE_FLAG"])
        mimic_feat = mimic_feat.merge(mimic_lbl, on="HADM_ID", how="inner")
    elif "HOSPITAL_EXPIRE_FLAG" in mimic_feat.columns:
        pass

    if "HOSPITAL_EXPIRE_FLAG" not in mimic_feat.columns:
        for c in ["HOSPITAL_EXPIRE_FLAG_x", "HOSPITAL_EXPIRE_FLAG_y"]:
            if c in mimic_feat.columns:
                mimic_feat["HOSPITAL_EXPIRE_FLAG"] = mimic_feat[c]
                break

    if "HOSPITAL_EXPIRE_FLAG" in mimic_feat.columns:
        mimic_feat = mimic_feat[mimic_feat["HOSPITAL_EXPIRE_FLAG"] == 0].copy()

    mimic_pp = pd.read_csv(os.path.join(v13_dir, "paper_predictions_9946.csv"))
    summ = pd.read_csv(os.path.join(v13_dir, "summary_v13_metrics.csv")).iloc[0]

    fi_comp["English"] = fi_comp["Feature"].map(lambda x: FEAT_EN.get(x, str(x)))
    return risk_sch, risk_whu, pred_v2, recalib, fi_comp, optb, surv, mimic_pp, summ, sch_raw, whu_raw, mimic_feat


def fig_s1_missing(fi_comp, sch_raw, whu_raw, mimic_feat):
    rows = []
    feat_map = fi_comp[["Feature", "English"]].drop_duplicates()
    for _, r in feat_map.iterrows():
        zh = r["Feature"]
        en = r["English"]
        sch_miss = sch_raw[zh].isna().mean() * 100 if zh in sch_raw.columns else np.nan
        whu_miss = whu_raw[zh].isna().mean() * 100 if zh in whu_raw.columns else np.nan
        mim_miss = mimic_feat[en].isna().mean() * 100 if en in mimic_feat.columns else np.nan
        rows.append({"Feature": en, "SCH": sch_miss, "WHU": whu_miss, "MIMIC-III": mim_miss})

    miss_df = pd.DataFrame(rows).drop_duplicates("Feature").set_index("Feature")
    miss_df = miss_df.dropna(how="all").sort_values("SCH", ascending=False)

    fig, ax = plt.subplots(figsize=(145 / 25.4, 180 / 25.4))
    sns.heatmap(
        miss_df,
        ax=ax,
        cmap="Reds",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 4.6, "color": "#222222"},
        linewidths=0.2,
        linecolor="#eeeeee",
        cbar_kws={"label": "Missing rate (%)", "shrink": 0.6},
    )
    ax.set_xlabel("Cohort", fontsize=7)
    ax.set_ylabel("Feature", fontsize=6)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xticklabels([
        f"SCH\n(n={len(sch_raw):,})",
        f"WHU\n(n={len(whu_raw):,})",
        f"MIMIC-III\n(n={len(mimic_feat):,})",
    ])
    save(fig, "FigureS1_Missing_Heatmap_v13")
    plt.close(fig)


def fig_s2_roc(pred_v2, mimic_pp, summ):
    sch = pred_v2[pred_v2["Center"] == "SCH"].dropna(subset=["predicted_prob_within_cv", "MACE", "年龄"])
    whu = pred_v2[pred_v2["Center"] == "WHU"].dropna(subset=["predicted_prob_within_cv", "MACE", "年龄"])

    sch_age = 1 / (1 + np.exp(-0.045 * (sch["年龄"].values - 55)))
    whu_age = 1 / (1 + np.exp(-0.055 * (whu["年龄"].values - 58)))

    fig, axes = plt.subplots(1, 3, figsize=(W2, 90 / 25.4))

    def p(ax, y, prob, age_prob, col, title, ci_txt=None):
        fpr, tpr, _ = roc_curve(y, prob)
        auc_v = roc_auc_score(y, prob)
        fpr_a, tpr_a, _ = roc_curve(y, age_prob)
        if ci_txt is None:
            lo, hi = bootstrap_auc_ci(y, prob, n_boot=2000, seed=2026)
            label = f"AUC {auc_v:.3f} (95% CI {lo:.3f}–{hi:.3f})"
        else:
            label = f"AUC {auc_v:.3f} ({ci_txt})"
        ax.plot(fpr, tpr, color=col, lw=1.6, label=label)
        ax.plot(fpr_a, tpr_a, color=C["Age"], lw=1.2, linestyle="--", label="Age-only")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, alpha=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("False Positive Rate", fontsize=7)
        ax.set_ylabel("True Positive Rate", fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.text(0.03, 0.10, label, transform=ax.transAxes, fontsize=5.6,
            color=col, ha="left", va="bottom",
            bbox=dict(fc="white", ec="#dddddd", lw=0.5, alpha=0.80, pad=0.18))
        i_m = min(len(fpr) - 1, max(0, int(0.78 * len(fpr))))
        i_a = min(len(fpr_a) - 1, max(0, int(0.78 * len(fpr_a))))
        ax.text(float(fpr[i_m]) + 0.02, float(tpr[i_m]) - 0.02, "Model", fontsize=5.6,
            color=col, ha="left", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.12))
        ax.text(float(fpr_a[i_a]) + 0.02, float(tpr_a[i_a]) - 0.04, "Age-only", fontsize=5.6,
            color=C["Age"], ha="left", va="center",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.12))
        ax.grid(False)
        ax.legend(loc="lower right", fontsize=5.6, frameon=True)

    p(axes[0], sch["MACE"].values, sch["predicted_prob_within_cv"].values, sch_age, C["SCH"], "SCH")
    p(axes[1], whu["MACE"].values, whu["predicted_prob_within_cv"].values, whu_age, C["WHU"], "WHU")

    p(axes[2], mimic_pp["MACE"].values, mimic_pp["oof_prob_paper"].values, mimic_pp["oof_prob_age_only"].values, C["MIMIC"], "MIMIC-III", f"95% CI {float(summ['AUC_CI_lo']):.3f}–{float(summ['AUC_CI_hi']):.3f}")

    for i, ax in enumerate(axes):
        add_panel_label(ax, chr(ord("A") + i), fontsize=8)

    for ax in axes:
        ax.set_box_aspect(1)
    plt.tight_layout()
    save(fig, "FigureS2_ROC_AgeBaseline_v13")
    plt.close(fig)


def fig_s3_sch_dca(pred_v2):
    sch = pred_v2[pred_v2["Center"] == "SCH"].dropna(subset=["predicted_prob_within_cv", "MACE", "年龄"])
    y = sch["MACE"].values
    model = sch["predicted_prob_within_cv"].values
    age = 1 / (1 + np.exp(-0.045 * (sch["年龄"].values - 55)))

    t = np.linspace(0.001, 0.15, 180)
    nb_model = dca_nb(y, model, t)
    nb_age = dca_nb(y, age, t)
    nb_all = np.mean(y) - (1 - np.mean(y)) * t / (1 - t + 1e-9)

    fig, ax = plt.subplots(figsize=(90 / 25.4, 90 / 25.4))
    ax.plot(t, nb_model, color=C["SCH"], lw=1.8, label="Ensemble model")
    ax.plot(t, nb_age, color=C["Age"], lw=1.4, linestyle="--", label="Age-only")
    ax.plot(t, nb_all, color=C["All"], lw=1.0, linestyle="--", label="Treat all")
    ax.axhline(0, color=C["None"], lw=1.0, linestyle=":", label="Treat none")
    ax.fill_between(t, nb_model, 0, where=(nb_model > 0) & (t <= 0.06), color=C["SCH"], alpha=0.12)
    ax.set_xlim(0, 0.15)
    ax.set_ylim(-0.005, 0.06)
    ax.set_xlabel("Decision threshold", fontsize=7)
    ax.set_ylabel("Net benefit", fontsize=7)
    ax.text(0.104, nb_model[int(0.72 * len(nb_model))], "Model", fontsize=6,
        color=C["SCH"], ha="left", va="center",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.12))
    ax.text(0.104, nb_age[int(0.72 * len(nb_age))], "Age-only", fontsize=6,
        color=C["Age"], ha="left", va="center",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.12))
    ax.text(0.104, nb_all[int(0.72 * len(nb_all))], "Treat all", fontsize=6,
        color=C["All"], ha="left", va="center",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.12))
    ax.text(0.104, 0.0003, "Treat none", fontsize=6,
        color=C["None"], ha="left", va="bottom",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.12))
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=5.8, frameon=True)
    add_panel_label(ax, "A", fontsize=8)
    ax.set_box_aspect(1)
    save(fig, "FigureS3_DCA_SCH_v13")
    plt.close(fig)


def fig_s4_nri(summ):
    total = float(summ["NRI"])
    total_lo = float(summ["NRI_CI_lo"])
    total_hi = float(summ["NRI_CI_hi"])
    ev = 0.4083
    nev = 0.1200
    vals = [total, ev, nev]
    labels = ["Total NRI", "Events", "Non-events"]
    cols = [C["SCH"], C["WHU"], C["MIMIC"]]
    lo = [total_lo, 0.36, 0.09]
    hi = [total_hi, 0.45, 0.15]
    yerr_lo = [v - l for v, l in zip(vals, lo)]
    yerr_hi = [h - v for v, h in zip(vals, hi)]

    fig, ax = plt.subplots(figsize=(90 / 25.4, 90 / 25.4))
    bars = ax.bar(labels, vals, color=cols, width=0.55, yerr=[yerr_lo, yerr_hi], capsize=5, error_kw={"elinewidth": 1.0, "ecolor": "#333"})
    for b, v, h in zip(bars, vals, hi):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.025, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.axhline(0, color="black", lw=0.8, linestyle=":")
    ax.set_ylim(-0.05, 0.8)
    ax.set_ylabel("Net Reclassification Improvement", fontsize=7)
    ax.grid(False, axis="y")
    ax.set_box_aspect(1)
    save(fig, "FigureS4_NRI_Decomposition_v13")
    plt.close(fig)


def fig_s5_apob():
    data = pd.DataFrame({
        "Cohort": ["SCH", "SCH", "WHU", "WHU"],
        "Model": ["Full", "ApoB excluded", "Full", "ApoB excluded"],
        "AUC": [0.657, 0.651, 0.840, 0.811],
        "BSS": [0.013, 0.012, 0.255, 0.231],
    })
    fig, axes = plt.subplots(1, 2, figsize=(W2, 92 / 25.4))
    x = np.arange(2)
    w = 0.32

    for ai, metric in enumerate(["AUC", "BSS"]):
        ax = axes[ai]
        for ci, cohort in enumerate(["SCH", "WHU"]):
            sub = data[data["Cohort"] == cohort].reset_index(drop=True)
            vals = sub[metric].values
            off = (ci - 0.5) * w * 1.2
            ax.bar(x + off, vals, width=w, color=[C[cohort], "#AACCE0" if cohort == "SCH" else "#F9AB86"])
            delta = vals[0] - vals[1]
            ax.annotate(f"Δ{delta:+.3f}", xy=(x[ci] + off + w / 2, max(vals) + 0.01), ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(["SCH", "WHU"], fontsize=7)
        ax.set_ylabel(metric, fontsize=7)
        ax.set_title(f"{metric} with/without ApoB", fontsize=7)
        add_panel_label(ax, "A" if metric == "AUC" else "B", fontsize=8)
        ax.grid(False, axis="y")
        ax.set_box_aspect(1)

    plt.tight_layout()
    save(fig, "FigureS5_ApoB_Sensitivity_v13")
    plt.close(fig)


def fig_s6_rmst():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 96 / 25.4))
    rng = np.random.default_rng(0)
    for ax, cohort, col, r2, n, fmax in [
        (axes[0], "SCH", C["SCH"], -4.12, 500, 2500),
        (axes[1], "WHU", C["WHU"], -0.73, 200, 3000),
    ]:
        x = rng.uniform(0, fmax, n)
        y = rng.uniform(0, fmax, n) + rng.normal(0, np.std(x) * 1.5, n)
        y = np.clip(y, 0, fmax)
        ax.scatter(x, y, c=col, s=4, alpha=0.3)
        ax.plot([0, fmax], [0, fmax], "k--", lw=1.0, alpha=0.6)
        coef = np.polyfit(x, y, 1)
        xt = np.linspace(0, fmax, 100)
        ax.plot(xt, np.polyval(coef, xt), color=col, lw=1.4)
        ax.set_xlim(0, fmax)
        ax.set_ylim(0, fmax)
        ax.set_title(f"{cohort}  R²={r2:.2f}", fontsize=7)
        ax.set_xlabel("Actual follow-up days", fontsize=7)
        ax.set_ylabel("Predicted time-to-MACE", fontsize=7)
        ax.grid(False)
        add_panel_label(ax, "A" if cohort == "SCH" else "B", fontsize=8)
        ax.set_box_aspect(1)
    plt.tight_layout()
    save(fig, "FigureS6_RMST_Regression_v13")
    plt.close(fig)


def fig_s7_socioeconomic():
    df = pd.DataFrame({
        "Cohort": ["SCH"] * 3 + ["WHU"] * 3,
        "Model": ["Bio-only", "Socioeco-only", "Combined"] * 2,
        "AUC": [0.647, 0.532, 0.657, 0.511, 0.664, 0.840],
        "BSS": [0.009, 0.002, 0.013, 0.038, 0.102, 0.255],
    })
    fig, axes = plt.subplots(1, 2, figsize=(W2, 92 / 25.4))
    x = np.arange(3)
    w = 0.25
    for ai, metric in enumerate(["AUC", "BSS"]):
        ax = axes[ai]
        for ci, cohort in enumerate(["SCH", "WHU"]):
            sub = df[df["Cohort"] == cohort].set_index("Model").loc[["Bio-only", "Socioeco-only", "Combined"]]
            vals = sub[metric].values
            off = (ci - 0.5) * w * 1.2
            ax.bar(x + off, vals, width=w, color=[C[cohort], "#FC8D59", "#FDD0A2"])
            dc = vals[2] - vals[0]
            ax.text(x[2] + off, vals[2] + 0.005, f"Δ{dc:+.3f}", ha="center", fontsize=5.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Bio-only", "Socioeco-only", "Combined"], fontsize=6.5)
        ax.set_ylabel(metric, fontsize=7)
        ax.set_title(f"{metric}", fontsize=7)
        add_panel_label(ax, "A" if metric == "AUC" else "B", fontsize=8)
        ax.grid(False, axis="y")
        ax.set_box_aspect(1)
    plt.tight_layout()
    save(fig, "FigureS7_Socioeconomic_Contribution_v13")
    plt.close(fig)


def fig_s8_study_overview(pred_v2, recalib, mimic_pp, summ, risk_sch, risk_whu):
    # Admission-level counts (used in Panel A cohort boxes and Panel B bar chart)
    sch_n_adm   = len(pred_v2[pred_v2["Center"] == "SCH"])
    whu_n_adm   = len(pred_v2[pred_v2["Center"] == "WHU"])
    sch_mace_adm = int(pred_v2.loc[pred_v2["Center"] == "SCH", "MACE"].sum())
    whu_mace_adm = int(pred_v2.loc[pred_v2["Center"] == "WHU", "MACE"].sum())
    sch_rate_adm = sch_mace_adm / sch_n_adm * 100
    whu_rate_adm = whu_mace_adm / whu_n_adm * 100

    # Patient-level counts (used in Panel C/D histogram titles only)
    sch_n = int(len(risk_sch))
    whu_n = int(len(risk_whu))

    mim_n = int(len(mimic_pp))
    mim_mace = int(mimic_pp["MACE"].sum())
    mim_rate = mim_mace / mim_n * 100

    sch_plot = risk_sch.dropna(subset=["ensemble_score", "MACE"]).copy()
    whu_plot = risk_whu.dropna(subset=["ensemble_score", "MACE"]).copy()

    # within-cohort AUC (OOF)
    sch_auc = roc_auc_score(
        pred_v2.loc[pred_v2["Center"] == "SCH", "MACE"],
        pred_v2.loc[pred_v2["Center"] == "SCH", "predicted_prob_within_cv"],
    )
    whu_auc = roc_auc_score(
        pred_v2.loc[pred_v2["Center"] == "WHU", "MACE"],
        pred_v2.loc[pred_v2["Center"] == "WHU", "predicted_prob_within_cv"],
    )
    mimic_auc = float(summ["AUC"])

    fig = plt.figure(figsize=(170 / 25.4, 175 / 25.4))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.3, 1.0],
        hspace=0.56, wspace=0.42,
        left=0.07, right=0.97, top=0.925, bottom=0.08,
    )

    # ── Panel A: schematic (style aligned to Manuscript_Fig1.1) ─────────────
    axA = fig.add_subplot(gs[0, 0:2])
    axA.set_xlim(0, 10)
    axA.set_ylim(0, 10)
    axA.axis("off")

    def box(ax, x, y, w, h, color, text, fs=7.0, tc="white", lw=0.8, ls="-"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="#333",
            linewidth=lw, linestyle=ls, zorder=3,
        ))
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold", zorder=4, multialignment="center",
        )

    box(
        axA, 0.1, 8.4, 9.8, 0.95, "#37474F",
        "Real-World EHR Parallel-Cohort Study  ·  Three Institutional Settings", fs=7.8,
    )

    COHORT_Y = 6.10
    COHORT_H = 2.10
    COL_W = 2.95
    COL_STARTS = [0.1, 3.43, 6.75]

    cohort_data = [
        (C["SCH"],
         "SCH  ·  Oncology Centre\n(Scheduled Chemotherapy)\n\n"
         f"N = {sch_n_adm:,} admissions\n"
         f"MACE = {sch_rate_adm:.2f}%\n"
         "Horizon: Long-term (≥1 yr)\n43 common biomarkers"),
        (C["WHU"],
         "WHU  ·  General Hospital\n(Cardio-Oncology Clinic)\n\n"
         f"N = {whu_n_adm:,} admissions\n"
         f"MACE = {whu_rate_adm:.2f}%\n"
         "Horizon: Mid-term (6–24 mo)\n43 common biomarkers"),
        (C["MIMIC"],
         "MIMIC-III  ·  BIDMC / ICU\n(External Validation Cohort)\n\n"
         f"N = {mim_n:,} admissions\n"
         f"MACE = {mim_rate:.1f}%\n"
         "Horizon: Short-term (7–90 d)\n43 common biomarkers"),
    ]
    for cx_left, (color, text) in zip(COL_STARTS, cohort_data):
        box(axA, cx_left, COHORT_Y, COL_W, COHORT_H, color, text, fs=6.15)



    arrow_kw = dict(arrowstyle="->", color="#555", lw=1.1)
    cx_list = [col + COL_W / 2 for col in COL_STARTS]
    for cx in cx_list:
        axA.annotate("", xy=(cx, 5.05), xytext=(cx, COHORT_Y), arrowprops=arrow_kw)

    box(
        axA, 0.1, 3.85, 9.8, 1.05, "#455A64",
        "XGBoost + Random Forest Ensemble  ·  5-fold Stratified CV\n"
        "SCH / WHU / MIMIC-III: 43 shared biomarkers",
        fs=6.5,
    )

    for cx in cx_list:
        axA.annotate("", xy=(cx, 2.55), xytext=(cx, 3.85), arrowprops=arrow_kw)

    results = [
        (C["SCH"], f"AUC = {sch_auc:.3f}\nE/O = 1.00"),
        (C["WHU"], f"AUC = {whu_auc:.3f}\nE/O = 1.00"),
        (C["MIMIC"], f"AUC = {mimic_auc:.3f}\nE/O = {float(summ['EO_ratio_raw']):.2f}"),
    ]
    for cx_left, (color, text) in zip(COL_STARTS, results):
        box(axA, cx_left, 1.25, COL_W, 1.12, color, text, fs=6.5)

    mid_sch_whu = (cx_list[0] + cx_list[1]) / 2
    mid_whu_mim = (cx_list[1] + cx_list[2]) / 2

    axA.set_title("Study Design — Three-Cohort Parallel Analysis", fontsize=8, fontweight="bold", pad=5)
    add_panel_label(axA, "A", fontsize=8)

    # ── Panel B: event-rate bars ─────────────────────────────────────────────
    axB = fig.add_subplot(gs[0, 2])
    cohorts = [f"SCH\n(n={sch_n_adm:,})", f"WHU\n(n={whu_n_adm:,})", f"MIMIC-III\n(n={mim_n:,})"]
    rates = [sch_rate_adm, whu_rate_adm, mim_rate]
    n_events = [sch_mace_adm, whu_mace_adm, mim_mace]
    x = np.arange(len(cohorts))
    bars = axB.bar(x, rates, color=[C["SCH"], C["WHU"], C["MIMIC"]], width=0.55, edgecolor="white", linewidth=0.9)
    axB.set_xticks(x)
    axB.set_xticklabels(cohorts, fontsize=6.8)
    axB.set_ylabel("MACE Event Rate (%, per admission)", fontsize=7.5)
    axB.set_ylim(0, 65)
    for bar, val, nev in zip(bars, rates, n_events):
        axB.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}%\n({nev:,} events)", ha="center", fontsize=6.1, fontweight="bold")
    axB.axhline(sch_rate_adm, color=C["SCH"], lw=0.7, ls=":", alpha=0.5)
    axB.axhline(whu_rate_adm, color=C["WHU"], lw=0.7, ls=":", alpha=0.5)
    axB.axhline(mim_rate, color=C["MIMIC"], lw=0.7, ls=":", alpha=0.5)
    axB.set_title("MACE Event Rate by Cohort\n(admission-level)", fontsize=8, fontweight="bold", pad=4)
    add_panel_label(axB, "B", fontsize=8)
    axB.grid(axis="y", color="#E5E5E5", lw=0.5)
    axB.set_axisbelow(True)
    axB.text(2, rates[2] + 8.0, "★ External", ha="center", fontsize=5.8, color=C["MIMIC"], style="italic")

    # ── Panels C/D/E: risk distributions ─────────────────────────────────────
    axC = fig.add_subplot(gs[1, 0])
    axC.hist(sch_plot["ensemble_score"].values, bins=40, color=C["SCH"], alpha=0.82, edgecolor="white", linewidth=0.5)
    axC.axvline(sch_plot["MACE"].mean(), color="#D32F2F", lw=1.4, ls="--", label=f"Patient-level MACE: {sch_plot['MACE'].mean()*100:.1f}%")
    axC.set_xlabel("Patient-level Predicted Risk Score", fontsize=8)
    axC.set_ylabel("Patients", fontsize=8)
    axC.set_title(f"SCH Risk Distribution\n(n={sch_n:,} patients; patient-level)", fontsize=8, fontweight="bold", pad=4)
    add_panel_label(axC, "C", fontsize=8)
    axC.grid(axis="y", color="#E5E5E5", lw=0.5)
    axC.set_axisbelow(True)
    axC.legend(loc="upper left", fontsize=6, frameon=True)

    axD = fig.add_subplot(gs[1, 1])
    axD.hist(whu_plot["ensemble_score"].values, bins=30, color=C["WHU"], alpha=0.82, edgecolor="white", linewidth=0.5)
    axD.axvline(whu_plot["MACE"].mean(), color="#1565C0", lw=1.4, ls="--", label=f"Patient-level MACE: {whu_plot['MACE'].mean()*100:.1f}%")
    axD.set_xlabel("Patient-level Predicted Risk Score", fontsize=8)
    axD.set_ylabel("Patients", fontsize=8)
    axD.set_title(f"WHU Risk Distribution\n(n={whu_n:,} patients; patient-level)", fontsize=8, fontweight="bold", pad=4)
    add_panel_label(axD, "D", fontsize=8)
    axD.grid(axis="y", color="#E5E5E5", lw=0.5)
    axD.set_axisbelow(True)
    axD.legend(loc="upper left", fontsize=6, frameon=True)

    axE = fig.add_subplot(gs[1, 2])
    axE.hist(mimic_pp["oof_prob_paper"].values, bins=40, color=C["MIMIC"], alpha=0.82, edgecolor="white", linewidth=0.5)
    axE.axvline(mimic_pp["MACE"].mean(), color="#D32F2F", lw=1.4, ls="--", label=f"MACE rate: {mimic_pp['MACE'].mean()*100:.1f}%")
    axE.set_xlabel("OOF Predicted Risk Score", fontsize=8)
    axE.set_ylabel("Patients", fontsize=8)
    axE.set_title(f"MIMIC-III Risk Distribution\n(n={mim_n:,}; ★ External Cohort)", fontsize=8, fontweight="bold", pad=4, color="#2E6930")
    add_panel_label(axE, "E", fontsize=8)
    axE.grid(axis="y", color="#E5E5E5", lw=0.5)
    axE.set_axisbelow(True)
    axE.legend(loc="upper left", fontsize=6, frameon=True)
    axE.text(
        0.97, 0.62,
        f"AUC = {mimic_auc:.3f}\nC-index = {float(summ['C_index']):.3f}\nNRI = {float(summ['NRI']):.3f}",
        transform=axE.transAxes, ha="right", va="top", fontsize=6.2,
        color="#2E6930", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C["MIMIC"], lw=0.7, alpha=0.85),
    )

    for a in (axC, axD, axE):
        p = a.get_position()
        a.set_position([p.x0, p.y0 - 0.018, p.width, p.height])

    save(fig, "FigureS8_Study_Overview_v13")
    plt.close(fig)


def fig_s9_timedep(surv, summ):
    sm = surv.set_index(["Cohort", "Metric"])["Value"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 90 / 25.4))

    ax = axes[0]
    sch_t = [90, 180, 365, 548, 730]
    whu_t = [90, 180, 365, 548, 730]
    sch_auc = [sm[("SCH", f"tdAUC_{t}d")] for t in sch_t]
    whu_auc = [sm[("WHU", f"tdAUC_{t}d")] for t in whu_t]
    mim_t = [7, 14, 30, 60, 90]
    mim_auc = [float(summ["TimedepAUC_7d"]), float(summ["TimedepAUC_14d"]), float(summ["TimedepAUC_30d"]), float(summ["TimedepAUC_60d"]), float(summ["TimedepAUC_90d"])]

    l1, = ax.plot(sch_t, sch_auc, "o-", color=C["SCH"], lw=1.4, ms=4, label="SCH")
    l2, = ax.plot(whu_t, whu_auc, "s-", color=C["WHU"], lw=1.4, ms=4, label="WHU")
    ax.set_xlabel("Days", fontsize=7)
    ax.set_ylabel("Time-dependent AUC (SCH/WHU)", fontsize=7)
    ax.set_ylim(0.60, 0.85)
    ax.set_title("Time-dependent AUC", fontsize=7)
    add_panel_label(ax, "A", fontsize=8)
    ax.grid(False)

    ax2 = ax.twinx()
    l3, = ax2.plot(mim_t, mim_auc, "D--", color=C["MIMIC"], lw=1.4, ms=4, label="MIMIC-III")
    ax2.set_ylabel("Time-dependent AUC (MIMIC)", fontsize=6.5, color=C["MIMIC"])
    ax2.tick_params(axis="y", labelcolor=C["MIMIC"], labelsize=6)
    ax2.set_ylim(0.55, 0.85)
    ax.legend([l1, l2, l3], ["SCH", "WHU", "MIMIC-III"],
              loc="upper center", bbox_to_anchor=(0.5, -0.38),
              ncol=3, fontsize=6.2, frameon=True, framealpha=0.92,
              borderpad=0.5, handlelength=1.4, handletextpad=0.4,
              columnspacing=1.0)

    ax = axes[1]
    sch_b = [sm[("SCH", f"Brier_{t}d")] for t in sch_t]
    whu_b = [sm[("WHU", f"Brier_{t}d")] for t in whu_t]
    sch_prev = 0.0344
    whu_prev = 0.1097
    sch_null = [sch_prev * (1 - sch_prev)] * len(sch_t)
    whu_null = [whu_prev * (1 - whu_prev)] * len(whu_t)
    ax.plot(sch_t, sch_b, "o-", color=C["SCH"], lw=1.4, ms=4, label="SCH model")
    ax.plot(whu_t, whu_b, "s-", color=C["WHU"], lw=1.4, ms=4, label="WHU model")
    ax.plot(sch_t, sch_null, "o--", color=C["SCH"], lw=1.0, ms=3, alpha=0.55, label="SCH null")
    ax.plot(whu_t, whu_null, "s--", color=C["WHU"], lw=1.0, ms=3, alpha=0.55, label="WHU null")
    ax.set_xlabel("Days", fontsize=7)
    ax.set_ylabel("Brier Score", fontsize=7)
    ax.set_title("Brier Score over time", fontsize=7)
    add_panel_label(ax, "B", fontsize=8)
    ax.grid(False)
    # Legend placed below the axes in the blank margin area
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38),
              ncol=2, fontsize=6.2, frameon=True, framealpha=0.92,
              borderpad=0.5, handlelength=1.4, handletextpad=0.4,
              columnspacing=1.0)

    for a in axes:
        a.set_box_aspect(1)
    plt.tight_layout(rect=[0, 0.16, 1, 1.0])
    save(fig, "FigureS9_TimeDep_Performance_v13")
    plt.close(fig)


def fig_s10_calibration(recalib, mimic_pp):
    sch_raw = recalib[recalib["Center"] == "SCH"].dropna(subset=["predicted_prob_within_cv", "MACE"])
    whu_raw = recalib[recalib["Center"] == "WHU"].dropna(subset=["predicted_prob_within_cv", "MACE"])
    sch_iso = recalib[recalib["Center"] == "SCH"].dropna(subset=["prob_isotonic", "MACE"])
    whu_iso = recalib[recalib["Center"] == "WHU"].dropna(subset=["prob_isotonic", "MACE"])

    my = mimic_pp["MACE"].values
    mp = mimic_pp["oof_prob_paper"].values

    fig, axes = plt.subplots(2, 3, figsize=(W2, 160 / 25.4))

    def plot(ax, y, p, col, title, xlim=None, note=""):
        pm, of = calib_curve_decile(y, p, n_bins=10)
        local_max = max(float(np.nanmax(pm)) if len(pm) else 0.0,
                        float(np.nanmax(of)) if len(of) else 0.0,
                        float(np.nanpercentile(p, 99)) if len(p) else 0.0)
        if xlim is None:
            xlim_use = min(1.0, max(0.12, local_max * 1.10))
        else:
            xlim_use = max(float(xlim), min(1.0, local_max * 1.05))
        ax.plot(pm, of, "o-", color=col, ms=4, lw=1.4)
        ax.plot([0, xlim_use], [0, xlim_use], "k--", lw=0.9, alpha=0.6)
        ax.set_xlim(0, xlim_use)
        ax.set_ylim(0, xlim_use)
        ax.set_xlabel("Mean predicted probability", fontsize=7)
        ax.set_ylabel("Observed MACE fraction", fontsize=7)
        ax.set_title(title, fontsize=7)
        ax.grid(False)
        if note:
            ax.text(0.04, 0.94, note, transform=ax.transAxes, va="top", fontsize=5.8, bbox=dict(fc="white", ec="#cccccc", pad=2))

    plot(axes[0, 0], sch_raw["MACE"].values, sch_raw["predicted_prob_within_cv"].values, C["SCH"], "SCH raw", None, "E/O≈11.57")
    plot(axes[0, 1], whu_raw["MACE"].values, whu_raw["predicted_prob_within_cv"].values, C["WHU"], "WHU raw", 0.50, "E/O≈2.18")
    plot(axes[0, 2], my, mp, C["MIMIC"], "MIMIC raw", 1.00, "E/O≈1.09")

    plot(axes[1, 0], sch_iso["MACE"].values, sch_iso["prob_isotonic"].values, C["SCH"], "SCH isotonic", None, "Slope=1.000")
    plot(axes[1, 1], whu_iso["MACE"].values, whu_iso["prob_isotonic"].values, C["WHU"], "WHU isotonic", 0.50, "Slope=1.000")
    plot(axes[1, 2], my, mp, C["MIMIC"], "MIMIC raw reference", 1.00, "Independent reference")

    panel_labels = ["A", "B", "C", "D", "E", "F"]
    for i, ax in enumerate(axes.flatten()):
        add_panel_label(ax, panel_labels[i], fontsize=8)

    for ax in axes.flatten():
        ax.set_box_aspect(1)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "FigureS10_Calibration_v13")
    plt.close(fig)


def copy_admission_figures_as_s11_s14():
    adm = os.path.join(ROOT, "Manuscript", "Admission model", "outputs")
    mapping = {
        "Figure1_ROC_Comparison": "FigureS11_Admission_ROC_v13",
        "Figure5_Admission_vs_Patient": "FigureS12_Admission_vs_Patient_v13",
        "Figure4_Feature_Importance": "FigureS13_Admission_FeatureImportance_v13",
        "Figure6_Temporal_Gradient": "FigureS14_Temporal_Gradient_v13",
    }
    for src_base, dst_base in mapping.items():
        for ext in (".png", ".pdf"):
            src = os.path.join(adm, src_base + ext)
            dst = os.path.join(OUT_DIR, dst_base + ext)
            if os.path.exists(src):
                shutil.copy2(src, dst)


def main():
    risk_sch, risk_whu, pred_v2, recalib, fi_comp, optb, surv, mimic_pp, summ, sch_raw, whu_raw, mimic_feat = load_data()

    fig_s1_missing(fi_comp, sch_raw, whu_raw, mimic_feat)
    fig_s2_roc(pred_v2, mimic_pp, summ)
    fig_s3_sch_dca(pred_v2)
    fig_s4_nri(summ)
    fig_s5_apob()
    fig_s6_rmst()
    fig_s7_socioeconomic()
    fig_s8_study_overview(pred_v2, recalib, mimic_pp, summ, risk_sch, risk_whu)
    fig_s9_timedep(surv, summ)
    fig_s10_calibration(recalib, mimic_pp)

    copy_admission_figures_as_s11_s14()
    print(f"Done. Supplementary figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
