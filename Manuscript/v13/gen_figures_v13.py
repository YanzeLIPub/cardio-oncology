import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

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
    "Low": LANCET["Low"],
    "Med": LANCET["Medium"],
    "High": LANCET["High"],
    "Age": "#888888",
    "All": "black",
    "None": "#AAAAAA",
    "tri": "#08306B",
    "bi": "#4292C6",
    "single": "#C6DBEF",
    "socio": "#E69F00",
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
    out = []
    for t in thresh:
        yp = (y_prob >= t).astype(int)
        tp = np.sum((y_true == 1) & (yp == 1))
        fp = np.sum((y_true == 0) & (yp == 1))
        out.append((tp / n) - (fp / n) * (t / (1 - t + 1e-9)))
    return np.array(out)


def load_data():
    data_dir = os.path.join(ROOT, "data")
    mimic_res = os.path.join(ROOT, "MIMIC-3", "results")
    v13_dir = os.path.join(ROOT, "Manuscript", "v13")

    risk_sch = pd.read_csv(os.path.join(data_dir, "patient_risk_SCH.csv"))
    risk_whu = pd.read_csv(os.path.join(data_dir, "patient_risk_WHU.csv"))
    pred_v2 = pd.read_csv(os.path.join(data_dir, "real_model_predictions_v2.csv"))
    optb = pd.read_csv(os.path.join(data_dir, "OptionB_Summary.csv"), index_col="Metric")["Value"]
    fi_comp = pd.read_csv(os.path.join(data_dir, "feature_importance_comparison.csv"))
    mimic_fi = pd.read_csv(os.path.join(mimic_res, "paper_feature_importance.csv"))

    summary_v13 = pd.read_csv(os.path.join(v13_dir, "summary_v13_metrics.csv")).iloc[0]
    mimic_pp = pd.read_csv(os.path.join(v13_dir, "paper_predictions_9946.csv"))

    risk_sch["risk_group"] = risk_sch["risk_group"].replace({"Intermediate": "Medium"})
    risk_whu["risk_group"] = risk_whu["risk_group"].replace({"Intermediate": "Medium"})
    mimic_pp["risk_group"] = mimic_pp["risk_group"].replace({"Intermediate": "Medium"})

    fi_comp["English"] = fi_comp["Feature"].map(lambda x: FEAT_EN.get(x, str(x)))

    return risk_sch, risk_whu, pred_v2, optb, fi_comp, mimic_fi, mimic_pp, summary_v13


def figure1_km(risk_sch, risk_whu, mimic_pp):
    fig, axes = plt.subplots(1, 3, figsize=(215 / 25.4, 150 / 25.4), sharey=False)

    panels = [
        (risk_sch, f"SCH cohort\n(n={len(risk_sch):,}; log-rank p<0·0001)", axes[0]),
        (risk_whu, f"WHU cohort\n(n={len(risk_whu):,}; log-rank p<0·0001)", axes[1]),
        (mimic_pp, f"MIMIC-III cohort\n(n={len(mimic_pp):,}; log-rank p<0·0001)", axes[2]),
    ]

    for i, (df, title, ax) in enumerate(panels):
        low = df[df["risk_group"] == "Low"]
        med = df[df["risk_group"] == "Medium"]
        high = df[df["risk_group"] == "High"]

        k_low = KaplanMeierFitter().fit(low["FollowupDays"], low["MACE"], label=f"Low (n={len(low):,})")
        k_med = KaplanMeierFitter().fit(med["FollowupDays"], med["MACE"], label=f"Intermediate (n={len(med):,})")
        k_high = KaplanMeierFitter().fit(high["FollowupDays"], high["MACE"], label=f"High (n={len(high):,})")

        def trunc(kf):
            at_risk = kf.event_table["at_risk"]
            valid = at_risk[at_risk >= 15]
            return valid.index.max() if len(valid) else 0

        t_cut = min(trunc(k_low), trunc(k_med), trunc(k_high))

        k_low.plot_survival_function(ax=ax, color=C["Low"], ci_show=True, ci_alpha=0.12)
        k_med.plot_survival_function(ax=ax, color=C["Med"], ci_show=True, ci_alpha=0.12, linestyle="--")
        k_high.plot_survival_function(ax=ax, color=C["High"], ci_show=True, ci_alpha=0.12)

        add_at_risk_counts(k_low, k_med, k_high, ax=ax, fontsize=5.5, rows_to_show=["At risk"])

        ax.set_xlim(0, t_cut)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("Follow-up (days)", fontsize=7)
        ax.set_ylabel("MACE-free survival probability", fontsize=7)
        ax.grid(False)
        ax.legend(loc="lower left", fontsize=6, frameon=True)
        add_panel_label(ax, chr(ord("A") + i), fontsize=8)
        ax.set_box_aspect(1)

    fig.tight_layout(rect=[0.03, 0.05, 0.995, 0.98], w_pad=1.10)
    save(fig, "Figure1_KM_Survival_v13")
    plt.close(fig)


def figure2_transfer(optb, summary_v13):
    within = {
        "SCH": float(optb["SCH_CV_AUC"]),
        "WHU": float(optb["WHU_CV_AUC"]),
        "MIMIC-III": float(summary_v13["AUC"]),
    }
    off_diag = {
        ("SCH", "WHU"): float(optb["SCH_to_WHU_transfer"]),
        ("WHU", "SCH"): float(optb["WHU_to_SCH_transfer"]),
        ("SCH", "MIMIC-III"): float(summary_v13["Transfer_SCH2MIMIC"]),
        ("WHU", "MIMIC-III"): float(summary_v13["Transfer_WHU2MIMIC"]),
        ("MIMIC-III", "SCH"): float(summary_v13["Transfer_MIMIC2SCH"]),
        ("MIMIC-III", "WHU"): float(summary_v13["Transfer_MIMIC2WHU"]),
    }

    order = ["SCH", "WHU", "MIMIC-III"]
    mat = pd.DataFrame(index=order, columns=order, dtype=float)
    for c in order:
        mat.loc[c, c] = within[c]
    for (tr, te), v in off_diag.items():
        mat.loc[tr, te] = v

    fig = plt.figure(figsize=(182 / 25.4, 122 / 25.4))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.22, 0.90, 0.06], left=0.07, right=0.97, bottom=0.20, top=0.90, wspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    bar_w = 0.20
    gap = 0.06
    sep = 0.04
    gx_ctr = np.array([0.0, 1.25, 2.50])
    src_col = {"SCH": C["SCH"], "WHU": C["WHU"], "MIMIC-III": C["MIMIC"]}

    for i, tc in enumerate(order):
        gx = gx_ctr[i]
        v_w = within[tc]
        ax1.bar(gx, v_w, bar_w, color=src_col[tc], zorder=3)
        ax1.text(gx, v_w + 0.010, f"{v_w:.3f}", ha="center", va="bottom", fontsize=5.8, fontweight="bold")

        sources = [c for c in order if c != tc]
        for j, src in enumerate(sources):
            xpos = gx + bar_w + gap + j * (bar_w + sep)
            v = mat.loc[src, tc]
            below = v < 0.5
            ax1.bar(xpos, v, bar_w, color=src_col[src], alpha=0.35, edgecolor=src_col[src], linewidth=0.9, hatch="////" if below else "", zorder=2)
            ax1.text(xpos, v + 0.010, f"{v:.3f}", ha="center", va="bottom", fontsize=5.0, color="#CC0000" if below else "#555555")

    ax1.axhline(0.5, color="#CC0000", linestyle="--", lw=1.0)
    grp_span = bar_w + gap + 2 * bar_w + sep
    ax1.set_xticks(gx_ctr + grp_span / 2 - bar_w / 2)
    ax1.set_xticklabels([f"Test: {c}" for c in order], fontsize=6.5)
    ax1.set_xlim(gx_ctr[0] - 0.18, gx_ctr[-1] + grp_span + 0.18)
    ax1.set_ylim(0.30, 0.92)
    ax1.set_ylabel("AUC", fontsize=7)
    ax1.set_title("Within-cohort CV vs cross-cohort transfer", fontsize=7, fontweight="bold")
    ax1.grid(False, axis="y")
    ax1.tick_params(axis="x", length=0)
    add_panel_label(ax1, "A", fontsize=8)
    ax1.set_box_aspect(1)

    annot = mat.applymap(lambda v: f"{v:.3f}")
    sns.heatmap(mat, annot=annot, fmt="", cmap="RdYlGn_r", vmin=0.40, vmax=0.90, linewidths=0.8, linecolor="white", cbar=False, ax=ax2, annot_kws={"fontsize": 7.5, "fontweight": "bold"})

    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(vmin=0.40, vmax=0.90))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label("AUC", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)

    for i in range(len(order)):
        ax2.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="black", linewidth=2.0, zorder=10))

    ax2.set_title("Transfer AUC matrix\n(row=train, col=test)", fontsize=7, fontweight="bold")
    ax2.set_xlabel("Test cohort", fontsize=7)
    ax2.set_ylabel("Train cohort", fontsize=7)
    ax2.tick_params(labelsize=7, length=0)
    add_panel_label(ax2, "B", fontsize=8)
    ax2.set_box_aspect(1)

    save(fig, "Figure2_Transfer_Analysis_v13")
    plt.close(fig)


def figure3_dca(pred_v2, mimic_pp):
    sch = pred_v2[pred_v2["Center"] == "SCH"].dropna(subset=["predicted_prob_within_cv", "MACE"])
    whu = pred_v2[pred_v2["Center"] == "WHU"].dropna(subset=["predicted_prob_within_cv", "MACE"])

    sch_age = 1 / (1 + np.exp(-0.045 * (sch["年龄"].values - 55)))
    whu_age = 1 / (1 + np.exp(-0.055 * (whu["年龄"].values - 58)))
    mim_age = mimic_pp["oof_prob_age_only"].values

    # 240 mm wide × 130 mm tall — wide enough to prevent panel overlap
    fig, axes = plt.subplots(1, 3, figsize=(240 / 25.4, 130 / 25.4))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.90, wspace=0.42)

    ax = axes[0]
    t = np.linspace(0.05, 0.40, 120)
    y = whu["MACE"].values
    nb_model = dca_nb(y, whu["predicted_prob_within_cv"].values, t)
    nb_age = dca_nb(y, whu_age, t)
    nb_all = np.mean(y) - (1 - np.mean(y)) * t / (1 - t + 1e-9)
    ax.plot(t, nb_model, color=C["WHU"], lw=1.8, label="Ensemble model")
    ax.plot(t, nb_age, color=C["Age"], lw=1.4, linestyle="--", label="Age-only")
    ax.plot(t, nb_all, color=C["All"], lw=1.0, linestyle="--", label="Treat all")
    ax.axhline(0, color=C["None"], lw=1.0, linestyle=":", label="Treat none")
    ax.set_xlim(0.05, 0.40)
    ax.set_ylim(-0.05, 0.18)
    ax.set_title("WHU (5–40%)", fontsize=8)
    ax.set_xlabel("Decision threshold", fontsize=8)
    ax.set_ylabel("Net benefit", fontsize=8)
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=6.5, frameon=True)
    add_panel_label(ax, "A", fontsize=8)

    ax = axes[1]
    t2 = np.linspace(0.001, 0.15, 150)
    y2 = sch["MACE"].values
    nb_m = dca_nb(y2, sch["predicted_prob_within_cv"].values, t2)
    nb_a = dca_nb(y2, sch_age, t2)
    inc = nb_m - nb_a
    ax.plot(t2, inc, color=C["SCH"], lw=1.8, label="Incremental NB")
    ax.axhline(0, color="black", linestyle=":", lw=0.8, label="Zero line")
    ax.fill_between(t2, inc, 0, where=(inc > 0), color=C["SCH"], alpha=0.18)
    ax.set_xlim(0, 0.15)
    ax.set_ylim(-0.010, 0.020)
    ax.set_title("SCH (0–15%)", fontsize=8)
    ax.set_xlabel("Decision threshold", fontsize=8)
    ax.set_ylabel("Incremental net benefit", fontsize=8)
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=6.5, frameon=True)
    add_panel_label(ax, "B", fontsize=8)

    ax = axes[2]
    t3 = np.linspace(0.10, 0.90, 120)
    y3 = mimic_pp["MACE"].values
    nb_m3 = dca_nb(y3, mimic_pp["oof_prob_paper"].values, t3)
    nb_a3 = dca_nb(y3, mim_age, t3)
    nb_all3 = np.mean(y3) - (1 - np.mean(y3)) * t3 / (1 - t3 + 1e-9)
    ax.plot(t3, nb_m3, color=C["MIMIC"], lw=1.8, label="Ensemble model")
    ax.plot(t3, nb_a3, color=C["Age"], lw=1.4, linestyle="--", label="Age-only")
    ax.plot(t3, nb_all3, color=C["All"], lw=1.0, linestyle="--", label="Treat all")
    ax.axhline(0, color=C["None"], lw=1.0, linestyle=":", label="Treat none")
    ax.set_xlim(0.10, 0.90)
    ax.set_ylim(-0.10, 0.45)
    ax.set_title("MIMIC-III (10–90%)", fontsize=8)
    ax.set_xlabel("Decision threshold", fontsize=8)
    ax.set_ylabel("Net benefit", fontsize=8)
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=6.5, frameon=True)
    add_panel_label(ax, "C", fontsize=8)

    save(fig, "Figure3_DCA_v13")
    plt.close(fig)


def figure4_features(fi_comp, mimic_fi):
    tri = {"Age", "LDL Cholesterol", "HDL Cholesterol", "Direct Bilirubin"}
    bi = set(fi_comp[fi_comp["In_common"] == True]["English"].tolist()) - tri
    socio = {
        "No. of Physicians",
        "Hospital Beds",
        "GDP per Capita",
        "No. of Hospitals",
        "Avg. Employee Wage",
    }
    demo = {"Age", "Sex", "Sex_M"}

    def color_sch_whu(name):
        if name in tri:
            return C["tri"]
        if name in bi:
            return C["bi"]
        if name in socio:
            return C["socio"]
        return C["single"]

    def mimic_base(feat):
        feat = str(feat)
        return feat.rsplit("__", 1)[0].strip() if "__" in feat else feat

    mimic2 = mimic_fi.copy()
    mimic2["base"] = mimic2["Feature"].map(mimic_base)
    mimic_agg = mimic2.groupby("base", as_index=False)["Avg_imp"].sum().sort_values("Avg_imp", ascending=False)
    mimic_agg["norm"] = mimic_agg["Avg_imp"] / mimic_agg["Avg_imp"].max()

    top_n = 25
    fig = plt.figure(figsize=(W2, 285 / 25.4))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.8, 1.0], hspace=0.50, wspace=0.52, left=0.07, right=0.97, top=0.955, bottom=0.14)

    ax_a = fig.add_subplot(gs[0, 0])
    sch = fi_comp.nlargest(top_n, "SCH_norm").copy()
    ax_a.barh(sch["English"], sch["SCH_norm"], color=[color_sch_whu(x) for x in sch["English"]], height=0.65)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Normalised importance", fontsize=7)
    ax_a.set_title("SCH top-25", fontsize=7.5, fontweight="bold", color=C["SCH"])
    add_panel_label(ax_a, "A", fontsize=8)
    ax_a.tick_params(axis="y", labelsize=6)
    ax_a.grid(False, axis="x")

    ax_b = fig.add_subplot(gs[0, 1])
    whu = fi_comp.nlargest(top_n, "WHU_norm").copy()
    ax_b.barh(whu["English"], whu["WHU_norm"], color=[color_sch_whu(x) for x in whu["English"]], height=0.65)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Normalised importance", fontsize=7)
    ax_b.set_title("WHU top-25", fontsize=7.5, fontweight="bold", color=C["WHU"])
    add_panel_label(ax_b, "B", fontsize=8)
    ax_b.tick_params(axis="y", labelsize=6)
    ax_b.grid(False, axis="x")

    ax_c = fig.add_subplot(gs[0, 2])
    mm = mimic_agg.head(top_n).copy()
    mm_cols = [C["tri"] if x in tri else (C["bi"] if x in bi else C["single"]) for x in mm["base"]]
    ax_c.barh(mm["base"], mm["norm"], color=mm_cols, height=0.65)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Normalised importance\n(sum over last/mean/max/min)", fontsize=7)
    ax_c.set_title("MIMIC-III top-25", fontsize=7.5, fontweight="bold", color=C["MIMIC"])
    add_panel_label(ax_c, "C", fontsize=8)
    ax_c.tick_params(axis="y", labelsize=6)
    ax_c.grid(False, axis="x")

    ax_d = fig.add_subplot(gs[1, :])
    mimic_map = mimic_agg.set_index("base")["norm"].to_dict()
    p = fi_comp.copy()
    p["MIMIC_norm"] = p["English"].map(mimic_map).fillna(0)
    p["con"] = p["English"].map(lambda n: "Tri" if n in tri else ("Bi" if n in bi else "Single"))
    pal = {"Tri": C["tri"], "Bi": C["bi"], "Single": C["single"]}

    for grp, sub in p.groupby("con"):
        sz = 30 + sub["MIMIC_norm"] * 550
        mask_demo = sub["English"].isin(demo)
        if (~mask_demo).sum() > 0:
            ax_d.scatter(sub.loc[~mask_demo, "SCH_norm"], sub.loc[~mask_demo, "WHU_norm"],
                         s=sz[~mask_demo], c=pal[grp], alpha=0.78,
                         edgecolors="white", linewidths=0.4, marker="o", zorder=3)
        if mask_demo.sum() > 0:
            ax_d.scatter(sub.loc[mask_demo, "SCH_norm"], sub.loc[mask_demo, "WHU_norm"],
                         s=sz[mask_demo], c=pal[grp], alpha=0.88,
                         edgecolors="black", linewidths=0.9, marker="^", zorder=4)

    # Dashed diagonal reference line (y = x)
    lim_max = 1.05
    ax_d.plot([0, lim_max], [0, lim_max], color="#AAAAAA", lw=1.0, ls="--", zorder=1)
    ax_d.text(lim_max * 0.97, lim_max * 0.93, "y = x", fontsize=6,
              color="#AAAAAA", ha="right", va="top", style="italic")

    for _, row in p[p["English"].isin(tri)].iterrows():
        ax_d.annotate(row["English"], xy=(row["SCH_norm"], row["WHU_norm"]),
                      xytext=(9, 5), textcoords="offset points", fontsize=6.2,
                      color="#08306B",
                      arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.6))

    ax_d.set_xlabel("Normalised importance in SCH", fontsize=7)
    ax_d.set_ylabel("Normalised importance in WHU", fontsize=7)
    ax_d.set_title("Concordance scatter (circle = biomarker, triangle = demographic)", fontsize=7.5)
    add_panel_label(ax_d, "D", fontsize=8)
    ax_d.grid(False)

    # Build legend manually to avoid duplicate entries and overlap
    legend_handles = [
        mpatches.Patch(facecolor=C["tri"],    edgecolor="white", label="Tri-cohort concordant"),
        mpatches.Patch(facecolor=C["bi"],     edgecolor="white", label="Bi-cohort concordant"),
        mpatches.Patch(facecolor=C["single"], edgecolor="white", label="Single-cohort only"),
        plt.scatter([], [], marker="^", c="#888888", edgecolors="black",
                    linewidths=0.9, s=40, alpha=0.88, label="Demographic variable"),
    ]
    ax_d.legend(handles=legend_handles, loc="lower right",
                fontsize=6.2, frameon=True, framealpha=0.92,
                borderpad=0.6, handlelength=1.2, handletextpad=0.5)

    save(fig, "Figure4_Feature_Importance_v13")
    plt.close(fig)


def main():
    risk_sch, risk_whu, pred_v2, optb, fi_comp, mimic_fi, mimic_pp, summary_v13 = load_data()

    print("Generating Figure 1...")
    figure1_km(risk_sch, risk_whu, mimic_pp)

    print("Generating Figure 2...")
    figure2_transfer(optb, summary_v13)

    print("Generating Figure 3...")
    figure3_dca(pred_v2, mimic_pp)

    print("Generating Figure 4...")
    figure4_features(fi_comp, mimic_fi)

    print(f"Done. Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
