"""
Deep-dive: NSE–Albumin collinearity and its clinical meaning in SCH vs WHU

Five analyses:
  D1. NSE–Albumin scatter coloured by MACE (both cohorts, side-by-side)
  D2. NSE-tertile × Albumin-tertile MACE-rate grid (3×3 heatmap per cohort)
  D3. Partial correlation: NSE→MACE with and without Albumin partialled out
  D4. Logistic regression: main vs interaction (NSE × Albumin)
  D5. "High-NSE / Low-Albumin" clinical subgroup characterisation

Outputs:
  FigureS23_NSE_Albumin_Collinearity.pdf/png
  TableS8_NSE_Albumin_Stats.csv
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import warnings
warnings.filterwarnings('ignore')
import datetime, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
BASE = Path(_release_path('pipeline', ''))
ADD_DIR = BASE / 'Manuscript\\v13\\figures\\add'
SUB_DIR = BASE / 'Manuscript\\Submit'
SUB_DIR.mkdir(parents=True, exist_ok=True)
SCH_CSV = BASE / 'SCH_processed_data.csv'
WHU_CSV = BASE / 'WHU_Cancer_processed_data.csv'
SCH_RISK = BASE / 'data' / 'patient_risk_SCH.csv'
WHU_RISK = BASE / 'data' / 'patient_risk_WHU.csv'
ID_COL = '病案号'
LOG = open(ADD_DIR / 'nse_alb_progress.txt', 'w', encoding='utf-8')

def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()
C_SCH = '#2166AC'
C_WHU = '#D6604D'
C_MACE = '#B2182B'
C_NOMA = '#4D9221'
C_HI = '#8B0000'
C_LO = '#1565C0'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 150})

def despine(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

def save_fig(fig, name):
    for ext, dpi in [('.pdf', None), ('.png', 600)]:
        p = SUB_DIR / f'{name}{ext}'
        kw = {'dpi': dpi} if dpi else {}
        fig.savefig(p, bbox_inches='tight', facecolor='white', **kw)
        log(f'  Saved → {p.name}')
    plt.close(fig)

def get_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lstrip('*★ ').strip() in candidates:
            return c
    return None
ADMIN_KW = ['时间', '号', '诊断', '科室', 'Unnamed', '上次', '入院', '出院', '年龄', '性别', '出生', '日期', '职工', '人均', '医生', '医院', '卫生院', '时间差', 'Center', 'MACE', 'FollowUp', 'ensemble', 'risk_group']

def is_lab(n):
    return not any((k in n for k in ADMIN_KW))
log('Loading WHU …')
whu_risk = pd.read_csv(WHU_RISK)
whu_raw = pd.read_csv(WHU_CSV, low_memory=False)
whu_raw.columns = [c.strip() for c in whu_raw.columns]
whu_raw = whu_raw.loc[:, ~whu_raw.columns.duplicated(keep='first')]
dept_df = whu_raw[[ID_COL, '入院科室']].copy() if '入院科室' in whu_raw.columns else None
whu_lab = [c for c in whu_raw.columns if is_lab(c)]
for c in whu_lab:
    whu_raw[c] = pd.to_numeric(whu_raw[c], errors='coerce')
whu_agg = whu_raw.groupby(ID_COL)[whu_lab].first().reset_index()
whu = whu_risk[[ID_COL, 'MACE']].merge(whu_agg, on=ID_COL, how='left')
if dept_df is not None:
    whu = whu.merge(dept_df.groupby(ID_COL)['入院科室'].first().reset_index(), on=ID_COL, how='left')
log(f"  WHU {whu.shape}  MACE={whu['MACE'].mean():.3f}")
log('Loading SCH …')
sch_risk = pd.read_csv(SCH_RISK)
for _chunk in pd.read_csv(SCH_CSV, chunksize=1, low_memory=False, nrows=1):
    _chunk.columns = [c.strip() for c in _chunk.columns]
    sch_cols_raw = _chunk.columns.tolist()
    break
sch_col_map = {c.lstrip('*★ ').strip(): c for c in sch_cols_raw if c.lstrip('*★ ').strip() not in {}}
target_s = [s for s in sch_col_map if is_lab(s) and s in set(whu_lab)]
o2s = {sch_col_map[s]: s for s in target_s}
chunks = []
for i, chunk in enumerate(pd.read_csv(SCH_CSV, chunksize=50000, low_memory=False)):
    chunk.columns = [c.strip() for c in chunk.columns]
    keep = [ID_COL] + [c for c in o2s if c in chunk.columns]
    sub = chunk[keep].rename(columns=o2s).copy()
    for c in sub.columns:
        if c != ID_COL:
            sub[c] = pd.to_numeric(sub[c], errors='coerce')
    chunks.append(sub)
    if (i + 1) % 3 == 0:
        log(f'  SCH chunk {i + 1} …')
sch_raw = pd.concat(chunks, ignore_index=True)
sch_agg = sch_raw.groupby(ID_COL)[[c for c in sch_raw.columns if c != ID_COL]].first().reset_index()
sch = sch_risk[[ID_COL, 'MACE']].merge(sch_agg, on=ID_COL, how='left')
log(f"  SCH {sch.shape}  MACE={sch['MACE'].mean():.3f}")
NSE_ZH = '神经元特异性烯醇化酶'
NSE_EN = 'NSE'
ALB_ZH = '白蛋白'
ALB_EN = 'Albumin'
WBC_ZH = '白细胞'
WBC_EN = 'WBC'
PREA_ZH = '前白蛋白'
PREA_EN = 'Prealbumin'

def pull(df, zh, en):
    col = get_col(df, zh, en)
    if col is None:
        return pd.Series(np.nan, index=df.index, name=en)
    return pd.to_numeric(df[col], errors='coerce').rename(en)

def build_anal_df(df):
    nse = pull(df, NSE_ZH, NSE_EN)
    alb = pull(df, ALB_ZH, ALB_EN)
    wbc = pull(df, WBC_ZH, WBC_EN)
    prea = pull(df, PREA_ZH, PREA_EN)
    mace = pd.to_numeric(df['MACE'], errors='coerce')
    dept = df['入院科室'] if '入院科室' in df.columns else pd.Series('', index=df.index)
    return pd.DataFrame({'NSE': nse, 'Albumin': alb, 'WBC': wbc, 'Prealbumin': prea, 'MACE': mace, 'Dept': dept})
sch_d = build_anal_df(sch).dropna(subset=['MACE'])
whu_d = build_anal_df(whu).dropna(subset=['MACE'])
log(f'  Analysis sets — SCH:{len(sch_d)}  WHU:{len(whu_d)}')

def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-09)
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(15.0, 12.5), constrained_layout=False)
fig.suptitle('NSE-Albumin Collinearity: Mechanistic Analysis', fontsize=9, fontweight='bold', y=0.998)
gs = GridSpec(3, 4, figure=fig, hspace=0.72, wspace=0.52)
ax_sc_sch = fig.add_subplot(gs[0, 0])
ax_sc_whu = fig.add_subplot(gs[0, 1])
ax_hm_sch = fig.add_subplot(gs[0, 2])
ax_hm_whu = fig.add_subplot(gs[0, 3])
ax_pcorr = fig.add_subplot(gs[1, :2])
ax_lr = fig.add_subplot(gs[1, 2:])
ax_sub_bar = fig.add_subplot(gs[2, :2])
ax_sub_km = fig.add_subplot(gs[2, 2:])
stat_rows = []
log('D1: Scatter plots …')
for ax, df, cname, cohort_color in [(ax_sc_sch, sch_d, 'SCH', C_SCH), (ax_sc_whu, whu_d, 'WHU', C_WHU)]:
    sub = df[['NSE', 'Albumin', 'MACE']].dropna()
    if len(sub) < 10:
        continue
    nse_z = zscore(sub['NSE'])
    alb_z = zscore(sub['Albumin'])
    rho, p = stats.spearmanr(sub['NSE'], sub['Albumin'])
    stat_rows.append({'Analysis': 'D1_corr', 'Cohort': cname, 'Metric': 'NSE_Albumin_Spearman_rho', 'Value': round(rho, 3), 'p': round(p, 4)})
    for mval, color, lbl in [(0, C_NOMA + '99', 'No MACE'), (1, C_MACE, 'MACE')]:
        m = sub['MACE'] == mval
        ax.scatter(nse_z[m], alb_z[m], c=color, s=6 if mval == 0 else 18, alpha=0.4 if mval == 0 else 0.7, label=lbl, rasterized=True)
    from numpy.polynomial.polynomial import polyfit
    ok = nse_z.notna() & alb_z.notna()
    if ok.sum() > 5:
        b0, b1 = polyfit(nse_z[ok], alb_z[ok], 1)
        xs = np.linspace(nse_z[ok].min(), nse_z[ok].max(), 100)
        ax.plot(xs, b0 + b1 * xs, color=cohort_color, lw=1.5, ls='--', alpha=0.9)
    ax.set_title(f'D1  {cname} — NSE vs Albumin\nSpearman ρ={rho:.3f}  p={p:.3e}', fontsize=8, fontweight='bold')
    ax.set_xlabel('NSE (z-score)')
    ax.set_ylabel('Albumin (z-score)')
    ax.legend(fontsize=7, markerscale=2)
    despine(ax)
log('D2: Tertile MACE-rate grids …')
grid_results = {}
for ax, df, cname in [(ax_hm_sch, sch_d, 'SCH'), (ax_hm_whu, whu_d, 'WHU')]:
    sub = df[['NSE', 'Albumin', 'MACE']].dropna()
    if len(sub) < 30:
        continue
    try:
        nse_t = pd.qcut(sub['NSE'].rank(method='first'), 3, labels=['NSE-Low', 'NSE-Mid', 'NSE-High'])
        alb_t = pd.qcut(sub['Albumin'].rank(method='first'), 3, labels=['Alb-Low', 'Alb-Mid', 'Alb-High'])
    except Exception as e:
        log(f'  {cname} qcut error: {e}')
        continue
    grid = sub.groupby([nse_t, alb_t])['MACE'].mean().unstack() * 100
    row_ord = [l for l in ['NSE-Low', 'NSE-Mid', 'NSE-High'] if l in grid.index]
    col_ord = [l for l in ['Alb-Low', 'Alb-Mid', 'Alb-High'] if l in grid.columns]
    grid = grid.reindex(index=row_ord, columns=col_ord)
    grid_results[cname] = grid
    annot = grid.applymap(lambda x: f'{x:.1f}%' if not pd.isna(x) else '')
    sns.heatmap(grid, ax=ax, annot=annot, fmt='', cmap='YlOrRd', vmin=0, vmax=grid.values[~np.isnan(grid.values)].max() * 1.1, linewidths=0.5, cbar_kws={'shrink': 0.55, 'pad': 0.02})
    ax.set_title(f'D2  {cname}  MACE rate (%)\nNSE tertile x Albumin tertile', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('Albumin tertile', fontsize=7)
    ax.set_ylabel('NSE tertile', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for rt, ct in [('NSE-High', 'Alb-Low'), ('NSE-Low', 'Alb-High')]:
        if rt in grid.index and ct in grid.columns:
            val = grid.loc[rt, ct]
            stat_rows.append({'Analysis': 'D2_grid', 'Cohort': cname, 'Metric': f'MACE_pct_{rt}_{ct}', 'Value': round(val, 2), 'p': np.nan})
log('D3: Partial correlations …')

def partial_corr_spearman(x, y, z):
    """Spearman partial r of x and y controlling for z."""
    from scipy.stats import spearmanr
    df_ = pd.DataFrame({'x': x, 'y': y, 'z': z}).dropna()
    if len(df_) < 20:
        return (np.nan, np.nan)
    rx = stats.rankdata(df_['x'])
    ry = stats.rankdata(df_['y'])
    rz = stats.rankdata(df_['z'])
    from numpy.linalg import lstsq
    Z = np.column_stack([np.ones(len(rz)), rz])
    rx_res = rx - Z @ lstsq(Z, rx, rcond=None)[0]
    ry_res = ry - Z @ lstsq(Z, ry, rcond=None)[0]
    r, p = stats.pearsonr(rx_res, ry_res)
    return (r, p)
pcorr_rows = []
for df, cname in [(sch_d, 'SCH'), (whu_d, 'WHU')]:
    sub = df[['NSE', 'Albumin', 'WBC', 'MACE']].dropna(subset=['MACE'])
    for feat, feat_zh in [('NSE', 'NSE'), ('WBC', 'WBC'), ('Albumin', 'Albumin')]:
        fv = sub[feat].dropna()
        mv = sub.loc[fv.index, 'MACE']
        if len(fv) < 20:
            continue
        rho, p = stats.spearmanr(fv, mv)
        pcorr_rows.append({'Cohort': cname, 'Feature': feat, 'Type': 'Raw', 'rho': round(rho, 4), 'p': round(p, 4)})
        stat_rows.append({'Analysis': 'D3_raw', 'Cohort': cname, 'Metric': f'Spearman_rho_{feat}_vs_MACE', 'Value': round(rho, 4), 'p': round(p, 4)})
    sub2 = sub[['NSE', 'Albumin', 'MACE']].dropna()
    if len(sub2) >= 20:
        pr, pp = partial_corr_spearman(sub2['NSE'], sub2['MACE'], sub2['Albumin'])
        pcorr_rows.append({'Cohort': cname, 'Feature': 'NSE', 'Type': 'Partial|Albumin', 'rho': round(pr, 4), 'p': round(pp, 4)})
        stat_rows.append({'Analysis': 'D3_partial', 'Cohort': cname, 'Metric': 'Partial_rho_NSE_MACE_given_Albumin', 'Value': round(pr, 4), 'p': round(pp, 4)})
    if len(sub2) >= 20:
        pr2, pp2 = partial_corr_spearman(sub2['Albumin'], sub2['MACE'], sub2['NSE'])
        pcorr_rows.append({'Cohort': cname, 'Feature': 'Albumin', 'Type': 'Partial|NSE', 'rho': round(pr2, 4), 'p': round(pp2, 4)})
        stat_rows.append({'Analysis': 'D3_partial', 'Cohort': cname, 'Metric': 'Partial_rho_Albumin_MACE_given_NSE', 'Value': round(pr2, 4), 'p': round(pp2, 4)})
pcorr_df = pd.DataFrame(pcorr_rows)
log(f'  Partial corr results:\n{pcorr_df.to_string(index=False)}')
features_plot = ['NSE', 'Albumin', 'WBC']
types_raw = ['Raw', 'Partial|Albumin', 'Partial|NSE']
type_labels = {'Raw': 'Raw rho', 'Partial|Albumin': 'Partial rho\n(| Albumin)', 'Partial|NSE': 'Partial rho\n(| NSE)'}
bw = 0.22
x_base = np.arange(len(features_plot))
for ci, (cname, color) in enumerate([('SCH', C_SCH), ('WHU', C_WHU)]):
    for ti, typ in enumerate(['Raw', 'Partial|Albumin', 'Partial|NSE']):
        offset = (ci * 3 + ti - 2.5) * bw
        for fi, feat in enumerate(features_plot):
            row = pcorr_df[(pcorr_df['Cohort'] == cname) & (pcorr_df['Feature'] == feat) & (pcorr_df['Type'] == typ)]
            if len(row) == 0:
                continue
            rho = row['rho'].values[0]
            p = row['p'].values[0]
            bar_c = color if typ == 'Raw' else color + '88'
            bar_lw = 1.5 if not pd.isna(p) and p < 0.05 else 0.5
            ax_pcorr.bar(x_base[fi] + offset, rho, width=bw * 0.85, color=bar_c, edgecolor='black', linewidth=bar_lw, label=f'{cname} {type_labels[typ]}' if fi == 0 else '')
ax_pcorr.axhline(0, color='black', lw=0.7, ls='--')
ax_pcorr.set_xticks(x_base)
ax_pcorr.set_xticklabels(features_plot, fontsize=9)
ax_pcorr.set_ylabel('Spearman rho (feature vs MACE)')
ax_pcorr.set_title('D3  Partial correlations: NSE->MACE before/after controlling Albumin', fontsize=9, fontweight='bold', loc='left')
handles = [mpatches.Patch(color=C_SCH, label='SCH - raw rho'), mpatches.Patch(color=C_SCH + '88', label='SCH - partial rho'), mpatches.Patch(color=C_WHU, label='WHU - raw rho'), mpatches.Patch(color=C_WHU + '88', label='WHU - partial rho')]
ax_pcorr.legend(handles=handles, fontsize=7, ncol=2)
despine(ax_pcorr)
log('D4: Logistic regression with interaction …')
lr_rows = []
for df, cname, color in [(sch_d, 'SCH', C_SCH), (whu_d, 'WHU', C_WHU)]:
    sub = df[['NSE', 'Albumin', 'WBC', 'MACE']].dropna()
    if len(sub) < 50:
        continue
    y = sub['MACE'].values.astype(int)
    sc = StandardScaler()
    X_s = sc.fit_transform(sub[['NSE', 'Albumin', 'WBC']].values)
    X_main = X_s
    X_int = np.column_stack([X_s, X_s[:, 0] * X_s[:, 1]])
    feat_names_main = ['NSE', 'Albumin', 'WBC']
    feat_names_int = ['NSE', 'Albumin', 'WBC', 'NSE x Albumin']
    for Xm, names, model_label in [(X_main, feat_names_main, 'Main'), (X_int, feat_names_int, 'Interaction')]:
        try:
            lr = LogisticRegression(penalty='l2', C=1.0, max_iter=500, solver='lbfgs', random_state=42)
            lr.fit(Xm, y)
            coefs = lr.coef_[0]
            for name, coef in zip(names, coefs):
                lr_rows.append({'Cohort': cname, 'Model': model_label, 'Feature': name, 'Coef': round(coef, 4)})
        except Exception as e:
            log(f'  LR error {cname} {model_label}: {e}')
lr_df = pd.DataFrame(lr_rows)
log(f'  LR results:\n{lr_df.to_string(index=False)}')
stat_rows.extend(lr_df.rename(columns={'Model': 'Analysis', 'Feature': 'Metric', 'Coef': 'Value'}).assign(p=np.nan).to_dict('records'))
feats_plot = ['NSE', 'Albumin', 'WBC', 'NSE x Albumin']
bw2 = 0.35
x2 = np.arange(len(feats_plot))
for ci, (cname, color) in enumerate([('SCH', C_SCH), ('WHU', C_WHU)]):
    sub_lr = lr_df[(lr_df['Cohort'] == cname) & (lr_df['Model'] == 'Interaction')]
    coefs = [sub_lr.loc[sub_lr['Feature'] == f, 'Coef'].values[0] if f in sub_lr['Feature'].values else 0 for f in feats_plot]
    xs = x2 + (ci - 0.5) * bw2
    ax_lr.bar(xs, coefs, width=bw2 * 0.85, color=color, alpha=0.85, label=cname)
    for xi, (xv, cv) in enumerate(zip(xs, coefs)):
        if feats_plot[xi] == 'NSE x Albumin':
            ax_lr.text(xv, cv + 0.01 * np.sign(cv), f'{cv:.3f}', ha='center', va='bottom' if cv >= 0 else 'top', fontsize=7)
ax_lr.axhline(0, color='k', lw=0.7, ls='--')
ax_lr.set_xticks(x2)
ax_lr.set_xticklabels(feats_plot, fontsize=8)
ax_lr.set_ylabel('Standardised log-odds (LR coef)')
ax_lr.set_title('D4  LR coefficients (incl. NSE x Albumin interaction)', fontsize=9, fontweight='bold', loc='left')
ax_lr.legend(fontsize=8)
despine(ax_lr)
log('D5: Clinical subgroup characterisation …')

def make_subgroup(df):
    sub = df[['NSE', 'Albumin', 'WBC', 'Prealbumin', 'MACE', 'Dept']].copy()
    sub = sub.dropna(subset=['NSE', 'Albumin'])
    nse_hi = sub['NSE'] >= sub['NSE'].quantile(0.67)
    alb_lo = sub['Albumin'] <= sub['Albumin'].quantile(0.33)
    sub['subgroup'] = 'Other'
    sub.loc[nse_hi & alb_lo, 'subgroup'] = 'NSE-High/Alb-Low'
    sub.loc[~nse_hi & ~alb_lo, 'subgroup'] = 'NSE-Low/Alb-High'
    return sub
sch_sg = make_subgroup(sch_d)
whu_sg = make_subgroup(whu_d)
sg_order = ['NSE-Low/Alb-High', 'Other', 'NSE-High/Alb-Low']
sg_colors = [C_NOMA, '#AAAAAA', C_MACE]
bw3 = 0.32
x3 = np.arange(len(sg_order))
for ci, (sg, cname, color) in enumerate([(sch_sg, 'SCH', C_SCH), (whu_sg, 'WHU', C_WHU)]):
    rates = sg.groupby('subgroup')['MACE'].mean() * 100
    vals = [rates.get(s, np.nan) for s in sg_order]
    ns = [sg[sg['subgroup'] == s]['MACE'].notna().sum() for s in sg_order]
    xs = x3 + (ci - 0.5) * bw3
    bars = ax_sub_bar.bar(xs, vals, width=bw3 * 0.85, color=color, alpha=0.85, label=cname)
    for xv, v, n in zip(xs, vals, ns):
        if not np.isnan(v):
            ax_sub_bar.text(xv, v + 0.5, f'n={n}', ha='center', fontsize=6, rotation=90)
    for s, v, n in zip(sg_order, vals, ns):
        if not np.isnan(v):
            stat_rows.append({'Analysis': 'D5_subgroup', 'Cohort': cname, 'Metric': f"MACE_pct_{s.replace('/', '_')}", 'Value': round(v, 2), 'p': np.nan})
ax_sub_bar.set_xticks(x3)
ax_sub_bar.set_xticklabels(sg_order, fontsize=8)
ax_sub_bar.set_ylabel('MACE rate (%)')
ax_sub_bar.set_ylim(0, ax_sub_bar.get_ylim()[1] * 1.25)
ax_sub_bar.set_title('D5a  MACE rate by NSE-Albumin subgroup', fontsize=9, fontweight='bold', loc='left')
ax_sub_bar.legend(fontsize=8)
despine(ax_sub_bar)
log('D5b: feature characterisation of NSE-High/Alb-Low …')
aux_feats = ['WBC', 'Prealbumin']
for ax, sg, cname, cohort_c in [(ax_sub_km, whu_sg, 'WHU', C_WHU)]:
    hi_lo = sg[sg['subgroup'] == 'NSE-High/Alb-Low']
    ref = sg[sg['subgroup'] == 'NSE-Low/Alb-High']
    positions = []
    labels = []
    medians_hi = []
    medians_ref = []
    feat_results = []
    for feat in aux_feats + ['NSE', 'Albumin']:
        v_hi = hi_lo[feat].dropna()
        v_ref = ref[feat].dropna()
        if len(v_hi) < 5 or len(v_ref) < 5:
            continue
        u, pu = stats.mannwhitneyu(v_hi, v_ref, alternative='two-sided')
        feat_results.append({'feat': feat, 'med_hi': v_hi.median(), 'med_ref': v_ref.median(), 'p': pu})
    feat_df = pd.DataFrame(feat_results)
    if feat_df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    else:
        feat_df['fc'] = feat_df['med_hi'] / (feat_df['med_ref'] + 1e-09)
        feat_df['log2fc'] = np.log2(feat_df['fc'].clip(0.1, 10))
        yp = np.arange(len(feat_df))
        colors = [C_MACE if v > 0 else C_NOMA for v in feat_df['log2fc']]
        ax_sub_km.barh(yp, feat_df['log2fc'], color=colors, alpha=0.8)
        ax_sub_km.set_yticks(yp)
        ax_sub_km.set_yticklabels(feat_df['feat'], fontsize=9)
        ax_sub_km.axvline(0, c='k', lw=0.8)
        for yi, (_, row) in enumerate(feat_df.iterrows()):
            p_lbl = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else 'ns'
            ax_sub_km.text(row['log2fc'] + 0.05 * np.sign(row['log2fc']) + 0.05, yi, p_lbl, va='center', fontsize=8, fontweight='bold')
        ax_sub_km.set_xlabel('log2 fold-change (NSE-Hi/Alb-Lo vs NSE-Lo/Alb-Hi)')
        ax_sub_km.set_title(f'D5b  Feature profile: NSE-High/Alb-Low vs NSE-Low/Alb-High  ({cname}, n={len(hi_lo)} vs n={len(ref)})', fontsize=9, fontweight='bold', loc='left')
        despine(ax_sub_km)
        for _, row in feat_df.iterrows():
            stat_rows.append({'Analysis': 'D5b_MWU', 'Cohort': cname, 'Metric': f"MWU_p_{row['feat']}_HiNSE_LoAlb_vs_ref", 'Value': round(row['p'], 4), 'p': round(row['p'], 4)})
        log(f'  {cname} NSE-Hi/Alb-Lo feature profile:\n{feat_df.to_string(index=False)}')
plt.tight_layout(rect=[0, 0, 1, 0.995])
save_fig(fig, 'FigureS23_NSE_Albumin_Collinearity')
stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(ADD_DIR / 'D_NSE_Albumin_stats.csv', index=False)
log(f'  Stats table saved ({len(stat_df)} rows).')
log('=' * 60)
for cname, grid in grid_results.items():
    log(f'\nD2 MACE rate grid — {cname}:')
    log(grid.to_string())
log('=' * 60)
log('NSE–Albumin deep-dive complete.')
LOG.close()
