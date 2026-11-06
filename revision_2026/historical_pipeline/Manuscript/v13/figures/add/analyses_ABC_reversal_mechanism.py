"""
Three core mechanism analyses for the SHAP-direction reversal story
====================================================================
Analysis A: Feature Ablation — which reversal features drive the AUC drop?
  Method: SCH-trained LightGBM (22 common features) applied to WHU.
  For each of the 9 reversal features (and cumulatively), replace WHU values
  with the SCH median → recompute AUC → record ΔAUC.
  Output: FigureS22_FeatureAblation.pdf/.png + TableS8_AblationResults.csv

Analysis B: Crude MACE rate by tertile — verifying raw-data direction
  For each of 9 reversal features, split SCH & WHU patients into T1/T2/T3,
  compute MACE rates, plot as grouped bar chart.
  Output: FigureS23_CrudeMACE_Tertiles.pdf/.png

Analysis C: Visit-selection effect — CVD admission vs. oncology admission in WHU
  Stratify WHU by 入院科室 (admission department):
    Cardiovascular: 心血管内科 or 心脏大血管外科
    Oncology/other: all remaining
  For each reversal feature: Spearman ρ(feature, MACE) in each stratum.
  Output: FigureS24_VisitSelection.pdf/.png + TableS9_StratifiedDirection.csv
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os, sys, warnings, time
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_LOG = open(_release_path('pipeline', 'abc_progress.txt'), 'w', encoding='utf-8')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    sys.stdout.flush()
    _LOG.write(line + '\n')
    _LOG.flush()
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import lightgbm as lgb
ROOT = _release_path('pipeline', '')
SUBMIT = os.path.join(ROOT, 'Manuscript', 'Submit')
SCH_RAW = os.path.join(ROOT, 'SCH_processed_data.csv')
WHU_RAW = os.path.join(ROOT, 'WHU_Cancer_processed_data.csv')
SCH_RISK = os.path.join(ROOT, 'data', 'patient_risk_SCH.csv')
WHU_RISK = os.path.join(ROOT, 'data', 'patient_risk_WHU.csv')
os.makedirs(SUBMIT, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
C_SCH = '#2C7BB6'
C_WHU = '#D7191C'
C_GAIN = '#1A9641'
C_NEUT = '#AAAAAA'
C_CV = '#E66101'
C_ONC = '#5E3C99'

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def save_fig(fig, name):
    for ext, dpi in [('.pdf', None), ('.png', 600)]:
        kw = {'dpi': dpi} if dpi else {}
        path = os.path.join(SUBMIT, name + ext)
        fig.savefig(path, bbox_inches='tight', facecolor='white', **kw)
        log(f'Saved: {path}')
    plt.close(fig)
FEAT_EN = {'前白蛋白': 'Prealbumin', '血红蛋白': 'Hemoglobin', '镁': 'Magnesium', '球蛋白': 'Globulin', '白细胞': 'WBC', '红细胞压积': 'Hematocrit', '碱性磷酸酶': 'ALP', '血小板压积': 'Plateletcrit', '神经元特异性烯醇化酶': 'NSE', 'D-二聚体': 'D-dimer', 'α-羟丁酸脱氢酶': 'α-HBDH', '尿比重': 'Urine SG', '尿素氮': 'BUN', '总胆汁酸': 'Total Bile Acids', '总胆红素': 'Total Bilirubin', '直接胆红素': 'Direct Bilirubin', '红细胞': 'RBC', '细胞角蛋白19片段': 'CYFRA21-1', '肌酸激酶': 'CK', '胱抑素C': 'Cystatin C', '载脂蛋白AⅠ': 'Apo AI', '载脂蛋白B': 'Apo B'}
en = lambda x: FEAT_EN.get(x, x)
REVERSAL_FEATS = ['前白蛋白', '血红蛋白', '镁', '球蛋白', '白细胞', '红细胞压积', '碱性磷酸酶', '血小板压积', '神经元特异性烯醇化酶']
REVERSAL_EN = [en(f) for f in REVERSAL_FEATS]
ADMIN_KW = ['时间', '号', '诊断', '科室', 'Unnamed', '上次', '入院', '出院', '年龄', '性别', '出生', '日期', '职工', '人均', '医生', '医院', '卫生院', '时间差', 'Center', 'MACE', 'FollowUp', 'ensemble', 'risk_group']
ADMIN_EXACT = {'病案号', 'Angle', 'C反应蛋白', 'EPL', 'G', 'IFN-γ', 'K', 'K轻链', 'LY30', 'MA', 'R'}

def is_lab(name):
    if name in ADMIN_EXACT:
        return False
    return not any((kw in name for kw in ADMIN_KW))

def strip_special(s):
    return s.lstrip('*★ ')
log('Loading WHU raw data...')
whu_risk = pd.read_csv(WHU_RISK)
id_col = '病案号'
whu_raw = pd.read_csv(WHU_RAW, low_memory=False)
whu_raw.columns = [c.strip() for c in whu_raw.columns]
whu_raw = whu_raw.loc[:, ~whu_raw.columns.duplicated(keep='first')]
log(f'WHU raw: {whu_raw.shape}')
dept_df = whu_raw[[id_col, '入院科室']].copy() if '入院科室' in whu_raw.columns else None
whu_lab_cols = [c for c in whu_raw.columns if is_lab(c)]
for c in whu_lab_cols:
    whu_raw[c] = pd.to_numeric(whu_raw[c], errors='coerce')
whu_agg = whu_raw.groupby(id_col)[whu_lab_cols].first().reset_index()
whu_lab_set = set(whu_lab_cols)
whu = whu_risk[[id_col, 'MACE']].merge(whu_agg, on=id_col, how='left')
if dept_df is not None:
    dept_first = dept_df.groupby(id_col)['入院科室'].first().reset_index()
    whu = whu.merge(dept_first, on=id_col, how='left')
log(f"WHU with MACE: {whu.shape}  MACE rate={whu['MACE'].mean():.3f}")
log('Loading SCH raw data (chunked)...')
sch_risk = pd.read_csv(SCH_RISK)
sch_cols_raw = None
for chunk in pd.read_csv(SCH_RAW, chunksize=1, low_memory=False, nrows=1):
    chunk.columns = [c.strip() for c in chunk.columns]
    sch_cols_raw = chunk.columns.tolist()
    break
sch_col_map = {}
for c in sch_cols_raw:
    s = strip_special(c)
    if s not in sch_col_map:
        sch_col_map[s] = c
target_stripped = [s for s in sch_col_map if is_lab(s) and s in whu_lab_set]
orig_to_stripped = {sch_col_map[s]: s for s in target_stripped}
canon_orig = list(orig_to_stripped.keys())
sch_chunks = []
for i, chunk in enumerate(pd.read_csv(SCH_RAW, chunksize=50000, low_memory=False)):
    chunk.columns = [c.strip() for c in chunk.columns]
    keep = [id_col] + [c for c in canon_orig if c in chunk.columns]
    sub = chunk[keep].copy().rename(columns=orig_to_stripped)
    for c in sub.columns:
        if c != id_col:
            sub[c] = pd.to_numeric(sub[c], errors='coerce')
    sch_chunks.append(sub)
    if (i + 1) % 3 == 0:
        log(f'  SCH chunk {i + 1}...')
sch_raw_all = pd.concat(sch_chunks, ignore_index=True)
sch_feat_cols = [c for c in sch_raw_all.columns if c != id_col]
sch_agg = sch_raw_all.groupby(id_col)[sch_feat_cols].first().reset_index()
sch = sch_risk[[id_col, 'MACE']].merge(sch_agg, on=id_col, how='left')
log(f"SCH with MACE: {sch.shape}  MACE rate={sch['MACE'].mean():.3f}")
log('Finding common features...')
sch_feats = set(sch.columns) - {id_col, 'MACE'}
whu_feats = set(whu.columns) - {id_col, 'MACE', '入院科室'}
COMMON_FEATS = sorted([f for f in sch_feats & whu_feats if sch[f].notna().mean() >= 0.2 and whu[f].notna().mean() >= 0.2])
log(f'Common features: {len(COMMON_FEATS)}')
REV_AVAIL = [f for f in REVERSAL_FEATS if f in COMMON_FEATS]
log(f'Reversal features available: {len(REV_AVAIL)} / {len(REVERSAL_FEATS)}')

def make_Xy(df, feats, min_nonmiss=5):
    d = df[[id_col, 'MACE'] + feats].dropna(subset=['MACE']).copy()
    d = d[d[feats].notna().sum(axis=1) >= min_nonmiss]
    y = d['MACE'].values.astype(int)
    imp = SimpleImputer(strategy='median')
    X = pd.DataFrame(imp.fit_transform(d[feats]), columns=feats, index=d.index)
    return (X, y, imp, d.index)
X_sch, y_sch, imp_sch, idx_sch = make_Xy(sch, COMMON_FEATS)
X_whu, y_whu, imp_whu, idx_whu = make_Xy(whu, COMMON_FEATS)
log(f'Model datasets — SCH: {X_sch.shape}, WHU: {X_whu.shape}')
log('Training LightGBM on SCH...')
spw = (y_sch == 0).sum() / max((y_sch == 1).sum(), 1)
mdl_sch = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, random_state=42, verbosity=-1, n_jobs=-1)
mdl_sch.fit(X_sch, y_sch)
auc_own = roc_auc_score(y_sch, mdl_sch.predict_proba(X_sch)[:, 1])
auc_cross_base = roc_auc_score(y_whu, mdl_sch.predict_proba(X_whu)[:, 1])
log(f'SCH model AUC — own: {auc_own:.3f}, cross→WHU baseline: {auc_cross_base:.3f}')
log('=' * 60)
log('Analysis A: Feature ablation...')
sch_medians = {f: float(X_sch[f].median()) for f in REV_AVAIL}
ablation_results = []
REV_ORDER = sorted(REV_AVAIL, key=lambda f: abs(X_whu[f].corr(pd.Series(mdl_sch.predict_proba(X_whu)[:, 1], index=X_whu.index))))

def ablation_auc(X_whu_mod, y):
    return roc_auc_score(y, mdl_sch.predict_proba(X_whu_mod)[:, 1])
for f in REV_AVAIL:
    X_mod = X_whu.copy()
    X_mod[f] = sch_medians[f]
    auc_new = ablation_auc(X_mod, y_whu)
    delta = auc_new - auc_cross_base
    ablation_results.append({'feature_cn': f, 'feature_en': en(f), 'auc_baseline': auc_cross_base, 'auc_ablated': auc_new, 'delta_auc': delta, 'type': 'single'})
    log(f'  {en(f):20s}  AUC: {auc_new:.4f}  Δ: {delta:+.4f}')
abl_df_single = pd.DataFrame([r for r in ablation_results if r['type'] == 'single'])
feat_by_gain = abl_df_single.sort_values('delta_auc', ascending=False)['feature_cn'].tolist()
X_cum = X_whu.copy()
auc_cum_prev = auc_cross_base
cum_added = []
for rank, f in enumerate(feat_by_gain, 1):
    X_cum[f] = sch_medians[f]
    cum_added.append(en(f))
    auc_new = ablation_auc(X_cum, y_whu)
    delta_cum = auc_new - auc_cross_base
    ablation_results.append({'feature_cn': '+'.join([en(ff) for ff in feat_by_gain[:rank]]), 'feature_en': f'Top-{rank} cumulative', 'auc_baseline': auc_cross_base, 'auc_ablated': auc_new, 'delta_auc': delta_cum, 'type': 'cumulative'})
    log(f'  Cumulative top-{rank}: AUC={auc_new:.4f}  Δ={delta_cum:+.4f}')
abl_df = pd.DataFrame(ablation_results)
abl_df.to_csv(os.path.join(SUBMIT, 'TableS8_AblationResults.csv'), index=False)
log('TableS8 saved.')
fig_a, axes_a = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.1, 1]})
single = abl_df[abl_df['type'] == 'single'].sort_values('delta_auc', ascending=True)
n_feats = len(single)
y_pos = np.arange(n_feats)
ax = axes_a[0]
colors = [C_GAIN if d > 0 else '#CC3333' for d in single['delta_auc']]
ax.barh(y_pos, single['delta_auc'], color=colors, height=0.6, edgecolor='white', lw=0.5)
ax.axvline(0, color='black', lw=0.8, ls='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(single['feature_en'].tolist(), fontsize=7.5)
ax.set_xlabel(f'ΔAUC (feature replaced with SCH median)\nBaseline AUC = {auc_cross_base:.3f}', labelpad=4)
ax.set_title('A  Individual feature ablation', loc='left', fontweight='bold', fontsize=9)
for yi, (_, row) in zip(y_pos, single.iterrows()):
    xpos = row['delta_auc']
    ha = 'left' if xpos >= 0 else 'right'
    offset = 0.0005 if xpos >= 0 else -0.0005
    ax.text(xpos + offset, yi, f'{xpos:+.3f}', va='center', fontsize=6.5, color=C_GAIN if xpos > 0 else '#CC3333', fontweight='bold')
despine(ax)
cum = abl_df[abl_df['type'] == 'cumulative'].copy()
x_cum = np.arange(len(cum) + 1)
aucs = [auc_cross_base] + cum['auc_ablated'].tolist()
xlabels = ['Baseline'] + [f'Top-{i + 1}' for i in range(len(cum))]
ax2 = axes_a[1]
ax2.plot(x_cum, aucs, 'o-', color='#333333', lw=1.5, ms=5, zorder=3)
ax2.fill_between(x_cum, auc_cross_base, aucs, where=[a >= auc_cross_base for a in aucs], color=C_GAIN, alpha=0.18, label='AUC gain')
ax2.fill_between(x_cum, auc_cross_base, aucs, where=[a < auc_cross_base for a in aucs], color='#CC3333', alpha=0.18, label='AUC loss')
ax2.axhline(auc_cross_base, color='#888', lw=0.8, ls='--', alpha=0.7, label=f'Baseline ({auc_cross_base:.3f})')
ax2.set_xticks(x_cum)
ax2.set_xticklabels(xlabels, rotation=40, ha='right', fontsize=7)
ax2.set_ylabel('AUC (SCH model on WHU)', labelpad=4)
ax2.set_title('B  Cumulative ablation (ranked by individual gain)', loc='left', fontweight='bold', fontsize=9)
ax2.annotate(f'All 9: {aucs[-1]:.3f}', xy=(x_cum[-1], aucs[-1]), xytext=(-25, 10), textcoords='offset points', fontsize=6.5, color=C_GAIN if aucs[-1] > auc_cross_base else '#CC3333', arrowprops=dict(arrowstyle='->', lw=0.7))
ax2.legend(fontsize=6.5, loc='upper left')
despine(ax2)
plt.tight_layout(pad=1.5)
save_fig(fig_a, 'FigureS16_FeatureAblation')
log('=' * 60)
log('Analysis B: Crude MACE rate by tertile...')

def tertile_mace(df, feat_col, mace_col='MACE'):
    """Return DataFrame with T1/T2/T3 MACE rates and 95% CI (Wilson)."""
    valid = df[[feat_col, mace_col]].dropna()
    if len(valid) < 30 or valid[mace_col].sum() == 0:
        return None
    try:
        valid = valid.copy()
        valid['tertile'] = pd.qcut(valid[feat_col], 3, labels=['T1 (Low)', 'T2', 'T3 (High)'])
    except ValueError:
        return None
    rows = []
    for t, g in valid.groupby('tertile', observed=True):
        n = len(g)
        k = int(g[mace_col].sum())
        p = k / n if n > 0 else np.nan
        z = 1.96
        denom = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
        rows.append({'tertile': str(t), 'n': n, 'mace_rate': p, 'ci_lo': max(0, center - half), 'ci_hi': min(1, center + half)})
    return pd.DataFrame(rows)
tertile_data = {}
for f in REV_AVAIL:
    t_sch = tertile_mace(sch, f)
    t_whu = tertile_mace(whu, f)
    if t_sch is not None and t_whu is not None:
        tertile_data[f] = {'SCH': t_sch, 'WHU': t_whu}
n_valid = len(tertile_data)
log(f'Features with valid tertile data: {n_valid}')
ncols = 3
nrows = int(np.ceil(n_valid / ncols))
fig_b, axes_b = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.6))
axes_b_flat = np.array(axes_b).flatten() if n_valid > 1 else [axes_b]
x_off = np.array([-0.15, 0.0, 0.15])
t_lbls = ['T1\n(Low)', 'T2', 'T3\n(High)']
x_base = np.arange(3)
for idx, (f, cohort_data) in enumerate(sorted(tertile_data.items(), key=lambda x: REVERSAL_FEATS.index(x[0]) if x[0] in REVERSAL_FEATS else 99)):
    ax = axes_b_flat[idx]
    for ci, (cohort, color) in enumerate([('SCH', C_SCH), ('WHU', C_WHU)]):
        td = cohort_data[cohort]
        xs = x_base + x_off[ci] if ci == 0 else x_base + x_off[2]
        xs = x_base + (x_off[0] if cohort == 'SCH' else x_off[2])
        rates = td['mace_rate'].values
        lo = rates - td['ci_lo'].values
        hi = td['ci_hi'].values - rates
        ax.bar(xs, rates * 100, width=0.25, color=color, alpha=0.85, label=cohort, edgecolor='white', lw=0.5)
        ax.errorbar(xs, rates * 100, yerr=[lo * 100, hi * 100], fmt='none', color='black', lw=0.8, capsize=2.5)
    ax.set_title(en(f), fontsize=8, fontweight='bold', pad=3)
    ax.set_xticks(x_base)
    ax.set_xticklabels(t_lbls, fontsize=6.5)
    ax.set_ylabel('MACE rate (%)', fontsize=6.5, labelpad=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    despine(ax)
    if idx == 0:
        ax.legend(fontsize=6, loc='upper right', frameon=False)
for idx in range(n_valid, len(axes_b_flat)):
    axes_b_flat[idx].set_visible(False)
fig_b.suptitle('Crude MACE rates by tertile: SCH vs WHU\n(Reversal features — Direction at data level)', fontsize=9, fontweight='bold')
fig_b.tight_layout(rect=[0, 0, 1, 0.93])
save_fig(fig_b, 'FigureS17_CrudeMACE_Tertiles')
log('=' * 60)
log('Analysis C: Visit-selection stratification...')
CV_DEPTS = {'心血管内科', '心脏大血管外科', '心内科', '冠心病科'}
whu_valid = whu.dropna(subset=['MACE', '入院科室'] if '入院科室' in whu.columns else ['MACE']).copy()
if '入院科室' in whu_valid.columns:
    whu_valid['visit_type'] = whu_valid['入院科室'].apply(lambda d: 'Cardiovascular' if str(d).strip() in CV_DEPTS else 'Oncology/Other')
    n_cv = (whu_valid['visit_type'] == 'Cardiovascular').sum()
    n_onc = (whu_valid['visit_type'] == 'Oncology/Other').sum()
    log(f'WHU stratification — CV: n={n_cv}, Oncology/Other: n={n_onc}')
else:
    log('WARNING: 入院科室 not found; skipping Analysis C')
    whu_valid = None
strat_rows = []
if whu_valid is not None:
    for f in REV_AVAIL:
        v = sch[[f, 'MACE']].dropna()
        rho, p = stats.spearmanr(v[f], v['MACE'])
        strat_rows.append({'feature': en(f), 'stratum': 'SCH (all)', 'rho': rho, 'p': p, 'n': len(v), 'cohort': 'SCH'})
        for stype in ['Cardiovascular', 'Oncology/Other']:
            sub = whu_valid[whu_valid['visit_type'] == stype][[f, 'MACE']].dropna()
            if len(sub) >= 15:
                rho, p = stats.spearmanr(sub[f], sub['MACE'])
                strat_rows.append({'feature': en(f), 'stratum': f'WHU ({stype})', 'rho': rho, 'p': p, 'n': len(sub), 'cohort': 'WHU'})
            else:
                strat_rows.append({'feature': en(f), 'stratum': f'WHU ({stype})', 'rho': np.nan, 'p': np.nan, 'n': len(sub), 'cohort': 'WHU'})
    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(os.path.join(SUBMIT, 'TableS9_StratifiedDirection.csv'), index=False)
    log('TableS9 saved.')
    STRATA = ['SCH (all)', 'WHU (Cardiovascular)', 'WHU (Oncology/Other)']
    S_COLORS = [C_SCH, C_CV, C_ONC]
    S_LABELS = ['SCH', 'WHU: CVD admission', 'WHU: Oncology/other']
    feat_order = [en(f) for f in REVERSAL_FEATS if en(f) in strat_df['feature'].values]
    n_f = len(feat_order)
    x = np.arange(n_f)
    bw = 0.22
    offsets = np.array([-bw, 0, bw])
    fig_c, axes_c = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={'width_ratios': [1.6, 1]})
    ax_bar = axes_c[0]
    for si, (stratum, color, label) in enumerate(zip(STRATA, S_COLORS, S_LABELS)):
        sub = strat_df[strat_df['stratum'] == stratum].set_index('feature')
        rhos = [sub.loc[f, 'rho'] if f in sub.index else np.nan for f in feat_order]
        ps = [sub.loc[f, 'p'] if f in sub.index else np.nan for f in feat_order]
        xs = x + offsets[si]
        bar_colors = []
        for r, p in zip(rhos, ps):
            if pd.isna(r):
                bar_colors.append('#DDDDDD')
            elif p < 0.05:
                bar_colors.append(color)
            else:
                bar_colors.append(color + '66')
        for xi, (xv, rv, bc) in enumerate(zip(xs, rhos, bar_colors)):
            if not pd.isna(rv):
                ax_bar.bar(xv, rv, width=bw, color=bc, edgecolor='white', lw=0.4)
        ax_bar.bar(-999, 0, width=bw, color=color, label=label)
    ax_bar.axhline(0, color='black', lw=0.7, ls='--', alpha=0.5)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(feat_order, rotation=35, ha='right', fontsize=7.5)
    ax_bar.set_xlim(-0.5, n_f - 0.5)
    _rho_abs_max = strat_df['rho'].abs().max(skipna=True)
    _rho_lim = max(float(_rho_abs_max) * 1.25, 0.15) if not pd.isna(_rho_abs_max) else 0.5
    ax_bar.set_ylim(-_rho_lim, _rho_lim)
    ax_bar.set_ylabel('Spearman ρ (feature vs MACE)', labelpad=4)
    ax_bar.set_title('A  MACE directionality by visit type (WHU)', loc='left', fontweight='bold', fontsize=9)
    ax_bar.legend(loc='lower left', fontsize=6.5)
    despine(ax_bar)
    ax_sc = axes_c[1]
    cv_sub = strat_df[strat_df['stratum'] == 'WHU (Cardiovascular)'].set_index('feature')
    onc_sub = strat_df[strat_df['stratum'] == 'WHU (Oncology/Other)'].set_index('feature')
    for f in feat_order:
        xcv = cv_sub.loc[f, 'rho'] if f in cv_sub.index else np.nan
        yonc = onc_sub.loc[f, 'rho'] if f in onc_sub.index else np.nan
        pcv = cv_sub.loc[f, 'p'] if f in cv_sub.index else np.nan
        ponc = onc_sub.loc[f, 'p'] if f in onc_sub.index else np.nan
        if pd.isna(xcv) or pd.isna(yonc):
            continue
        is_reversed = np.sign(xcv) != np.sign(yonc)
        col = C_ONC if is_reversed else '#555555'
        ax_sc.scatter(xcv, yonc, s=45, color=col, alpha=0.85, edgecolors='white', lw=0.4, zorder=3)
        ax_sc.text(xcv + 0.012, yonc, f, fontsize=6, color=col, va='center', fontstyle='italic')
    lim = max(0.5, strat_df['rho'].abs().max() + 0.1)
    ax_sc.axhline(0, color='black', lw=0.7, ls='--', alpha=0.4)
    ax_sc.axvline(0, color='black', lw=0.7, ls='--', alpha=0.4)
    ax_sc.set_xlim(-lim, lim)
    ax_sc.set_ylim(-lim, lim)
    ax_sc.set_xlabel('WHU: Cardiovascular admission (ρ)', labelpad=4)
    ax_sc.set_ylabel('WHU: Oncology/Other admission (ρ)', labelpad=4)
    ax_sc.set_title('B  CV vs. oncology direction (WHU)', loc='left', fontweight='bold', fontsize=9)
    ax_sc.text(lim * 0.7, lim * 0.75, 'Both\npositive', fontsize=6, ha='center', color='#666')
    ax_sc.text(-lim * 0.7, -lim * 0.75, 'Both\nnegative', fontsize=6, ha='center', color='#666')
    ax_sc.text(lim * 0.7, -lim * 0.75, 'Split\n(CV+, Onc−)', fontsize=6, ha='center', color=C_ONC, fontweight='bold')
    ax_sc.text(-lim * 0.7, lim * 0.75, 'Split\n(CV−, Onc+)', fontsize=6, ha='center', color=C_ONC, fontweight='bold')
    despine(ax_sc)
    plt.tight_layout(pad=1.5)
    save_fig(fig_c, 'FigureS18_VisitSelection')
log('=' * 60)
log('All three analyses complete.')
_LOG.close()
