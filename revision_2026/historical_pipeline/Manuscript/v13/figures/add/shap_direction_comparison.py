"""
SHAP Directionality Comparison: SCH vs WHU
===========================================
For all common lab features (sufficient coverage in both cohorts):
1. Aggregate patient-level data from each cohort
2. Train LightGBM on each cohort separately
3. Compute SHAP values (TreeExplainer)
4. For each feature: SHAP slope = Spearman rho(feature_value, shap_value)
5. Identify features where direction is OPPOSITE and BOTH significant (p<0.05)
6. Publication-quality figure:
   - Main: scatter SCH_slope vs WHU_slope, all common features; label reversals
   - Panel B: lollipop of just the reversal features showing both slopes

Output:
  Manuscript/Submit/FigureS21_SHAP_Direction_Comparison.pdf/.png
  Manuscript/Submit/TableS7_SHAP_Direction_Reversals.csv
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os, sys, warnings, time
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
_LOG_PATH = _release_path('pipeline', 'shap_progress.txt')
_log_f = open(_LOG_PATH, 'w', encoding='utf-8')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    sys.stdout.flush()
    _log_f.write(line + '\n')
    _log_f.flush()
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
import shap
import lightgbm as lgb
from sklearn.impute import SimpleImputer
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
C_BOTH = '#7B2D8B'
C_GRY = '#AAAAAA'

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(which='both', length=3)

def save_fig(fig, name):
    pdf_path = os.path.join(SUBMIT, f'{name}.pdf')
    png_path = os.path.join(SUBMIT, f'{name}.png')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log(f'Saved: {pdf_path}')
ADMIN_KW = ['时间', '号', '诊断', '科室', 'Unnamed', '上次', '入院', '出院', '年龄', '性别', '出生', '日期', '职工', '人均', '医生', '医院', '卫生院', '时间差', 'Center', 'MACE', 'FollowUp', 'ensemble', 'risk_group']
ADMIN_EXACT = {'病案号', 'Angle', 'C反应蛋白', 'EPL', 'G', 'IFN-γ', 'K', 'K轻链', 'LY30', 'MA', 'R'}

def is_lab_feature(name: str) -> bool:
    if name in ADMIN_EXACT:
        return False
    for kw in ADMIN_KW:
        if kw in name:
            return False
    return True
log('Loading WHU raw data...')
whu_risk = pd.read_csv(WHU_RISK)
id_col_whu = '病案号'
whu_raw = pd.read_csv(WHU_RAW, low_memory=False)
whu_raw.columns = [c.strip() for c in whu_raw.columns]
whu_raw = whu_raw.loc[:, ~whu_raw.columns.duplicated(keep='first')]
log(f'WHU raw: {whu_raw.shape}')
whu_lab_cols = [c for c in whu_raw.columns if is_lab_feature(c)]
whu_lab_set = set(whu_lab_cols)
log(f'Lab cols in WHU: {len(whu_lab_cols)}')
for c in whu_lab_cols:
    whu_raw[c] = pd.to_numeric(whu_raw[c], errors='coerce')
whu_agg = whu_raw.groupby(id_col_whu)[whu_lab_cols].first().reset_index()
log(f'WHU patient-level: {whu_agg.shape}')
whu = whu_risk[[id_col_whu, 'MACE']].merge(whu_agg, on=id_col_whu, how='left')
log(f"WHU with MACE: {whu.shape}  |  MACE rate: {whu['MACE'].mean():.3f}")
log('=' * 60)
log('Loading SCH raw data (chunked, filtered to WHU-common features)...')
sch_risk = pd.read_csv(SCH_RISK)
id_col_sch = '病案号'

def strip_special(s):
    return s.lstrip('*★ ')
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
target_stripped = [s for s in sch_col_map if is_lab_feature(s) and s in whu_lab_set]
log(f'WHU-common lab features to load from SCH: {len(target_stripped)}')
orig_to_stripped = {sch_col_map[s]: s for s in target_stripped}
canon_orig_cols = list(orig_to_stripped.keys())
sch_chunks = []
for i, chunk in enumerate(pd.read_csv(SCH_RAW, chunksize=50000, low_memory=False)):
    chunk.columns = [c.strip() for c in chunk.columns]
    keep_orig = [id_col_sch] + [c for c in canon_orig_cols if c in chunk.columns]
    sub = chunk[keep_orig].copy().rename(columns=orig_to_stripped)
    for c in sub.columns:
        if c != id_col_sch:
            sub[c] = pd.to_numeric(sub[c], errors='coerce')
    sch_chunks.append(sub)
    if (i + 1) % 3 == 0:
        log(f'  SCH chunk {i + 1} loaded...')
sch_raw_all = pd.concat(sch_chunks, ignore_index=True)
log(f'SCH raw concatenated: {sch_raw_all.shape}')
sch_feat_cols = [c for c in sch_raw_all.columns if c != id_col_sch]
sch_agg = sch_raw_all.groupby(id_col_sch)[sch_feat_cols].first().reset_index()
log(f'SCH patient-level: {sch_agg.shape}')
sch = sch_risk[[id_col_sch, 'MACE']].merge(sch_agg, on=id_col_sch, how='left')
log(f"SCH with MACE: {sch.shape}  |  MACE rate: {sch['MACE'].mean():.3f}")
log('=' * 60)
log('Identifying common features with ≥20% coverage in both cohorts...')
sch_feats = set(sch.columns) - {id_col_sch, 'MACE'}
whu_feats = set(whu.columns) - {id_col_whu, 'MACE'}
common_raw = sch_feats & whu_feats
COVERAGE_THRESH = 0.2
common_feats = []
for f in sorted(common_raw):
    sch_cov = sch[f].notna().mean()
    whu_cov = whu[f].notna().mean()
    if sch_cov >= COVERAGE_THRESH and whu_cov >= COVERAGE_THRESH:
        common_feats.append(f)
log(f'Common features after coverage filter: {len(common_feats)}')
log(common_feats)

def prepare_dataset(df, id_col, feats, min_feat_nonmiss=3):
    """Return X (imputed), y arrays; rows with MACE missing are dropped."""
    d = df[[id_col, 'MACE'] + feats].dropna(subset=['MACE']).copy()
    feat_df = d[feats]
    d = d[feat_df.notna().sum(axis=1) >= min_feat_nonmiss].copy()
    y = d['MACE'].values.astype(int)
    imp = SimpleImputer(strategy='median')
    X = pd.DataFrame(imp.fit_transform(d[feats]), columns=feats)
    return (X, y, imp)
log('\nPreparing datasets...')
X_sch, y_sch, imp_sch = prepare_dataset(sch, id_col_sch, common_feats, min_feat_nonmiss=5)
X_whu, y_whu, imp_whu = prepare_dataset(whu, id_col_whu, common_feats, min_feat_nonmiss=5)
log(f'SCH model set: X={X_sch.shape}, MACE rate={y_sch.mean():.3f}')
log(f'WHU model set: X={X_whu.shape}, MACE rate={y_whu.mean():.3f}')

def train_lgbm(X, y, seed=42):
    pos = y.sum()
    neg = len(y) - pos
    spw = neg / max(pos, 1)
    params = dict(objective='binary', metric='auc', n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, random_state=seed, verbosity=-1, n_jobs=-1)
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model
log('\nTraining LightGBM on SCH...')
mdl_sch = train_lgbm(X_sch, y_sch)
log('Training LightGBM on WHU...')
mdl_whu = train_lgbm(X_whu, y_whu)
log('\nComputing SHAP values...')
explainer_sch = shap.TreeExplainer(mdl_sch)
shap_sch = explainer_sch.shap_values(X_sch)
if isinstance(shap_sch, list):
    shap_sch = shap_sch[1]
shap_sch = pd.DataFrame(shap_sch, columns=common_feats)
explainer_whu = shap.TreeExplainer(mdl_whu)
shap_whu = explainer_whu.shap_values(X_whu)
if isinstance(shap_whu, list):
    shap_whu = shap_whu[1]
shap_whu = pd.DataFrame(shap_whu, columns=common_feats)
log(f'SHAP matrices: SCH {shap_sch.shape}, WHU {shap_whu.shape}')
log('\nComputing Spearman slopes...')
rows = []
for f in common_feats:
    rho_sch, p_sch = stats.spearmanr(X_sch[f], shap_sch[f])
    rho_whu, p_whu = stats.spearmanr(X_whu[f], shap_whu[f])
    rows.append({'feature': f, 'rho_SCH': rho_sch, 'p_SCH': p_sch, 'rho_WHU': rho_whu, 'p_WHU': p_whu})
slope_df = pd.DataFrame(rows)
slope_df['sig_SCH'] = slope_df['p_SCH'] < 0.05
slope_df['sig_WHU'] = slope_df['p_WHU'] < 0.05
slope_df['both_sig'] = slope_df['sig_SCH'] & slope_df['sig_WHU']
slope_df['opp_sign'] = np.sign(slope_df['rho_SCH']) != np.sign(slope_df['rho_WHU'])
slope_df['reversal'] = slope_df['both_sig'] & slope_df['opp_sign']
log('\nAll features Spearman slopes:')
log(slope_df.sort_values('reversal', ascending=False).to_string(index=False))
reversals = slope_df[slope_df['reversal']].sort_values('rho_SCH')
log(f'\n>>> Reversal features (both sig + opposite sign): {len(reversals)}')
log(reversals[['feature', 'rho_SCH', 'p_SCH', 'rho_WHU', 'p_WHU']].to_string(index=False))
slope_df_out = slope_df.copy()
slope_df_out.to_csv(os.path.join(SUBMIT, 'TableS7_SHAP_Direction_Slopes.csv'), index=False)
rev_table = reversals[['feature', 'rho_SCH', 'p_SCH', 'rho_WHU', 'p_WHU']].copy()
rev_table.to_csv(os.path.join(SUBMIT, 'TableS7b_SHAP_Direction_Reversals.csv'), index=False)
log('TableS7 saved.')
FEAT_EN = {'D-二聚体': 'D-dimer', 'α-羟丁酸脱氢酶': 'α-HBDH', '前白蛋白': 'Prealbumin', '尿比重': 'Urine SG', '尿素氮': 'BUN', '总胆汁酸': 'Total Bile Acids', '总胆红素': 'Total Bilirubin', '球蛋白': 'Globulin', '白细胞': 'WBC', '直接胆红素': 'Direct Bilirubin', '碱性磷酸酶': 'ALP', '神经元特异性烯醇化酶': 'NSE', '红细胞': 'RBC', '红细胞压积': 'Hematocrit', '细胞角蛋白19片段': 'CYFRA21-1', '肌酸激酶': 'CK', '胱抑素C': 'Cystatin C', '血小板压积': 'Plateletcrit', '血红蛋白': 'Hemoglobin', '载脂蛋白AⅠ': 'Apo AI', '载脂蛋白B': 'Apo B', '镁': 'Magnesium', '乳酸脱氢酶': 'LDH', '白蛋白': 'Albumin', '肌酐': 'Creatinine', '甘油三脂': 'Triglycerides', '总胆固醇': 'Total Cholesterol', '高密度脂蛋白胆固醇': 'HDL-C', '低密度脂蛋白胆固醇': 'LDL-C', '载脂蛋白B': 'Apo B', '甲胎蛋白': 'AFP', '癌胚抗原': 'CEA', '糖类抗原CA125': 'CA-125', '糖类抗原CA15-3': 'CA15-3', '糖类抗原CA19-9': 'CA19-9', '葡萄糖': 'Glucose', '血红蛋白': 'Hemoglobin', '钙': 'Calcium', '钾': 'Potassium', '钠': 'Sodium', '氯': 'Chloride', '磷': 'Phosphorus', '铁蛋白': 'Ferritin', '转铁蛋白': 'Transferrin', '补体C3': 'Complement C3', '补体C4': 'Complement C4', '尿酸': 'Uric Acid', '尿素': 'Urea', '总蛋白': 'Total Protein', '同型半胱氨酸': 'Homocysteine', '糖化血红蛋白': 'HbA1c', '降钙素原': 'PCT', 'C反应蛋白': 'CRP', '淀粉酶': 'Amylase', '腺苷脱氨酶': 'ADA', '胆碱酯酶': 'Cholinesterase'}

def en(name: str) -> str:
    """Return English name for a feature, falling back to original."""
    return FEAT_EN.get(name, name)
log('\nBuilding figure...')
n_rev = len(reversals)
if n_rev > 0:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={'width_ratios': [1.2, 1]})
    ax_sc, ax_lo = axes
else:
    fig, ax_sc = plt.subplots(figsize=(5.5, 4.5))
    ax_lo = None

def assign_color(row):
    if row['reversal']:
        return C_BOTH
    if row['both_sig']:
        return '#555555'
    if row['sig_SCH'] or row['sig_WHU']:
        return '#888888'
    return C_GRY
slope_df['color'] = slope_df.apply(assign_color, axis=1)
slope_df['size'] = slope_df['reversal'].map({True: 55, False: 22})
for _, row in slope_df[~slope_df['reversal']].iterrows():
    ax_sc.scatter(row['rho_SCH'], row['rho_WHU'], s=row['size'], color=row['color'], alpha=0.65, edgecolors='none', zorder=2)
for _, row in slope_df[slope_df['reversal']].iterrows():
    ax_sc.scatter(row['rho_SCH'], row['rho_WHU'], s=row['size'], color=row['color'], alpha=0.92, edgecolors='white', linewidths=0.5, zorder=4)
    dx = 0.013
    ha = 'left' if row['rho_SCH'] < 0 else 'left'
    ax_sc.text(row['rho_SCH'] + dx, row['rho_WHU'], en(row['feature']), fontsize=6.3, color=C_BOTH, ha=ha, va='center', zorder=5, fontstyle='italic')
lim = max(abs(slope_df['rho_SCH']).max(), abs(slope_df['rho_WHU']).max()) + 0.08
ax_sc.axhline(0, color='black', lw=0.7, ls='--', alpha=0.5)
ax_sc.axvline(0, color='black', lw=0.7, ls='--', alpha=0.5)
x_ref = np.linspace(-lim, lim, 200)
ax_sc.fill_between(x_ref, x_ref - 0.05, x_ref + 0.05, color='#DDDDDD', alpha=0.35, zorder=0)
ax_sc.set_xlim(-lim, lim)
ax_sc.set_ylim(-lim, lim)
ax_sc.set_xlabel('SHAP directionality (Spearman ρ): SCH', labelpad=4)
ax_sc.set_ylabel('SHAP directionality (Spearman ρ): WHU', labelpad=4)
ax_sc.set_title('A  SHAP directionality: SCH vs WHU', loc='left', fontweight='bold', fontsize=9)
despine(ax_sc)
ax_sc.text(lim * 0.85, lim * 0.85, 'Risk↑\nboth', fontsize=6, ha='center', va='center', color='#666')
ax_sc.text(-lim * 0.85, -lim * 0.85, 'Risk↓\nboth', fontsize=6, ha='center', va='center', color='#666')
ax_sc.text(lim * 0.85, -lim * 0.85, 'SCH↑\nWHU↓', fontsize=6, ha='center', va='center', color=C_BOTH, fontweight='bold')
ax_sc.text(-lim * 0.85, lim * 0.85, 'SCH↓\nWHU↑', fontsize=6, ha='center', va='center', color=C_BOTH, fontweight='bold')
handles = [mpatches.Patch(color=C_BOTH, label=f'Reversal (both sig, opposite sign, n={n_rev})'), mpatches.Patch(color='#555555', label='Same direction, both sig'), mpatches.Patch(color='#888888', label='One cohort sig'), mpatches.Patch(color=C_GRY, label='Neither sig')]
ax_sc.legend(handles=handles, loc='lower right', frameon=True, framealpha=0.85, edgecolor='#CCC', fontsize=6)
ax_sc.text(0.02, 0.98, f'n={len(common_feats)} common features', transform=ax_sc.transAxes, fontsize=6.5, va='top', color='#555')
if ax_lo is not None and n_rev > 0:
    rev_sorted = reversals.sort_values('rho_SCH')
    ypos = np.arange(n_rev)
    for yi, (_, row) in zip(ypos, rev_sorted.iterrows()):
        ax_lo.plot([0, row['rho_SCH']], [yi + 0.08, yi + 0.08], color=C_SCH, lw=2.2, solid_capstyle='round')
        ax_lo.scatter([row['rho_SCH']], [yi + 0.08], s=55, color=C_SCH, zorder=3, edgecolors='white', lw=0.5)
        stars = '***' if row['p_SCH'] < 0.001 else '**' if row['p_SCH'] < 0.01 else '*'
        ax_lo.text(row['rho_SCH'] + (0.007 if row['rho_SCH'] >= 0 else -0.007), yi + 0.08, stars, fontsize=6, color=C_SCH, ha='left' if row['rho_SCH'] >= 0 else 'right', va='center')
        ax_lo.plot([0, row['rho_WHU']], [yi - 0.08, yi - 0.08], color=C_WHU, lw=2.2, solid_capstyle='round')
        ax_lo.scatter([row['rho_WHU']], [yi - 0.08], s=55, color=C_WHU, zorder=3, edgecolors='white', lw=0.5)
        stars = '***' if row['p_WHU'] < 0.001 else '**' if row['p_WHU'] < 0.01 else '*'
        ax_lo.text(row['rho_WHU'] + (0.007 if row['rho_WHU'] >= 0 else -0.007), yi - 0.08, stars, fontsize=6, color=C_WHU, ha='left' if row['rho_WHU'] >= 0 else 'right', va='center')
    ax_lo.axvline(0, color='black', lw=0.7, ls='--', alpha=0.5)
    ax_lo.set_yticks(ypos)
    ax_lo.set_yticklabels([en(f) for f in rev_sorted['feature']], fontsize=7.5)
    ax_lo.set_xlabel('SHAP directionality\n(Spearman ρ with feature value)', labelpad=4)
    ax_lo.set_title('B  Reversal features (both cohorts, p<0·05)', loc='left', fontweight='bold', fontsize=9)
    despine(ax_lo)
    leg_b = [Line2D([0], [0], marker='o', lw=0, markerfacecolor=C_SCH, ms=7, label='SCH'), Line2D([0], [0], marker='o', lw=0, markerfacecolor=C_WHU, ms=7, label='WHU')]
    ax_lo.legend(handles=leg_b, loc='lower right', fontsize=6.5)
    xabs = max(abs(rev_sorted['rho_SCH']).max(), abs(rev_sorted['rho_WHU']).max()) + 0.12
    ax_lo.set_xlim(-xabs, xabs)
elif ax_lo is not None:
    ax_lo.text(0.5, 0.5, 'No reversal features\nfound (both p<0·05,\nopposite sign)', ha='center', va='center', transform=ax_lo.transAxes, fontsize=8, color='#999')
    despine(ax_lo)
plt.tight_layout(pad=1.5)
save_fig(fig, 'FigureS15_SHAP_Direction_Comparison')
log('Done. All outputs saved to Manuscript/Submit/')
_log_f.close()
