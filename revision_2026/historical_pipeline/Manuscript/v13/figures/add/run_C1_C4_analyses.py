"""
C1: Reversal feature conditional distributions (violin + Bhattacharyya)
C2: Feature correlation network comparison (SCH vs WHU)
C3: UMAP joint embedding
C4: NSE + WBC mechanistic deep-dive
"""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import warnings
warnings.filterwarnings('ignore')
import os, sys, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
import seaborn as sns
BASE = Path(_release_path('pipeline', ''))
ADD_DIR = BASE / 'Manuscript\\v13\\figures\\add'
SUB_DIR = BASE / 'Manuscript\\Submit'
SUB_DIR.mkdir(parents=True, exist_ok=True)
SCH_CSV = BASE / 'SCH_processed_data.csv'
WHU_CSV = BASE / 'WHU_Cancer_processed_data.csv'
SCH_RISK = BASE / 'data' / 'patient_risk_SCH.csv'
WHU_RISK = BASE / 'data' / 'patient_risk_WHU.csv'
ID_COL = '病案号'
LOG_PATH = ADD_DIR / 'c1c4_progress.txt'
_LOG = open(LOG_PATH, 'w', encoding='utf-8')

def log(msg):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    _LOG.write(line + '\n')
    _LOG.flush()
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 150})
C_SCH = '#2166AC'
C_WHU = '#D6604D'
C_MACE = '#B2182B'
C_NOMA = '#4D9221'
REVERSAL_FEATS_ZH = ['前白蛋白', '血红蛋白', '镁', '球蛋白', '白细胞', '红细胞压积', '碱性磷酸酶', '血小板压积', '神经元特异性烯醇化酶']
REVERSAL_FEATS_EN = ['Prealbumin', 'Hemoglobin', 'Magnesium', 'Globulin', 'WBC', 'Hematocrit', 'ALP', 'Plateletcrit', 'NSE']

def despine(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

def get_col(df, zh, en):
    """Return first matching column name."""
    for c in [zh, en]:
        if c in df.columns:
            return c
    for c in df.columns:
        stripped = c.lstrip('*★ ').strip()
        if stripped in (zh, en):
            return c
    return None

def save_fig(fig, name):
    for ext, dpi in [('.pdf', None), ('.png', 600)]:
        p = SUB_DIR / f'{name}{ext}'
        kw = {'dpi': dpi} if dpi else {}
        fig.savefig(p, bbox_inches='tight', facecolor='white', **kw)
        log(f'  Saved → {p.name}')
    plt.close(fig)
ADMIN_KW = ['时间', '号', '诊断', '科室', 'Unnamed', '上次', '入院', '出院', '年龄', '性别', '出生', '日期', '职工', '人均', '医生', '医院', '卫生院', '时间差', 'Center', 'MACE', 'FollowUp', 'ensemble', 'risk_group']

def is_lab(name):
    return not any((kw in name for kw in ADMIN_KW))
log('Loading WHU …')
whu_risk = pd.read_csv(WHU_RISK)
whu_raw = pd.read_csv(WHU_CSV, low_memory=False)
whu_raw.columns = [c.strip() for c in whu_raw.columns]
whu_raw = whu_raw.loc[:, ~whu_raw.columns.duplicated(keep='first')]
dept_df = whu_raw[[ID_COL, '入院科室']].copy() if '入院科室' in whu_raw.columns else None
whu_lab_cols = [c for c in whu_raw.columns if is_lab(c)]
for c in whu_lab_cols:
    whu_raw[c] = pd.to_numeric(whu_raw[c], errors='coerce')
whu_agg = whu_raw.groupby(ID_COL)[whu_lab_cols].first().reset_index()
whu = whu_risk[[ID_COL, 'MACE']].merge(whu_agg, on=ID_COL, how='left')
if dept_df is not None:
    dept_first = dept_df.groupby(ID_COL)['入院科室'].first().reset_index()
    whu = whu.merge(dept_first, on=ID_COL, how='left')
whu_mace = 'MACE'
log(f"  WHU shape: {whu.shape}  MACE rate={whu['MACE'].mean():.3f}")
log('Loading SCH …')
sch_risk = pd.read_csv(SCH_RISK)
for _chunk in pd.read_csv(SCH_CSV, chunksize=1, low_memory=False, nrows=1):
    _chunk.columns = [c.strip() for c in _chunk.columns]
    sch_cols_raw = _chunk.columns.tolist()
    break
sch_col_map = {}
for c in sch_cols_raw:
    s = c.lstrip('*★ ').strip()
    if s not in sch_col_map:
        sch_col_map[s] = c
whu_lab_set = set(whu_lab_cols)
target_stripped = [s for s in sch_col_map if is_lab(s) and s in whu_lab_set]
orig_to_stripped = {sch_col_map[s]: s for s in target_stripped}
canon_orig = list(orig_to_stripped.keys())
sch_chunks = []
for i, chunk in enumerate(pd.read_csv(SCH_CSV, chunksize=50000, low_memory=False)):
    chunk.columns = [c.strip() for c in chunk.columns]
    keep = [ID_COL] + [c for c in canon_orig if c in chunk.columns]
    sub = chunk[keep].copy().rename(columns=orig_to_stripped)
    for c in sub.columns:
        if c != ID_COL:
            sub[c] = pd.to_numeric(sub[c], errors='coerce')
    sch_chunks.append(sub)
    if (i + 1) % 3 == 0:
        log(f'  SCH chunk {i + 1} …')
sch_raw_all = pd.concat(sch_chunks, ignore_index=True)
sch_feat_cols = [c for c in sch_raw_all.columns if c != ID_COL]
sch_agg = sch_raw_all.groupby(ID_COL)[sch_feat_cols].first().reset_index()
sch = sch_risk[[ID_COL, 'MACE']].merge(sch_agg, on=ID_COL, how='left')
sch_mace = 'MACE'
log(f"  SCH shape: {sch.shape}  MACE rate={sch['MACE'].mean():.3f}")
log('=' * 60)
log('C1: Conditional distributions …')
fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.0))
fig.suptitle('Reversal Feature Distributions: MACE=1 vs MACE=0\n(z-scored within cohort;  BC = Bhattacharyya coefficient for MACE=1 overlap)', fontsize=9, fontweight='bold', y=0.995)
bhatt_rows = []
for idx, (feat_zh, feat_en) in enumerate(zip(REVERSAL_FEATS_ZH, REVERSAL_FEATS_EN)):
    ax = axes[idx // 3][idx % 3]
    plot_data = []
    mace1_vecs = {}
    for cohort_df, mace_col, cname in [(sch, sch_mace, 'SCH'), (whu, whu_mace, 'WHU')]:
        col = get_col(cohort_df, feat_zh, feat_en)
        if col is None:
            log(f'    {feat_en}: missing in {cname}')
            continue
        v = pd.to_numeric(cohort_df[col], errors='coerce')
        m = pd.to_numeric(cohort_df[mace_col], errors='coerce')
        mu, sd = (v.mean(), v.std())
        if sd < 1e-09:
            continue
        v_z = (v - mu) / sd
        for mval, mlabel in [(0, 'No MACE'), (1, 'MACE')]:
            sub = v_z[(m == mval) & v_z.notna()]
            if len(sub) < 5:
                continue
            for s in sub:
                plot_data.append({'group': f'{cname}\n{mlabel}', 'value': s})
            if mval == 1:
                mace1_vecs[cname] = sub.values
    if plot_data:
        pdf = pd.DataFrame(plot_data)
        order = ['SCH\nNo MACE', 'SCH\nMACE', 'WHU\nNo MACE', 'WHU\nMACE']
        order = [o for o in order if o in pdf['group'].values]
        pal = {'SCH\nNo MACE': C_SCH + 'AA', 'SCH\nMACE': C_MACE, 'WHU\nNo MACE': C_WHU + 'AA', 'WHU\nMACE': '#FF6F00'}
        sns.violinplot(data=pdf, x='group', y='value', order=order, palette=pal, ax=ax, cut=0, linewidth=0.7, inner='quartile')
    bc_str = ''
    if 'SCH' in mace1_vecs and 'WHU' in mace1_vecs:
        lo = min(mace1_vecs['SCH'].min(), mace1_vecs['WHU'].min())
        hi = max(mace1_vecs['SCH'].max(), mace1_vecs['WHU'].max())
        bins = np.linspace(lo, hi, 30)
        h1, _ = np.histogram(mace1_vecs['SCH'], bins=bins, density=True)
        h2, _ = np.histogram(mace1_vecs['WHU'], bins=bins, density=True)
        h1 /= h1.sum() + 1e-09
        h2 /= h2.sum() + 1e-09
        bc = float(np.sum(np.sqrt(h1 * h2)))
        bc_str = f'  BC={bc:.2f}'
        bhatt_rows.append({'Feature': feat_en, 'BC_MACE1_overlap': round(bc, 3)})
    panel_letter = chr(ord('A') + idx)
    ax.set_title(f'{panel_letter}  {feat_en}{bc_str}', fontsize=8, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('z-score', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.axhline(0, c='grey', lw=0.5, ls='--')
    despine(ax)
plt.tight_layout(rect=[0, 0, 1, 0.97])
save_fig(fig, 'FigureS19_ConditionalDistributions')
bc_df = pd.DataFrame(bhatt_rows)
bc_df.to_csv(ADD_DIR / 'C1_bhattacharyya.csv', index=False)
log(f'  Bhattacharyya coefficients:\n{bc_df.to_string(index=False)}')
log('=' * 60)
log('C2: Feature correlation networks …')
ALL_ZH = REVERSAL_FEATS_ZH + ['总胆固醇', '甘油三酯', '高密度脂蛋白胆固醇', 'LDL-C', '白蛋白', '乳酸脱氢酶', '肌酐', 'C反应蛋白']
ALL_EN = REVERSAL_FEATS_EN + ['T-Chol', 'TG', 'HDL-C', 'LDL-C', 'Albumin', 'LDH', 'Creatinine', 'CRP']

def build_corr(df, zh_list, en_list):
    fdf = {}
    used = []
    for zh, en in zip(zh_list, en_list):
        col = get_col(df, zh, en)
        if col:
            fdf[en] = pd.to_numeric(df[col], errors='coerce')
            used.append(en)
    if not fdf:
        return (pd.DataFrame(), [])
    fdf = pd.DataFrame(fdf).dropna(how='all')
    return (fdf.corr(method='spearman'), used)
sch_corr, sch_used = build_corr(sch, ALL_ZH, ALL_EN)
whu_corr, whu_used = build_corr(whu, ALL_ZH, ALL_EN)
common = [f for f in sch_used if f in whu_used]
log(f'  Common features for correlation: {len(common)}')
sch_c = sch_corr.loc[common, common]
whu_c = whu_corr.loc[common, common]
dif_c = whu_c - sch_c
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle('Feature Correlation Structure: SCH vs WHU  (Spearman rho)', fontsize=9, fontweight='bold')
kw = dict(annot=True, fmt='.1f', linewidths=0.3, annot_kws={'size': 6}, cmap='RdBu_r', vmin=-1, vmax=1, square=True, cbar=False)
kwd = dict(annot=True, fmt='.1f', linewidths=0.3, annot_kws={'size': 6}, cmap='PuOr', vmin=-1, vmax=1, square=True)
sns.heatmap(sch_c, ax=axes[0], **kw)
sns.heatmap(whu_c, ax=axes[1], **kw)
sns.heatmap(dif_c, ax=axes[2], **kwd)
for ax, title in zip(axes, ['A  SCH', 'B  WHU', 'C  Δ (WHU − SCH)']):
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=6.5)
rev_set = set(REVERSAL_FEATS_EN)
for ax in axes:
    for lbl in ax.get_xticklabels():
        if lbl.get_text() in rev_set:
            lbl.set_color(C_MACE)
            lbl.set_fontweight('bold')
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in rev_set:
            lbl.set_color(C_MACE)
            lbl.set_fontweight('bold')
plt.tight_layout()
save_fig(fig, 'FigureS20_CorrelationNetworks')
dif_flat = dif_c.where(np.triu(np.ones(dif_c.shape), k=1).astype(bool))
dif_stack = dif_flat.stack().reset_index()
dif_stack.columns = ['Feat1', 'Feat2', 'Delta_rho']
dif_stack['abs_delta'] = dif_stack['Delta_rho'].abs()
dif_stack = dif_stack.sort_values('abs_delta', ascending=False).head(20)
dif_stack.to_csv(ADD_DIR / 'C2_top_correlation_shifts.csv', index=False)
log(f'  Top 5 correlation shifts:\n{dif_stack.head(5).to_string(index=False)}')
log('=' * 60)
log('C3: UMAP joint embedding …')
try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    log('  umap-learn not installed — installing …')
    import subprocess
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', 'umap-learn'], capture_output=True, text=True)
    log(r.stdout[-300:] if r.stdout else '')
    try:
        import umap as umap_lib
        HAS_UMAP = True
    except Exception as e:
        log(f'  UMAP unavailable: {e}')
        HAS_UMAP = False
if HAS_UMAP:

    def build_mat(df, mace_col):
        cols = {}
        for zh, en in zip(REVERSAL_FEATS_ZH, REVERSAL_FEATS_EN):
            col = get_col(df, zh, en)
            if col:
                cols[en] = pd.to_numeric(df[col], errors='coerce')
        fdf = pd.DataFrame(cols)
        mask = fdf.notna().sum(axis=1) >= max(4, len(cols) // 2)
        fdf = fdf[mask].fillna(fdf.median())
        fdf = (fdf - fdf.mean()) / (fdf.std() + 1e-09)
        mace = pd.to_numeric(df.loc[fdf.index, mace_col], errors='coerce').fillna(0).astype(int)
        return (fdf.reset_index(drop=True), mace.reset_index(drop=True))
    sch_fmat, sch_mv = build_mat(sch, sch_mace)
    whu_fmat, whu_mv = build_mat(whu, whu_mace)
    np.random.seed(42)
    n_sch = min(5000, len(sch_fmat))
    idx = np.random.choice(len(sch_fmat), n_sch, replace=False)
    sch_s, sch_ms = (sch_fmat.iloc[idx], sch_mv.iloc[idx])
    common_f = [f for f in sch_s.columns if f in whu_fmat.columns]
    joint = pd.concat([sch_s[common_f], whu_fmat[common_f]], ignore_index=True)
    cohort_l = ['SCH'] * len(sch_s) + ['WHU'] * len(whu_fmat)
    mace_l = pd.concat([sch_ms, whu_mv], ignore_index=True).values
    log(f'  UMAP input: {len(joint)} patients × {len(common_f)} features …')
    reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.3, random_state=42, n_jobs=1)
    emb = reducer.fit_transform(joint.values)
    log('  UMAP done.')
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    fig.suptitle('UMAP Joint Embedding of Reversal Features (SCH + WHU)', fontsize=9, fontweight='bold')
    col_arr = [C_SCH if c == 'SCH' else C_WHU for c in cohort_l]
    axes[0].scatter(emb[:, 0], emb[:, 1], c=col_arr, s=3, alpha=0.35, rasterized=True)
    axes[0].set_title('A  Coloured by cohort', fontsize=9, fontweight='bold')
    axes[0].legend(handles=[mpatches.Patch(color=C_SCH, label=f'SCH (n={n_sch})'), mpatches.Patch(color=C_WHU, label=f'WHU (n={len(whu_fmat)})')], fontsize=7)
    axes[0].set_xlabel('UMAP-1')
    axes[0].set_ylabel('UMAP-2')
    despine(axes[0])
    for cname, marker, ms in [('SCH', 'o', 4), ('WHU', '^', 10)]:
        mask = np.array(cohort_l) == cname
        for mval, color, mlabel in [(0, C_NOMA + '88', 'No MACE'), (1, C_MACE, 'MACE')]:
            m2 = mask & (mace_l == mval)
            axes[1].scatter(emb[m2, 0], emb[m2, 1], c=color, marker=marker, s=ms, alpha=0.6, rasterized=True, label=f'{cname} {mlabel}')
    axes[1].set_title('B  Coloured by MACE outcome', fontsize=9, fontweight='bold')
    axes[1].legend(fontsize=7, markerscale=2, ncol=2)
    axes[1].set_xlabel('UMAP-1')
    axes[1].set_ylabel('UMAP-2')
    despine(axes[1])
    plt.tight_layout()
    save_fig(fig, 'FigureS21_UMAP_JointEmbedding')
    pd.DataFrame({'UMAP1': emb[:, 0], 'UMAP2': emb[:, 1], 'cohort': cohort_l, 'MACE': mace_l}).to_csv(ADD_DIR / 'C3_umap_embeddings.csv', index=False)
    log('  UMAP embeddings saved.')
    for cohort in ['SCH', 'WHU']:
        mask = (np.array(cohort_l) == cohort) & (mace_l == 1)
        if mask.sum() > 0:
            cx, cy = (emb[mask, 0].mean(), emb[mask, 1].mean())
            log(f'  MACE=1 centroid [{cohort}]: ({cx:.2f}, {cy:.2f})')
log('=' * 60)
log('C4: NSE + WBC mechanism …')
fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.5))
fig.suptitle('NSE and WBC: Mechanistic Analysis of Top Ablation Drivers', fontsize=9, fontweight='bold')
c4_rows = []
for row_i, (feat_zh, feat_en) in enumerate([('神经元特异性烯醇化酶', 'NSE'), ('白细胞', 'WBC')]):
    ax = axes[row_i][0]
    for cohort_df, mace_col, cname, color in [(sch, sch_mace, 'SCH', C_SCH), (whu, whu_mace, 'WHU', C_WHU)]:
        col = get_col(cohort_df, feat_zh, feat_en)
        if col is None:
            continue
        v = pd.to_numeric(cohort_df[col], errors='coerce')
        m = pd.to_numeric(cohort_df[mace_col], errors='coerce')
        valid = v.notna() & m.notna()
        v, m = (v[valid], m[valid])
        if len(v) < 30:
            continue
        try:
            t = pd.qcut(v, 3, labels=False, duplicates='drop')
        except Exception:
            continue
        n_bins = t.nunique()
        if n_bins < 2:
            continue
        bin_labels = {0: 'T1 (Low)', 1: 'T2 (Mid)', 2: 'T3 (High)'}
        bin_lbl_list = [bin_labels[i] for i in range(n_bins)]
        t_named = t.map({i: lbl for i, lbl in enumerate(bin_lbl_list)})
        rate = m.groupby(t_named).mean() * 100
        all_lbls = ['T1 (Low)', 'T2 (Mid)', 'T3 (High)']
        rate = rate.reindex(all_lbls).fillna(0)
        xs = np.arange(len(rate)) + (0.2 if cname == 'WHU' else -0.2)
        ax.bar(xs, rate.values, width=0.35, color=color, alpha=0.85, label=cname)
        rho, p = stats.spearmanr(v, m)
        c4_rows.append({'Feature': feat_en, 'Cohort': cname, 'rho': round(rho, 3), 'p': round(p, 4), 'T1_pct': round(rate.iloc[0], 2) if len(rate) > 0 else np.nan, 'T3_pct': round(rate.iloc[-1], 2) if len(rate) > 0 else np.nan})
    _panel = chr(ord('A') + row_i * 3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['T1\n(Low)', 'T2\n(Mid)', 'T3\n(High)'])
    ax.set_ylabel('MACE rate (%)')
    ax.set_xlabel('')
    ax.set_title(f'{_panel}  {feat_en} - Tertile MACE rates', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    despine(ax)
    ax = axes[row_i][1]
    corr_vals, corr_feats = ([], [])
    for zh2, en2 in zip(REVERSAL_FEATS_ZH, REVERSAL_FEATS_EN):
        if en2 == feat_en:
            continue
        col1 = get_col(whu, feat_zh, feat_en)
        col2 = get_col(whu, zh2, en2)
        if col1 is None or col2 is None:
            continue
        v1 = pd.to_numeric(whu[col1], errors='coerce')
        v2 = pd.to_numeric(whu[col2], errors='coerce')
        ok = v1.notna() & v2.notna()
        if ok.sum() < 20:
            continue
        rho, _ = stats.spearmanr(v1[ok], v2[ok])
        corr_vals.append(rho)
        corr_feats.append(en2)
    if corr_vals:
        colors = [C_MACE if v > 0 else C_SCH for v in corr_vals]
        yp = np.arange(len(corr_feats))
        ax.barh(yp, corr_vals, color=colors, alpha=0.85)
        ax.set_yticks(yp)
        ax.set_yticklabels(corr_feats, fontsize=7)
        ax.axvline(0, c='k', lw=0.8)
        _panel_b = chr(ord('A') + row_i * 3 + 1)
        ax.set_title(f'{_panel_b}  {feat_en} vs reversal features (WHU rho)', fontsize=9, fontweight='bold')
        ax.set_xlabel('Spearman rho')
    despine(ax)
    ax = axes[row_i][2]
    for cohort_df, mace_col, cname, ls in [(sch, sch_mace, 'SCH', '-'), (whu, whu_mace, 'WHU', '--')]:
        col = get_col(cohort_df, feat_zh, feat_en)
        if col is None:
            continue
        v = pd.to_numeric(cohort_df[col], errors='coerce')
        m = pd.to_numeric(cohort_df[mace_col], errors='coerce')
        mu, sd = (v[v.notna()].mean(), v[v.notna()].std())
        if sd < 1e-09:
            continue
        v_z = (v - mu) / sd
        for mval, color, mlabel in [(0, C_NOMA, 'No MACE'), (1, C_MACE, 'MACE')]:
            sub = v_z[(m == mval) & v_z.notna()]
            if len(sub) < 10:
                continue
            sub.plot.kde(ax=ax, color=color, ls=ls, lw=1.8, label=f'{cname} {mlabel}')
    ax.set_xlabel('z-score')
    ax.set_ylabel('Density')
    _panel_c = chr(ord('A') + row_i * 3 + 2)
    ax.set_title(f'{_panel_c}  {feat_en} - KDE by MACE status', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6.5)
    ax.axvline(0, c='grey', lw=0.5, ls=':')
    despine(ax)
plt.tight_layout()
save_fig(fig, 'FigureS22_NSE_WBC_Mechanism')
c4_df = pd.DataFrame(c4_rows)
c4_df.to_csv(ADD_DIR / 'C4_NSE_WBC_results.csv', index=False)
log(f'  C4 results:\n{c4_df.to_string(index=False)}')
log('=' * 60)
log('All C1–C4 analyses complete.')
_LOG.close()
