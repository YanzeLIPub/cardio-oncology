"""Recover the two existing S25 scatter layers using the historical data rules."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import sys, json
R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pymupdf as fitz
sys.path.pop(0)
BASE = Path(_release_path('pipeline', ''))
ID = '病案号'
LABS = ['神经元特异性烯醇化酶', '白蛋白']
Q = R / 'review'
Q.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'figure.dpi': 150})
fig = plt.figure(figsize=(15, 12.5))
gs = GridSpec(3, 4, figure=fig, hspace=0.72, wspace=0.52)
axes = [fig.add_subplot(gs[0, i]) for i in [0, 1]]
records = []
paths = []
for i, c in enumerate(['SCH', 'WHU']):
    src = BASE / ('SCH_processed_data.csv' if c == 'SCH' else 'WHU_Cancer_processed_data.csv')
    header = list(pd.read_csv(src, nrows=0).columns)
    if c == 'SCH':
        mapping = {x.lstrip('*★ ').strip(): x for x in header}
        cols = [mapping[x] for x in LABS]
    else:
        cols = LABS
    frames = []
    for chunk in pd.read_csv(src, usecols=[ID] + cols, chunksize=50000, low_memory=False):
        chunk = chunk.rename(columns=dict(zip(cols, LABS)))
        for col in LABS:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        frames.append(chunk)
    agg = pd.concat(frames, ignore_index=True).groupby(ID)[LABS].first().reset_index()
    risk = BASE / f'data/patient_risk_{c}.csv'
    d = pd.read_csv(risk, usecols=[ID, 'MACE']).merge(agg, on=ID, how='left').dropna(subset=LABS + ['MACE'])
    x = (d[LABS[0]] - d[LABS[0]].mean()) / (d[LABS[0]].std() + 1e-09)
    y = (d[LABS[1]] - d[LABS[1]].mean()) / (d[LABS[1]].std() + 1e-09)
    rho = d[LABS[0]].rank().corr(d[LABS[1]].rank())
    assert round(rho, 3) == {'SCH': -0.073, 'WHU': 0.114}[c], (c, rho)
    ax = axes[i]
    for val, col in [(0, '#4D922199'), (1, '#B2182B')]:
        keep = d.MACE == val
        ax.scatter(x[keep], y[keep], c=col, s=6 if val == 0 else 18, alpha=0.4 if val == 0 else 0.7, rasterized=True)
    b0, b1 = np.polynomial.polynomial.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, b0 + b1 * xs, color=['#2166AC', '#D6604D'][i], lw=1.5, ls='--', alpha=0.9)
    records.append({'cohort': c, 'complete_pairs': len(d), 'positive_pairs': int(d.MACE.sum()), 'spearman_rho': rho, 'x_limits': list(ax.get_xlim()), 'y_limits': list(ax.get_ylim()), 'source_csv': src.as_posix(), 'source_risk': risk.as_posix(), 'source_columns': cols, 'aggregation': 'First nonmissing value in original file order, within original risk subset'})
    print(c, 'scatter pairs', len(d), 'rho', round(rho, 6), flush=True)
    paths += [src.as_posix(), risk.as_posix()]
fig.savefig(Q / 'S25_raster_replay.pdf', facecolor='white')
for ax in axes:
    for line in ax.lines:
        line.set_visible(False)
    ax.set_axis_off()
    ax.patch.set_visible(False)
    for collection in ax.collections:
        collection.set_rasterized(False)
fig.patch.set_visible(False)
fig.savefig(Q / 'S25_scatter_layers.pdf', transparent=True)
plt.close(fig)
old = fitz.open(ROOT / 'FigureS9_NSE_Albumin_Collinearity.pdf')
rep = fitz.open(Q / 'S25_raster_replay.pdf')
old_images = [v for v in old[0].get_images() if v[2] > 100 and v[3] > 100]
rep_images = [v for v in rep[0].get_images() if v[2] > 100 and v[3] > 100]
assert len(old_images) == len(rep_images) == 2
for record, a, b in zip(records, old_images, rep_images):
    pa, pb = (fitz.Pixmap(old, a[0]), fitz.Pixmap(rep, b[0]))
    record['old_image_size'] = [pa.width, pa.height]
    record['replay_image_size'] = [pb.width, pb.height]
    record['raster_rgb_exact_match'] = pa.samples == pb.samples
    record['source_plot_box'] = list(rep[0].get_image_rects(b[0])[0])
    pa.save(Q / f"S25_{record['cohort']}_old.png")
    pb.save(Q / f"S25_{record['cohort']}_replay.png")
print(json.dumps(records, ensure_ascii=False), flush=True)
(Q / 'scatter_recovery.json').write_text(json.dumps({'panels': records, 'historical_script': _release_source_path(BASE / 'Manuscript/v13/figures/add/nse_albumin_deep_dive.py').as_posix(), 'model_fitting': False, 'versions': {'python': sys.version.split()[0], 'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__}}, ensure_ascii=False, indent=2), encoding='utf-8')
