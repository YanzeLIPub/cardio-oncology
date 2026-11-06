"""Re-export three historical SVG-only plots, preserving dashes and alpha."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import sys, ast, csv, json
R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
Q = R / 'review'
LAND = ROOT / 'revision_2026-10-15/landmark_completion_v1'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
METRICS = ROOT / 'revision_2026-10-15/original_v5_independent_validation_v1/results/validation_metrics.csv'
metrics = list(csv.DictReader(METRICS.open(encoding='utf-8-sig')))

def metric(run, model, kind='auc'):
    r = next((r for r in metrics if r['run'] == run and r['model'] == model and (r['metric'] == kind)))
    return np.array([float(r[k]) for k in ['estimate', 'ci_low', 'ci_high']])

def save(fig, n):
    assert n == 3
    fig.savefig(Q / 'replotted_S4.pdf', facecolor='white')
    fig.savefig(Q / 'replotted_S4.svg', facecolor='white')
    plt.close(fig)
code = _release_source_path(LAND / 'main/build_figures.py').read_text(encoding='utf-8')
chunk = code[code.index('models = ["primary_raw"'):code.index('calmodels =')]
env = {'np': np, 'plt': plt, 'metric': metric, 'save': save, 'RUNS': ['SCH_original', 'WHU_original', 'MIMIC_original'], 'TITLES': ['SCH\n38,509 patients', 'WHU\n2,562 patients', 'MIMIC-III\n9,946 admissions / 6,783 patients']}
exec(compile(chunk, 'frozen_S4_plot', 'exec'), env)
source = _release_source_path(LAND / 'mimic/summarize_windows.py')
tree = ast.parse(source.read_text(encoding='utf-8'))
names = next((ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'CSV_NAMES' for t in n.targets))))
tables = {name: pd.read_csv(LAND / 'mimic/results' / name) for name in names}
fun = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'figures'))
original_save = plt.Figure.savefig

def redirected_save(fig, path, *args, **kw):
    path = Path(path)
    if path.suffix != '.png':
        return
    key = 'S14' if path.stem.endswith('_AUC') else 'S13'
    kw.pop('dpi', None)
    original_save(fig, Q / f'replotted_{key}.pdf', **kw)
    original_save(fig, Q / f'replotted_{key}.svg', **kw)
plt.Figure.savefig = redirected_save
env = {'np': np, 'ROOT': LAND / 'mimic', 'CSV_NAMES': names}
exec(compile(ast.Module(body=[fun], type_ignores=[]), str(source), 'exec'), env)
env['figures'](tables)
plt.Figure.savefig = original_save
(Q / 'replotted_source_files.json').write_text(json.dumps({'S4': {'script': _release_source_path(LAND / 'main/build_figures.py').as_posix(), 'aggregate': METRICS.as_posix()}, 'S13_S14': {'script': source.as_posix(), 'aggregates': [(LAND / 'mimic/results' / name).as_posix() for name in names]}, 'model_fitting': False, 'confidence_intervals_recalculated': False}, indent=2), encoding='utf-8')
print('Three plots exported directly from historical plot code and frozen aggregate tables.')
