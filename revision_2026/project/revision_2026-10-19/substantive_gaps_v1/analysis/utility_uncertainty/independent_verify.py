"""Direct threshold-count replay; does not import the analysis implementation."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
SOURCE = Path(_release_path('project', 'revision_2026-10-15/original_v5_independent_validation_v1'))

def sha(p):
    return None

def records(name):
    with (ROOT / name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))
cfg = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text()))
curve = records('dca_curves.csv')
contrast = records('dca_contrasts.csv')
nri = records('nri_continuous.csv')
cat = records('nri_categorical.csv')
summaries = []
for run in cfg['runs']:
    with np.load(SOURCE / 'private' / 'runs' / run / 'predictions.npz') as z:
        y, groups = (z['y'], z['groups'])
        probabilities = {m: z[m] for m in cfg['decision_curve_models']}
    with np.load(ROOT / f'{run}_aggregate_bootstrap.npz') as z:
        boot = z['net_benefit']
        nb_nri, nb_cat = (z['continuous_nri'], z['categorical_nri'])
        thresholds, strategies = (z['thresholds'], z['strategies'].tolist())
    positive, negative = (y == 1, y == 0)
    points = {}
    max_error = 0.0
    for model in probabilities:
        p = probabilities[model]
        points[model] = np.array([(np.count_nonzero(positive & (p >= t)) - np.count_nonzero(negative & (p >= t)) * t / (1 - t)) / len(y) for t in thresholds])
        for r in [a for a in curve if a['run'] == run and a['model'] == model]:
            ix = int(round(float(r['threshold']) * 100)) - 1
            lo, hi = np.percentile(boot[:, strategies.index(model), ix], [2.5, 97.5])
            error = max(abs(float(r['estimate']) - points[model][ix]), abs(float(r['ci_low']) - lo), abs(float(r['ci_high']) - hi))
            max_error = max(max_error, error)
    points['treat_all'] = y.mean() - (1 - y.mean()) * thresholds / (1 - thresholds)
    points['treat_none'] = thresholds * 0
    for r in [a for a in contrast if a['run'] == run]:
        a, b = r['contrast'].split('_minus_')
        ix = int(round(float(r['threshold']) * 100)) - 1
        lo, hi = np.percentile(boot[:, strategies.index(a), ix] - boot[:, strategies.index(b), ix], [2.5, 97.5])
        error = max(abs(float(r['estimate']) - (points[a][ix] - points[b][ix])), abs(float(r['ci_low']) - lo), abs(float(r['ci_high']) - hi))
        max_error = max(max_error, error)
    p, q = (probabilities['primary_raw'], probabilities['age'])
    direct_nri = np.array([np.mean(p[positive] > q[positive]) - np.mean(p[positive] < q[positive]), np.mean(p[negative] < q[negative]) - np.mean(p[negative] > q[negative])])
    direct_nri = np.r_[direct_nri, direct_nri.sum()]
    components = ['positive_label_component', 'negative_label_component', 'total']
    for r in [a for a in nri if a['run'] == run]:
        ix = components.index(r['component'])
        lo, hi = np.percentile(nb_nri[:, ix], [2.5, 97.5])
        max_error = max(max_error, abs(float(r['estimate']) - direct_nri[ix]), abs(float(r['ci_low']) - lo), abs(float(r['ci_high']) - hi))
    for r in [a for a in cat if a['run'] == run]:
        t = float(r['threshold'])
        ix = int(round(t * 100)) - 1
        delta = (p >= t).astype(int) - (q >= t).astype(int)
        values = [np.mean(delta[positive]), -np.mean(delta[negative])]
        values.append(sum(values))
        j = components.index(r['component'])
        lo, hi = np.percentile(nb_cat[:, j, ix], [2.5, 97.5])
        max_error = max(max_error, abs(float(r['estimate']) - values[j]), abs(float(r['ci_low']) - lo), abs(float(r['ci_high']) - hi))
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(cfg['bootstrap']['seed'])
    for rep in range(20):
        sampled = rng.choice(len(unique), size=len(unique), replace=True)
        multiplicity = np.bincount(sampled, minlength=len(unique))[inverse]
        expanded = np.repeat(np.arange(len(y)), multiplicity)
        ey = y[expanded]
        for model in probabilities:
            ep = probabilities[model][expanded]
            for ix in (0, 4, 9, 19, 29, 49, 98):
                t = thresholds[ix]
                direct = (np.count_nonzero((ey == 1) & (ep >= t)) - np.count_nonzero((ey == 0) & (ep >= t)) * t / (1 - t)) / len(expanded)
                max_error = max(max_error, abs(direct - boot[rep, strategies.index(model), ix]))
        ep, eq = (p[expanded], q[expanded])
        a, b = (ey == 1, ey == 0)
        c = [np.mean(ep[a] > eq[a]) - np.mean(ep[a] < eq[a]), np.mean(ep[b] < eq[b]) - np.mean(ep[b] > eq[b])]
        c.append(sum(c))
        max_error = max(max_error, float(np.max(np.abs(c - nb_nri[rep]))))
    assert max_error < 1e-12
    summaries.append({'run': run, 'max_absolute_error': max_error, 'all_99_threshold_points_checked': True, 'all_reported_percentile_intervals_checked': True, 'explicit_repeated_patient_rows_bootstraps_checked': 20, 'continuous_and_categorical_nri_components_checked': True})
result = {'passed': True, 'analysis_module_imported': False, 'runs': summaries, 'privacy': 'Aggregate results only'}
with (ROOT / 'independent_verification.json').open('x', encoding='utf-8') as f:
    json.dump(result, f, indent=2)
    f.write('\n')
manifest = [{'path': p.name, 'bytes': p.stat().st_size} for p in sorted(ROOT.iterdir()) if p.is_file() and p.name != 'output_manifest.json']
with (ROOT / 'output_manifest.json').open('x', encoding='utf-8') as f:
    json.dump({'files': manifest}, f, indent=2)
    f.write('\n')
print(json.dumps(result))
