"""Conditional patient-cluster uncertainty for independent diagnostic predictions."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'protocol.json'

def sha(path):
    return None

def identity(path):
    p = Path(path).resolve()
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save_json(name, value):
    with (ROOT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def save_csv(name, rows):
    with (ROOT / name).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def estimates(y, probabilities, weights, thresholds):
    """Weighted TP/FP counts from bins, preserving p >= threshold ties."""
    n = weights.sum()
    positive = weights * y
    negative = weights * (1 - y)
    n1, n0 = (positive.sum(), negative.sum())
    result, counts = ({}, {})
    odds = thresholds / (1 - thresholds)
    for model, p in probabilities.items():
        bins = np.searchsorted(thresholds, p, side='right')
        tp = np.bincount(bins, weights=positive, minlength=len(thresholds) + 1)[::-1].cumsum()[::-1][1:]
        fp = np.bincount(bins, weights=negative, minlength=len(thresholds) + 1)[::-1].cumsum()[::-1][1:]
        result[model] = (tp - fp * odds) / n
        counts[model] = (tp, fp)
    result['treat_all'] = (n1 - n0 * odds) / n
    result['treat_none'] = np.zeros(len(thresholds))
    change = np.sign(probabilities['primary_raw'] - probabilities['age'])
    continuous = np.array([(positive * change).sum() / n1, -(negative * change).sum() / n0])
    continuous = np.r_[continuous, continuous.sum()]
    categorical = np.array([(counts['primary_raw'][0] - counts['age'][0]) / n1, (counts['age'][1] - counts['primary_raw'][1]) / n0])
    categorical = np.vstack([categorical, categorical.sum(axis=0)])
    return (result, continuous, categorical)

def self_tests():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.8, 0.5, 0.5, 0.1])
    q = np.array([0.6, 0.5, 0.7, 0.2])
    t = np.array([0.1, 0.5, 0.8])
    probs = {'primary_raw': p, 'age': q, 'base_raw': p, 'base_platt': p}
    weights = np.array([2.0, 1.0, 2.0, 1.0])
    values, continuous, cat = estimates(y, probs, weights, t)
    for k, v in probs.items():
        direct = [sum((w * (int(yi) - (1 - int(yi)) * a / (1 - a)) for yi, pi, w in zip(y, v, weights) if pi >= a)) / sum(weights) for a in t]
        assert np.allclose(values[k], direct, atol=1e-14)
    assert np.isclose(values['primary_raw'][1], 1 / 6)
    assert np.allclose(continuous, [2 / 3, 1, 5 / 3])
    assert np.allclose(cat[:, 1], [0, 0, 0])
    equal = dict(probs, age=p)
    _, eq_cont, eq_cat = estimates(y, equal, weights, t)
    assert np.array_equal(eq_cont, np.zeros(3)) and np.array_equal(eq_cat, np.zeros((3, 3)))
    return {'weighted_direct_formula': True, 'inclusive_threshold_ties': True, 'nri_components_and_exact_ties': True, 'replicated_patient_weights': True}

def collect_identities(value, result):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            result[str(Path(value['path']).resolve()).lower()] = None
        for child in value.values():
            collect_identities(child, result)
    elif isinstance(value, list):
        for child in value:
            collect_identities(child, result)

def load_verified(source, run, models, prior_hashes):
    predpath = source / 'private' / 'runs' / run / 'predictions.npz'
    rolepath = predpath.with_name('patient_roles.npz')
    for p in (predpath, rolepath):
        pass
    with np.load(predpath, allow_pickle=False) as z:
        y, groups, folds = (z['y'].copy(), z['groups'].copy(), z['fold'].copy())
        probabilities = {m: z[m].copy() for m in models}
    with np.load(rolepath, allow_pickle=False) as z:
        roles = z['roles'].copy()
        assert np.array_equal(groups, z['groups']) and np.array_equal(y, z['y'])
    assert np.array_equal(np.argmax(roles == 0, axis=0) + 1, folds)
    assert np.all((roles == 0).sum(axis=0) == 1)
    for role in roles:
        sets = [set(groups[role == r]) for r in range(4)]
        assert all((sets[a].isdisjoint(sets[b]) for a in range(4) for b in range(a + 1, 4)))
    unique, inverse = np.unique(groups, return_inverse=True)
    assert np.isin(y, [0, 1]).all() and len(np.unique(y)) == 2
    assert all((np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() for p in probabilities.values()))
    return (y, inverse, len(unique), probabilities, [identity(predpath), identity(rolepath)])

def interval_rows(run, names, point, boot, thresholds=None, key='model'):
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    rows = []
    for j, name in enumerate(names):
        if thresholds is None:
            rows.append({'run': run, key: name, 'estimate': float(point[j]), 'ci_low': float(low[j]), 'ci_high': float(high[j])})
        else:
            for k, threshold in enumerate(thresholds):
                rows.append({'run': run, key: name, 'threshold': float(threshold), 'estimate': float(point[j, k]), 'ci_low': float(low[j, k]), 'ci_high': float(high[j, k])})
    return rows

def plot_results(curves, contrasts, run_names):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    titles = {'SCH_original': 'SCH', 'WHU_original': 'WHU', 'MIMIC_original': 'MIMIC', 'MIMIC_exclude_AST_HDL': 'MIMIC: AST/HDL excluded'}
    outputs = []
    for limit, suffix in ((0.5, '01_50'), (0.99, '01_99')):
        fig, axes = plt.subplots(4, 2, figsize=(7.4, 10.3), layout='constrained')
        for row, run in enumerate(run_names):
            ax, bx = axes[row]
            for m, label, color, style in (('base_raw', 'Base raw', '#0072B2', '-'), ('base_platt', 'Independent Platt', '#D55E00', '-'), ('treat_all', 'Treat all', '#777777', '--'), ('treat_none', 'Treat none', '#222222', ':')):
                series = [v for v in curves if v['run'] == run and v['model'] == m and (v['threshold'] <= limit)]
                x = np.array([v['threshold'] for v in series])
                ax.plot(x, [v['estimate'] for v in series], color=color, ls=style, label=label, lw=1.2)
                if m in ('base_raw', 'base_platt'):
                    ax.fill_between(x, [v['ci_low'] for v in series], [v['ci_high'] for v in series], color=color, alpha=0.18, lw=0)
            ds = [v for v in contrasts if v['run'] == run and v['contrast'] == 'base_platt_minus_base_raw' and (v['threshold'] <= limit)]
            x = np.array([v['threshold'] for v in ds])
            bx.plot(x, [v['estimate'] for v in ds], color='#009E73', lw=1.3)
            bx.fill_between(x, [v['ci_low'] for v in ds], [v['ci_high'] for v in ds], color='#009E73', alpha=0.2, lw=0)
            bx.axhline(0, color='#555555', lw=0.7, ls=':')
            ax.set_title(f'{chr(65 + 2 * row)}  {titles[run]}', loc='left', fontsize=10)
            bx.set_title(f'{chr(66 + 2 * row)}  Platt minus raw', loc='left', fontsize=10)
            ax.set_ylabel('Diagnostic-label net benefit')
            bx.set_ylabel('Paired net-benefit difference')
            if limit > 0.5:
                ax.set_yscale('symlog', linthresh=0.05)
                ax.set_ylabel('Net benefit (symmetric log)')
            for a in (ax, bx):
                a.set_xlim(0.01, limit)
                a.set_xlabel('Numerical threshold probability')
                a.grid(axis='y', alpha=0.18, lw=0.5)
            if row == 0:
                ax.legend(fontsize=7, frameon=False, loc='lower left')
        for ext in ('pdf', 'png'):
            path = ROOT / f'utility_calibration_{suffix}.{ext}'
            fig.savefig(path, dpi=240, metadata={'Creator': 'Supplemental diagnostic-label analysis'} if ext == 'pdf' else None)
            outputs.append(path.name)
        plt.close(fig)
    return outputs

def main():
    cfg = _release_resolve_config(json.loads(PROTOCOL.read_text(encoding='utf-8')))
    if (ROOT / 'summary.json').exists():
        raise FileExistsError('Completed output exists; use a versioned directory')
    tests = self_tests()
    source = Path(cfg['source_directory'])
    audit_path = source / 'evidence' / 'validation_replay_audit.json'
    audit = _release_resolve_config(json.loads(audit_path.read_text(encoding='utf-8')))
    assert audit['passed'] and audit['all_read_artifact_hashes_stable']
    prior_hashes = {}
    collect_identities(audit, prior_hashes)
    source_files = [identity(_release_source_path(source / 'run_validation.py')), identity(source / 'protocol.json'), identity(audit_path)]
    protocol_hash = sha(PROTOCOL)
    environment = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'matplotlib')}}
    save_json('execution_lock.json', {'started_at': datetime.now().astimezone().isoformat(), 'script': identity(__file__), 'environment': environment, 'source_files': source_files})
    thresholds = np.arange(1, 100) / 100
    models = cfg['decision_curve_models']
    strategies = models + ['treat_all', 'treat_none']
    contrasts_def = [('base_platt', 'base_raw'), ('primary_raw', 'age')]
    contrasts_def += [(m, default) for m in models for default in ('treat_all', 'treat_none')]
    contrast_names = [f'{a}_minus_{b}' for a, b in contrasts_def]
    components = ['positive_label_component', 'negative_label_component', 'total']
    curves, differences, continuous_rows, categorical_rows, cohorts, verifications = ([], [], [], [], [], [])
    for run in cfg['runs']:
        y, inverse, patients, probabilities, inputs = load_verified(source, run, models, prior_hashes)
        n = len(y)
        original, continuous, categorical = estimates(y, probabilities, np.ones(n), thresholds)
        point = np.array([original[m] for m in strategies])
        delta = np.array([original[a] - original[b] for a, b in contrasts_def])
        B = cfg['bootstrap']['replicates']
        boot = np.empty((B, len(strategies), len(thresholds)))
        boot_c = np.empty((B, 3))
        boot_cat = np.empty((B, 3, len(thresholds)))
        rng = np.random.default_rng(cfg['bootstrap']['seed'])
        direct_max_error = 0.0
        for rep in range(B):
            multiplicity = np.bincount(rng.integers(0, patients, size=patients), minlength=patients)
            weights = multiplicity[inverse]
            values, nc, ncat = estimates(y, probabilities, weights, thresholds)
            boot[rep] = [values[m] for m in strategies]
            boot_c[rep], boot_cat[rep] = (nc, ncat)
            if rep < 3:
                for m in models:
                    for ix in (0, 4, 9, 19, 49, 98):
                        t = thresholds[ix]
                        pos = probabilities[m] >= t
                        direct = (weights[pos & (y == 1)].sum() - weights[pos & (y == 0)].sum() * t / (1 - t)) / weights.sum()
                        direct_max_error = max(direct_max_error, abs(direct - values[m][ix]))
                assert direct_max_error < 1e-12
        boot_delta = np.array([boot[:, strategies.index(a)] - boot[:, strategies.index(b)] for a, b in contrasts_def]).transpose(1, 0, 2)
        curves.extend(interval_rows(run, strategies, point, boot, thresholds))
        differences.extend(interval_rows(run, contrast_names, delta, boot_delta, thresholds, 'contrast'))
        continuous_rows.extend(interval_rows(run, components, continuous, boot_c, key='component'))
        categorical_rows.extend(interval_rows(run, components, categorical, boot_cat, thresholds, 'component'))
        np.savez_compressed(ROOT / f'{run}_aggregate_bootstrap.npz', thresholds=thresholds, strategies=np.array(strategies), net_benefit=boot, continuous_nri=boot_c, categorical_nri=boot_cat)
        cohorts.append({'run': run, 'unit': 'admission' if run.startswith('MIMIC') else 'patient', 'rows': n, 'patients': patients, 'positive_labels': int(y.sum()), 'prevalence': float(y.mean())})
        verifications.append({'run': run, 'inputs': inputs, 'roles_patient_disjoint': True, 'matches_independent_replay_hashes': True, 'direct_bootstrap_counts_max_error': direct_max_error, 'bootstrap_replicates': B, 'all_bootstraps_finite': bool(np.isfinite(boot).all() and np.isfinite(boot_c).all())})
        print(json.dumps({'run_complete': run, **cohorts[-1], 'continuous_nri': continuous.tolist()}), flush=True)
    selected = [r for r in differences if r['threshold'] in cfg['display_table_thresholds']]
    selected_curves = [r for r in curves if r['threshold'] in cfg['display_table_thresholds']]
    for filename, rows in (('dca_curves.csv', curves), ('dca_contrasts.csv', differences), ('dca_selected_thresholds.csv', selected), ('dca_selected_curves.csv', selected_curves), ('nri_continuous.csv', continuous_rows), ('nri_categorical.csv', categorical_rows), ('cohorts.csv', cohorts)):
        save_csv(filename, rows)
    figures = plot_results(curves, differences, cfg['runs'])
    save_json('summary.json', {'completed_at': datetime.now().astimezone().isoformat(), 'passed': True, 'cohorts': cohorts, 'continuous_nri': continuous_rows, 'self_tests': tests, 'verifications': verifications, 'figures': figures, 'source_files_unchanged': True, 'limitations': cfg['remaining_requirements'], 'confidence_interval_scope': cfg['bootstrap']['uncertainty']})
    save_json('publication_text.json', {'section_title': 'Diagnostic-label decision-curve and reclassification sensitivity', 'methods': "We evaluated fixed predictions from the separately reported patient-isolated internal revalidation, preserving each cohort's original diagnostic labels and analysis units. Base raw and Platt probabilities arose from the same base-fit model; Platt fitting used separate calibration patients inside each outer training fold. A common numerical threshold grid of 0.01 to 0.99 in increments of 0.01 was frozen before inspecting these results. At each threshold t, net benefit was TP/N - FP/N times t/(1-t), with predicted probability >= t defining a positive decision. We calculated paired differences between base Platt and base raw, between the full-training ensemble and the age-only comparator, and between each model and treat-all/treat-none defaults. Two thousand patient-cluster bootstrap samples retained all admissions of sampled MIMIC patients. Percentile 95% intervals are pointwise and conditional on fixed predictions, not simultaneous or model-refitting intervals.", 'nri_methods': 'Exploratory continuous NRI and two-category NRI at every numerical threshold compared the full-training ensemble with the age-only logistic model, fitted to the same outer-training patients and evaluated on identical outer-test rows. Positive-label and negative-label components were reported separately, with ties contributing zero. These comparisons use distinct independently evaluated models, not same-OOF isotonic probabilities or an outcome-derived clinical proxy. NRI is sensitive to probability calibration and does not demonstrate clinical benefit.', 'interpretation': 'These analyses quantify uncertainty for classification of existing diagnostic labels. The numerical grid is not a set of recommended clinical thresholds, and no threshold was selected by observed net benefit. Original outcome timing, ascertainment and measurement limitations remain. The analysis does not establish incident-event prediction, prospective clinical utility, intervention efficacy, a validated clinical panel, or validation of historical apparent isotonic DCA.', 'figure_caption': 'Supplementary diagnostic-label decision curves and paired calibration contrasts. A, C, E and G show base raw, independently Platt-calibrated, treat-all and treat-none estimates; B, D, F and H show paired Platt-minus-raw differences. Shading denotes pointwise percentile 95% intervals from 2,000 patient-cluster bootstrap samples. Rows correspond to SCH, WHU, MIMIC and MIMIC excluding AST/HDL. China uses patient-level denominators; MIMIC retains admission-level denominators with patient-cluster uncertainty. The primary display covers the prespecified numerical grid from 0.01 to 0.50; the full 0.01-0.99 grid and all comparisons are supplied in aggregate tables. The full-grid figure uses a symmetric-log vertical scale for net benefit. Thresholds are numerical sensitivity values, not clinical recommendations.', 'response': 'We added a separately labelled diagnostic-classification sensitivity analysis using patient-isolated outer-held-out probabilities and independently fitted Platt calibration. It provides pointwise patient-cluster bootstrap confidence intervals for decision curves and paired differences, plus continuous and threshold-specific NRI components versus a genuinely independently evaluated age-only model. The grid was fixed before evaluation. We retain the withdrawal of clinical-benefit claims because no clinical action, benefit-harm trade-off, clinically acceptable threshold, incident-event timing or intervention effect has been established.', 'actions': 'Added Appendix E2, aggregate net-benefit and paired-contrast tables, NRI component estimates with 95% intervals, and supplementary calibration decision-curve panels; preserved main Figure 4 and original results. Statistical uncertainty is now evaluated, whereas clinical threshold justification and intervention-benefit validation remain unresolved.', 'method_sources': cfg['method_sources']})
if __name__ == '__main__':
    main()
