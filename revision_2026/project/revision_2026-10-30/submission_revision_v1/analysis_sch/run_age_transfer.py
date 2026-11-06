"""Source-fitted shared-age transport stress test; no original files are edited.

Public outputs contain aggregates only. Individual predictions/model objects stay
in private/. This does not authenticate SCH laboratory units or future outcomes.
"""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import argparse, hashlib, importlib.metadata, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PRIOR = PROJECT / 'analysis/r3_leakage_2026-10-21_v1'
PREP = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1/private/prepared'
CENTRES = ('SCH', 'WHU', 'MIMIC')
SOURCES = {c: PRIOR / ('mimic' if c == 'MIMIC' else 'china') / 'private' / f'{c}_demographics.npz' for c in CENTRES}
EXPECTED = {'SCH': (38509, 38509, 1323), 'WHU': (2562, 2562, 281), 'MIMIC': (9946, 6783, 4462)}

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def prepare():
    assert not (ROOT / 'protocol.json').exists(), 'Do not overwrite a frozen protocol.'
    ROOT.mkdir(parents=True, exist_ok=True)
    chinese = load(PRIOR / 'china/demographic_preparation.json')
    mimic = load(PRIOR / 'mimic/independent_verification_and_validity_correction.json')
    certified = {x['center']: x['output'] for x in chinese['cohorts']}
    certified['MIMIC'] = mimic['source_rebuilt_demographic_matrix']
    rows = []
    for c in CENTRES:
        assert identity(SOURCES[c]) == certified[c], f'{c}: source-demographic certificate mismatch'
        with np.load(SOURCES[c], allow_pickle=False) as z:
            age = z['X'][:, 0].astype(float)
            groups = z['groups']
        with np.load(PREP / f'{c}.npz', allow_pickle=False) as z:
            assert np.array_equal(groups, z['groups'])
            y = z['y'].astype(np.int8)
        assert (len(y), len(np.unique(groups)), int(y.sum())) == EXPECTED[c]
        assert np.isfinite(age).all() and (age >= 0).all() and (age <= 120).all()
        rows.append({'centre': c, 'rows': len(y), 'patients': len(np.unique(groups)), 'positive_labels': int(y.sum()), 'unit': 'admission' if c == 'MIMIC' else 'patient', 'age_min_before_common_cap': float(age.min()), 'age_max_before_common_cap': float(age.max()), 'rows_above_91_before_common_cap': int((age > 91).sum()), 'age_missing': int((~np.isfinite(age)).sum()), 'source_demographics': identity(SOURCES[c]), 'label_array': identity(PREP / f'{c}.npz')})
    protocol = {'version': 'shared_age_six_direction_transport_v1', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'purpose': 'Shared-age recorded-diagnosis transport stress test, including SCH, with source-fitted preprocessing.', 'population': 'Retain every original row and original label, including the 304 WHU patients without linked raw JSON. No exclusions by source availability or diagnosis label.', 'feature': 'One predictor: age in years, at earliest dated recorded observation in SCH/WHU and admission in MIMIC.', 'age_rule': 'Use certified source-rebuilt demographics, not the age/sex columns of copied prediction CSVs. Cap age above 91 in every centre before modelling, consistently with the retained MIMIC de-identification handling. The MIMIC age is admission calendar year minus birth calendar year, capped above at 91; Chinese age is the recorded numeric age at the earliest timestamp. These definitions are close but not identical exact-age measurements.', 'sex_exclusion': 'Chinese codes 1/2 lack an authenticated male/female mapping; no sex transfer or assumed recoding is performed.', 'source_unit': 'SCH and WHU patients; MIMIC admissions, with all original repeated admissions retained.', 'labels': 'Original centre-specific stored diagnostic labels, not a common incident clinical endpoint.', 'source_training': 'Each centre trains on its entire original source population. MIMIC admits each original admission as a training row. Target labels are unavailable to fit_source; there is no target retraining, scaling, imputation, calibration or threshold selection.', 'primary_model': {'pipeline': ['StandardScaler fitted only to source age', 'L2 LogisticRegression'], 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 5000, 'tol': 1e-10, 'class_weight': None}, 'sensitivity_model': {'pipeline': ['SplineTransformer fitted only to source age', 'L2 LogisticRegression'], 'n_knots': 5, 'degree': 3, 'knots': 'quantile', 'include_bias': False, 'extrapolation': 'linear', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 5000, 'tol': 1e-10, 'class_weight': None}, 'directions': [f'{s}->{t}' for s in CENTRES for t in CENTRES if s != t], 'metrics': ['ROC_AUC', 'average_precision', 'Brier', 'expected_to_observed'], 'bootstrap': {'replicates': 2000, 'seed_base': 20261029, 'unit': "target patient; all of a sampled patient's admissions receive the same multiplicity", 'type': 'non-stratified multinomial patient-cluster resampling', 'interval': 'percentile 2.5 and 97.5 percentiles', 'same_target_weights_shared_across_models': True, 'conditional_on_fixed_source_fit': True, 'includes_source_training_uncertainty': False}, 'model_selection': 'Both models and hyperparameters fixed before any target performance is calculated. No choice by target AUC.', 'scope_limits': ['Does not complete the four SCH common-laboratory transfers or authenticate SCH assay identity/units/scale.', 'Does not make native outcomes, observation windows or patient/admission units equivalent.', 'Does not establish prospective prediction, clinical adjudication, complete negative capture or exclude all indirect leakage in the original biomarker models.', 'Does not provide independent source-centre validation: only external target scores are evaluated.', 'Differences from historical multivariable AUCs cannot be attributed to any single feature family.'], 'no_original_artifacts_or_labels_changed': True, 'cohorts': rows, 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib')}}}
    dump(ROOT / 'protocol.json', protocol)
    evidence = [PRIOR / 'china/demographic_preparation.json', PRIOR / 'china/preparation_verification.json', PRIOR / 'mimic/independent_verification_and_validity_correction.json', PROJECT / 'audit/omop_environment_followup_2026-10-24_v1/mapping/sch_source_refresh_and_upstream_search.json', PROJECT / 'revision_2026-10-28/source_eligible_v1/reproducibility/source_evidence/sch_units/sch_transfer_gate.json']
    dump(ROOT / 'input_lock.json', {'frozen_at_utc': protocol['frozen_at_utc'], 'protocol': identity(ROOT / 'protocol.json'), 'script': identity(__file__), 'inputs': [identity(SOURCES[c]) for c in CENTRES] + [identity(PREP / f'{c}.npz') for c in CENTRES], 'provenance_evidence': [identity(p) for p in evidence]})
    print(json.dumps({'protocol_frozen': True, 'cohorts': [{k: x[k] for k in ('centre', 'rows', 'patients', 'positive_labels')} for x in rows]}), flush=True)

def datasets():
    lock = load(ROOT / 'input_lock.json')
    assert identity(ROOT / 'protocol.json') == lock['protocol'] and identity(__file__) == lock['script']
    for entry in lock['inputs']:
        assert identity(entry['path']) == entry
    result = {}
    for c in CENTRES:
        with np.load(SOURCES[c], allow_pickle=False) as z:
            X = np.minimum(z['X'][:, [0]].astype(float), 91)
            groups = z['groups'].copy()
        with np.load(PREP / f'{c}.npz', allow_pickle=False) as z:
            y = z['y'].astype(np.int8)
        result[c] = (X, y, groups)
    return result

def fit_source(X, y, kind):
    transform = StandardScaler() if kind == 'linear' else SplineTransformer(n_knots=5, degree=3, knots='quantile', include_bias=False, extrapolation='linear')
    model = make_pipeline(transform, LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000, tol=1e-10, class_weight=None))
    model.fit(X, y)
    assert int(model[-1].n_iter_.max()) < 5000
    return model

def metrics_fast(y, p, w, levels):
    idx = levels
    pos = np.bincount(idx, weights=w * y, minlength=int(idx.max()) + 1)
    neg = np.bincount(idx, weights=w * (1 - y), minlength=len(pos))
    P, N = (pos.sum(), neg.sum())
    if P <= 0 or N <= 0:
        return np.full(4, np.nan)
    auc = float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (P * N))
    reverse_pos = pos[::-1]
    reverse_neg = neg[::-1]
    tp = np.cumsum(reverse_pos)
    total = tp + np.cumsum(reverse_neg)
    ap = float(np.sum(np.divide(tp, total, out=np.zeros_like(tp), where=total > 0) * reverse_pos) / P)
    return np.array([auc, ap, float(np.sum(w * (p - y) ** 2) / w.sum()), float(np.sum(w * p) / P)])

def run():
    assert not (ROOT / 'private').exists(), 'Never overwrite any prior or partial model run.'
    data = datasets()
    protocol = load(ROOT / 'protocol.json')
    private = ROOT / 'private'
    private.mkdir()
    source_models = {}
    model_reports = []
    all_scores = {}
    for source in CENTRES:
        X, y, _ = data[source]
        for kind in ('linear', 'spline'):
            model = fit_source(X, y, kind)
            source_models[source, kind] = model
            path = private / f'{source}_{kind}.joblib'
            joblib.dump(model, path, compress=3)
            info = {'source': source, 'model': kind, 'source_rows': len(y), 'source_positive_labels': int(y.sum()), 'iterations': int(model[-1].n_iter_.max()), 'coefficient': model[-1].coef_.tolist(), 'intercept': model[-1].intercept_.tolist(), 'model_identity': identity(path)}
            if kind == 'linear':
                info['source_mean'] = model[0].mean_.tolist()
                info['source_scale'] = model[0].scale_.tolist()
            else:
                info['source_knots'] = model[0].bsplines_[0].t.tolist()
            model_reports.append(info)
            for target in CENTRES:
                if target == source:
                    continue
                p = model.predict_proba(data[target][0])[:, 1]
                assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                all_scores[source, target, kind] = p
            print(f'{source} {kind} fitted from source rows only.', flush=True)
    dump(ROOT / 'model_specifications.json', {'protocol': identity(ROOT / 'protocol.json'), 'models': model_reports, 'model_count': len(model_reports)})
    metric_names = protocol['metrics']
    rows = []
    paired = []
    verification = []
    bootstrap_checks = []
    for target_idx, target in enumerate(CENTRES):
        X, y, groups = data[target]
        unique, inverse = np.unique(groups, return_inverse=True)
        keys = [k for k in all_scores if k[1] == target]
        scores = [all_scores[k] for k in keys]
        levels = [np.unique(p, return_inverse=True)[1] for p in scores]
        points = np.stack([metrics_fast(y, p, np.ones(len(y)), lev) for p, lev in zip(scores, levels)])
        for j, p in enumerate(scores):
            sklearn_point = np.array([roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p), p.sum() / y.sum()])
            assert np.allclose(points[j], sklearn_point, rtol=1e-12, atol=1e-12)
            verification.append({'target': target, 'source': keys[j][0], 'model': keys[j][2], 'point_max_abs_difference_vs_sklearn': float(np.max(np.abs(points[j] - sklearn_point)))})
        seed = protocol['bootstrap']['seed_base'] + target_idx
        rng = np.random.default_rng(seed)
        B = protocol['bootstrap']['replicates']
        draws = np.full((B, len(keys), 4), np.nan)
        weight_hash = hashlib.sha256()
        for b in range(B):
            counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
            weight_hash.update(counts.astype('<i4').tobytes())
            w = counts[inverse]
            for j, (p, lev) in enumerate(zip(scores, levels)):
                draws[b, j] = metrics_fast(y, p, w, lev)
            if b in (0, 1, 1999):
                for j, p in enumerate(scores):
                    ref = np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w), np.dot(w, p) / np.dot(w, y)])
                    assert np.allclose(draws[b, j], ref, rtol=1e-11, atol=1e-11)
                    bootstrap_checks.append({'target': target, 'resample': b, 'source': keys[j][0], 'model': keys[j][2], 'max_abs_difference_vs_sklearn': float(np.max(np.abs(draws[b, j] - ref)))})
        assert np.isfinite(draws).all()
        np.savez_compressed(private / f'{target}_predictions.npz', y=y, groups=groups, age=X[:, 0], scores=np.column_stack(scores), score_names=np.array(['_'.join(k) for k in keys]))
        np.savez_compressed(ROOT / f'{target}_aggregate_bootstrap.npz', draws=draws, points=points, score_names=np.array(['_'.join(k) for k in keys]), metric_names=np.array(metric_names))
        for j, key in enumerate(keys):
            for m, metric in enumerate(metric_names):
                low, high = np.percentile(draws[:, j, m], [2.5, 97.5])
                rows.append({'source': key[0], 'target': target, 'model': key[2], 'metric': metric, 'estimate': float(points[j, m]), 'ci_low': float(low), 'ci_high': float(high), 'target_rows': len(y), 'target_patients': len(unique), 'target_positive_labels': int(y.sum()), 'target_unit': 'admission' if target == 'MIMIC' else 'patient', 'bootstrap_replicates': B, 'bootstrap_seed': seed})
        for source in CENTRES:
            if source == target:
                continue
            a = keys.index((source, target, 'linear'))
            b = keys.index((source, target, 'spline'))
            for m, metric in enumerate(metric_names):
                low, high = np.percentile(draws[:, b, m] - draws[:, a, m], [2.5, 97.5])
                paired.append({'source': source, 'target': target, 'comparison': 'spline_minus_linear', 'metric': metric, 'estimate': float(points[b, m] - points[a, m]), 'ci_low': float(low), 'ci_high': float(high)})
        print(f'{target}: {len(keys)} transferred scores, {B} valid patient-cluster resamples.', flush=True)
        dump(ROOT / f'{target}_bootstrap_audit.json', {'target': target, 'valid_replicates': B, 'seed': seed, 'conditional_on_fixed_source_models': True})
    pd.DataFrame(rows).to_csv(ROOT / 'age_transfer_performance.csv', index=False)
    pd.DataFrame(paired).to_csv(ROOT / 'age_transfer_model_sensitivity.csv', index=False)
    dump(ROOT / 'numerical_verification.json', {'point_checks': verification, 'weighted_bootstrap_checks': bootstrap_checks, 'all_passed': True, 'no_target_fitting': True, 'source_fit_function_accepts_only_source_X_y_and_fixed_kind': True})
    data2 = datasets()
    assert all((np.array_equal(data[c][i], data2[c][i]) for c in CENTRES for i in range(3)))
    dump(ROOT / 'summary.json', {'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol': identity(ROOT / 'protocol.json'), 'input_lock': identity(ROOT / 'input_lock.json'), 'models_fitted': len(model_reports), 'directions_evaluated': 6, 'model_variants': 2, 'all_original_cohort_rows_and_labels_retained': True, 'source_full_population_training': True, 'no_source_performance_claim': True, 'no_sch_assay_unit_resolution': True, 'limits': protocol['scope_limits'], 'results': rows, 'paired_model_sensitivity': paired, 'public_patient_identifiers_or_individual_rows': False, 'inputs_unchanged_after_run': True})
    print(json.dumps({'complete': True, 'AUC': [r for r in rows if r['metric'] == 'ROC_AUC']}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('prepare', 'run'))
    args = p.parse_args()
    prepare() if args.mode == 'prepare' else run()
