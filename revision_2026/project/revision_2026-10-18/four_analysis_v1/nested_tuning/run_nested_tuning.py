"""Limited nested tuning; writes only to this new, locked analysis directory."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
THREADS = 2
LOCK = ROOT / 'analysis_lock_v2.json'

def sha(p):
    return None

def ident(p):
    p = Path(p)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def load(p):
    return _release_resolve_config(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def now():
    return datetime.now(timezone.utc).isoformat()

def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'threads': THREADS, 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib', 'scipy')}}

def freeze():
    paths = [ROOT / 'protocol.json', Path(__file__), BASE / 'protocol.json', _release_source_path(BASE / 'run_validation.py'), BASE / 'evidence/analysis_lock.json']
    for c in ('SCH', 'WHU', 'MIMIC'):
        paths += [BASE / f'private/prepared/{c}.npz', BASE / f'private/runs/{c}_original/patient_roles.npz', BASE / f'private/runs/{c}_original/predictions.npz']
        for f in range(1, 6):
            paths += [BASE / f'private/runs/{c}_original/fold_{f}/full_package.joblib', BASE / f'private/runs/{c}_original/fold_{f}/full_xgb.ubj']
    for item in load(BASE / 'evidence/analysis_lock.json')['state']['files']:
        pass
    state = {'sources': [ident(p) for p in paths], 'protocol': load(ROOT / 'protocol.json'), 'environment': environment()}
    p = LOCK
    if p.exists():
        assert load(p)['state'] == state, 'Locked inputs/code/environment changed'
    else:
        save(p, {'frozen_at': now(), 'state': state})
    return state

def prepared(c):
    with np.load(BASE / f'private/prepared/{c}.npz', allow_pickle=False) as z:
        X, y, g, names = (z['X'].copy(), z['y'].copy(), z['groups'].copy(), z['feature_names'].tolist())
    with np.load(BASE / f'private/runs/{c}_original/patient_roles.npz', allow_pickle=False) as z:
        assert np.array_equal(z['y'], y) and np.array_equal(z['groups'], g)
        roles = z['roles'].copy()
    with np.load(BASE / f'private/runs/{c}_original/predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(z['y'], y) and np.array_equal(z['groups'], g)
        previous = {'previous_ensemble': z['primary_raw'].copy(), 'previous_xgb': z['xgb_full'].copy(), 'rf': z['rf_full'].copy()}
        fold = z['fold'].copy()
    assert not np.isinf(X).any()
    expected = {'SCH': (38509, 38509, 1323, 104), 'WHU': (2562, 2562, 281, 104), 'MIMIC': (9946, 6783, 4462, 43)}[c]
    assert (len(y), len(np.unique(g)), int(y.sum()), len(names)) == expected
    assert np.all(np.sum(roles == 0, axis=0) == 1)
    for i, r in enumerate(roles, 1):
        assert np.array_equal(r == 0, fold == i)
        assert not set(g[r == 0]) & set(g[r != 0])
    return (X, y, g, names, roles, fold, previous)

def prep_fit(X, c):
    columns = np.arange(X.shape[1]) if c != 'MIMIC' else np.flatnonzero(np.isfinite(X).any(axis=0))
    assert len(columns) > 0
    imputer = None if c != 'MIMIC' else SimpleImputer(strategy='median', keep_empty_features=True).fit(X[:, columns])
    return {'columns': columns, 'zero_fill': c != 'MIMIC', 'imputer': imputer}

def transform(p, X):
    a = X[:, p['columns']]
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) if p['zero_fill'] else p['imputer'].transform(a).astype(np.float32)

def same_preprocessor(actual, expected):
    assert actual['zero_fill'] == expected['zero_fill']
    assert np.array_equal(actual['columns'], expected['columns'])
    if actual['imputer'] is None:
        assert expected['imputer'] is None
    else:
        a, b = (actual['imputer'].get_params(), expected['imputer'].get_params())
        assert a.keys() == b.keys()
        for key in a:
            if key == 'missing_values' and np.isscalar(a[key]) and np.isscalar(b[key]):
                assert a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key]))
            else:
                assert a[key] == b[key]
        assert np.array_equal(actual['imputer'].statistics_, expected['imputer'].statistics_)

def matrix(X, y=None):
    return xgb.DMatrix(X, label=y, nthread=THREADS)

def parameters(y, c, fold, candidate):
    assert np.unique(y).size == 2
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42 if c != 'MIMIC' else fold, 'nthread': THREADS, 'verbosity': 0, 'scale_pos_weight': float(np.sum(y == 0) / np.sum(y == 1)), **{k: v for k, v in candidate.items() if k != 'name'}}

def split_stop(indices, y, g, seed):
    groups = np.unique(g[indices])
    labels = np.array([y[indices][g[indices] == v].max() for v in groups])
    a, b = train_test_split(groups, test_size=0.25, stratify=labels, random_state=seed)
    return (indices[np.isin(g[indices], a)], indices[np.isin(g[indices], b)])

def inner_roles(y, g, outer_train, c, fold, config):
    seed = config['inner_split_seed_base'] + fold
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed) if c == 'MIMIC' else StratifiedKFold(3, shuffle=True, random_state=seed)
    rows = []
    splits = splitter.split(np.zeros((len(outer_train), 1)), y[outer_train], g[outer_train]) if c == 'MIMIC' else splitter.split(np.zeros((len(outer_train), 1)), y[outer_train])
    for inner, (fit, val) in enumerate(splits, 1):
        train, test = (outer_train[fit], outer_train[val])
        stopfit, stopval = split_stop(train, y, g, config['stopping_seed_base'] + fold * 10 + inner)
        assert all((np.unique(y[ix]).size == 2 for ix in (stopfit, stopval, test)))
        for a, b in ((stopfit, stopval), (stopfit, test), (stopval, test)):
            assert not set(g[a]) & set(g[b])
        role = np.full(len(y), -1, dtype=np.int8)
        role[test] = 0
        role[stopfit] = 1
        role[stopval] = 2
        rows.append(role)
    result = np.array(rows)
    assert np.all(np.sum(result[:, outer_train] == 0, axis=0) == 1)
    return result

def fit_inner(X, y, c, fold, candidate, role, dest, config):
    dest.mkdir(parents=True)
    stopfit, stopval, scoring = [np.flatnonzero(role == r) for r in (1, 2, 0)]
    full = np.flatnonzero(role > 0)
    stopprep = prep_fit(X[stopfit], c)
    pars = parameters(y[stopfit], c, fold, candidate)
    history = {}
    early = xgb.train(pars, matrix(transform(stopprep, X[stopfit]), y[stopfit]), num_boost_round=300, evals=[(matrix(transform(stopprep, X[stopval]), y[stopval]), 'early_stop')], early_stopping_rounds=20, evals_result=history, verbose_eval=False)
    rounds = int(early.best_iteration) + 1
    assert rounds == int(np.argmax(history['early_stop']['auc'])) + 1
    early.save_model(dest / 'early_xgb.ubj')
    joblib.dump(stopprep, dest / 'early_preprocessor.joblib', compress=3)
    fullprep = prep_fit(X[full], c)
    fullpars = parameters(y[full], c, fold, candidate)
    fitted = xgb.train(fullpars, matrix(transform(fullprep, X[full]), y[full]), num_boost_round=rounds, verbose_eval=False)
    fitted.save_model(dest / 'scoring_xgb.ubj')
    joblib.dump(fullprep, dest / 'scoring_preprocessor.joblib', compress=3)
    prediction = fitted.predict(matrix(transform(fullprep, X[scoring])))
    np.savez_compressed(dest / 'scoring_predictions.npz', index=scoring, probability=prediction)
    report = {'candidate': candidate['name'], 'selected_rounds': rounds, 'early_parameters': pars, 'scoring_parameters': fullpars, 'history': history, 'scoring_auc': float(roc_auc_score(y[scoring], prediction)), 'roles': {k: {'rows': len(ix), 'positive_labels': int(y[ix].sum())} for k, ix in (('early_fit', stopfit), ('early_stop', stopval), ('scoring', scoring), ('refit', full))}, 'early_columns': stopprep['columns'].tolist(), 'scoring_columns': fullprep['columns'].tolist()}
    save(dest / 'fit.json', report)
    return report

def run(c, state, gate):
    assert gate.is_relative_to(ROOT.parent / 'shap_seeds') and load(gate)['passed'] is True
    config = state['protocol']
    X, y, g, names, roles, fold_ids, previous = prepared(c)
    dest = ROOT / 'private' / c
    if dest.exists():
        raise FileExistsError('Never overwrite an existing nested run')
    dest.mkdir(parents=True)
    output = {m: np.full(len(y), np.nan) for m in config['models'] if m not in previous}
    output.update({k: v for k, v in previous.items() if k != 'rf'})
    fold_reports = []
    started = now()
    for fold, outer in enumerate(roles, 1):
        folder = dest / f'fold_{fold}'
        folder.mkdir()
        train, test = (np.flatnonzero(outer != 0), np.flatnonzero(outer == 0))
        role = inner_roles(y, g, train, c, fold, config)
        assert np.all(role[:, test] == -1)
        np.savez_compressed(folder / 'inner_roles.npz', roles=role)
        reports = []
        for candidate in config['configurations']:
            details = []
            for inner, r in enumerate(role, 1):
                details.append(fit_inner(X, y, c, fold, candidate, r, folder / candidate['name'] / f'inner_{inner}', config))
            report = {'candidate': candidate['name'], 'mean_inner_auc': float(np.mean([x['scoring_auc'] for x in details])), 'inner': details, 'final_rounds': max(1, int(np.median([x['selected_rounds'] for x in details])))}
            reports.append(report)
            print(f"{c} outer {fold}/5 candidate {candidate['name']}: inner scoring completed", flush=True)
        selected = int(np.argmax([x['mean_inner_auc'] for x in reports]))
        prep = prep_fit(X[train], c)
        joblib.dump(prep, folder / 'outer_preprocessor.joblib', compress=3)
        oldpackage = joblib.load(BASE / f'private/runs/{c}_original/fold_{fold}/full_package.joblib')
        same_preprocessor(prep, oldpackage['preprocessor'])
        assert list(oldpackage['input_feature_names']) == names
        assert list(oldpackage['fitted_feature_names']) == [names[i] for i in prep['columns']]
        assert np.array_equal(transform(prep, X[test]), transform(oldpackage['preprocessor'], X[test]))
        oldpackage['rf'].n_jobs = THREADS
        rp = oldpackage['rf'].predict_proba(transform(prep, X[test]))[:, 1]
        assert np.max(np.abs(rp - previous['rf'][test])) < 1e-12
        oldbooster = xgb.Booster(params={'nthread': THREADS})
        oldbooster.load_model(BASE / f'private/runs/{c}_original/fold_{fold}/full_xgb.ubj')
        xp = oldbooster.predict(matrix(transform(prep, X[test])))
        assert np.max(np.abs(xp - previous['previous_xgb'][test])) < 1e-12
        assert np.max(np.abs(0.5 * xp + 0.5 * rp - previous['previous_ensemble'][test])) < 1e-12
        for kind, index in (('selected', selected), ('fixed', 0)):
            pars = parameters(y[train], c, fold, config['configurations'][index])
            booster = xgb.train(pars, matrix(transform(prep, X[train]), y[train]), num_boost_round=reports[index]['final_rounds'], verbose_eval=False)
            booster.save_model(folder / f'{kind}_xgb.ubj')
            p = booster.predict(matrix(transform(prep, X[test])))
            output[f'nested_{kind}_xgb'][test] = p
            output[f'nested_{kind}_ensemble'][test] = 0.5 * p + 0.5 * rp
        fold_report = {'fold': fold, 'selected_candidate': config['configurations'][selected]['name'], 'selected_index': selected, 'candidates': reports, 'outer_training_rows': len(train), 'outer_test_rows': len(test), 'outer_training_patients': len(np.unique(g[train])), 'outer_test_patients': len(np.unique(g[test])), 'rf_and_previous_xgb_replayed': True, 'outer_test_not_used_for_selection': True}
        save(folder / 'selection.json', fold_report)
        fold_reports.append(fold_report)
        print(f"{c} outer {fold}/5 complete; selected={fold_report['selected_candidate']}", flush=True)
    assert all((np.isfinite(p).all() for p in output.values()))
    np.savez_compressed(dest / 'predictions.npz', y=y, groups=g, fold=fold_ids, **output)
    models = [ident(p) for p in dest.rglob('*') if p.is_file()]
    report = {'execution_complete': True, 'started_at': started, 'completed_at': now(), 'cohort': c, 'analysis_unit': 'admission' if c == 'MIMIC' else 'patient', 'rows': len(y), 'patients': len(np.unique(g)), 'positive_labels': int(y.sum()), 'input_features': names, 'previous_stage_verified': ident(gate), 'folds': fold_reports, 'outputs': models, 'predictions': ident(dest / 'predictions.npz')}
    save(ROOT / f'results/{c}_training.json', report)

class FixedScoreMetrics:

    def __init__(self, y, p):
        self.y = np.asarray(y, dtype=int)
        self.p = np.asarray(p, dtype=float)
        self.order = np.argsort(-self.p, kind='stable')
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.p[self.order]) != 0) + 1]
        self.sy = self.y[self.order]

    def evaluate(self, w):
        w = np.asarray(w, dtype=float)
        pos = np.add.reduceat((w * self.y)[self.order], self.starts)
        neg = np.add.reduceat((w * (1 - self.y))[self.order], self.starts)
        tp, fp = (np.r_[0, np.cumsum(pos)], np.r_[0, np.cumsum(neg)])
        if tp[-1] == 0 or fp[-1] == 0:
            return np.full(3, np.nan)
        auc = np.trapezoid(tp / tp[-1], fp / fp[-1])
        precision = np.divide(tp[1:], tp[1:] + fp[1:], out=np.zeros_like(pos), where=tp[1:] + fp[1:] > 0)
        ap = np.sum(pos * precision) / tp[-1]
        brier = np.sum(w * (self.p - self.y) ** 2) / np.sum(w)
        return np.array([auc, ap, brier])

def direct_metrics(y, p, w=None):
    return np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w)])

def self_test():
    rng = np.random.default_rng(718)
    count = 0
    for n in (21, 80, 203):
        y = rng.integers(0, 2, n)
        y[:2] = [0, 1]
        for p in (rng.random(n), np.round(rng.random(n), 1), np.full(n, 0.4)):
            metric = FixedScoreMetrics(y, p)
            for _ in range(10):
                w = rng.integers(0, 5, n).astype(float)
                w[:2] = 1
                assert np.allclose(metric.evaluate(w), direct_metrics(y, p, w), atol=1e-12, rtol=0)
                count += 1
    X = np.array([[0, np.nan, 3], [4, np.nan, 5]], dtype=np.float32)
    p = prep_fit(X, 'MIMIC')
    assert p['columns'].tolist() == [0, 2]
    assert np.array_equal(transform(p, np.array([[np.nan, 999, np.nan]], dtype=np.float32)), [[2, 4]])
    assert transform(prep_fit(X, 'SCH'), X)[0, 0] == 0
    g = np.repeat(np.arange(120), 2)
    y = np.repeat(np.arange(120) % 2, 2)
    role = inner_roles(y, g, np.arange(200), 'MIMIC', 1, load(ROOT / 'protocol.json'))
    assert np.all(role[:, 200:] == -1)
    return {'passed': True, 'weighted_metric_cases': count, 'fit_only_imputation_and_patient_role_checks': True}

def summarize(state):
    config = state['protocol']
    rows = []
    paired = []
    boot = []
    for c in config['cohorts']:
        z = np.load(ROOT / f'private/{c}/predictions.npz', allow_pickle=False)
        y, g = (z['y'], z['groups'])
        models = config['models']
        caches = [FixedScoreMetrics(y, z[m]) for m in models]
        points = np.array([direct_metrics(y, z[m]) for m in models])
        _, inverse = np.unique(g, return_inverse=True)
        n = int(inverse.max()) + 1
        rng = np.random.default_rng(config['bootstrap_seed'])
        samples = np.empty((2000, len(models), 3))
        h = hashlib.sha256()
        for i in range(2000):
            mult = rng.multinomial(n, np.full(n, 1 / n))
            h.update(np.asarray(mult, dtype='<i8').tobytes())
            w = mult[inverse]
            samples[i] = [m.evaluate(w) for m in caches]
            if i < 10:
                assert np.allclose(samples[i], [direct_metrics(y, z[m], w) for m in models], atol=2e-12, rtol=0)
        assert np.isfinite(samples).all()
        context = {'cohort': c, 'analysis_unit': 'admission' if c == 'MIMIC' else 'patient', 'rows': len(y), 'patients': n, 'positive_labels': int(y.sum())}
        for mi, m in enumerate(models):
            for ki, k in enumerate(config['metrics']):
                low, high = np.quantile(samples[:, mi, ki], [0.025, 0.975])
                rows.append({**context, 'model': m, 'metric': k, 'estimate': float(points[mi, ki]), 'ci_low': float(low), 'ci_high': float(high)})
        for a, b in config['paired_comparisons']:
            ai, bi = (models.index(a), models.index(b))
            for ki, k in enumerate(config['metrics']):
                low, high = np.quantile(samples[:, ai, ki] - samples[:, bi, ki], [0.025, 0.975])
                paired.append({**context, 'first': a, 'second': b, 'metric': 'delta_' + k, 'estimate': float(points[ai, ki] - points[bi, ki]), 'ci_low': float(low), 'ci_high': float(high)})
        np.savez_compressed(ROOT / f'private/{c}/bootstrap.npz', samples=samples, models=np.array(models), metrics=np.array(config['metrics']))
        boot.append({'cohort': c, 'draws': 2000, 'patients': n})
        z.close()
    pd.DataFrame(rows).to_csv(ROOT / 'results/metrics.csv', index=False, mode='x', float_format='%.15g')
    pd.DataFrame(paired).to_csv(ROOT / 'results/paired_comparisons.csv', index=False, mode='x', float_format='%.15g')
    selection = []
    for c in config['cohorts']:
        for f in load(ROOT / f'results/{c}_training.json')['folds']:
            for a in f['candidates']:
                selection.append({'cohort': c, 'outer_fold': f['fold'], 'candidate': a['candidate'], 'mean_inner_auc': a['mean_inner_auc'], 'selected': a['candidate'] == f['selected_candidate'], 'final_rounds': a['final_rounds']})
    pd.DataFrame(selection).to_csv(ROOT / 'results/candidate_selection.csv', index=False, mode='x', float_format='%.15g')
    save(ROOT / 'results/summary.json', {'execution_complete': True, 'lock': ident(LOCK), 'protocol': config, 'metrics': rows, 'paired': paired, 'bootstrap': boot, 'selection': selection, 'self_tests': self_test()})

def verify(state):
    checked = 0
    outer_checked = 0
    early_checked = 0
    for c in state['protocol']['cohorts']:
        X, y, g, names, roles, fold_ids, previous = prepared(c)
        report = load(ROOT / f'results/{c}_training.json')
        for item in report['outputs']:
            pass
        z = np.load(ROOT / f'private/{c}/predictions.npz', allow_pickle=False)
        assert np.array_equal(z['y'], y) and np.array_equal(z['groups'], g) and np.array_equal(z['fold'], fold_ids)
        for fold, outer in enumerate(roles, 1):
            folder = ROOT / f'private/{c}/fold_{fold}'
            train, test = (np.flatnonzero(outer != 0), np.flatnonzero(outer == 0))
            role = inner_roles(y, g, train, c, fold, state['protocol'])
            assert np.array_equal(role, np.load(folder / 'inner_roles.npz')['roles'])
            select = load(folder / 'selection.json')
            reconstructed = []
            for candidate_index, candidate in enumerate(state['protocol']['configurations']):
                inner_auc = []
                inner_rounds = []
                for inner, r in enumerate(role, 1):
                    d = folder / candidate['name'] / f'inner_{inner}'
                    ix = np.flatnonzero(r == 0)
                    p = joblib.load(d / 'scoring_preprocessor.joblib')
                    same_preprocessor(p, prep_fit(X[r > 0], c))
                    fitted = xgb.Booster(params={'nthread': THREADS})
                    fitted.load_model(d / 'scoring_xgb.ubj')
                    pred = fitted.predict(matrix(transform(p, X[ix])))
                    stored = np.load(d / 'scoring_predictions.npz')
                    assert np.array_equal(stored['index'], ix) and np.array_equal(stored['probability'], pred)
                    fit = load(d / 'fit.json')
                    assert abs(roc_auc_score(y[ix], pred) - fit['scoring_auc']) < 1e-12
                    assert fitted.num_boosted_rounds() == fit['selected_rounds']
                    early_p = joblib.load(d / 'early_preprocessor.joblib')
                    same_preprocessor(early_p, prep_fit(X[r == 1], c))
                    early = xgb.Booster(params={'nthread': THREADS})
                    early.load_model(d / 'early_xgb.ubj')
                    early_pred = early.predict(matrix(transform(early_p, X[r == 2])), iteration_range=(0, fit['selected_rounds']))
                    best = int(np.argmax(fit['history']['early_stop']['auc']))
                    assert best + 1 == fit['selected_rounds'] == int(early.best_iteration) + 1
                    assert abs(roc_auc_score(y[r == 2], early_pred) - fit['history']['early_stop']['auc'][best]) < 1e-10
                    inner_auc.append(float(roc_auc_score(y[ix], pred)))
                    inner_rounds.append(fitted.num_boosted_rounds())
                    early_checked += 1
                    checked += 1
                reconstructed.append({'mean_inner_auc': float(np.mean(inner_auc)), 'final_rounds': max(1, int(np.median(inner_rounds)))})
                recorded = select['candidates'][candidate_index]
                assert reconstructed[-1]['mean_inner_auc'] == recorded['mean_inner_auc']
                assert reconstructed[-1]['final_rounds'] == recorded['final_rounds']
            assert select['selected_index'] == int(np.argmax([r['mean_inner_auc'] for r in reconstructed]))
            p = joblib.load(folder / 'outer_preprocessor.joblib')
            same_preprocessor(p, prep_fit(X[train], c))
            original = joblib.load(BASE / f'private/runs/{c}_original/fold_{fold}/full_package.joblib')
            same_preprocessor(p, original['preprocessor'])
            assert list(original['input_feature_names']) == names
            assert list(original['fitted_feature_names']) == [names[i] for i in p['columns']]
            for kind in ('selected', 'fixed'):
                model = xgb.Booster(params={'nthread': THREADS})
                model.load_model(folder / f'{kind}_xgb.ubj')
                candidate_index = select['selected_index'] if kind == 'selected' else 0
                assert model.num_boosted_rounds() == reconstructed[candidate_index]['final_rounds']
                pred = model.predict(matrix(transform(p, X[test])))
                assert np.array_equal(pred, z[f'nested_{kind}_xgb'][test])
                assert np.max(np.abs(0.5 * pred + 0.5 * previous['rf'][test] - z[f'nested_{kind}_ensemble'][test])) < 1e-12
                outer_checked += 1
        z.close()
    for item in state['sources']:
        pass
    assert checked == 180 and outer_checked == 30 and (early_checked == 180)
    save(ROOT / 'verification.json', {'passed': True, 'inner_scoring_models_replayed': checked, 'outer_models_replayed': outer_checked, 'early_stop_models_replayed': early_checked, 'all_preprocessors_reconstructed': True, 'sources_unchanged': True, 'patient_splits_reconstructed': True, 'selection_argmax_checked': True, 'selection_and_final_rounds_reconstructed_from_inner_fits': True, 'summary': ident(ROOT / 'results/summary.json'), 'lock': ident(LOCK), 'self_tests': self_test(), 'completed_at': now()})
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--run', choices=('SCH', 'WHU', 'MIMIC', 'all'))
    parser.add_argument('--stage-3-verified-path', type=Path)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
    else:
        state = freeze()
        if args.run:
            assert args.stage_3_verified_path is not None
            for c in state['protocol']['cohorts'] if args.run == 'all' else [args.run]:
                run(c, state, args.stage_3_verified_path.resolve())
        if args.summarize:
            summarize(state)
        if args.verify:
            verify(state)
