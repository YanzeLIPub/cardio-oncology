"""Fixed-patient-fold Chinese XGBoost seed sensitivity; new outputs only."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from scipy.stats import spearmanr, false_discovery_control
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
OLD = PROJECT / 'revision_2026-10-16/reviewer_priority_analyses_v1/shap'
COHORTS = ('SCH', 'WHU')
SEEDS = (42, 43, 44, 45)
METRICS = ('auc', 'average_precision', 'brier')

def now():
    return datetime.now(timezone.utc).isoformat()

def read(p):
    return _release_resolve_config(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def identity(p):
    p = Path(p)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save(p, value):
    p = Path(p)
    assert p.resolve().is_relative_to(ROOT)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'threads': 2, 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'joblib', 'threadpoolctl')}}

def source_paths():
    paths = [BASE / n for n in ('protocol.json', 'run_validation.py', 'evidence/analysis_lock.json', 'evidence/validation_replay_audit.json')]
    paths += [OLD / n for n in ('protocol.json', 'run_heldout_shap.py', 'analysis_lock.json', 'verification.json', 'results/summary.json', 'results/feature_associations.csv', 'results/fold_associations.csv', 'results/cross_cohort_associations.csv')]
    for c in COHORTS:
        run = BASE / f'private/runs/{c}_original'
        paths += [BASE / f'private/prepared/{c}.npz', run / 'patient_roles.npz', run / 'predictions.npz', BASE / f'results/{c}_original_training.json']
        for fold in range(1, 6):
            paths += [run / f'fold_{fold}/{n}' for n in ('full_xgb.ubj', 'full_package.joblib', 'fit_report.json')]
    return paths

def lock():
    for item in read(BASE / 'evidence/analysis_lock.json')['state']['files']:
        assert identity(item['path']) == item
    assert read(BASE / 'evidence/validation_replay_audit.json')['passed']
    assert read(OLD / 'verification.json')['passed']
    state = {'protocol': identity(ROOT / 'protocol.json'), 'runner': identity(__file__), 'sources': [identity(p) for p in source_paths()], 'environment': environment()}
    dest = ROOT / 'analysis_lock.json'
    if dest.exists():
        assert read(dest)['state'] == state, 'Frozen source/code/environment changed'
    else:
        save(dest, {'frozen_before_new_training_at': now(), 'state': state})
    return state

def check_sources(state):
    for item in state['sources']:
        assert identity(item['path']) == item

def split(indices, y, groups, fraction, seed):
    unique = np.unique(groups[indices])
    labels = np.array([y[indices][groups[indices] == g].max() for g in unique])
    a, b = train_test_split(unique, test_size=fraction, stratify=labels, random_state=seed)
    return (indices[np.isin(groups[indices], a)], indices[np.isin(groups[indices], b)])

def reproduce_roles(y, g):
    config = read(BASE / 'protocol.json')
    result = np.full((5, len(y)), -1, dtype=np.int8)
    for fold, (train, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y), 1):
        base, cal = split(train, y, g, 0.25, config['calibration_seed_base'] + fold)
        inner, stop = split(base, y, g, 0.25, config['inner_seed_base'] + fold)
        for r, ix in enumerate((test, cal, inner, stop)):
            result[fold - 1, ix] = r
    return result

def data(c):
    with np.load(BASE / f'private/prepared/{c}.npz', allow_pickle=False) as z:
        X, y, g, names = (z['X'].copy(), z['y'].copy(), z['groups'].copy(), z['feature_names'].tolist())
    with np.load(BASE / f'private/runs/{c}_original/patient_roles.npz', allow_pickle=False) as z:
        assert np.array_equal(z['groups'], g) and np.array_equal(z['y'], y)
        roles = z['roles'].copy()
    with np.load(BASE / f'private/runs/{c}_original/predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(z['groups'], g) and np.array_equal(z['y'], y)
        baseline = z['xgb_full'].copy()
    assert [len(y), len(np.unique(g)), int(y.sum()), len(names)] == read(ROOT / 'protocol.json')['expected_counts'][c]
    assert np.array_equal(roles, reproduce_roles(y, g)) and (not np.isinf(X).any())
    assert np.all((roles == 0).sum(axis=0) == 1)
    for role in roles:
        sets = [set(g[role == r]) for r in range(4)]
        assert all((not sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)))
    return (X, y, g, names, roles, baseline)

def transform(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def matrix(X, y=None):
    return xgb.DMatrix(transform(X), label=y, nthread=2)

def params(y, seed):
    assert np.unique(y).size == 2
    config = read(ROOT / 'protocol.json')['xgb']
    return {'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist', 'verbosity': 0, 'max_depth': config['max_depth'], 'learning_rate': config['learning_rate'], 'subsample': config['subsample'], 'colsample_bytree': config['colsample_bytree'], 'scale_pos_weight': float((y == 0).sum() / (y == 1).sum()), 'seed': seed, 'nthread': 2}

def explain(model, X):
    model.set_param({'nthread': 2})
    dm = matrix(X)
    prob = model.predict(dm)
    margin = model.predict(dm, output_margin=True)
    contrib = model.predict(dm, pred_contribs=True, approx_contribs=False)
    a = float(np.max(np.abs(contrib.sum(axis=1) - margin)))
    b = float(np.max(np.abs(expit(margin) - prob)))
    assert contrib.shape == (len(X), X.shape[1] + 1) and np.isfinite(contrib).all()
    assert a < 0.0002 and b < 1e-07
    return (prob, margin, contrib, {'additivity_max_abs_error': a, 'sigmoid_probability_max_abs_error': b})

def train(state, gate):
    assert gate.resolve().is_relative_to(ROOT.parent / 'high_coverage') and read(gate)['passed'] is True
    config = read(ROOT / 'protocol.json')
    for c in COHORTS:
        X, y, g, names, roles, baseline = data(c)
        for seed in SEEDS:
            folder = ROOT / f'private/{c}/seed_{seed}'
            assert not folder.exists(), 'Do not overwrite or silently resume a partial run'
            folder.mkdir(parents=True)
            predictions = np.full(len(y), np.nan)
            contributions = np.full((len(y), len(names) + 1), np.nan, dtype=np.float32)
            margins = np.full(len(y), np.nan, dtype=np.float32)
            folds = np.zeros(len(y), dtype=np.int8)
            reports = []
            started = time.monotonic()
            for fold, role in enumerate(roles, 1):
                dest = folder / f'fold_{fold}'
                dest.mkdir()
                test, inner, stop, full = [np.flatnonzero(condition) for condition in (role == 0, role == 2, role == 3, role != 0)]
                if seed == 42:
                    old = BASE / f'private/runs/{c}_original/fold_{fold}'
                    package = joblib.load(old / 'full_package.joblib')
                    assert package['input_feature_names'] == names and package['fitted_feature_names'] == names
                    prep = package['preprocessor']
                    assert prep['zero_fill'] and prep['imputer'] is None and np.array_equal(prep['columns'], np.arange(len(names)))
                    assert package['parameters']['xgb_parameters']['seed'] == 42
                    model_path = old / 'full_xgb.ubj'
                    model = xgb.Booster(params={'nthread': 2})
                    model.load_model(model_path)
                    rounds = read(old / 'fit_report.json')['selection']['selected_rounds']
                    assert model.num_boosted_rounds() == rounds
                    model_files = [identity(model_path), identity(old / 'full_package.joblib')]
                else:
                    history = {}
                    early = xgb.train(params(y[inner], seed), matrix(X[inner], y[inner]), num_boost_round=config['xgb']['max_rounds'], evals=[(matrix(X[stop], y[stop]), 'inner_stop')], early_stopping_rounds=config['xgb']['early_stopping_rounds'], evals_result=history, verbose_eval=False)
                    rounds = int(early.best_iteration) + 1
                    assert rounds == int(np.argmax(history['inner_stop']['auc'])) + 1
                    early.save_model(dest / 'inner_xgb.ubj')
                    save(dest / 'inner_evaluation.json', {'selected_rounds': rounds, 'history': history, 'parameters': params(y[inner], seed), 'inner_rows': len(inner), 'stop_rows': len(stop)})
                    model = xgb.train(params(y[full], seed), matrix(X[full], y[full]), num_boost_round=rounds, verbose_eval=False)
                    model_path = dest / 'full_xgb.ubj'
                    model.save_model(model_path)
                    joblib.dump({'preprocessor': {'zero_fill': True, 'columns': np.arange(len(names)), 'imputer': None}, 'input_feature_names': names, 'fitted_feature_names': names, 'parameters': params(y[full], seed), 'selected_rounds': rounds}, dest / 'full_package.joblib', compress=3)
                    model_files = [identity(p) for p in sorted(dest.glob('*'))]
                probability, margin, contrib, checks = explain(model, X[test])
                if seed == 42:
                    assert np.max(np.abs(probability - baseline[test])) < 1e-07
                predictions[test], margins[test], contributions[test], folds[test] = (probability, margin, contrib, fold)
                report = {'cohort': c, 'seed': seed, 'fold': fold, 'selected_rounds': rounds, 'model_path': model_path.as_posix(), 'model_identities': model_files, 'inner_fit_rows': len(inner), 'inner_stop_rows': len(stop), 'outer_train_rows': len(full), 'outer_test_rows': len(test), 'patient_role_intersections': 0, **checks}
                save(dest / 'fit_report.json', report)
                reports.append(report)
                print(f'{c} seed {seed} fold {fold}/5 complete; rounds={rounds}', flush=True)
            assert np.isfinite(predictions).all() and np.isfinite(contributions).all() and np.all(folds > 0)
            np.savez_compressed(folder / 'heldout.npz', probability=predictions, margin=margins, contributions=contributions, fold=folds, feature_names=np.array(names + ['BIAS']))
            save(ROOT / f'results/{c}_seed_{seed}_training.json', {'execution_complete': True, 'cohort': c, 'seed': seed, 'rows': len(y), 'patients': len(np.unique(g)), 'positive_labels': int(y.sum()), 'new_full_models_fitted': 0 if seed == 42 else 5, 'seconds': time.monotonic() - started, 'analysis_lock': identity(ROOT / 'analysis_lock.json'), 'previous_stage_verification': identity(gate), 'fold_reports': reports, 'heldout': identity(folder / 'heldout.npz')})
    check_sources(state)

def association(raw, s):
    valid = np.isfinite(raw)
    a, b = (raw[valid], s[valid])
    reason = 'fewer_than_three_observations' if len(a) < 3 else 'constant_raw_values' if np.unique(a).size < 2 else 'constant_contributions' if np.unique(b).size < 2 else None
    result = {'n_observed': len(a), 'n_observed_zero': int((a == 0).sum()), 'testable': reason is None, 'not_testable_reason': reason, 'rho': None, 'p_nominal': None}
    if reason is None:
        rho, p = spearmanr(a, b)
        result.update(rho=float(rho), p_nominal=float(p))
    return result

def bh(p):
    p = np.asarray(p)
    order = np.argsort(p, kind='stable')
    q = np.empty(len(p))
    q[order] = np.minimum(1.0, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    assert np.allclose(q, false_discovery_control(p, method='bh'), atol=1e-14, rtol=1e-14)
    return q

def direct_metrics(y, p, w=None):
    return np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w)])

class Metric:

    def __init__(self, y, p):
        self.y, self.p = (y, p)
        self.order = np.argsort(-p, kind='stable')
        self.starts = np.r_[0, np.flatnonzero(np.diff(p[self.order])) + 1]

    def evaluate(self, w):
        pos = np.add.reduceat((w * self.y)[self.order], self.starts)
        neg = np.add.reduceat((w * (1 - self.y))[self.order], self.starts)
        tp, fp = (np.r_[0.0, np.cumsum(pos)], np.r_[0.0, np.cumsum(neg)])
        assert tp[-1] > 0 and fp[-1] > 0
        precision = np.divide(tp[1:], tp[1:] + fp[1:], out=np.zeros_like(pos, dtype=float), where=tp[1:] + fp[1:] > 0)
        return np.array([np.trapezoid(tp / tp[-1], fp / fp[-1]), np.sum(pos * precision) / tp[-1], np.average((self.p - self.y) ** 2, weights=w)])

def load_result(c, seed):
    report = read(ROOT / f'results/{c}_seed_{seed}_training.json')
    assert report['analysis_lock'] == identity(ROOT / 'analysis_lock.json')
    assert identity(report['heldout']['path']) == report['heldout']
    for fold in report['fold_reports']:
        for item in fold['model_identities']:
            assert identity(item['path']) == item
    with np.load(report['heldout']['path'], allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def summarize(state):
    config = read(ROOT / 'protocol.json')
    associations, foldstats, pairs, ranks, metricrows, deltas, families = ([], [], [], [], [], [], [])
    mean_abs = {}
    for c in COHORTS:
        X, y, g, names, roles, baseline = data(c)
        probabilities = []
        for seed in SEEDS:
            z = load_result(c, seed)
            assert z['feature_names'].tolist() == names + ['BIAS']
            probabilities.append(z['probability'])
            s = z['contributions'][:, :-1]
            mean_abs[c, seed] = np.abs(s).mean(axis=0, dtype=np.float64)
            for j, name in enumerate(names):
                row = {'seed': seed, 'cohort': c, 'feature_index': j, 'feature': name, **association(X[:, j], s[:, j]), 'mean_abs_contribution_all': float(mean_abs[c, seed][j])}
                signs = []
                for fold, role in enumerate(roles, 1):
                    stat = association(X[role == 0, j], s[role == 0, j])
                    foldstats.append({'seed': seed, 'cohort': c, 'feature_index': j, 'feature': name, 'fold': fold, **stat})
                    if stat['testable']:
                        signs.append(int(np.sign(stat['rho'])))
                row.update(folds_positive=signs.count(1), folds_negative=signs.count(-1), folds_zero=signs.count(0), folds_untestable=5 - len(signs), all_five_fold_sign_agreement=bool(row['testable'] and row['rho'] != 0 and (len(signs) == 5) and all((sg == np.sign(row['rho']) for sg in signs))))
                associations.append(row)
        score = [Metric(y, p) for p in probabilities]
        points = np.array([direct_metrics(y, p) for p in probabilities])
        rng = np.random.default_rng(config['bootstrap']['seeds'][c])
        draws = np.empty((2000, 4, 3))
        for b in range(2000):
            weights = rng.multinomial(len(y), np.full(len(y), 1 / len(y)))
            draws[b] = [m.evaluate(weights) for m in score]
            if b < 10:
                assert np.allclose(draws[b], [direct_metrics(y, p, weights) for p in probabilities], atol=2e-12, rtol=0)
        np.savez_compressed(ROOT / f'private/{c}/bootstrap.npz', draws=draws, points=points, seeds=np.array(SEEDS), metrics=np.array(METRICS))
        for i, seed in enumerate(SEEDS):
            for k, metric in enumerate(METRICS):
                low, high = np.quantile(draws[:, i, k], [0.025, 0.975])
                metricrows.append({'cohort': c, 'seed': seed, 'metric': metric, 'estimate': float(points[i, k]), 'ci_low': float(low), 'ci_high': float(high), 'rows': len(y), 'patients': len(y), 'positive_labels': int(y.sum())})
                if i:
                    low, high = np.quantile(draws[:, i, k] - draws[:, 0, k], [0.025, 0.975])
                    deltas.append({'cohort': c, 'seed': seed, 'reference_seed': 42, 'metric': 'delta_' + metric, 'estimate': float(points[i, k] - points[0, k]), 'ci_low': float(low), 'ci_high': float(high)})
        for a, b in itertools.combinations(SEEDS, 2):
            aa, bb = (mean_abs[c, a], mean_abs[c, b])
            nonzero = (aa > 0) | (bb > 0)
            result = {'cohort': c, 'first_seed': a, 'second_seed': b, 'rho_all': float(spearmanr(aa, bb).statistic), 'rho_nonzero': float(spearmanr(aa[nonzero], bb[nonzero]).statistic), 'nonzero_union_n': int(nonzero.sum())}
            for k in (10, 20):
                ia, ib = (np.argsort(-aa, kind='stable'), np.argsort(-bb, kind='stable'))
                result[f'top_{k}_overlap_n'] = len(set(ia[aa[ia] > 0][:k]) & set(ib[bb[ib] > 0][:k]))
            ranks.append(result)
    for seed in SEEDS:
        family = [r for r in associations if r['seed'] == seed and r['testable']]
        for row, q in zip(family, bh([r['p_nominal'] for r in family])):
            row['q_bh_joint_family'] = float(q)
        families.append({'seed': seed, 'testable_family_n': len(family), **{c + '_testable_n': sum((r['cohort'] == c for r in family)) for c in COHORTS}})
    for row in associations:
        row.setdefault('q_bh_joint_family', None)
    lookup = {(r['seed'], r['cohort'], r['feature']): r for r in associations}
    for seed in SEEDS:
        for feature in names:
            a, b = (lookup[seed, 'SCH', feature], lookup[seed, 'WHU', feature])
            testable = a['testable'] and b['testable']
            opposite = bool(testable and a['rho'] * b['rho'] < 0)
            bothq = bool(testable and a['q_bh_joint_family'] < 0.05 and (b['q_bh_joint_family'] < 0.05))
            folds = a['all_five_fold_sign_agreement'] and b['all_five_fold_sign_agreement']
            pairs.append({'seed': seed, 'feature': feature, 'testable_both': testable, 'SCH_rho': a['rho'], 'WHU_rho': b['rho'], 'SCH_q_joint': a['q_bh_joint_family'], 'WHU_q_joint': b['q_bh_joint_family'], 'opposite_signs': opposite, 'both_q_below_0_05': bothq, 'both_all_five_fold_sign_agreement': folds, 'opposite_both_q_and_five_fold_agreement': bool(opposite and bothq and folds)})
    old = pd.read_csv(OLD / 'results/feature_associations.csv')
    for row in [r for r in associations if r['seed'] == 42]:
        match = old[(old.cohort == row['cohort']) & (old.feature == row['feature'])].iloc[0]
        for key in ('rho', 'p_nominal', 'q_bh_joint_family', 'mean_abs_contribution_all'):
            assert row[key] is None and pd.isna(match[key]) or np.isclose(row[key], match[key], atol=1e-13, rtol=1e-12)
    oldpairs = pd.read_csv(OLD / 'results/cross_cohort_associations.csv')
    selected = oldpairs.loc[oldpairs.opposite_both_q_and_five_fold_agreement, 'feature'].tolist()
    assert len(selected) == 8 and families[0]['testable_family_n'] == 156
    baseline_selected = [r['feature'] for r in pairs if r['seed'] == 42 and r['opposite_both_q_and_five_fold_agreement']]
    assert baseline_selected == selected
    examples = ['神经元特异性烯醇化酶', '白蛋白']
    assert all((f in names for f in examples))
    featured = [r | {'prespecified_group': 'original_eight' if r['feature'] in selected else 'always_display_example'} for r in associations if r['feature'] in selected + examples]
    retention = []
    for r in pairs:
        if r['feature'] in selected:
            base = next((x for x in pairs if x['seed'] == 42 and x['feature'] == r['feature']))
            retention.append(r | {'baseline_selected': True, 'same_pooled_signs_as_seed42': bool(r['testable_both'] and np.sign(r['SCH_rho']) == np.sign(base['SCH_rho']) and (np.sign(r['WHU_rho']) == np.sign(base['WHU_rho'])))})
    seedsummary = []
    sets = []
    for seed in SEEDS:
        chosen = [r['feature'] for r in pairs if r['seed'] == seed and r['opposite_both_q_and_five_fold_agreement']]
        sets.append(set(chosen))
        seedsummary.append({'seed': seed, 'stable_opposite_features': chosen, 'stable_opposite_n': len(chosen), 'original_eight_retained': [f for f in selected if f in chosen], 'original_eight_retained_n': len(set(selected) & set(chosen)), 'common_testable_features_n': sum((r['seed'] == seed and r['testable_both'] for r in pairs))})
    rankranges = {c: {key: {'min': min((r[key] for r in ranks if r['cohort'] == c)), 'max': max((r[key] for r in ranks if r['cohort'] == c))} for key in ('rho_all', 'rho_nonzero', 'top_10_overlap_n', 'top_20_overlap_n')} for c in COHORTS}
    tables = {'feature_associations.csv': associations, 'fold_associations.csv': foldstats, 'cross_cohort_associations.csv': pairs, 'seed_rank_stability.csv': ranks, 'seed_metrics.csv': metricrows, 'seed_paired_comparisons.csv': deltas, 'prespecified_feature_stability.csv': featured, 'retention.csv': retention}
    outputs = []
    for filename, rows in tables.items():
        p = ROOT / 'results' / filename
        pd.DataFrame(rows).to_csv(p, index=False, mode='x', float_format='%.17g')
        outputs.append(identity(p))
    publication = {'seeds': list(SEEDS), 'baseline_seed': 42, 'cohorts': list(COHORTS), 'BH_families': families, 'seed_results': seedsummary, 'four_seed_intersection': [f for f in names if all((f in s for s in sets))], 'four_seed_intersection_n': len(set.intersection(*sets)), 'original_eight': selected, 'rank_ranges': rankranges, 'NSE_Albumin': [r for r in featured if r['feature'] in examples], 'metrics': metricrows, 'paired_vs_seed42': deltas, 'limits': config['limits']}
    save(ROOT / 'results/publication_summary.json', publication)
    save(ROOT / 'results/summary.json', {'execution_complete': True, 'completed_at': now(), 'analysis_lock': identity(ROOT / 'analysis_lock.json'), 'environment': environment(), 'BH_families': families, 'seed_results': seedsummary, 'rank_ranges': rankranges, 'outputs': outputs + [identity(ROOT / 'results/publication_summary.json')], 'bootstrap_draws': 2000, 'model_scope': 'XGBoost component only', 'new_full_models': 30, 'reused_baseline_models': 10, 'patient_rows_or_ids_in_public_outputs': False, 'limits': config['limits']})
    check_sources(state)

def self_test():
    rng = np.random.default_rng(2818)
    count = 0
    for n in (31, 110, 203):
        y = rng.integers(0, 2, n)
        y[:2] = [0, 1]
        for p in (rng.random(n), np.round(rng.random(n), 1), np.full(n, 0.4)):
            for _ in range(10):
                w = rng.integers(0, 4, n)
                w[:2] = 1
                assert np.allclose(Metric(y, p).evaluate(w), direct_metrics(y, p, w), atol=2e-12, rtol=0)
                count += 1
    for n in (1, 10, 156, 208):
        bh(rng.random(n))
    r = association(np.array([np.nan, 0.0, 1.0, 2.0]), np.array([99.0, 0.0, 2.0, 3.0]))
    assert r['n_observed'] == 3 and r['n_observed_zero'] == 1 and (r['rho'] == 1)
    assert not association(np.zeros(5), np.arange(5))['testable']
    assert not association(np.arange(5), np.zeros(5))['testable']
    assert params(np.array([0, 0, 0, 1]), 44)['scale_pos_weight'] == 3
    return {'passed': True, 'weighted_sklearn_metric_cases': count, 'observed_zero_and_missingness': True, 'full_BH_reference': True, 'fit_only_weight': True}

def verify(state):
    checks = []
    for c in COHORTS:
        X, y, g, names, roles, baseline = data(c)
        for seed in SEEDS:
            z = load_result(c, seed)
            report = read(ROOT / f'results/{c}_seed_{seed}_training.json')
            for fold, role in enumerate(roles, 1):
                test = np.flatnonzero(role == 0)
                info = report['fold_reports'][fold - 1]
                model = xgb.Booster(params={'nthread': 2})
                model.load_model(info['model_path'])
                assert model.num_boosted_rounds() == info['selected_rounds']
                p, m, s, errors = explain(model, X[test])
                assert np.array_equal(p, z['probability'][test]) and np.array_equal(s, z['contributions'][test]) and np.array_equal(m, z['margin'][test])
                if seed != 42:
                    inner = read(ROOT / f'private/{c}/seed_{seed}/fold_{fold}/inner_evaluation.json')
                    assert info['selected_rounds'] == int(np.argmax(inner['history']['inner_stop']['auc'])) + 1
                    assert inner['parameters'] == params(y[role == 2], seed)
                    pack = joblib.load(ROOT / f'private/{c}/seed_{seed}/fold_{fold}/full_package.joblib')
                    assert pack['parameters'] == params(y[role != 0], seed) and pack['input_feature_names'] == names
                checks.append({'cohort': c, 'seed': seed, 'fold': fold, 'probabilities_exact': True, 'contributions_exact': True, **errors})
    check_sources(state)
    summary = read(ROOT / 'results/summary.json')
    for item in summary['outputs']:
        assert identity(item['path']) == item
    save(ROOT / 'verification.json', {'passed': True, 'completed_at': now(), 'analysis_lock': identity(ROOT / 'analysis_lock.json'), 'summary': identity(ROOT / 'results/summary.json'), 'saved_models_replayed': 40, 'checks': checks, 'roles_reconstructed': True, 'source_files_unchanged': True, 'self_tests': self_test(), 'scope': 'Separate saved-model invocation; full aggregate independent verification recorded separately, not a second TreeSHAP implementation'})
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('self-test', 'lock', 'train', 'summarize', 'verify'))
    parser.add_argument('--stage-2-verification', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.mode == 'self-test':
            print(json.dumps(self_test()))
        else:
            state = lock()
            if args.mode == 'lock':
                print(json.dumps({'locked': True, 'source_count': len(state['sources']), 'lock': identity(ROOT / 'analysis_lock.json')}))
            elif args.mode == 'train':
                assert args.stage_2_verification is not None
                train(state, args.stage_2_verification.resolve())
            elif args.mode == 'summarize':
                summarize(state)
            else:
                verify(state)
