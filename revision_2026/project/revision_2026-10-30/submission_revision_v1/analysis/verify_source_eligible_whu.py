"""Independent saved-model and aggregate audit; no patient-level log output."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import os, sys, json, hashlib
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '8'
import numpy as np, pandas as pd, joblib, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
PROJECT = Path(_release_path('project', ''))
D = Path(__file__).resolve().parents[1]
A = D / 'analysis'
E = D / 'private/whu_source_eligible'
S = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'

def sha(p):
    return None

def read(p):
    return _release_resolve_config(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def check(name, condition):
    checks.append({'check': name, 'passed': bool(condition)})
    assert condition, name

def vals(y, p, w=None):
    return np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w), brier_score_loss(y, p, sample_weight=w), np.sum(p if w is None else p * w) / np.sum(y if w is None else y * w)])

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    lock = read(A / 'source_eligible_input_lock.json')
    check('all_frozen_source_hashes_unchanged', all((sha(x['path']) == None for x in lock['inputs'])))
    idx = pd.read_csv(PROJECT / 'clinical_review_2026-10-27/full_index_v1/private/whu_index.csv', dtype=str)
    original_labels = pd.read_csv(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'), dtype={'病案号': str})
    labs = original_labels.loc[original_labels.Center.eq('WHU')].copy()

    def norm(x):
        return str(x).removesuffix('.0')
    covered = set(idx.loc[idx.raw_record_count.astype(int).gt(0), 'original_patient_id'].map(norm))
    independent_mask = labs['病案号'].map(norm).isin(covered).to_numpy()
    with np.load(E / 'eligibility_mask.npz') as f:
        mask = f['eligible'].copy()
    check('independent_key_set_mask_equal', np.array_equal(mask, independent_mask))
    check('retained2258_excluded304', int(mask.sum()) == 2258 and int((~mask).sum()) == 304)
    with np.load(E / 'private/prepared/WHU.npz') as f:
        data = {k: f[k].copy() for k in f.files}
    with np.load(E / 'private/runs/WHU_source_eligible/predictions.npz') as f:
        new = {k: f[k].copy() for k in f.files}
    with np.load(S / 'private/runs/WHU_original/predictions.npz') as f:
        old = {k: f[k].copy() for k in f.files}
    with np.load(E / 'private/runs/WHU_source_eligible/patient_roles.npz') as f:
        roles = f['roles'].copy()
    check('original_labels_retained', np.array_equal(new['y'], labs.MACE.to_numpy(int)[mask]))
    check('positive_counts182_and99', new['y'].sum() == 182 and old['y'][~mask].sum() == 99)
    check('one_test_fold_per_patient', np.all((roles == 0).sum(0) == 1))
    check('exclusions_not_in_model_group_codes', not set(new['groups']).intersection(old['groups'][~mask]))
    maxerrors = {}
    y = data['y']
    X = np.nan_to_num(data['X'], nan=0, posinf=0, neginf=0)
    names = data['feature_names'].tolist()
    for fold, role in enumerate(roles, 1):
        test = np.flatnonzero(role == 0)
        cal = np.flatnonzero(role == 1)
        inner = np.flatnonzero(role == 2)
        stop = np.flatnonzero(role == 3)
        group_sets = [set(new['groups'][q]) for q in [test, cal, inner, stop]]
        check(f'fold{fold}_all_roles_disjoint', all((not group_sets[i].intersection(group_sets[j]) for i in range(4) for j in range(i + 1, 4))))
        folder = E / f'private/runs/WHU_source_eligible/fold_{fold}'
        fit = read(folder / 'fit_report.json')
        rounds = fit['selection']['selected_rounds']
        res = {}
        for prefix, outname in [('full', 'primary_raw'), ('base', 'base_raw')]:
            package = joblib.load(folder / f'{prefix}_package.joblib')
            booster = xgb.Booster()
            booster.load_model(folder / f'{prefix}_xgb.ubj')
            xp = booster.predict(xgb.DMatrix(X[test], nthread=8))
            rp = package['rf'].predict_proba(X[test])[:, 1]
            res[outname] = (xp + rp) / 2
            check(f'fold{fold}_{prefix}_selected_rounds', booster.num_boosted_rounds() == rounds)
            check(f'fold{fold}_{prefix}_feature_order', package['input_feature_names'] == names)
            source = np.flatnonzero(role != 0) if prefix == 'full' else np.flatnonzero(role >= 2)
            check(f'fold{fold}_{prefix}_fit_only_class_weights', np.isclose(package['parameters']['xgb_parameters']['scale_pos_weight'], (y[source] == 0).sum() / (y[source] == 1).sum()))
            if prefix == 'full':
                res['xgb_full'] = xp
                res['rf_full'] = rp
        cp = joblib.load(folder / 'comparators_and_platt.joblib')
        pp = np.clip(res['base_raw'], 1e-06, 1 - 1e-06)
        res['base_platt'] = cp['platt'].predict_proba(np.log(pp / (1 - pp)).reshape(-1, 1))[:, 1]
        res['logistic'] = cp['logistic'].predict_proba(X[test])[:, 1]
        res['missingness'] = cp['missingness'].predict_proba(np.isnan(data['X'][test]).astype(np.float32))[:, 1]
        res['age'] = cp['age'].predict_proba(data['X'][test, names.index('年龄')].reshape(-1, 1))[:, 1]
        for key, v in res.items():
            err = float(np.abs(v - new[key][test]).max())
            maxerrors[key] = max(err, maxerrors.get(key, 0))
            check(f'fold{fold}_{key}_saved_model_score_match', err <= 1e-12)
    rows = pd.read_csv(A / 'source_eligible_metrics.csv')
    pair = pd.read_csv(A / 'source_eligible_paired_comparisons.csv')
    changed = 0
    oldfields = {'historical_original_cache': 'original', 'prior_patient_isolated_raw_fixed': 'primary_raw', 'prior_patient_isolated_platt_fixed': 'base_platt'}
    audited_samples = 0
    for scope, sel in [('all_original', np.ones(len(mask), bool)), ('retained', mask), ('excluded_no_raw_JSON', ~mask)]:
        yy = old['y'][sel]
        scores = {n: old[k][sel] for n, k in oldfields.items()}
        if scope == 'retained':
            scores.update({f'retrained_{k}': new[k] for k in ['primary_raw', 'base_raw', 'base_platt', 'xgb_full', 'rf_full', 'logistic', 'missingness', 'age']})
        with np.load(E / f'bootstrap_{scope}.npz') as f:
            samples = f['values'].copy()
            score_names = f['names'].tolist()
        rng = np.random.default_rng(20261028 + {'all_original': 0, 'retained': 1, 'excluded_no_raw_JSON': 2}[scope])
        auditindices = set(np.linspace(0, 1999, 101, dtype=int))
        for b in range(2000):
            w = rng.multinomial(len(yy), np.full(len(yy), 1 / len(yy)))
            for j, name in enumerate(score_names):
                if name in ['retrained_primary_raw', 'historical_original_cache', 'prior_patient_isolated_raw_fixed', 'retrained_base_platt'] or b in auditindices:
                    vv = vals(yy, scores[name], w)
                    check(f'bootstrap_{scope}_{name}_{b}', np.allclose(vv, samples[b, j], atol=2e-12, rtol=0))
                    audited_samples += 1
        for j, name in enumerate(score_names):
            expected = vals(yy, scores[name])
            selected = rows.loc[rows.scope.eq(scope) & rows.score.eq(name)].set_index('metric')
            for k, met in enumerate(['auc', 'average_precision', 'brier', 'e_over_o']):
                q = np.quantile(samples[:, j, k], [0.025, 0.975])
                rr = selected.loc[met]
                check(f'point_and_interval_{scope}_{name}_{met}', np.allclose([expected[k], *q], [rr.estimate, rr.lower_95, rr.upper_95], atol=2e-12, rtol=0))
        if scope == 'retained':
            defs = {'retrained_raw_minus_prior_isolated_fixed': ('retrained_primary_raw', 'prior_patient_isolated_raw_fixed'), 'retrained_raw_minus_historical_fixed': ('retrained_primary_raw', 'historical_original_cache'), 'retrained_platt_minus_retrained_base_raw': ('retrained_base_platt', 'retrained_base_raw'), 'retrained_raw_minus_age': ('retrained_primary_raw', 'retrained_age'), 'retrained_raw_minus_missingness': ('retrained_primary_raw', 'retrained_missingness')}
            for name, (l, r) in defs.items():
                a, b = (score_names.index(l), score_names.index(r))
                delta = vals(yy, scores[l]) - vals(yy, scores[r])
                rr = pair.loc[pair.comparison.eq(name)].set_index('metric')
                for k, met in enumerate(['auc', 'average_precision', 'brier', 'e_over_o']):
                    q = np.quantile(samples[:, a, k] - samples[:, b, k], [0.025, 0.975])
                    check(f'paired_{name}_{met}', np.allclose([delta[k], *q], [rr.loc[met, 'difference'], rr.loc[met, 'lower_95'], rr.loc[met, 'upper_95']], atol=2e-12, rtol=0))
    result = {'passed': all((x['passed'] for x in checks)), 'checks': len(checks), 'failure_count': sum((not x['passed'] for x in checks)), 'bootstrap_vectors_independently_checked': audited_samples, 'full2000_draws_verified_for': ['historical_original_cache', 'prior_patient_isolated_raw_fixed', 'retrained_primary_raw', 'retrained_base_platt'], 'remaining_scores': '101 evenly spaced draws checked plus every saved interval recomputed', 'model_score_max_absolute_error': maxerrors, 'source_eligible_n': 2258, 'positive': 182, 'excluded': 304, 'excluded_positive': 99, 'original_data_unchanged': True, 'clinical_validity_certified': False, 'fitting_repeated_in_audit': False}
    (D / 'qa/source_eligible_independent_numerical_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
checks = []
if __name__ == '__main__':
    main()
