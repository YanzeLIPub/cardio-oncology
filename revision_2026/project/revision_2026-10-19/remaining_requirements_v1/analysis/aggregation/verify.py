"""Replay every stored model and cross-check bootstrap calculations with sklearn."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from prepare import ROOT, OLD, identity, save_json
from summarize import VARIANTS, METRICS, STRATA, masks

def zero(X):
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def close(a, b, atol=2e-14):
    np.testing.assert_allclose(a, b, rtol=0, atol=atol)

def main():
    lock = _release_resolve_config(json.loads((ROOT / 'analysis_lock_v2.json').read_text(encoding='utf-8')))
    for item in lock['state']['files']:
        assert identity(item['path']) == item
    protocol = lock['protocol']
    model_checks = []
    maximum_rf_replay_discrepancy = 0.0
    maximum_mean_reference_discrepancy = 0.0
    for center in ('SCH', 'WHU'):
        data = dict(np.load(ROOT / f'private/prepared/{center}.npz', allow_pickle=False))
        pred = dict(np.load(ROOT / f'private/runs/{center}/predictions.npz', allow_pickle=False))
        original = dict(np.load(OLD / f'private/prepared/{center}.npz', allow_pickle=False))
        oldroles = dict(np.load(OLD / f'private/runs/{center}_original/patient_roles.npz', allow_pickle=False))
        y = data['y']
        assert pred['processed_record_count'].dtype.kind in 'iu'
        assert np.array_equal(pred['processed_record_count'], data['record_count'])
        assert (pred['processed_record_count'] >= 1).all()
        assert ((pred['record_count'] >= 0) & (pred['record_count'] <= 1)).all()
        for actual, expected in zip(masks(pred['processed_record_count']), masks(data['record_count'])):
            assert np.array_equal(actual, expected)
        assert np.array_equal(data['X_mean'], original['X'], equal_nan=True)
        for name in ('y', 'groups'):
            assert np.array_equal(data[name], original[name])
        assert np.array_equal(data['roles'], oldroles['roles'])
        assert np.array_equal(data['roles'], pred['roles'])
        assert np.all((data['roles'] == 0).sum(axis=0) == 1)
        meta = np.setdiff1d(np.arange(104), data['lab_indices'])
        for variant in VARIANTS[:4]:
            assert np.array_equal(data['X_' + variant][:, meta], data['X_mean'][:, meta], equal_nan=True)
        assert np.array_equal(np.isnan(data['X_first_observed']), np.isnan(data['X_mean']))
        one = data['record_count'] == 1
        for variant in VARIANTS[:4]:
            assert np.array_equal(data['X_' + variant][one], data['X_mean'][one], equal_nan=True)
        for fold, roles in enumerate(data['roles'], 1):
            test = np.flatnonzero(roles == 0)
            train = np.flatnonzero(roles != 0)
            inner, stop = (np.flatnonzero(roles == 2), np.flatnonzero(roles == 3))
            assert len(set(test) & set(train)) == 0
            assert len(set(inner) & set(stop)) == 0
            countpath = ROOT / f'private/runs/{center}/fold_{fold}_record_count.joblib'
            count = joblib.load(countpath)
            log_count = np.log1p(data['record_count']).reshape(-1, 1)
            close(count[0].mean_, np.mean(log_count[train], axis=0))
            close(count[0].var_, np.var(log_count[train], axis=0))
            close(count.predict_proba(log_count[test])[:, 1], pred['record_count'][test])
            for variant in VARIANTS[:4]:
                X = data['X_' + variant]
                if variant == 'mean':
                    dest = OLD / f'private/runs/{center}_original/fold_{fold}'
                    pack = joblib.load(dest / 'full_package.joblib')
                    sourcepred = dict(np.load(OLD / f'private/runs/{center}_original/predictions.npz', allow_pickle=False))
                    assert np.array_equal(pred[variant][test], sourcepred['primary_raw'][test])
                    maximum_mean_reference_discrepancy = max(maximum_mean_reference_discrepancy, float(np.max(np.abs(pred['mean_rf'][test] - sourcepred['rf_full'][test]))))
                else:
                    dest = ROOT / f'private/runs/{center}/fold_{fold}_{variant}'
                    pack = joblib.load(dest / 'full_package.joblib')
                    fit = _release_resolve_config(json.loads((dest / 'fit_report.json').read_text(encoding='utf-8')))
                    early = xgb.Booster()
                    early.load_model(dest / 'inner_xgb.ubj')
                    early.set_param({'nthread': 2})
                    history = np.asarray(fit['inner_history']['inner_stop']['auc'])
                    assert fit['selected_rounds'] == int(np.argmax(history)) + 1 == int(early.best_iteration) + 1
                    for end in range(1, early.num_boosted_rounds() + 1):
                        p = early.predict(xgb.DMatrix(zero(X[stop]), nthread=2), iteration_range=(0, end))
                        close(roc_auc_score(y[stop], p), history[end - 1], atol=1e-12)
                    assert fit['inner_parameters']['scale_pos_weight'] == float((y[inner] == 0).sum() / (y[inner] == 1).sum())
                    assert fit['final_parameters']['scale_pos_weight'] == float((y[train] == 0).sum() / (y[train] == 1).sum())
                    assert fit['n_inner_fit'] == len(inner) and fit['n_inner_stop'] == len(stop)
                    assert fit['n_outer_train'] == len(train) and fit['n_outer_test'] == len(test)
                assert pack['input_feature_names'] == data['feature_names'].tolist()
                booster = xgb.Booster()
                booster.load_model(dest / 'full_xgb.ubj')
                booster.set_param({'nthread': 2})
                if variant != 'mean':
                    assert booster.num_boosted_rounds() == pack['selected_rounds']
                rf = pack['rf']
                assert rf.n_estimators == 300 and rf.max_depth == 8 and (rf.min_samples_leaf == 5) and (rf.random_state == 42)
                assert rf.class_weight == {0: 1, 1: int((y[train] == 0).sum() / (y[train] == 1).sum())}
                assert len(rf.estimators_) == 300 and rf.n_features_in_ == 104
                rf.n_jobs = 2
                xp = booster.predict(xgb.DMatrix(zero(X[test]), nthread=2))
                rp = rf.predict_proba(zero(X[test]))[:, 1]
                assert np.array_equal(xp, pred[variant + '_xgb'][test])
                close(rp, pred[variant + '_rf'][test], atol=2e-15)
                maximum_rf_replay_discrepancy = max(maximum_rf_replay_discrepancy, float(np.max(np.abs(rp - pred[variant + '_rf'][test]))))
                close(0.5 * xp + 0.5 * rp, pred[variant][test], atol=2e-15)
                model_checks.append({'center': center, 'fold': fold, 'variant': variant, 'held_out_probability_replay_passed': True, 'input_features': rf.n_features_in_, 'xgb_rounds': booster.num_boosted_rounds()})
            print(json.dumps({'verified_models': len(model_checks), 'center': center, 'fold': fold}), flush=True)
    bootstrap = dict(np.load(ROOT / 'private/bootstrap.npz', allow_pickle=False))
    all_metrics = pd.read_csv(ROOT / 'results/metrics.csv')
    all_pairs = pd.read_csv(ROOT / 'results/paired_comparisons.csv')
    cross = pd.read_csv(ROOT / 'results/between_centre_gains.csv')
    direct_bootstrap_checks = 0
    for cidx, center in enumerate(('SCH', 'WHU')):
        pred = dict(np.load(ROOT / f'private/runs/{center}/predictions.npz', allow_pickle=False))
        for sidx, (stratum, mask) in enumerate(zip(STRATA, masks(pred['processed_record_count']))):
            y = pred['y'][mask]
            n = len(y)
            key = f'{center}_{sidx}'
            rng = np.random.default_rng(protocol['bootstrap_seed'] + cidx * 10000 + sidx)
            for b in range(protocol['bootstrap_replicates']):
                draws = rng.integers(0, n, n)
                if b not in (0, 1, 2, 3, 4, 19, 39, 99, 199, 499, 999, 1499, 1999):
                    continue
                if len(np.unique(y[draws])) < 2:
                    assert not np.isfinite(bootstrap[key][b]).any()
                    continue
                for j, v in enumerate(VARIANTS):
                    p = pred[v][mask][draws]
                    target = [roc_auc_score(y[draws], p), average_precision_score(y[draws], p), brier_score_loss(y[draws], p), p.sum() / y[draws].sum()]
                    close(bootstrap[key][b, j], target, atol=2e-14)
                    direct_bootstrap_checks += 1
            for row in all_metrics.loc[(all_metrics.center == center) & (all_metrics.record_stratum == stratum)].itertuples():
                j, k = (VARIANTS.index(row.variant), METRICS.index(row.metric))
                vals = bootstrap[key][:, j, k]
                close([row.estimate, row.ci_low, row.ci_high], [bootstrap[key + '_point'][j, k], *np.nanquantile(vals, [0.025, 0.975])])
                assert row.bootstrap_valid == np.isfinite(vals).sum()
            for row in all_pairs.loc[(all_pairs.center == center) & (all_pairs.record_stratum == stratum)].itertuples():
                j, k, m = (VARIANTS.index(row.variant), VARIANTS.index(row.reference), METRICS.index(row.metric))
                vals = bootstrap[key][:, j, m] - bootstrap[key][:, k, m]
                target = bootstrap[key + '_point'][j, m] - bootstrap[key + '_point'][k, m]
                close([row.difference, row.ci_low, row.ci_high], [target, *np.nanquantile(vals, [0.025, 0.975])])
    for row in cross.itertuples():
        sidx = STRATA.index(row.record_stratum)
        j, k, m = (VARIANTS.index(row.variant), VARIANTS.index(row.reference), METRICS.index(row.metric))
        wd = bootstrap[f'WHU_{sidx}'][:, j, m] - bootstrap[f'WHU_{sidx}'][:, k, m]
        sd = bootstrap[f'SCH_{sidx}'][:, j, m] - bootstrap[f'SCH_{sidx}'][:, k, m]
        wp, sp = (bootstrap[f'WHU_{sidx}_point'], bootstrap[f'SCH_{sidx}_point'])
        target = wp[j, m] - wp[k, m] - sp[j, m] + sp[k, m]
        close([row.difference_of_differences, row.ci_low, row.ci_high], [target, *np.nanquantile(wd - sd, [0.025, 0.975])])
    for item in lock['state']['files']:
        assert identity(item['path']) == item
    save_json(ROOT / 'verification.json', {'passed': True, 'script': identity(__file__), 'locked_sources_unchanged': True, 'prepared_means_exactly_match_prior': True, 'roles_and_labels_exactly_match_prior': True, 'nonlaboratory_inputs_held_fixed': True, 'raw_count_and_probability_schema_distinct': True, 'strata_exactly_reconstructed_from_original_integer_counts': True, 'one_record_input_variants_identical': True, 'model_checks': model_checks, 'outer_ensembles_replayed': len(model_checks), 'new_inner_boosters_all_round_histories_replayed': 30, 'record_count_models_replayed': 10, 'direct_sklearn_resample_checks': direct_bootstrap_checks, 'metric_rows_verified': len(all_metrics), 'paired_rows_verified': len(all_pairs), 'between_centre_rows_verified': len(cross), 'maximum_rf_replay_discrepancy': maximum_rf_replay_discrepancy, 'maximum_mean_reference_rf_discrepancy': maximum_mean_reference_discrepancy, 'floating_point_note': 'RF parallel reduction differences tolerated to 2e-15; XGBoost replay exact. Historical mean ensemble probabilities are preserved verbatim.', 'output_inputs': [identity(ROOT / name) for name in ('private/bootstrap.npz', 'results/metrics.csv', 'results/paired_comparisons.csv', 'results/between_centre_gains.csv', 'results/summary.json')]})
    print(json.dumps({'passed': True, 'ensembles_replayed': len(model_checks), 'direct_sklearn_resamples': direct_bootstrap_checks}), flush=True)
if __name__ == '__main__':
    main()
