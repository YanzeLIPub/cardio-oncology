"""Cross-agent verification without importing any analysis-author functions."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
LABELS = Path(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'))
RAW = {'SCH': Path(_release_path('china', 'SCH_processed_data.csv')), 'WHU': Path(_release_path('china', 'WHU_Cancer_processed_data.csv'))}
KEY, CHECK, ADMIT, EXIT = ('病案号', '检查时间', '入院时间', '出院时间')
REGIONAL = ['人均地区生产总值(元)', '职工平均工资(元)', '医院、卫生院数(个)', '医院、卫生院床位数(张)', '医生数(人)']

def ident(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def read(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8')))

def close(a, b, atol=2e-11):
    np.testing.assert_allclose(a, b, rtol=0, atol=atol)

def independent_metrics(y, p, weight=None):
    return np.array([roc_auc_score(y, p, sample_weight=weight), average_precision_score(y, p, sample_weight=weight), brier_score_loss(y, p, sample_weight=weight)])

def main():
    protocol, lock = (read(ROOT / 'protocol.json'), read(ROOT / 'analysis_lock.json'))
    context = read(ROOT / 'context_audit.json')
    precision = read(ROOT / 'clock_precision_audit.json')
    state = lock['state']
    assert state['protocol'] == protocol
    for item in state['sources'] + [state['script']] + context['sources']:
        assert ident(item['path']) == item
    tracked = [Path(item['path']) for item in state['sources'] + [state['script']] + context['sources']]
    tracked += [ROOT / name for name in ('protocol.json', 'analysis_lock.json', 'results/era_metrics.csv', 'results/era_paired_comparisons.csv', 'results/summary.json', 'clock_precision_audit.json')]
    for cohort in ('SCH', 'WHU'):
        tracked += list((ROOT / f'private/{cohort}').rglob('*.npz'))
        tracked += list((ROOT / f'private/{cohort}').rglob('*.ubj'))
        tracked += list((ROOT / f'private/{cohort}').rglob('*.joblib'))
        tracked += list((ROOT / f'private/{cohort}').rglob('fit_report.json'))
    before = [ident(p) for p in dict.fromkeys(tracked)]
    labels = pd.read_csv(LABELS, usecols=[KEY, 'Center', 'MACE'])
    metrics = pd.read_csv(ROOT / 'results/era_metrics.csv')
    paired = pd.read_csv(ROOT / 'results/era_paired_comparisons.csv')
    model_checks, date_checks, stats_checks = ([], [], [])
    for cohort in ('SCH', 'WHU'):
        print(json.dumps({'independent_stage': 'raw_dates', 'cohort': cohort}), flush=True)
        raw = pd.read_csv(RAW[cohort], usecols=[KEY, CHECK, ADMIT, EXIT], low_memory=False)
        local = labels.loc[labels.Center.eq(cohort)].reset_index(drop=True)
        assert raw[KEY].notna().all() and local[KEY].is_unique and (set(raw[KEY]) == set(local[KEY]))
        hhmm = {name: int(raw[name].astype('string').str.contains('\\d{1,2}:\\d{2}', regex=True, na=False).sum()) for name in (CHECK, ADMIT, EXIT)}
        for name in (CHECK, ADMIT, EXIT):
            raw[name] = pd.to_datetime(raw[name], format='mixed', errors='coerce')
        valid = raw[CHECK].between(pd.Timestamp('1970-01-01'), pd.Timestamp('2026-10-19'))
        complete = valid.groupby(raw[KEY]).all().reindex(local[KEY]).to_numpy()
        first = raw.groupby(KEY)[CHECK].min().reindex(local[KEY])
        last = raw.groupby(KEY)[CHECK].max().reindex(local[KEY])
        lower80 = first.iloc[np.flatnonzero(complete)].sort_values().iloc[int(np.floor(0.8 * (int(complete.sum()) - 1)))]
        boundary = pd.Timestamp(year=lower80.year, month=lower80.month, day=1)
        assert boundary.date().isoformat() == protocol['cutoffs'][cohort]
        role = np.full(len(local), -1, dtype=np.int8)
        earlier = complete & (last < boundary).to_numpy()
        later = complete & (first >= boundary).to_numpy()
        crossing = complete & ~earlier & ~later
        role[earlier], role[later], role[crossing] = (1, 0, -2)
        saved_dates = dict(np.load(ROOT / f'private/{cohort}_date_roles.npz', allow_pickle=False))
        assert np.array_equal(role, saved_dates['roles'])
        assert np.array_equal(first.to_numpy(dtype='datetime64[D]'), saved_dates['min_lab_day'])
        assert np.array_equal(last.to_numpy(dtype='datetime64[D]'), saved_dates['max_lab_day'])
        assert np.array_equal(raw.groupby(KEY).size().reindex(local[KEY]).to_numpy(), saved_dates['record_count'])
        discharge_complete = raw[EXIT].notna().groupby(raw[KEY]).all().reindex(local[KEY]).to_numpy()
        discharge_before = discharge_complete & (raw.groupby(KEY)[EXIT].max().reindex(local[KEY]) < boundary).to_numpy()
        assert np.array_equal(discharge_before, saved_dates['full_discharge_before_cutoff'])
        old = dict(np.load(BASE / f'private/prepared/{cohort}.npz', allow_pickle=False))
        assert np.array_equal(local.MACE.to_numpy(), old['y']) and np.array_equal(old['y'], saved_dates['y'])
        selected = np.flatnonzero(role >= 0)
        train = np.flatnonzero(role[selected] == 1)
        test = np.flatnonzero(role[selected] == 0)
        y = old['y'][selected]
        inner, stop = train_test_split(train, test_size=0.25, stratify=y[train], random_state=2043)
        expected_roles = np.full(len(selected), -1, dtype=np.int8)
        expected_roles[test], expected_roles[inner], expected_roles[stop] = (0, 2, 3)
        roledata = dict(np.load(ROOT / f'private/{cohort}/roles.npz', allow_pickle=False))
        assert np.array_equal(selected, roledata['selected_index'])
        assert np.array_equal(expected_roles, roledata['roles'])
        assert np.array_equal(old['groups'][selected], roledata['groups'])
        assert len(set(roledata['groups'][test]) & set(roledata['groups'][train])) == 0
        expected = protocol['expected_counts'][cohort]
        assert [len(train), int(y[train].sum()), len(test), int(y[test].sum()), int(crossing.sum())] == [expected[k] for k in ('train', 'train_positive', 'test', 'test_positive', 'crossing_excluded')]
        precision_row = next((r for r in precision['cohorts'] if r['cohort'] == cohort))
        assert hhmm == {CHECK: precision_row['laboratory_source_strings_with_hhmm'], ADMIT: precision_row['admission_source_strings_with_hhmm'], EXIT: precision_row['discharge_source_strings_with_hhmm']}
        assert int((raw[CHECK] < raw[ADMIT]).sum()) == precision_row['laboratory_calendar_day_before_admission_day']
        date_checks.append({'cohort': cohort, 'cutoff': boundary.date().isoformat(), 'train': len(train), 'test': len(test), 'test_positive': int(y[test].sum()), 'crossing_excluded': int(crossing.sum()), 'date_roles_reconstructed_from_raw': True, 'inner_roles_exactly_reconstructed': True, 'train_missing_or_later_discharge': int((earlier & ~discharge_before).sum()), 'raw_clock_strings_with_hhmm': hhmm, 'lab_day_before_attached_admission_day': int((raw[CHECK] < raw[ADMIT]).sum())})
        predictions = dict(np.load(ROOT / f'private/{cohort}/predictions.npz', allow_pickle=False))
        assert np.array_equal(predictions['y'], y[test]) and np.array_equal(predictions['groups'], old['groups'][selected][test])
        names = old['feature_names'].tolist()
        columns = {'full_104': np.arange(104), 'no_regional_99': np.array([i for i, name in enumerate(names) if name not in REGIONAL]), 'regional_only_5': np.array([names.index(name) for name in REGIONAL])}
        for model in protocol['models']:
            folder = ROOT / f'private/{cohort}/{model}'
            package = joblib.load(folder / 'full_package.joblib')
            fit = read(folder / 'fit_report.json')
            cols = columns[model]
            assert np.array_equal(package['columns'], cols)
            assert package['feature_names'] == [names[i] for i in cols] == fit['feature_names']
            assert package['zero_fill'] and fit['input_column_indices'] == cols.tolist()
            assert not np.isinf(old['X']).any()
            X = np.nan_to_num(old['X'][selected][:, cols], nan=0.0).astype(np.float32)
            booster, early = (xgb.Booster(), xgb.Booster())
            booster.load_model(folder / 'full_xgb.ubj')
            early.load_model(folder / 'inner_xgb.ubj')
            booster.set_param({'nthread': 2})
            early.set_param({'nthread': 2})
            rf = package['rf']
            rf.n_jobs = 2
            assert (rf.n_estimators, rf.max_depth, rf.min_samples_leaf, rf.random_state, rf.n_features_in_) == (300, 8, 5, 42, len(cols))
            assert rf.class_weight == {0: 1, 1: int((y[train] == 0).sum() / (y[train] == 1).sum())}
            assert fit['inner_xgb_parameters']['scale_pos_weight'] == (y[inner] == 0).sum() / (y[inner] == 1).sum()
            assert fit['full_xgb_parameters']['scale_pos_weight'] == (y[train] == 0).sum() / (y[train] == 1).sum()
            for params in (fit['inner_xgb_parameters'], fit['full_xgb_parameters']):
                assert [params[k] for k in ('max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'seed', 'tree_method')] == [5, 0.05, 0.8, 0.8, 42, 'hist']
            hist = np.array(fit['inner_history']['inner_stop']['auc'])
            assert booster.num_boosted_rounds() == fit['selected_rounds'] == int(np.argmax(hist)) + 1 == int(early.best_iteration) + 1
            dmstop = xgb.DMatrix(X[stop], nthread=2)
            for end in range(1, early.num_boosted_rounds() + 1):
                e = early.predict(dmstop, iteration_range=(0, end))
                close(roc_auc_score(y[stop], e), hist[end - 1], atol=1e-12)
            p = 0.5 * booster.predict(xgb.DMatrix(X[test], nthread=2)) + 0.5 * rf.predict_proba(X[test])[:, 1]
            close(p, predictions[model], atol=2e-15)
            model_checks.append({'cohort': cohort, 'model': model, 'features': len(cols), 'selected_rounds': fit['selected_rounds'], 'inner_history_rounds_recomputed': len(hist), 'prediction_max_abs_error': float(np.max(np.abs(p - predictions[model])))})
        models, metric_names = (protocol['models'], protocol['metrics'])
        target = predictions['y']
        points = np.array([independent_metrics(target, predictions[m]) for m in models])
        rng = np.random.default_rng(protocol['bootstrap']['seed'])
        draw_metrics = np.full((2000, 3, 3), np.nan)
        for b in range(2000):
            w = rng.multinomial(len(target), np.full(len(target), 1 / len(target)))
            if not 0 < np.dot(w, target) < w.sum():
                continue
            for j, model in enumerate(models):
                draw_metrics[b, j] = independent_metrics(target, predictions[model], w)
            if b % 500 == 0:
                print(json.dumps({'independent_stage': 'direct_sklearn_bootstrap', 'cohort': cohort, 'draw': b}), flush=True)
        valid = np.isfinite(draw_metrics).all(axis=(1, 2))
        draw_metrics = draw_metrics[valid]
        for row in metrics.loc[metrics.cohort == cohort].itertuples():
            j, k = (models.index(row.model), metric_names.index(row.metric))
            lo, hi = np.quantile(draw_metrics[:, j, k], [0.025, 0.975])
            close([row.estimate, row.ci_low, row.ci_high], [points[j, k], lo, hi])
            assert row.test_patients == len(target) and row.test_positive_labels == int(target.sum()) and (row.valid_bootstrap_replicates == int(valid.sum()))
        for row in paired.loc[paired.cohort == cohort].itertuples():
            a, b, k = (models.index(row.first), models.index(row.second), metric_names.index(row.metric.removeprefix('delta_')))
            lo, hi = np.quantile(draw_metrics[:, a, k] - draw_metrics[:, b, k], [0.025, 0.975])
            close([row.estimate, row.ci_low, row.ci_high], [points[a, k] - points[b, k], lo, hi])
        stats_checks.append({'cohort': cohort, 'valid_bootstrap_draws': int(valid.sum()), 'direct_sklearn_weighted_metric_calls': int(valid.sum()) * 3, 'metric_rows_verified': int((metrics.cohort == cohort).sum()), 'paired_rows_verified': int((paired.cohort == cohort).sum())})
    assert before == [ident(item['path']) for item in before]
    output = {'passed': True, 'verification_type': 'cross-agent independent external verification', 'author_analysis_functions_imported': False, 'script': ident(__file__), 'python': sys.version, 'all_source_and_result_files_unchanged': True, 'inputs': before, 'raw_date_and_role_checks': date_checks, 'model_replay_checks': model_checks, 'bootstrap_checks': stats_checks, 'original_patient_labels_unchanged': True, 'scientific_temporal_requirement_fully_resolved': False, 'scope_limits': ['Confirms only retrospective laboratory acquisition-era stress-test calculations', 'All processed clocks are date-only, not result-return times', 'Unknown label and nonlaboratory exposure availability is not resolved by these roles', 'Crossing exclusions create a duration-selected cohort; WHU test has 18 labels', 'This report does not independently certify geographic assignment, clinical event dates, or patient-level treatment variables']}
    with (ROOT / 'independent_external_verification.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'passed': True, 'cross_agent': True, 'six_models': len(model_checks), 'direct_sklearn_calls': sum((r['direct_sklearn_weighted_metric_calls'] for r in stats_checks))}), flush=True)
if __name__ == '__main__':
    main()
