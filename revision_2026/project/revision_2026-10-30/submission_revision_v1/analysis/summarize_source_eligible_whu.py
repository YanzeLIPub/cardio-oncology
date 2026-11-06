"""Summarize the frozen source-availability sensitivity, with fixed-score comparisons."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import os, sys, json, hashlib, importlib.util
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[k] = '8'
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
PROJECT = Path(_release_path('project', ''))
DEST = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
EXEC = DEST / 'private/whu_source_eligible'
AGG = DEST / 'analysis'

def read(p):
    return _release_resolve_config(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def save(p, v):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)

def sha(p):
    return None

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    spec = importlib.util.spec_from_file_location('original_summary', _release_source_path(SOURCE / 'summarize_validation.py'))
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    with np.load(EXEC / 'eligibility_mask.npz') as a:
        mask = a['eligible'].copy()
    with np.load(SOURCE / 'private/runs/WHU_original/predictions.npz') as a:
        old = {k: a[k].copy() for k in a.files}
    with np.load(EXEC / 'private/runs/WHU_source_eligible/predictions.npz') as a:
        new = {k: a[k].copy() for k in a.files}
    assert np.array_equal(old['y'][mask], new['y']) and np.array_equal(old['groups'][mask], new['groups'])
    oldfields = {'historical_original_cache': 'original', 'prior_patient_isolated_raw_fixed': 'primary_raw', 'prior_patient_isolated_platt_fixed': 'base_platt'}
    metricrows = []
    calrows = []
    binrows = []
    paired = []
    meta = []
    samplepacks = {}
    points = {}
    for scope, selection in [('all_original', np.ones(len(mask), bool)), ('retained', mask), ('excluded_no_raw_JSON', ~mask)]:
        y = old['y'][selection]
        scores = {label: old[key][selection] for label, key in oldfields.items()}
        if scope == 'retained':
            scores.update({f'retrained_{key}': new[key] for key in ['primary_raw', 'base_raw', 'base_platt', 'xgb_full', 'rf_full', 'logistic', 'missingness', 'age']})
        names = list(scores)
        caches = {k: sm.WeightedBinaryMetrics(y, p) for k, p in scores.items()}
        seed = 20261028 + {'all_original': 0, 'retained': 1, 'excluded_no_raw_JSON': 2}[scope]
        rng = np.random.default_rng(seed)
        samples = np.empty((2000, len(names), 4))
        digest = hashlib.sha256()
        for b in range(2000):
            weights = rng.multinomial(len(y), np.full(len(y), 1 / len(y)))
            digest.update(weights.astype('<i8').tobytes())
            for j, name in enumerate(names):
                samples[b, j] = caches[name].evaluate(weights)
        samplepacks[scope] = samples
        np.savez_compressed(EXEC / f'bootstrap_{scope}.npz', values=samples, names=np.array(names), y=y)
        for j, name in enumerate(names):
            p = scores[name]
            point = sm.sklearn_point_metrics(y, p)
            assert np.allclose(point, caches[name].evaluate(np.ones(len(y))), rtol=0, atol=2e-12)
            points[scope, name] = point
            for m, metric in enumerate(sm.METRIC_KEYS):
                lo, hi, valid = sm.interval(samples[:, j, m])
                metricrows.append({'cohort': 'WHU', 'scope': scope, 'unit': 'patient', 'n': len(y), 'positive': int(y.sum()), 'score': name, 'metric': metric, 'estimate': point[m], 'lower_95': lo, 'upper_95': hi, 'bootstrap_valid': valid, 'bootstrap_total': 2000, 'uncertainty': 'conditional_on_fixed_scores_not_refit', 'evaluation': 'new_exclusion_before_training' if name.startswith('retrained_') else 'fixed_prior_scores_no_refit'})
            calrows.append({'cohort': 'WHU', 'scope': scope, 'n': len(y), 'score': name, **sm.calibration_parameters(y, p)})
            binrows.extend(({'cohort': 'WHU', 'scope': scope, 'score': name, **b} for b in sm.calibration_bins(y, p)))
        if scope == 'retained':
            for desc, left, right in [('retrained_raw_minus_prior_isolated_fixed', 'retrained_primary_raw', 'prior_patient_isolated_raw_fixed'), ('retrained_raw_minus_historical_fixed', 'retrained_primary_raw', 'historical_original_cache'), ('retrained_platt_minus_retrained_base_raw', 'retrained_base_platt', 'retrained_base_raw'), ('retrained_raw_minus_age', 'retrained_primary_raw', 'retrained_age'), ('retrained_raw_minus_missingness', 'retrained_primary_raw', 'retrained_missingness')]:
                li, ri = (names.index(left), names.index(right))
                for m, metric in enumerate(sm.METRIC_KEYS):
                    lo, hi, valid = sm.interval(samples[:, li, m] - samples[:, ri, m])
                    paired.append({'cohort': 'WHU', 'scope': scope, 'n': len(y), 'comparison': desc, 'metric': metric, 'difference': points[scope, left][m] - points[scope, right][m], 'lower_95': lo, 'upper_95': hi, 'bootstrap_valid': valid, 'same_evaluation_patients': True, 'interpretation': 'not_causal_effect_of_exclusion; fitting_samples_splits_early_stopping_change' if 'fixed' in desc else 'paired_fixed_scores'})
        meta.append({'scope': scope, 'n': len(y), 'positive': int(y.sum()), 'replicates': 2000, 'seed': seed, 'score_names': names})
        print(json.dumps({'summarized_scope': scope, 'n': len(y), 'scores': len(names), 'bootstrap': 2000}), flush=True)
    for name, rows in [('source_eligible_metrics.csv', metricrows), ('source_eligible_paired_comparisons.csv', paired), ('source_eligible_calibration.csv', calrows), ('source_eligible_calibration_bins.csv', binrows)]:
        pd.DataFrame(rows).to_csv(AGG / name, index=False, encoding='utf-8-sig')
    keyrows = [r for r in metricrows if r['metric'] == 'auc' and r['score'] in ['historical_original_cache', 'prior_patient_isolated_raw_fixed', 'retrained_primary_raw', 'retrained_base_platt']]
    summary = {'completed': True, 'cohort': 'WHU', 'unit': 'patient', 'source_eligible_n': len(new['y']), 'source_eligible_algorithm_positive': int(new['y'].sum()), 'excluded_n': int((~mask).sum()), 'excluded_algorithm_positive': int(old['y'][~mask].sum()), 'label_changes': 0, 'feature_columns': 104, 'primary_model': '50:50 XGBoost/RandomForest;5fold;inner_training_earlystop;unmodified_original_run_one', 'primary_probability': 'primary_raw', 'calibrated_probability': 'base_platt', 'changed_outer_test_fold_assignment_among_retained': int((old['fold'][mask] != new['fold']).sum()), 'bootstrap': meta, 'key_auc_results': keyrows, 'limits': ['Eligibility applied before all training/calibration/test splits, not only at scoring.', 'Missing an individual laboratory or discharge summary is not an exclusion under the author-approved rule.', 'Fixed prior score slices are composition-only descriptions, not models retrained on source-eligible patients.', 'Paired score differences use the same2258 evaluation patients but training populations,outer folds and selected boosting rounds differ; not isolated causal effects of record exclusion.', 'Negative label means no positive match under specified record rule; not a clinician-adjudicated absence.', 'Full-record aggregation retained; event-before-feature timing and future-event validity are not established.', 'SCH38509 and MIMIC9946 retained as source-linked; old analyses on original cohorts remain historical references.', 'Confidence intervals condition on saved out-of-fold scores and omit model refitting uncertainty.'], 'original_data_unchanged': True, 'clinical_review_executed': False}
    save(AGG / 'source_eligible_summary.json', summary)
    print(json.dumps({'completed': True, 'key_auc_results': keyrows}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
