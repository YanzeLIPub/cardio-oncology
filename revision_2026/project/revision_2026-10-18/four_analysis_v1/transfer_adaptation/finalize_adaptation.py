"""Finalize verified aggregates without modifying frozen fitting code or inputs."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent

def load(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def main():
    assert not (ROOT / 'final_seal.json').exists(), 'Preserve completed seals'
    lock = load(ROOT / 'analysis_lock.json')
    replay = load(ROOT / 'aggregate/saved_model_replay.json')
    training = load(ROOT / 'aggregate/training_manifest.json')
    summary = load(ROOT / 'aggregate/summary.json')
    assert replay['passed'] and replay['estimated_adapter_refits'] == 360
    assert summary['execution_complete'] and (not summary['failed_adapter_fold_fits'])
    for expected in lock['inputs'] + training['model_packages']:
        assert identity(expected['path']) == expected
    expected_rows = {'metrics_by_repeat': 444, 'mean_repeat_metrics': 144, 'paired_by_repeat': 756, 'mean_repeat_paired': 252, 'evaluation_populations': 4}
    tables = {}
    for stem, expected in expected_rows.items():
        tables[stem] = pd.read_csv(ROOT / f'aggregate/{stem}.csv')
        assert len(tables[stem]) == expected
    estimates = tables['metrics_by_repeat']
    for _, row in tables['mean_repeat_metrics'].iterrows():
        group = estimates.loc[(estimates.source == row.source) & (estimates.target == row.target) & (estimates.population == row.population) & (estimates.method == row.method) & (estimates.sample_size_patients_per_fold == row.sample_size_patients_per_fold) & (estimates.metric == row.metric)]
        assert len(group) == 3 and set(group.repeat) == {1, 2, 3}
        assert abs(group.estimate.mean() - row.estimate) < 1e-14
        assert abs(group.estimate.min() - row.repeat_min) < 1e-14
        assert abs(group.estimate.max() - row.repeat_max) < 1e-14
    summary['independent_package_replay_pending'] = False
    summary['standalone_saved_model_replay_complete'] = True
    summary['saved_model_replay_identity'] = identity(ROOT / 'aggregate/saved_model_replay.json')
    summary['external_independent_numerical_verification'] = 'Separate parent-assigned review, not asserted by this writer.'
    summary['repeat_mean_and_range_recomputed_from_three_actual_repeat_metrics'] = True
    save(ROOT / 'aggregate/summary.json', summary)
    lines = ['# Restricted Laboratory Transfer Adaptation', '', 'Completed on 2026-10-18 in a new directory. Original manuscripts, raw sources, labels, matrices, folds and prior model packages were read-only.', '', '## Design', '', 'Two directions only: WHU to MIMIC and MIMIC to WHU. The four existing reviewed laboratory slots and original NaN values were retained. The original patient/admission label tasks were not made equivalent.', '', 'The existing full-source median-ensemble packages remained frozen. For each of the original five target patient outer folds, 50/200/800 target training patients were sampled in nested proportional max-label strata with three fixed seeds. All selected MIMIC admissions were retained. No target test patient entered any adapter, target-local transform, fitting decision or sample draw.', '', 'Five methods: direct frozen source; unconstrained unpenalized Platt; C=1 logistic update on source logit plus four source-transformed lab values; C=1 local logistic using those four source-transformed lab values; and truly target-local C=1 logistic with imputation/scaling fitted on selected target training rows only. No target-labelled hyperparameter search was performed.', '', 'Ninety private packages contain 360 fitted adapters. All fitted successfully. The standalone replay imports no fitting writer, reconstructs selected roles and transforms, refits all 360 estimators and replays 450,288 held-out row scores. It passed with every original source/package hash unchanged.', '', '## Results', '', 'AUC below is the mean of the three repeat-specific OOF AUCs, not AUC of averaged probabilities. Intervals use common target-patient weights to average the three recalculated bootstrap metrics. Repeats are never concatenated as 3N patients.', '', '| Source | Target | Population | Target patients per fold | Method | Mean AUC | Conditional 95% CI | Repeat range |', '| --- | --- | --- | ---: | --- | ---: | --- | --- |']
    for _, row in tables['mean_repeat_metrics'].query("metric == 'auc'").iterrows():
        lines.append(f'| {row.source} | {row.target} | {row.population} | {row.sample_size_patients_per_fold} | {row.method} | {row.estimate:.3f} | {row.ci_low:.3f} to {row.ci_high:.3f} | {row.repeat_min:.3f} to {row.repeat_max:.3f} |')
    lines += ['', 'Direct-source point AUCs remain 0.387481 in full WHU and 0.524713 in full MIMIC. The new bootstrap seed gives new conditional intervals; it does not change those predictions or authenticate unrelated historical transfer vectors.', '', 'At 800 target patients/fold, the MIMIC-to-WHU update minus direct-source mean AUC difference is +0.245742, but in the covered WHU subset it is +0.053393 with an interval crossing zero. In covered WHU, update minus truly target-local logistic is -0.022315 (interval crossing zero). For WHU-to-MIMIC, update minus truly target-local logistic is -0.012320 with a negative interval. These findings do not demonstrate a universally superior transferred model.', '', 'Platt slopes were negative in 39/45 WHU-target fits and 5/45 MIMIC-target fits. Negative slope reverses ordering within the corresponding held-out fold. The pooled OOF combines five distinct calibrators, so even positive within-fold slopes do not guarantee invariant pooled AUC. This procedure is not described as purely order-preserving calibration.', '', '## Limitations', '', '- The native WHU ever-recorded patient label and MIMIC same-admission survivor label remain different; all-record laboratory medians do not establish pre-event prediction, event-free risk sets or real result availability.', '- The four measurements passed a limited exact-name/context/physical-unit gate, not complete specimen, assay or platform equivalence. SCH is not part of this experiment because source-certified units remain absent.', '- WHU has 318 all-four-missing patients including 102 positive labels. The predeclared at-least-one-lab secondary evaluation changes the estimand without refitting; coverage is not clinical absence of testing or disease.', '- Source-transform local logistic is deliberately distinct from fully target-local logistic. Source-logit incremental value depends on which baseline and evaluation population are used; this is not a causal decomposition of transferred knowledge.', '- The learning curves use only three fixed sample sizes and three draws from already labelled training pools, with stratification using pool patient max-label prevalence. They do not estimate prospective label-acquisition efficiency, an optimal sample size or a minimum clinically acceptable training cohort.', '- MIMIC remains admission-weighted for fitting/evaluation, with patient-isolated sampling and patient-cluster uncertainty; it is not silently converted to one outcome per patient.', '- The 2,000-draw percentile intervals condition on fitted source/adapter models, selected target training samples, source extraction and coverage. They exclude model refitting, source-sampling, target-sample selection and multiplicity uncertainty. Repeat min/max is sampling variability, not an independent confidence interval.', '- Brier scores assess these native labels, including different prevalences. Better Brier after local calibration is not proof of clinical risk calibration, utility, prospective transportability or a care-architecture mechanism.', '- No external institution, treatment/stage adjustment, true WHU geographical validation, clinical net-benefit threshold or three-centre harmonized task was added.', '', '## Outputs', '', '- `aggregate/metrics_by_repeat.csv`: complete denominator AUC/AP/Brier, each repeat and frozen direct source, with conditional intervals.', '- `aggregate/mean_repeat_metrics.csv`: three-repeat arithmetic metric means, common-weight intervals and actual repeat min/max.', '- `aggregate/paired_by_repeat.csv` and `mean_repeat_paired.csv`: nominal paired differences, first minus second; negative Brier differences favour the first method.', '- `aggregate/calibration_and_update_coefficients.csv`: Platt unscaled logit slopes and signs; update coefficient order is source-logit, albumin, total protein, hemoglobin, hematocrit in source-SD units. Local source/target coefficients use their respective fitted scaling.', '- `aggregate/training_manifest.json`: aggregate fold/sample counts, class counts, transform identities and all private package identities.', '- `aggregate/saved_model_replay.json`: standalone selection/preprocessing/refit/probability replay, not a substitute for the parent independent numerical review.', '- `private/`: role/sample positions, model packages, original labels/groups, OOF scores and bootstrap arrays; no patient rows are published.', '']
    (ROOT / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    execution = load(ROOT / 'aggregate/execution.json')
    execution['package_replay_pending'] = False
    execution['saved_model_replay_complete'] = True
    execution['inputs_unchanged'] = True
    execution['outputs'] = [identity(p) for p in sorted((ROOT / 'aggregate').glob('*')) if p.name != 'execution.json']
    execution['saved_model_replay_identity'] = identity(ROOT / 'aggregate/saved_model_replay.json')
    save(ROOT / 'aggregate/execution.json', execution)
    output_paths = [p for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and (p.name != 'final_seal.json')]
    save(ROOT / 'final_seal.json', {'completed_utc': datetime.now(timezone.utc).isoformat(), 'analysis_complete': True, 'fitting_and_evaluation_complete': True, 'standalone_model_replay_passed': True, 'external_independent_numeric_review_asserted': False, 'input_identities_before_and_after_equal': True, 'input_identities': lock['inputs'], 'output_identities': [identity(p) for p in output_paths], 'no_source_or_prior_branch_modified': True, 'no_patient_rows_in_public_outputs': True, 'pending_execution_jobs': False, 'scope': 'Technical execution seal, not clinical task equivalence or complete reviewer-resolution certification.'})
    print(json.dumps({'sealed': True, 'packages': 90, 'adapter_fits': 360, 'patient_isolated_scores_replayed': 450288, 'metrics': expected_rows, 'source_inputs_unchanged': True}))
if __name__ == '__main__':
    main()
