"""Post-hoc target-coverage evaluation only; no model fitting or changes."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
from evaluate_common_v2 import FloatingClusterMetrics
from finalize_common import identity, write_json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

def main():
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol_amendment.json').read_text(encoding='utf-8-sig')))
    assert protocol['locked_before_subset_performance_calculation']
    paths = [ROOT / 'protocol_amendment.json', Path(__file__), PARENT / 'protocol.json', PARENT / 'aggregate/transfer_metrics.csv', PARENT / 'aggregate/local_metrics.csv', PARENT / 'analysis_lock.json']
    paths += [PARENT / f'private/{c}{suffix}.npz' for c in ('WHU', 'MIMIC') for suffix in ('', '_predictions')]
    before = [identity(p) for p in paths]
    write_json(ROOT / 'amendment_lock.json', {'before_subset_performance': True, 'locked_utc': datetime.now(timezone.utc).isoformat(), 'input_identities': before})
    original = pd.concat([pd.read_csv(PARENT / 'aggregate/transfer_metrics.csv'), pd.read_csv(PARENT / 'aggregate/local_metrics.csv')], ignore_index=True)
    outputs, differences, populations, checks = ([], [], [], [])
    for centre in ('WHU', 'MIMIC'):
        with np.load(PARENT / f'private/{centre}.npz', allow_pickle=False) as z:
            covered = np.isfinite(z['X']).any(axis=1)
        with np.load(PARENT / f'private/{centre}_predictions.npz', allow_pickle=False) as z:
            y, groups = (z['y'], z['groups'])
            predictions = {k: z[k] for k in z.files if k not in {'y', 'groups'}}
        full_calculators = {k: FloatingClusterMetrics(y, p) for k, p in predictions.items()}
        subset_calculators = {k: FloatingClusterMetrics(y[covered], p[covered]) for k, p in predictions.items()}
        full_points = {k: c.compute(np.ones(len(y))) for k, c in full_calculators.items()}
        subset_points = {k: c.compute(np.ones(int(covered.sum()))) for k, c in subset_calculators.items()}
        full_unique, full_codes = np.unique(groups, return_inverse=True)
        subset_unique, subset_codes = np.unique(groups[covered], return_inverse=True)
        rng_subset = np.random.default_rng(protocol['subset_intervals']['seed'])
        rng_full = np.random.default_rng(protocol['population_difference_intervals']['seed'])
        count = protocol['subset_intervals']['replicates']
        conditional = {k: np.empty((count, 3)) for k in predictions}
        full_replay = {k: np.empty((count, 3)) for k in predictions}
        delta = {k: np.empty((count, 3)) for k in predictions}
        draw_hash = hashlib.sha256()
        for i in range(count):
            m_subset = rng_subset.multinomial(len(subset_unique), np.full(len(subset_unique), 1 / len(subset_unique)))
            subset_weights = m_subset[subset_codes]
            m_full = rng_full.multinomial(len(full_unique), np.full(len(full_unique), 1 / len(full_unique)))
            draw_hash.update(m_full.tobytes())
            full_weights = m_full[full_codes]
            for key in predictions:
                conditional[key][i] = subset_calculators[key].compute(subset_weights)
                full_replay[key][i] = full_calculators[key].compute(full_weights)
                delta[key][i] = subset_calculators[key].compute(full_weights[covered]) - full_replay[key][i]
        for key in predictions:
            scope, model = key.split('_', 1)
            expected_scope = 'source_frozen_transfer' if scope == 'transfer' else 'local_patient_oof'
            reference = original[original.target.eq(centre) & original.model.eq(model) & original.scope.eq(expected_scope)]
            assert len(reference) == 1
            reference = reference.iloc[0]
            interval = np.nanquantile(conditional[key], [0.025, 0.975], axis=0)
            full_interval = np.nanquantile(full_replay[key], [0.025, 0.975], axis=0)
            delta_interval = np.nanquantile(delta[key], [0.025, 0.975], axis=0)
            row = {'source': reference.source, 'target': centre, 'model': model, 'scope': expected_scope, 'evaluation_population': 'posthoc_at_least_one_accepted_laboratory_no_refit', 'observations': int(covered.sum()), 'patients': len(subset_unique), 'positive_labels': int(y[covered].sum()), 'full_target_observations': len(y), 'full_source_training_and_frozen_predictions_unchanged': True, 'interval_scope': protocol['subset_intervals']['scheme']}
            for j, metric in enumerate(('roc_auc', 'average_precision', 'brier')):
                assert abs(reference[metric] - full_points[key][j]) < 2e-12
                assert abs(reference[metric + '_ci_low'] - full_interval[0, j]) < 2e-12
                assert abs(reference[metric + '_ci_high'] - full_interval[1, j]) < 2e-12
                row[metric] = float(subset_points[key][j])
                row[metric + '_ci_low'] = float(interval[0, j])
                row[metric + '_ci_high'] = float(interval[1, j])
                differences.append({'source': reference.source, 'target': centre, 'model': model, 'scope': expected_scope, 'metric': metric, 'subset_estimate': float(subset_points[key][j]), 'full_estimate': float(full_points[key][j]), 'subset_minus_full': float(subset_points[key][j] - full_points[key][j]), 'ci_low': float(delta_interval[0, j]), 'ci_high': float(delta_interval[1, j]), 'interval_scope': protocol['population_difference_intervals']['scheme']})
            outputs.append(row)
        populations.append({'target': centre, 'full_observations': len(y), 'full_patients': len(full_unique), 'full_positive_labels': int(y.sum()), 'subset_observations': int(covered.sum()), 'subset_patients': len(subset_unique), 'subset_positive_labels': int(y[covered].sum()), 'excluded_observations': int((~covered).sum()), 'excluded_positive_labels': int(y[~covered].sum())})
        checks.append({'target': centre, 'all_eight_original_metric_sets_and_intervals_replayed': True})
        print(json.dumps({'phase': 'coverage_sensitivity', 'target_completed': centre, 'subset_observations': int(covered.sum())}), flush=True)
    pd.DataFrame(outputs).to_csv(ROOT / 'metrics.csv', index=False, mode='x')
    pd.DataFrame(differences).to_csv(ROOT / 'population_differences.csv', index=False, mode='x')
    pd.DataFrame(populations).to_csv(ROOT / 'populations.csv', index=False, mode='x')
    after = [identity(p) for p in paths]
    assert before == after
    primary = pd.DataFrame(outputs)
    primary = primary[primary.model.eq('ensemble_median')]
    text = ['# Post-Hoc Coverage Caution and Fixed-Prediction Sensitivity', '', 'This amendment was authorized after discovering the coverage/label imbalance, not before the primary experiment. It does not replace or modify any primary metric, fitted model, feature, label or cohort row.', '', 'WHU has 318/2,562 patients with all four accepted laboratories missing, including 102/281 original positive labels. MIMIC has 174/9,946 all-missing admissions, including 60 positives. Missingness here combines source availability/linkage, report context, literal unit and numeric-result acceptance. It is not evidence that clinicians did not order a test or that patients had no disease.', '', 'We evaluate the unchanged full-source transfer and original local OOF scores only among targets with at least one accepted value. WHU then contains 2,244 patients/179 positives; MIMIC contains 9,772 admissions/4,402 positives. See populations.csv for the actual patient count. The source and local model fitting populations are not reduced or refitted.', '', '| Source | Target | Frozen prediction scope | Subset AUC (95% CI) |', '|---|---|---|---|']
    for _, row in primary.iterrows():
        text.append(f'| {row.source} | {row.target} | {row.scope} | {row.roc_auc:.4f} ({row.roc_auc_ci_low:.4f}-{row.roc_auc_ci_high:.4f}) |')
    text += ['', 'Subset intervals resample covered target patients; paired subset-minus-full population contrasts use shared full-target cluster draws and reproduce the original full-target intervals. Neither includes training/refit uncertainty, source sampling, coverage-selection uncertainty or multiplicity correction. Differences between populations are not causal coverage effects.', '', 'Source availability is a major competing explanation, especially for the WHU evaluation. Even persistence of inverse ranking in a conditional subset would not identify semantic drift: labels, patient/admission units, case mix, selection and remaining assay-method differences are still not harmonized.']
    with (ROOT / 'README.md').open('x', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(text) + '\n')
    write_json(ROOT / 'execution.json', {'passed': True, 'execution_complete': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'post_hoc_explicit': True, 'new_fitting_or_model_selection': False, 'primary_files_changed': False, 'input_identities_before': before, 'input_identities_after': after, 'replay_checks': checks, 'outputs': [identity(ROOT / name) for name in ('metrics.csv', 'population_differences.csv', 'populations.csv', 'README.md', 'amendment_lock.json')]})
if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
