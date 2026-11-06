"""Aggregate-only provenance wording from frozen preparation artifacts."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def main():
    target, readout = (HERE / 'provenance_brief.json', HERE / 'PROVENANCE_BRIEF.md')
    assert not target.exists() and (not readout.exists())
    paths = [HERE / 'preparation_report.json', HERE / 'preparation_contract.json', HERE / 'private/full_matrix.npz', HERE / 'private/historical43_audit.npz']
    before = [identity(path) for path in paths]
    report = _release_resolve_config(json.loads(paths[0].read_text(encoding='utf-8')))
    for path in paths[1:]:
        declared = [item for item in report['outputs'] if Path(item['path']).resolve() == path.resolve()]
        assert len(declared) == 1 and identity(path) == declared[0]
    with np.load(paths[2], allow_pickle=False) as z, np.load(paths[3], allow_pickle=False) as h:
        names, historical_names = (z['feature_names'].tolist(), h['feature_names'].tolist())
        labs = [index for index, name in enumerate(names) if name not in ('Age', 'Sex_M')]
        old_labs = [index for index, name in enumerate(historical_names) if name not in ('Age', 'Sex_M')]
        profiles = {}
        for arm in ('ever_all', 'ever_survivor', 'prior_all', 'prior_survivor'):
            x, history = (z['X'][z[arm]], h['X_historical43_audit_only'][z[arm]])
            profiles[arm] = {'nominal_predictor_slots': len(names), 'nonempty_predictor_slots_in_full_arm': int(np.isfinite(x).any(axis=0).sum()), 'nonempty_laboratory_slots_in_full_arm': int(np.isfinite(x[:, labs]).any(axis=0).sum()), 'admissions_without_any_historical_cached_lab': int((~np.isfinite(history[:, old_labs]).any(axis=1)).sum()), 'admissions_without_any_remaining_lab_after_uniform_drop': int((~np.isfinite(x[:, labs]).any(axis=1)).sum()), 'admissions_missing_age_or_sex': int((~np.isfinite(x[:, [names.index('Age'), names.index('Sex_M')]]).all(axis=1)).sum())}
            assert profiles[arm]['nominal_predictor_slots'] == 41
            assert profiles[arm]['nonempty_predictor_slots_in_full_arm'] == 24
            assert profiles[arm]['nonempty_laboratory_slots_in_full_arm'] == 22
            assert profiles[arm]['admissions_without_any_historical_cached_lab'] == profiles[arm]['admissions_without_any_remaining_lab_after_uniform_drop']
        assert profiles['ever_all']['admissions_without_any_historical_cached_lab'] == 171
    timing = report['laboratory_replay']['recorded_time_scope']
    assert report['laboratory_replay']['cached_rows'] == 1194579
    assert timing['before_admission_rows'] == 62143 and timing['after_discharge_rows'] == 1012
    proxy_only = {arm: report['cohorts'][arm]['source_death_proxy_only_positive'] for arm in profiles}
    assert proxy_only['ever_all'] == 662 and proxy_only['prior_all'] == 161
    assert before == [identity(path) for path in paths]
    result = {'completed_utc': datetime.now(timezone.utc).isoformat(), 'script': identity(Path(__file__)), 'inputs': before, 'cached_row_denominator_before_AST_HDL_exclusion': 1194579, 'recorded_time_scope_marginal_counts': timing, 'counts_are_not_assumed_mutually_exclusive': True, 'chronology_definition': 'Compare source cached CHARTTIME with recorded source ADMITTIME/DISCHTIME; retain original admission-ID-linked median construction without time filtering.', 'source_death_proxy_only_positive_by_arm': proxy_only, 'predictor_and_missing_lab_profiles': profiles, 'clinical_negative_guarantee': False, 'onset_or_clinician_result_availability_verified': False, 'source_or_frozen_preparation_changed': False, 'training_performed': False, 'interpretation': ['Source MACE=0 means no original composite code/death-rule match in the available source diagnoses, not clinically adjudicated absence or verified complete negative capture.', 'Source death proxy requires hospital death and broad cardiac diagnosis; it is not cardiovascular cause-of-death adjudication.', '24 nonempty slots is a full-arm data profile, not a promise that every fitting partition uses exactly 24 columns.', 'The before/after chronology counts refer to all historical cached rows before AST/HDL column exclusion. Do not add these marginal counts as a unique out-of-bounds count.']}
    lines = ['# Aggregate Analysis Provenance Brief', '', '## Laboratory Construction', '', 'The source cache contains 1,194,579 admission-linked laboratory rows. Of these, 62,143 have recorded CHARTTIME before ADMITTIME, 1,012 after DISCHTIME, and 953,122 after admission plus 24 hours. These are marginal counts over the full historical cache before AST/HDL column exclusion; they are not assumed mutually exclusive.', '', 'All original rows remain in the historical feature reconstruction. Use **admission-ID-linked, time-unrestricted medians**, not **restricted within-stay measurements**. Globulin is the difference of the separately aggregated Total Protein and Albumin medians, not a verified paired specimen result. A timestamp comparison establishes neither clinical onset nor first clinician availability.', '', '## Predictor Coverage', '', 'The new matrix has 41 nominal slots after uniformly dropping AST and HDL Cholesterol. Each whole arm has 24 actually nonempty slots: 22 laboratory slots plus Age and Sex_M. The other 17 nominal slots remain missing. No new item mapping or numeric unit conversion is performed; CK/CK-MB and other semantic/time limitations remain.', '', '| Arm | No Historical Cached Lab | No Remaining Lab After Drop | Missing Age/Sex |', '| --- | ---: | ---: | ---: |']
    for arm, profile in profiles.items():
        lines.append('| ' + arm + ' | ' + ' | '.join((str(profile[key]) for key in ('admissions_without_any_historical_cached_lab', 'admissions_without_any_remaining_lab_after_uniform_drop', 'admissions_missing_age_or_sex'))) + ' |')
    lines += ['', 'These admissions are retained with missing laboratories; they are not lab-complete-case exclusions. Empty-column selection and imputation must still be learned only from each eligible fitting partition.', '', '## Source Label Mechanism', '', 'Among source-defined positives, 662 ever-all admissions and 161 prior-all admissions are positive only because of the historical hospital-death-plus-broad-cardiac rule, not because of any of the six non-death code groups. These are source death-proxy-only labels, not adjudicated cardiovascular deaths. The primary labels are unchanged, and separate non-death code-union flags are available.', '', 'MACE=0 is a recorded-source rule no-hit. It supplies **no guarantee of clinically confirmed absence**, complete same-admission negative capture, first-event status or an event-free baseline. Prior completed coded history documents an earlier coded stay, not an event onset or a verified prospective inclusion criterion.', '', '## Use In New Manuscript', '', 'Describe the experiment as source-defined cohort-selection sensitivity in retrospective diagnosis-record classification. Neither altered AUC nor successful source replay can measure the amount of indirect temporal leakage or validate future-risk prediction. Original historical results are not overwritten.', '']
    with readout.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    result['readout'] = identity(readout)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=True, indent=2, allow_nan=False)
    print(json.dumps({'provenance_brief_complete': True, 'nominal_features': 41, 'full_arm_nonempty_features': 24, 'no_lab_source_rows_retained': 171, 'source_death_proxy_only_ever_all': 662, 'source_death_proxy_only_prior_all': 161}))
if __name__ == '__main__':
    main()
