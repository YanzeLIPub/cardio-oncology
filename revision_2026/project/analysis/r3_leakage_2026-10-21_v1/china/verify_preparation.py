"""Verify restriction behavior and the source-bound private demographic outputs."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import json
import numpy as np
import pandas as pd
from audit_feature_lineage import AGE, KEY, ROOT, SEX, TIME, identity, save
from prepare_demographic_control import choose_earliest_demographics

def main():
    checks = []
    fixture = pd.DataFrame({KEY: [0, 0, 1, 1, 1, 2, 2], TIME: ['2020-01-01', '2021-01-01', '2020-01-01', '2020-01-01', '2021-01-01', '2020-01-01', '2021-01-01'], AGE: [50, 90, 40, 41, 20, None, 70], SEX: [1, 2, 1, 2, 1, 9, 2]})
    result, counts = choose_earliest_demographics(fixture)
    assert np.array_equal(result.loc[0].to_numpy(), np.asarray([50.0, 0.0]))
    checks.append('Later-record age and sex cannot change earliest values')
    assert result.loc[1].isna().all()
    checks.append('Discordant same-earliest-time ages and sex codes become missing')
    assert result.loc[2].isna().all()
    checks.append('Missing age and invalid earliest sex are not filled from later values')
    assert counts['age_earliest_conflicts'] == counts['sex_earliest_conflicts'] == 1
    permuted, _ = choose_earliest_demographics(fixture.sample(frac=1, random_state=42))
    assert np.array_equal(result.to_numpy(), permuted.to_numpy(), equal_nan=True)
    checks.append('Source row permutation does not change tied-time selections')
    expanded = fixture.copy()
    expanded['MACE'] = [0, 1, 1, 0, 1, 1, 0]
    expanded['diagnosis'] = 'synthetic unused field'
    unchanged, _ = choose_earliest_demographics(expanded)
    assert np.array_equal(result.to_numpy(), unchanged.to_numpy(), equal_nan=True)
    checks.append('Extraneous synthetic outcomes and diagnoses do not influence selection')
    invalid = fixture.copy()
    invalid.loc[0, TIME] = 'invalid'
    try:
        choose_earliest_demographics(invalid)
    except AssertionError:
        checks.append('Any invalid source timestamp aborts rather than selecting a later row')
    else:
        raise AssertionError('Invalid timestamp should have aborted')
    preparation = _release_resolve_config(json.loads((ROOT / 'demographic_preparation.json').read_text(encoding='utf-8')))
    lineage = _release_resolve_config(json.loads((ROOT / 'feature_lineage_audit.json').read_text(encoding='utf-8')))
    for cohort in preparation['cohorts']:
        data = dict(np.load(cohort['output']['path'], allow_pickle=False))
        assert set(data) == {'X', 'groups', 'feature_names'}
        assert data['X'].shape == (cohort['n'], 2)
        assert len(np.unique(data['groups'])) == cohort['n']
        assert np.isfinite(data['X']).all()
        assert set(np.unique(data['X'][:, 1])) <= {0.0, 1.0}
        assert data['feature_names'].tolist() == ['age_at_earliest_check', 'sex_source_code2_indicator']
        assert identity(cohort['output']['path']) == cohort['output']
        assert identity(cohort['frozen_roles']['path']) == cohort['frozen_roles']
        assert all((v['numeric_nonmissing_rows'] == 0 for v in next((c for c in lineage['cohorts'] if c['center'] == cohort['center']))['direct_text_fields'].values()))
    checks.append('All private output shapes, groups, allowed columns, code ranges and hashes match')
    checks.append('Three historical diagnostic/department fields have zero numeric input rows in both sources')
    assert identity(_release_source_path(ROOT / 'prepare_demographic_control.py')) == preparation['script']
    assert identity(_release_source_path(ROOT / 'audit_feature_lineage.py')) == preparation['helper_script']
    checks.append('Executed preparation and helper script hashes are unchanged')
    save(ROOT / 'preparation_verification.json', {'passed': True, 'checks': checks, 'check_count': len(checks), 'synthetic_fixture_not_patient_data': True, 'script': identity(__file__), 'models_fitted': 0})
    print(json.dumps({'passed': True, 'checks': len(checks), 'models_fitted': 0}))
if __name__ == '__main__':
    main()
