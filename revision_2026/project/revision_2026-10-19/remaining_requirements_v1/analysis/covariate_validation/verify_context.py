"""Independent date-role reconstruction and row-wise source-year checks."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[3]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
KEY = '病案号'
TIME = '检查时间'
YEAR = '年份'
AREA = '地区'
ADMIT = '入院时间'
DISCHARGE = '出院时间'
REGIONAL = ['人均地区生产总值(元)', '职工平均工资(元)', '医院、卫生院数(个)', '医院、卫生院床位数(张)', '医生数(人)']

def identity(p):
    p = Path(p)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def main():
    audit = _release_resolve_config(json.loads((ROOT / 'context_audit.json').read_text(encoding='utf-8')))
    for s in audit['sources']:
        assert identity(s['path']) == s
    label = pd.read_csv(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'), usecols=[KEY, 'Center', 'MACE'])
    db = pd.read_excel(_release_path('china', 'city_database.xlsx'), sheet_name='原始数据', usecols=[YEAR, AREA, *REGIONAL])
    db[AREA] = db[AREA].astype('string').str.strip()
    db[YEAR] = pd.to_numeric(db[YEAR], errors='coerce')
    for f in REGIONAL:
        db[f] = pd.to_numeric(db[f], errors='coerce')
    assert not db.duplicated([AREA, YEAR]).any()
    vectors = db.dropna(subset=[YEAR, AREA, *REGIONAL]).set_index(REGIONAL)
    vectors = vectors.loc[~vectors.index.duplicated(keep=False)]
    assert vectors.index.is_unique
    findings, geo_year_rows, clock_rows = ([], [], [])
    for cohort, filename in [('SCH', 'SCH_processed_data.csv'), ('WHU', 'WHU_Cancer_processed_data.csv')]:
        fields = [KEY, TIME, ADMIT, DISCHARGE, *REGIONAL] + (['city', YEAR] if cohort == 'SCH' else [])
        data = pd.read_csv(Path(_release_path('china', '')) / filename, usecols=fields, low_memory=False)
        labs = pd.to_datetime(data[TIME], format='mixed', errors='raise')
        admits = pd.to_datetime(data[ADMIT], format='mixed', errors='raise')
        discharges = pd.to_datetime(data[DISCHARGE], format='mixed', errors='raise')
        patients = label.loc[label.Center.eq(cohort), KEY]
        minimum = labs.groupby(data[KEY]).min().reindex(patients)
        maximum = labs.groupby(data[KEY]).max().reindex(patients)
        first = sorted(minimum)
        t = first[math.floor(0.8 * (len(first) - 1))]
        cutoff = pd.Timestamp(year=t.year, month=t.month, day=1)
        roles = np.where(maximum < cutoff, 1, np.where(minimum >= cutoff, 0, -2)).astype(np.int8)
        with np.load(ROOT / f'private/{cohort}_date_roles.npz', allow_pickle=False) as z:
            assert np.array_equal(roles, z['roles'])
            assert np.array_equal(minimum.to_numpy(dtype='datetime64[D]'), z['min_lab_day'])
            assert np.array_equal(maximum.to_numpy(dtype='datetime64[D]'), z['max_lab_day'])
            assert np.array_equal(label.loc[label.Center.eq(cohort), 'MACE'].to_numpy(), z['y'])
        result = next((c for c in audit['cohorts'] if c['cohort'] == cohort))
        assert cutoff.date().isoformat() == result['temporal_feasibility']['cutoff']
        for field, values in [(TIME, labs), (ADMIT, admits), (DISCHARGE, discharges)]:
            expected = next((x for x in result['clock_audit'] if x['field'] == field))
            assert int(values.notna().sum()) == expected['parseable']
        clock_rows.append({'cohort': cohort, 'records': len(data), 'laboratory_before_stored_admission': int((labs < admits).sum()), 'laboratory_after_stored_discharge': int((labs > discharges).sum()), 'stored_discharge_before_stored_admission': int((discharges < admits).sum())})
        for f in REGIONAL:
            data[f] = pd.to_numeric(data[f], errors='coerce')
        row_vector = pd.MultiIndex.from_frame(data[REGIONAL])
        matched = vectors.reindex(row_vector)
        matchyear = matched[YEAR].to_numpy(dtype=float)
        complete = data[REGIONAL].notna().all(axis=1).to_numpy()
        matched_mask = np.isfinite(matchyear) & complete
        exact_count = int(matched_mask.sum())
        raw_sheet = audit['ecological_assignment']['sheets'][0]
        expected = next((c for c in raw_sheet['cohorts'] if c['cohort'] == cohort))
        assert exact_count == expected['complete_rows_with_exact_vector_in_sheet']
        for year in sorted(labs.dt.year.unique()):
            subset = labs.dt.year.to_numpy() == year
            geo_year_rows.append({'cohort': cohort, 'laboratory_year': int(year), 'records': int(subset.sum()), 'complete_regional_vector': int((subset & complete).sum()), 'exact_database_vector_match': int((subset & matched_mask).sum()), 'matched_vector_year_equal_lab_year': int((subset & matched_mask & (matchyear == year)).sum()), 'matched_vector_year_later_than_lab_year': int((subset & matched_mask & (matchyear > year)).sum()), 'matched_vector_year_earlier_than_lab_year': int((subset & matched_mask & (matchyear < year)).sum())})
        findings.append({'cohort': cohort, 'date_roles_exactly_replayed': True, 'rows_with_unique_complete_regional_vector_match': exact_count, 'matched_vector_year_equals_lab_year': int((matched_mask & (matchyear == labs.dt.year.to_numpy())).sum()), 'matched_vector_year_later_than_lab_year': int((matched_mask & (matchyear > labs.dt.year.to_numpy())).sum()), 'matched_vector_year_earlier_than_lab_year': int((matched_mask & (matchyear < labs.dt.year.to_numpy())).sum())})
    for s in audit['sources']:
        assert identity(s['path']) == s
    (ROOT / 'results').mkdir(exist_ok=True)
    pd.DataFrame(geo_year_rows).to_csv(ROOT / 'results/regional_year_consistency.csv', index=False, mode='x')
    pd.DataFrame(clock_rows).to_csv(ROOT / 'results/processed_clock_consistency.csv', index=False, mode='x')
    result = {'passed': True, 'context_audit': identity(ROOT / 'context_audit.json'), 'script': identity(__file__), 'independent_implementation': 'No import of audit_context.py. Reconstruct patient date roles and row-wise complete-vector mapping directly from source tables.', 'cohorts': findings, 'processed_clock_consistency': clock_rows, 'original_sources_unchanged': True, 'limitations': ['Complete-vector correspondence is not proof of correct patient residence or original mapping provenance.', 'Stored admission/discharge may have been attached through patient-wide matching. Cross-clock counts are inconsistency flags, not adjudicated errors.', 'No unmatched vector is assigned a city or year, and no original values or labels were changed.'], 'aggregate_outputs': [identity(ROOT / 'results/regional_year_consistency.csv'), identity(ROOT / 'results/processed_clock_consistency.csv')]}
    with (ROOT / 'context_verification.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
