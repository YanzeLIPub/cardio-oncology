"""Audit historical feature identity using headers and aggregate unit evidence."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[4]
BASE = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
LABS = PROJECT / 'revision_2026-10-12/njm_same_source_completion_2026-10-12/labs'

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def main():
    result = {'scope': 'Schema and aggregate unit evidence only. No source specimen identity is inferred from numeric values or similar names.', 'sources': [], 'cohorts': {}, 'WHU_exact_name_unit_evidence': {}}
    for centre, rawpath in (('SCH', _release_path('china', 'SCH_processed_data.csv')), ('WHU', _release_path('china', 'WHU_Cancer_processed_data.csv'))):
        result['sources'].append(identity(rawpath))
        headers = pd.read_csv(rawpath, nrows=0).columns.tolist()
        prepared = BASE / f'private/prepared/{centre}.npz'
        result['sources'].append(identity(prepared))
        with np.load(prepared, allow_pickle=False) as z:
            names = list(z['feature_names'])
            values = z['X'][:, names.index('白细胞')]
            finite = values[np.isfinite(values)]
            unique, frequencies = np.unique(finite, return_counts=True)
        result['cohorts'][centre] = {'processed_header_count': len(headers), 'distinct_source_header_fields_containing_white_cell_term': [x for x in headers if '白细胞' in x], 'model_feature_selected_exact_name': '白细胞', 'model_fields_containing_white_cell_term': [x for x in names if '白细胞' in x], 'historical_field_patient_mean_finite_n': int(len(finite)), 'historical_field_patient_mean_zero_n': int((finite == 0).sum()), 'historical_field_patient_mean_distinct_n': int(len(unique)), 'historical_field_maximum_tie_n': int(frequencies.max()), 'interpretation': 'Distinct header names are not interchangeable assays. The selected historical feature is retained without substitution. Numeric distributions alone cannot determine blood or urine origin.'}
    for filename in ('historical_104_inputs_exact_whu_units.csv', 'historical_104_inputs_same_stay_whu_units.csv'):
        path = LABS / filename
        result['sources'].append(identity(path))
        frame = pd.read_csv(path)
        selected = frame.loc[frame.input_name.eq('白细胞')]
        assert len(selected) == 1
        row = selected.iloc[0].to_dict()
        result['WHU_exact_name_unit_evidence'][filename] = {k: v.item() if isinstance(v, np.generic) else v for k, v in row.items()}
    result['clinical_interpretation'] = 'WHU exact-name unit evidence records G/L. It does not establish cross-center equivalence or validate SCH specimen/project identity. SCH contains separate white-cell-count, high-power-field and esterase headers. Its historical exact-name slot must not be presented as a validated blood WBC count, nor guessed to be a urine-derived measure. No original model inputs or historical Figure 3 data were altered.'
    output = ROOT / 'measurement_semantics.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
    print('Measurement semantics evidence saved; aggregate/schema evidence only.')
if __name__ == '__main__':
    main()
