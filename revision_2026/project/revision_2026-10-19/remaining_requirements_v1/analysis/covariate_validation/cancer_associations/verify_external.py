"""Cross-agent replay without importing analysis-author implementation functions."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '2'
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
OLD = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
STAGE = PROJECT / 'revision_2026-10-13/chinese_omop_diagnosis_alignment_2026-10-13'
LABELS = Path(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'))
KEY = '病案号'
RAW = {'SCH': Path(_release_path('china', 'SCH_processed_data.csv')), 'WHU': Path(_release_path('china', 'WHU_Cancer_processed_data.csv'))}

def read(path):
    return _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def identity(path):
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def canonical(value):
    value = str(value).strip()
    if re.fullmatch('[0-9]+(?:\\.0+)?', value):
        return str(int(value.partition('.')[0]))
    return value

class Categories(HTMLParser):

    def __init__(self):
        super().__init__()
        self.codes = set()

    def handle_starttag(self, tag, attributes):
        data = dict(attributes)
        code = data.get('id', '')
        if tag == 'a' and re.fullmatch('C[0-9]{2}', code) and (data.get('name') == code) and (data.get('class') == 'code'):
            self.codes.add(code)

def rank_correlation(a, b, axis=-1):
    ra = rankdata(a, axis=axis, method='average')
    rb = rankdata(b, axis=axis, method='average')
    ra -= ra.mean(axis=axis, keepdims=True)
    rb -= rb.mean(axis=axis, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (ra * rb).sum(axis=axis) / np.sqrt((ra * ra).sum(axis=axis) * (rb * rb).sum(axis=axis))

def close(actual, expected):
    assert np.allclose(actual, expected, rtol=0, atol=2e-12, equal_nan=True)

def main():
    protocol = read(HERE / 'protocol.json')
    preparation = read(HERE / 'preparation_manifest.json')
    execution = read(HERE / 'execution.json')
    protected = {}
    for document in (preparation, execution, read(HERE / 'analysis_lock.json')):
        for item in document['inputs'] + document.get('outputs', []):
            current = identity(item['path'])
            protected[current['path']] = current
    dictionary = read(HERE / 'dictionary_verification.json')
    dictionary_codes = set()
    for entry in dictionary['sources']:
        current = identity(entry['path'])
        protected[current['path']] = current
        parser = Categories()
        parser.feed(Path(entry['path']).read_text(encoding='utf-8'))
        assert parser.codes == set(entry['categories'])
        dictionary_codes |= parser.codes
    assert dictionary_codes == set(protocol['valid_categories']) and len(dictionary_codes) == 89
    assert len(protocol['valid_categories']) == len(set(protocol['valid_categories']))
    primary = {'lung_bronchus': {'C34'}, 'breast': {'C50'}, 'digestive': {f'C{i:02d}' for i in range(15, 27)}, 'hematologic': {c for c in dictionary_codes if 81 <= int(c[1:]) <= 96}}
    secondary = {f'C{i}' for i in range(76, 81)}
    primary['other_defined_primary_site'] = dictionary_codes - set.union(*primary.values()) - secondary - {'C97'}
    assert {k: set(v) for k, v in protocol['primary_groups'].items()} == primary
    assert set(protocol['allowed_roles']) == {'principal_discharge', 'secondary_discharge', 'discharge', 'current_diagnosis'}
    current_script = _release_source_path(HERE / 'run_correlations.py').read_bytes()
    assert current_script.count(b'assert len(finite) > 0') == 1
    old_script = current_script.replace(b'assert len(finite) > 0', b'assert len(finite) >= 1900')
    failed_lock = read(HERE / 'analysis_lock_attempt1_failed.json')
    failed_entry = next((x for x in failed_lock['inputs'] if x['path'].endswith('run_correlations.py')))
    assert len(old_script) == failed_entry['bytes']
    active_lock = read(HERE / 'analysis_lock.json')
    assert [v for v in failed_lock['inputs'] if not v['path'].endswith('run_correlations.py')] == [v for v in active_lock['inputs'] if not v['path'].endswith('run_correlations.py')]
    labels = pd.read_csv(LABELS, usecols=[KEY, 'Center', 'MACE'], dtype={KEY: str})
    coverage = read(HERE / 'coverage.json')['cohorts']
    pairs = [('NSE', 'albumin'), ('NSE', 'WBC'), ('WBC', 'albumin')]
    assert protocol['pairs'] == [list(p) for p in pairs]
    assert protocol['biomarkers'] == {'NSE': '神经元特异性烯醇化酶', 'albumin': '白蛋白', 'WBC': '白细胞'}
    assert protocol['bootstrap']['replicates'] == 2000 and protocol['bootstrap']['seed_base'] == 202610190
    public = pd.read_csv(HERE / 'correlations.csv').set_index(['centre', 'stratum', 'pair'])
    public_coverage = pd.read_csv(HERE / 'coverage.csv').set_index(['centre', 'stratum', 'pair'])
    assert len(public) == 60 and public.index.is_unique and (len(public_coverage) == 60)
    json_rows = {(r['centre'], r['stratum'], r['pair']): r for r in read(HERE / 'correlations.json')['rows']}
    draws = np.load(HERE / 'private/bootstrap_rhos.npz', allow_pickle=False)
    source_audits, row_audits = ([], [])
    expected_draw_keys = set()
    total_replayed = 0
    for ci, centre in enumerate(('SCH', 'WHU')):
        local = labels[labels.Center.eq(centre)].reset_index(drop=True)
        patient_keys = local[KEY].map(canonical)
        assert patient_keys.is_unique
        prefix = 'SCH:person:' if centre == 'SCH' else 'WHU|person|'
        position = {prefix + k: i for i, k in enumerate(patient_keys)}
        private = dict(np.load(HERE / f'private/{centre}_strata.npz', allow_pickle=False))
        frozen = dict(np.load(OLD / f'private/prepared/{centre}.npz', allow_pickle=False))
        assert np.array_equal(frozen['y'], local.MACE.to_numpy())
        assert np.array_equal(private['groups'], np.arange(len(local)))
        names = list(protocol['biomarkers'])
        assert list(private['biomarker_names']) == names
        fields = [protocol['biomarkers'][name] for name in names]
        feature_names = list(frozen['feature_names'])
        assert len(feature_names) == 104 and all((feature_names.count(v) == 1 for v in fields))
        indices = [feature_names.index(v) for v in fields]
        assert np.array_equal(private['biomarkers'], frozen['X'][:, indices], equal_nan=True)
        protected[RAW[centre].as_posix()] = identity(RAW[centre])
        raw = pd.read_csv(RAW[centre], usecols=[KEY, *fields], dtype={KEY: str}, low_memory=False)
        raw[KEY] = raw[KEY].map(canonical)
        assert set(raw[KEY]) == set(patient_keys)
        for field in fields:
            raw[field] = pd.to_numeric(raw[field], errors='coerce')
        rebuilt = raw.groupby(KEY)[fields].mean().reindex(patient_keys).to_numpy(dtype=np.float32)
        assert np.array_equal(rebuilt, private['biomarkers'], equal_nan=True)
        assert not np.isinf(rebuilt).any()
        del raw
        n = len(local)
        codes = [set() for _ in range(n)]
        unfiltered_codes = [set() for _ in range(n)]
        any_source = np.zeros(n, bool)
        eligible_source = np.zeros(n, bool)
        counts, roles, categories = (Counter(), Counter(), Counter())
        source = STAGE / centre.lower() / 'private/staging/conditions.jsonl'
        with source.open(encoding='utf-8') as stream:
            for line in stream:
                row = _release_resolve_config(json.loads(line))
                counts['all_condition_rows'] += 1
                p = position.get(row['person_source_key'])
                if p is None:
                    counts['unlinked_person_rows'] += 1
                    continue
                any_source[p] = True
                role = row['diagnosis_role']
                roles[role] += 1
                value = unicodedata.normalize('NFKC', str(row.get('diagnosis_code') or '')).strip().upper()
                match = re.fullmatch('(C[0-9]{2})(?:\\.[A-Z0-9]+)?', value)
                valid_code = match is not None and match[1] in dictionary_codes
                if role in protocol['allowed_roles'] and valid_code:
                    unfiltered_codes[p].add(match[1])
                if row['source_value_conflict']:
                    counts['source_conflict_rows_excluded'] += 1
                    continue
                if role not in protocol['allowed_roles']:
                    counts['role_rows_excluded'] += 1
                    continue
                eligible_source[p] = True
                counts['eligible_condition_rows'] += 1
                if not value:
                    counts['eligible_missing_code_rows'] += 1
                    continue
                counts['eligible_nonempty_code_rows'] += 1
                if match is None:
                    counts['eligible_nonmalignant_or_noncanonical_code_rows'] += 1
                elif not valid_code:
                    counts['eligible_unassigned_C_category_rows'] += 1
                else:
                    counts['eligible_valid_C_category_rows'] += 1
                    codes[p].add(match[1])
                    categories[match[1]] += 1
        category_membership = np.zeros((n, 89), bool)
        strata = []
        for p, observed in enumerate(codes):
            category_membership[p] = [c in observed for c in protocol['valid_categories']]
            membership = [g for g, members in primary.items() if observed.intersection(members)]
            if 'C97' in observed or len(membership) > 1:
                group = 'multiple_broad_groups_or_C97'
            elif membership:
                group = membership[0]
            elif observed.intersection(secondary):
                group = 'secondary_or_unspecified_only'
            else:
                group = 'no_interpretable_malignant_code' if eligible_source[p] else 'no_eligible_diagnostic_source'
            strata.append(group)
        strata = np.asarray(strata)
        for key, value in (('category_membership', category_membership), ('strata', strata), ('eligible_source', eligible_source), ('any_source', any_source)):
            assert np.array_equal(value, private[key])
        expected_coverage = {'patients': n, 'patients_any_diagnostic_source': int(any_source.sum()), 'patients_eligible_diagnostic_source': int(eligible_source.sum()), 'patients_valid_malignant_category': int(category_membership.any(axis=1).sum()), 'source_row_counts': dict(counts), 'all_linked_source_roles': dict(roles), 'valid_malignant_category_row_counts': dict(categories), 'strata_counts': dict(Counter(strata))}
        expected_coverage['strata_counts'] = {g: int((strata == g).sum()) for g in protocol['strata_order']}
        assert expected_coverage == coverage[centre]
        source_audits.append({'centre': centre, 'raw_means_exact_and_missingness_preserved': True, 'native_code_membership_and_all_counts_rebuilt': True, 'valid_malignant_patients_before_conflict_filter': sum((bool(x) for x in unfiltered_codes)), 'valid_malignant_patients_after_conflict_filter': sum((bool(x) for x in codes)), 'observed_biomarker_patients': {name: int(np.isfinite(rebuilt[:, j]).sum()) for j, name in enumerate(names)}, 'zero_patient_means': {name: int((rebuilt[:, j] == 0).sum()) for j, name in enumerate(names)}, 'numeric_input_identity': dict(zip(names, fields))})
        print(centre, 'source categories, mean inputs and coverage independently matched', flush=True)
        for si, group in enumerate(['all_original_patients', *protocol['strata_order']]):
            members = np.ones(n, bool) if si == 0 else strata == group
            for pi, (a, b) in enumerate(pairs):
                index = (centre, group, a + '--' + b)
                row, covrow, jrow = (public.loc[index], public_coverage.loc[index], json_rows[index])
                pair = rebuilt[:, [names.index(a), names.index(b)]]
                values = pair[members & np.isfinite(pair).all(axis=1)].astype(float)
                size = len(values)
                for column, value in (('patients', int(members.sum())), ('pair_complete_n', size), ('pair_missing_n', int(members.sum()) - size)):
                    assert row[column] == covrow[column] == jrow[column] == value
                seed = 202610190 + 1000 * ci + 10 * si + pi
                assert row.bootstrap_seed == jrow['bootstrap_seed'] == seed
                assert row.bootstrap_requested == jrow['bootstrap_requested'] == 2000
                if size < 50:
                    status = 'not_estimated_pair_n_below_50'
                elif np.any(np.ptp(values, axis=0) == 0):
                    status = 'not_estimated_constant_variable'
                else:
                    status = 'estimated'
                assert row.status == jrow['status'] == status
                audit = {'centre': centre, 'stratum': group, 'pair': index[2], 'status': status, 'n': size}
                if status != 'estimated':
                    assert row[['rho', 'ci_low', 'ci_high']].isna().all()
                    assert all((jrow[c] is None for c in ('rho', 'ci_low', 'ci_high')))
                    assert row.bootstrap_valid == row.bootstrap_undefined == 0
                    row_audits.append(audit)
                    continue
                rho = float(rank_correlation(values[:, 0], values[:, 1]))
                close(rho, row.rho)
                close(rho, jrow['rho'])
                rng = np.random.default_rng(seed)
                replayed = np.full(2000, np.nan)
                for start in range(0, 2000, 16):
                    stop = min(start + 16, 2000)
                    sample = values[rng.integers(0, size, size=(stop - start, size))]
                    replayed[start:stop] = rank_correlation(sample[:, :, 0], sample[:, :, 1], axis=1)
                draw_key = f'{centre}_{si}_{pi}'
                expected_draw_keys.add(draw_key)
                close(replayed, draws[draw_key])
                finite = np.isfinite(replayed)
                assert int(finite.sum()) == row.bootstrap_valid == jrow['bootstrap_valid']
                assert int((~finite).sum()) == row.bootstrap_undefined == jrow['bootstrap_undefined']
                quantiles = np.quantile(replayed[finite], [0.025, 0.975])
                close(quantiles, row[['ci_low', 'ci_high']].to_numpy(float))
                close(quantiles, [jrow['ci_low'], jrow['ci_high']])
                total_replayed += 2000
                audit.update({'rho': rho, 'bootstrap_recomputed': 2000, 'undefined': int((~finite).sum()), 'bootstrap_and_point_max_abs_difference': float(np.nanmax(np.abs(replayed - draws[draw_key])))})
                row_audits.append(audit)
            print(centre, group, 'all pair statistics and resamples checked', flush=True)
    assert expected_draw_keys == set(draws.files)
    for original in protected.values():
        assert identity(original['path']) == original
    report = {'passed': True, 'verification_kind': 'cross-agent independent implementation and source reconstruction', 'author_functions_imported': False, 'protected_inputs_and_author_outputs_unchanged': True, 'official_dictionary_categories_reparsed': 89, 'source_audits': source_audits, 'rows_checked': len(row_audits), 'bootstrap_samples_recomputed': total_replayed, 'rows': row_audits, 'failed_script_single_assertion_repair_verified': True, 'scientific_gates': {'WBC_identity': 'Exact original 104-input feature named 白细胞 verified; blood/urine identity and original units are not established by this numerical check. Do not call it blood WBC or a mechanistic validation.', 'coverage': 'WHU native interpretable malignant code coverage is limited. Unknown and unavailable strata are retained; neither means no cancer. Multigroup/C97 is recorded coding, not adjudicated multiple primary cancers.', 'undefined_bootstraps': 'Intervals exclude explicitly counted constant resamples according to the frozen protocol; high undefined fractions condition these intervals and reduce their evidential usefulness.', 'timing': 'Full-record codes and means do not establish baseline exposures, pre-event measurement, or outcome-independent ascertainment.', 'causality': 'Unadjusted within-source-category associations do not control stage, treatment, acute severity or selective measurement.'}, 'protected_files': list(protected.values()), 'script': identity(__file__), 'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__}}
    with (HERE / 'independent_external_verification.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('PASS', len(row_audits), 'rows;', total_replayed, 'independent bootstrap samples', flush=True)
if __name__ == '__main__':
    main()
