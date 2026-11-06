"""Compare original regional inputs against independently sourced residence city."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import ast
import collections
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from recover_nested_source import HERE, OLD, RAW, ROOT, identity, load, sha
from audit_residence_fields import clean, classify, safe_path
BOOK = Path(_release_path('china', 'city_database.xlsx'))
COHORT = Path(_release_path('china', 'WHU_Cancer_processed_data.csv'))
LINK = _release_source_path(ROOT / 'revision_2026-10-12/analysis/audit_a1_linkage.py')
SOURCE = _release_source_path(ROOT / 'revision_2026-10-12/execution_2026-10-12/whu/rebuild_whu_inpatient_candidates_v1.py')
REGIONAL = ['人均地区生产总值(元)', '职工平均工资(元)', '医院、卫生院数(个)', '医院、卫生院床位数(张)', '医生数(人)']
FIELDS = {'现住址市', '户口市'}

def helpers():
    h = {'pd': pd, 're': re, 'collections': collections, 'hashlib': hashlib}
    tree = ast.parse(LINK.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'blank', 'key', 'date', 'dicts'}]
    assert len(nodes) == 4
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LINK), 'exec'), h)
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    constants = {'EMR', 'PERSONAL', 'STAY', 'PID', 'ADM', 'DIS'}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and (n.targets[0].id in constants):
            h[n.targets[0].id] = ast.literal_eval(n.value)
    names = {'private_token', 'unique_normalized', 'source_sections', 'source_link'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 4 and constants.issubset(h)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), h)
    return h

def main():
    target = HERE / 'native_city_original_value_comparison.json'
    if target.exists():
        raise RuntimeError('Refusing to overwrite completed comparison')
    h = helpers()
    book = pd.ExcelFile(BOOK)
    tables = {}
    for sheet in book.sheet_names:
        headers = pd.read_excel(book, sheet_name=sheet, nrows=0).columns
        if set(['年份', '地区', *REGIONAL]).issubset(headers):
            df = pd.read_excel(book, sheet_name=sheet, usecols=['年份', '地区', *REGIONAL])
            df['地区'] = df['地区'].map(clean)
            df['年份'] = pd.to_numeric(df['年份'], errors='coerce')
            for c in REGIONAL:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            assert not df.duplicated(['地区', '年份']).any()
            tables[sheet] = df
    cities = set(tables['原始数据']['地区'])
    original = pd.read_csv(COHORT, usecols=['病案号', '检查时间', '入院时间', '出院时间', *REGIONAL], dtype={'病案号': str})
    original['_patient'] = original['病案号'].map(h['key'])
    for c in ['检查时间', '入院时间', '出院时间']:
        original[c] = pd.to_datetime(original[c], format='mixed', errors='coerce')
    original['_stay'] = [h['private_token'](p, a.normalize(), d.normalize()) if pd.notna(a) and pd.notna(d) and (p is not None) else None for p, a, d in original[['_patient', '入院时间', '出院时间']].itertuples(index=False, name=None)]
    original['_pt'] = original['_patient'].map(h['private_token'])
    index_path = OLD / 'labs/private/source_index.json'
    capture_path = OLD / 'whu/private/diagnosis_capture_record_audit_v2.jsonl'
    contract = load(OLD / 'whu/diagnosis_capture_contract_v2.json')
    assert sha(capture_path) == contract['private_output']['sha256']
    targets = collections.defaultdict(lambda: collections.defaultdict(list))
    with capture_path.open(encoding='utf-8') as f:
        for line in f:
            row = _release_resolve_config(json.loads(line))
            targets[row['source_content_sha256']][row['record_ordinal']].append((row['patient_token'], row['stay_candidate_token']))
    locators = {}
    for x in sorted(load(index_path), key=lambda x: x['ordinal']):
        if x['sha256'] in targets:
            locators.setdefault(x['sha256'], x)
    by_stay = collections.defaultdict(set)
    by_patient = collections.defaultdict(set)
    field_status = collections.Counter()
    path_counts = collections.Counter()
    sibling_date_counts = collections.Counter()
    safe_key_dictionary = {}
    totals = collections.Counter()
    static_path_hashes = {None, None, None}
    sequence = hashlib.sha256()
    for number, (content_hash, locator) in enumerate(sorted(locators.items()), 1):
        data = (RAW / locator['basename']).read_bytes()
        assert len(data) == locator['bytes'] and hashlib.sha256(data).hexdigest() == content_hash
        sequence.update(bytes.fromhex(content_hash))
        obj = _release_resolve_config(json.loads(data.decode('utf-8-sig')))
        if isinstance(obj, str):
            obj = _release_resolve_config(json.loads(obj))
        records = h['dicts'](obj)
        totals['contents'] += 1
        totals['bytes'] += len(data)
        for ordinal, references in targets[content_hash].items():
            record = records[ordinal]
            pid, key, _, flags = h['source_link'](record, h)
            assert key is not None and (not any(flags.values()))
            pt, st = (h['private_token'](pid), h['private_token'](*key))
            assert all(((pt, st) == pair for pair in references))
            totals['source_link_token_replays'] += len(references)
            stack = [(record, (), None)]
            while stack:
                node, keypath, parent = stack.pop()
                if isinstance(node, dict):
                    for field, value in node.items():
                        if not isinstance(field, str):
                            continue
                        digest = hashlib.sha256(field.encode()).hexdigest()
                        if digest in static_path_hashes:
                            if len(field) < 50 and re.search('信息|记录|菜单|详情|首页|病案', field):
                                safe_key_dictionary[digest] = field
                            else:
                                totals['ancestor_key_kept_hashed_by_privacy_filter'] += 1
                        if field in FIELDS:
                            status, exact = classify(value, cities)
                            field_status[field, status] += len(references)
                            path_counts[field, safe_path(keypath), status] += len(references)
                            if exact is not None:
                                by_stay[field, st].add(exact)
                                by_patient[field, pt].add(exact)
                                if isinstance(parent, list) and all((isinstance(x, dict) and len(x) == 1 for x in parent)):
                                    dates = []
                                    for item in parent:
                                        name, v = next(iter(item.items()))
                                        if name in {'入院日期', '入院时间'}:
                                            dt = h['date'](v)
                                            dates.append(dt)
                                    if dates:
                                        valid = [x for x in dates if x is not None]
                                        same = bool(valid) and all((x.normalize() == key[1] for x in valid))
                                        sibling_date_counts[field, 'singleton_form_list', bool(valid), same] += len(references)
                        stack.append((value, keypath + (field,), node))
                elif isinstance(node, list):
                    stack.extend(((v, keypath + ('[]',), node) for v in node))
        if number % 500 == 0:
            print(f'Original-city identity/value replay checked {number}/{len(locators)} contents', flush=True)
    assert totals['source_link_token_replays'] == 5744
    comparisons, coverage = ([], [])
    for field in sorted(FIELDS):
        mapping = {st: next(iter(v)) for (f, st), v in by_stay.items() if f == field and len(v) == 1}
        left = original.copy()
        left['_native_city'] = left['_stay'].map(mapping)
        valid = left['_native_city'].notna()
        ps = [v for (f, _), v in by_patient.items() if f == field]
        coverage.append({'field': field, 'patients_with_exact_original_workbook_city': len(ps), 'patients_one_city': sum((len(v) == 1 for v in ps)), 'patients_multiple_cities': sum((len(v) > 1 for v in ps)), 'source_stay_day_keys_one_city': len(mapping), 'source_stay_day_keys_city_conflicts': sum((f == field and len(v) > 1 for (f, _), v in by_stay.items())), 'matched_processed_rows': int(valid.sum()), 'matched_processed_patients': int(left.loc[valid, '_pt'].nunique()), 'matched_processed_stay_day_keys': int(left.loc[valid, '_stay'].nunique())})
        left = left.loc[valid].copy()
        for date_col in ['检查时间', '入院时间']:
            left['_candidate_year'] = left[date_col].dt.year
            for sheet, table in tables.items():
                joined = left.merge(table, left_on=['_native_city', '_candidate_year'], right_on=['地区', '年份'], how='left', validate='many_to_one', suffixes=('_stored', '_book'), indicator=True)
                matched = joined['_merge'].eq('both')
                complete, equal = (np.ones(len(joined), bool), np.ones(len(joined), bool))
                per_field = []
                for name in REGIONAL:
                    a = pd.to_numeric(joined[name + '_stored'], errors='coerce').to_numpy(float)
                    b = pd.to_numeric(joined[name + '_book'], errors='coerce').to_numpy(float)
                    both = np.isfinite(a) & np.isfinite(b)
                    same = both & (a == b)
                    per_field.append({'field': name, 'both_finite_rows': int(both.sum()), 'exact_equal_rows': int(same.sum()), 'different_rows': int((both & ~same).sum())})
                    complete &= both
                    equal &= same
                comparisons.append({'residence_field': field, 'candidate_year_basis': date_col, 'workbook_sheet': sheet, 'eligible_processed_rows': len(joined), 'eligible_patients': int(joined['_pt'].nunique()), 'exact_city_year_key_matched_rows': int(matched.sum()), 'unmatched_city_year_rows': int((~matched).sum()), 'all_five_finite_rows': int(complete.sum()), 'all_five_exact_equal_rows': int(equal.sum()), 'all_five_exact_equal_patients': int(joined.loc[equal, '_pt'].nunique()), 'complete_rows_any_value_different': int((complete & ~equal).sum()), 'per_field': per_field})
    report = {'passed': True, 'inputs': [identity(BOOK), identity(COHORT), identity(index_path), identity(capture_path), identity(LINK), identity(SOURCE)], 'counts': dict(totals), 'source_hash_sequence': sequence.hexdigest(), 'original_workbook_city_names': len(cities), 'native_city_coverage': coverage, 'comparisons': comparisons, 'field_value_statuses': [dict(field=f, status=s, occurrences=n) for (f, s), n in sorted(field_status.items())], 'safe_source_path_dictionary': safe_key_dictionary, 'exact_city_source_paths': [dict(field=f, path=list(p), occurrences=n) for (f, p, s), n in sorted(path_counts.items()) if s == 'exact_city_workbook_name'], 'same_singleton_form_admission_date_context': [dict(field=f, container=c, date_parses=p, date_matches_verified_source_stay=d, occurrences=n) for (f, c, p, d), n in sorted(sibling_date_counts.items())], 'rules': ['City identity comes only from exact explicit source-field values matching the original workbook name dictionary.', 'All original workbook sheets with five regional columns are evaluated separately, without selection by match rate.', 'Exact internal patient plus admission/discharge day linkage is replayed against frozen source tokens.', 'Lab year and admission year are candidate comparison keys, not recovered historical assignment rules.', 'Different stays may have different cities; a stay with multiple exact source cities is excluded.', 'No city inferred by reversing regional values; no nearest city, fuzzy name or missing city imputation.'], 'historical_assignment_fully_validated': False, 'models_trained': 0, 'patient_identifiers_or_city_values_exported': False, 'private_rows_written': False, 'script': identity(__file__), 'environment': {'python': sys.version, 'executable': sys.executable, 'pandas': pd.__version__, 'numpy': np.__version__, 'platform': platform.platform()}}
    with target.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
        f.write('\n')
    print(json.dumps({'passed': True, 'coverage': coverage, 'comparison_sets': len(comparisons)}, ensure_ascii=True))
if __name__ == '__main__':
    main()
