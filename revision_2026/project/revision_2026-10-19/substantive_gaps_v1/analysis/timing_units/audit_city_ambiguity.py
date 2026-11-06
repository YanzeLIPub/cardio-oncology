"""Stricter source-city sensitivity including nonempty unrecognized values."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import collections
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from compare_native_city_values import BOOK, COHORT, FIELDS, REGIONAL, helpers
from audit_residence_fields import classify, clean
from recover_nested_source import HERE, OLD, RAW, identity, load

def main():
    output = HERE / 'strict_ambiguity_sensitivity.json'
    if output.exists():
        raise RuntimeError('Refusing to overwrite completed strict sensitivity')
    h = helpers()
    tables = {s: pd.read_excel(BOOK, sheet_name=s, usecols=['年份', '地区', *REGIONAL]) for s in ['原始数据', '线性插值', 'Arima填补']}
    for table in tables.values():
        table['地区'] = table['地区'].map(clean)
        for c in ['年份', *REGIONAL]:
            table[c] = pd.to_numeric(table[c], errors='coerce')
        assert not table.duplicated(['地区', '年份']).any()
    cities = set(tables['原始数据']['地区'])
    index = OLD / 'labs/private/source_index.json'
    capture = OLD / 'whu/private/diagnosis_capture_record_audit_v2.jsonl'
    targets = collections.defaultdict(lambda: collections.defaultdict(list))
    with capture.open(encoding='utf-8') as f:
        for line in f:
            x = _release_resolve_config(json.loads(line))
            targets[x['source_content_sha256']][x['record_ordinal']].append((x['patient_token'], x['stay_candidate_token']))
    locators = {}
    for x in sorted(load(index), key=lambda x: x['ordinal']):
        if x['sha256'] in targets:
            locators.setdefault(x['sha256'], x)
    exact, other = (collections.defaultdict(set), collections.defaultdict(set))
    source_sequence = hashlib.sha256()
    total = 0
    for number, (digest, x) in enumerate(sorted(locators.items()), 1):
        data = (RAW / x['basename']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest and len(data) == x['bytes']
        source_sequence.update(bytes.fromhex(digest))
        obj = _release_resolve_config(json.loads(data.decode('utf-8-sig')))
        if isinstance(obj, str):
            obj = _release_resolve_config(json.loads(obj))
        records = h['dicts'](obj)
        for ordinal, links in targets[digest].items():
            pid, key, _, flags = h['source_link'](records[ordinal], h)
            assert key is not None and (not any(flags.values()))
            pt, st = (h['private_token'](pid), h['private_token'](*key))
            assert all(((pt, st) == v for v in links))
            total += len(links)
            stack = [records[ordinal]]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    for field, value in node.items():
                        if field in FIELDS:
                            status, city = classify(value, cities)
                            if city is not None:
                                exact[field, st].add(city)
                            elif status not in {'empty', 'empty_container', 'empty_or_placeholder'}:
                                other[field, st].add(status)
                        stack.append(value)
                elif isinstance(node, list):
                    stack.extend(node)
        if number % 500 == 0:
            print(f'Strict residence ambiguity checked {number}/{len(locators)} contents', flush=True)
    assert total == 5744
    previous = load(HERE / 'native_city_original_value_comparison.json')
    assert source_sequence.hexdigest() == previous['source_hash_sequence']
    raw = pd.read_csv(COHORT, usecols=['病案号', '检查时间', '入院时间', '出院时间', *REGIONAL], dtype={'病案号': str})
    raw['_patient'] = raw['病案号'].map(h['key'])
    for c in ['检查时间', '入院时间', '出院时间']:
        raw[c] = pd.to_datetime(raw[c], format='mixed', errors='coerce')
    raw['_stay'] = [h['private_token'](p, a.normalize(), d.normalize()) if p is not None and pd.notna(a) and pd.notna(d) else None for p, a, d in raw[['_patient', '入院时间', '出院时间']].itertuples(index=False, name=None)]
    coverage, comparisons = ([], [])
    for field in sorted(FIELDS):
        broad = {st: next(iter(v)) for (f, st), v in exact.items() if f == field and len(v) == 1}
        strict = {st: city for st, city in broad.items() if not other.get((field, st))}
        excluded = set(broad) - set(strict)
        before = raw['_stay'].isin(broad)
        after = raw['_stay'].isin(strict)
        excluded_rows = raw['_stay'].isin(excluded)
        coverage.append({'field': field, 'broad_exact_city_stays': len(broad), 'strict_city_stays': len(strict), 'exact_city_stays_with_nonempty_unrecognized_city_values': len(excluded), 'broad_processed_rows': int(before.sum()), 'strict_processed_rows': int(after.sum()), 'strict_patients': int(raw.loc[after, '_patient'].nunique()), 'excluded_processed_rows': int(excluded_rows.sum()), 'excluded_patients': int(raw.loc[excluded_rows, '_patient'].nunique()), 'excluded_value_statuses': dict(collections.Counter((s for st in excluded for s in other[field, st])))})
        left = raw.loc[after].copy()
        left['_city'] = left['_stay'].map(strict)
        for basis in ['检查时间', '入院时间']:
            left['_year'] = left[basis].dt.year
            for sheet, table in tables.items():
                joined = left.merge(table, left_on=['_city', '_year'], right_on=['地区', '年份'], how='left', validate='many_to_one', suffixes=('_stored', '_book'), indicator=True)
                complete, equal, close = (np.ones(len(joined), bool) for _ in range(3))
                per_field = []
                for name in REGIONAL:
                    a = pd.to_numeric(joined[name + '_stored'], errors='coerce').to_numpy(float)
                    b = pd.to_numeric(joined[name + '_book'], errors='coerce').to_numpy(float)
                    both = np.isfinite(a) & np.isfinite(b)
                    same = both & (a == b)
                    near = both & np.isclose(a, b, rtol=1e-10, atol=1e-08)
                    complete &= both
                    equal &= same
                    close &= near
                    per_field.append({'field': name, 'both_finite_rows': int(both.sum()), 'exact_equal_rows': int(same.sum()), 'floating_tolerance_equal_rows': int(near.sum()), 'different_rows': int((both & ~same).sum()), 'beyond_floating_tolerance_rows': int((both & ~near).sum())})
                comparisons.append({'field': field, 'candidate_year_basis': basis, 'workbook_sheet': sheet, 'eligible_processed_rows': len(joined), 'eligible_patients': int(joined['_patient'].nunique()), 'city_year_key_matched_rows': int(joined['_merge'].eq('both').sum()), 'all_five_finite_rows': int(complete.sum()), 'all_five_exact_equal_rows': int(equal.sum()), 'all_five_exact_equal_patients': int(joined.loc[equal, '_patient'].nunique()), 'all_five_floating_tolerance_equal_rows': int(close.sum()), 'complete_rows_any_beyond_floating_tolerance': int((complete & ~close).sum()), 'complete_rows_any_different': int((complete & ~equal).sum()), 'per_field': per_field})
    result = {'passed': True, 'source_contents_replayed': len(locators), 'source_record_token_replays': total, 'source_hash_sequence': source_sequence.hexdigest(), 'strict_rule': 'Exclude a stay if any selected field has any nonempty unrecognized value in any linked source record, including numeric/container/template values. Empty placeholders alone do not exclude.', 'floating_tolerance': {'rtol': 1e-10, 'atol': 1e-08, 'purpose': 'Numerical representation sensitivity only, not an allowed substantive value replacement.'}, 'coverage': coverage, 'comparisons': comparisons, 'inputs': [identity(BOOK), identity(COHORT), identity(index), identity(capture), identity(HERE / 'native_city_original_value_comparison.json')], 'script': identity(__file__), 'environment': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__, 'pandas': pd.__version__}, 'originals_written': False, 'patient_rows_exported': False, 'historical_assignment_fully_validated': False}
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=True, indent=2)
        f.write('\n')
    print(json.dumps({'passed': True, 'coverage': coverage}, ensure_ascii=True))
if __name__ == '__main__':
    main()
