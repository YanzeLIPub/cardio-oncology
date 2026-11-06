"""Aggregate native residence-field provenance; never exports addresses or keys."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import collections
import hashlib
import json
import platform
import re
import sys
import unicodedata
from pathlib import Path
import pandas as pd
from recover_nested_source import HERE, OLD, RAW, identity, load, sha
CITY_BOOK = Path(_release_path('njm', 'city_database_official_update_2025_working.xlsx'))
FIELDS = {'现住址省', '现住址市', '现住址县', '现住址', '居住地址', '户口省', '户口市', '户口县', '户口地址', '户籍地址', '家庭地址'}
CITY_FIELDS = {'现住址市', '户口市'}
SAFE_CONTEXT = {'病案首页', '住院病案首页', '基本信息', '患者基本信息', '个人信息', '数据', '内容', '病历内容', '入院记录', '住院信息', '出院记录', '死亡记录', '病程记录', '病历', '病案', '患者信息', '病人信息', '登记信息', '基本资料', '病历首页', '病案首页信息', '病案首页主表', '病案首页记录', '病案信息', '住院病人病案首页'}
PLACEHOLDERS = {'', '-', '--', '/', '无', '不详', '未知', '请选择', '未填写', 'null', 'none', 'nan', '省', '市', '县', '省份', '城市', '地区', '0'}
DATE_KEY = re.compile('^(?:入院时间|入院日期|入院年月日|住院日期|住院时间)$')
DATE_VALUE = re.compile('^\\s*(?:19|20)\\d{2}[-/.年]\\d{1,2}[-/.月]\\d{1,2}')

def clean(v):
    return unicodedata.normalize('NFKC', str(v)).strip()

def classify(value, cities):
    if value is None:
        return ('empty', None)
    if isinstance(value, (dict, list, tuple)):
        return ('empty_container' if not value else 'unresolved_container', None)
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return ('numeric_unmapped', None)
    v = clean(value)
    if v.lower() in PLACEHOLDERS:
        return ('empty_or_placeholder', None)
    if re.search('<[^>]+>|\\{\\{|\\}\\}|\\[\\[|\\]\\]|\\$\\{|请选择|请填写', v):
        return ('template_or_html', None)
    if v.startswith(('[', '{')):
        return ('unresolved_serialized_container', None)
    if v in cities:
        return ('exact_city_workbook_name', v)
    if re.fullmatch('\\d+(?:\\.0+)?', v):
        return ('numeric_string_unmapped', None)
    return ('plain_text_not_city_name', None)

def safe_path(path):
    return tuple((k if k in SAFE_CONTEXT else '[]' if k == '[]' else 'key_sha256:' + hashlib.sha256(k.encode()).hexdigest() for k in path))

def date_context(node):
    valid = []
    for key, value in node.items():
        if not isinstance(key, str) or not DATE_KEY.fullmatch(key) or (not isinstance(value, str)) or (not DATE_VALUE.match(value)):
            continue
        try:
            parsed = pd.to_datetime(value, errors='raise')
        except (ValueError, TypeError, OverflowError):
            continue
        valid.append((key, int(parsed.year)))
    return valid

def main():
    output = HERE / 'residence_source_provenance.json'
    if output.exists():
        raise RuntimeError('Refusing to overwrite completed geography audit')
    city = pd.read_excel(CITY_BOOK, sheet_name='原始数据', usecols=['年份', '行政区划代码', '地区'])
    cities = {clean(x) for x in city['地区'].dropna()}
    years_by_city = collections.defaultdict(set)
    for row in city.itertuples(index=False, name=None):
        try:
            year = int(row[0])
        except (ValueError, TypeError):
            continue
        years_by_city[clean(row[2])].add(year)
    index_path = OLD / 'labs/private/source_index.json'
    capture_path = OLD / 'whu/private/diagnosis_capture_record_audit_v2.jsonl'
    contract = load(OLD / 'whu/diagnosis_capture_contract_v2.json')
    assert sha(capture_path) == contract['private_output']['sha256']
    targets = collections.defaultdict(lambda: collections.defaultdict(list))
    patients, stays = (set(), set())
    with capture_path.open(encoding='utf-8') as f:
        for line in f:
            row = _release_resolve_config(json.loads(line))
            targets[row['source_content_sha256']][row['record_ordinal']].append((row['patient_token'], row['stay_candidate_token']))
            patients.add(row['patient_token'])
            stays.add(row['stay_candidate_token'])
    locators = {}
    for x in sorted(load(index_path), key=lambda x: x['ordinal']):
        if x['sha256'] in targets:
            locators.setdefault(x['sha256'], x)
    counts, paths, totals = (collections.Counter(), collections.Counter(), collections.Counter())
    patient_sets, stay_sets, record_sets = (collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set))
    valid_patient_values, valid_stay_values = (collections.defaultdict(set), collections.defaultdict(set))
    exact_year_patient_values = collections.defaultdict(set)
    distinct_values = collections.defaultdict(set)
    patient_years = collections.defaultdict(set)
    source_sequence = hashlib.sha256()
    for count, (content_hash, locator) in enumerate(sorted(locators.items()), 1):
        path = RAW / locator['basename']
        data = path.read_bytes()
        assert len(data) == locator['bytes'] and hashlib.sha256(data).hexdigest() == content_hash
        source_sequence.update(bytes.fromhex(content_hash))
        obj = _release_resolve_config(json.loads(data.decode('utf-8-sig')))
        if isinstance(obj, str):
            obj = _release_resolve_config(json.loads(obj))
        records = [obj] if isinstance(obj, dict) else [x for x in obj if isinstance(x, dict)]
        totals['unique_contents'] += 1
        totals['bytes'] += len(data)
        for ordinal, links in targets[content_hash].items():
            record_id = (content_hash, ordinal)
            totals['record_occurrences'] += len(links)
            stack = [(records[ordinal], (), ())]
            while stack:
                node, keypath, dates = stack.pop()
                if isinstance(node, dict):
                    local_dates = date_context(node)
                    active_dates = tuple(local_dates) or dates
                    for key, value in node.items():
                        if not isinstance(key, str):
                            continue
                        if key in FIELDS:
                            status, exact_city = classify(value, cities)
                            has_date = bool(active_dates)
                            signature = (key, status, has_date)
                            counts[signature] += len(links)
                            paths[key, safe_path(keypath), status, has_date] += len(links)
                            for patient, stay in links:
                                patient_sets[signature].add(patient)
                                stay_sets[signature].add(stay)
                            record_sets[signature].add(record_id)
                            if isinstance(value, str) and status not in {'empty_or_placeholder', 'template_or_html'}:
                                distinct_values[key, status].add(clean(value))
                            if exact_city and key in CITY_FIELDS:
                                for patient, stay in links:
                                    valid_patient_values[key, patient].add(exact_city)
                                    valid_stay_values[key, stay].add(exact_city)
                                    for _, year in active_dates:
                                        patient_years[key, patient].add(year)
                                        if year in years_by_city[exact_city]:
                                            exact_year_patient_values[key, patient].add((exact_city, year))
                        stack.append((value, keypath + (key,), active_dates))
                elif isinstance(node, list):
                    stack.extend(((value, keypath + ('[]',), dates) for value in node))
        if count % 500 == 0:
            print(f'Native residence provenance checked {count}/{len(locators)} contents', flush=True)
    assert totals['record_occurrences'] == 5744 and len(patients) == 2246 and (len(stays) == 5415)
    summaries = []
    for field in sorted(CITY_FIELDS):
        ps = [v for (k, _), v in valid_patient_values.items() if k == field]
        ss = [v for (k, _), v in valid_stay_values.items() if k == field]
        ys = [v for (k, _), v in exact_year_patient_values.items() if k == field]
        summaries.append({'field': field, 'patients_with_exact_workbook_city': len(ps), 'patients_one_distinct_exact_city': sum((len(v) == 1 for v in ps)), 'patients_multiple_exact_cities': sum((len(v) > 1 for v in ps)), 'stay_day_keys_with_exact_workbook_city': len(ss), 'stay_day_keys_one_distinct_exact_city': sum((len(v) == 1 for v in ss)), 'stay_day_keys_multiple_exact_cities': sum((len(v) > 1 for v in ss)), 'patients_with_ancestor_admission_year_matching_workbook_city_year': len(ys), 'patients_one_distinct_ancestor_city_year_pair': sum((len(v) == 1 for v in ys))})
    report = {'passed': True, 'inputs': [identity(index_path), identity(capture_path), identity(CITY_BOOK)], 'raw_source_hash_sequence': source_sequence.hexdigest(), 'counts': dict(totals), 'matched_patients': len(patients), 'matched_stay_day_keys': len(stays), 'workbook_sheet': '原始数据', 'workbook_city_field': '地区', 'workbook_distinct_names': len(cities), 'scope': 'Exact whitelisted native residence fields. No geocoding, address extraction, fuzzy city matching or historical assignment assumed.', 'field_status_coverage': [dict(field=k, value_status=s, ancestor_or_same_node_admission_date=d, occurrences=n, patients=len(patient_sets[k, s, d]), stay_day_keys=len(stay_sets[k, s, d]), content_records=len(record_sets[k, s, d])) for (k, s, d), n in sorted(counts.items())], 'source_node_contexts': [dict(field=k, safe_ancestor_path=list(p), value_status=s, ancestor_or_same_node_admission_date=d, occurrences=n) for (k, p, s, d), n in sorted(paths.items())], 'distinct_values_by_field_status': [dict(field=k, value_status=s, count=len(v)) for (k, s), v in sorted(distinct_values.items())], 'exact_city_binding_summary': summaries, 'date_context_semantics': 'Only strictly parsed explicit admission dates from the same dictionary or an ancestor are carried; no dates borrowed across sibling records. Admission year is not the verified historical economic-year assignment.', 'historical_regional_assignment_validated': False, 'patient_addresses_or_city_values_exported': False, 'private_values_retained_on_disk': False, 'originals_written': False, 'script': identity(__file__), 'environment': {'python': sys.version, 'executable': sys.executable, 'pandas': pd.__version__, 'platform': platform.platform()}}
    with output.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=True, indent=2)
        f.write('\n')
    print(json.dumps({'passed': True, 'counts': dict(totals), 'exact_city_binding_summary': summaries}, ensure_ascii=True))
if __name__ == '__main__':
    main()
