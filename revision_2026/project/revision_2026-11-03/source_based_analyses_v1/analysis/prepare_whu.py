from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import csv, json, re, sys
import numpy as np
import pandas as pd
ROOT = Path(_release_path('project', ''))
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'revision_2026-11-03/source_gap_search_v1'))
from scan_sources import records, key, link
PRIVATE = OUT / 'private'
AGG = OUT / 'aggregate'
PRIVATE.mkdir(exist_ok=True)
AGG.mkdir(exist_ok=True)

def save(name, v):
    (AGG / name).write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')

def clock(v):
    if not isinstance(v, str):
        return None
    s = re.sub('(\\d{4})年(\\d{1,2})月(\\d{1,2})日', '\\1-\\2-\\3', v.strip()).replace('/', '-')
    m = re.fullmatch('(\\d{4})-(\\d{1,2})-(\\d{1,2})(?:[ T](\\d{1,2}):(\\d{2})(?::(\\d{2}))?)?', s)
    if not m:
        return None
    try:
        return datetime(*[int(x or 0) for x in m.groups()])
    except ValueError:
        return None

def unique_clock(values):
    vals = [clock(v) for v in values if v]
    if not vals or any((v is None for v in vals)):
        return None
    exact = [clock(v) for v in values if ':' in str(v)]
    if not exact or len(set(exact)) != 1:
        return None
    c = exact[0]
    return c if all((v.date() == c.date() for v in vals)) else None

def numeric(v):
    s = str(v).strip()
    return float(s) if re.fullmatch('[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?', s) and np.isfinite(float(s)) else None

def demographics(visit):
    vals = defaultdict(set)
    for block in records(visit.get('电子病历菜单及详情')):
        for section, payload in block.items():
            if section not in ['个人基本信息', '住院基本信息']:
                continue
            for d in records(payload):
                for field in ['年龄', '性别']:
                    if field in d and d[field] is not None:
                        vals[field].add(str(d[field]).strip())
    ages = set()
    sexes = set()
    for a in vals['年龄']:
        m = re.fullmatch('(\\d{1,3}(?:\\.\\d+)?)\\s*(?:岁)?', a)
        if m and 0 <= float(m[1]) <= 120:
            ages.add(float(m[1]))
    for s in vals['性别']:
        if s in ['男', '女']:
            sexes.add(float(s == '男'))
    return [next(iter(ages)) if len(ages) == 1 else np.nan, next(iter(sexes)) if len(sexes) == 1 else np.nan]

def main():
    labels = pd.read_csv(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'), usecols=['Center', '病案号', 'MACE'], dtype={'病案号': 'string'})
    labels = labels.loc[labels.Center.eq('WHU')].copy()
    labels['key'] = labels['病案号'].map(key)
    labmap = dict(zip(labels.key, labels.MACE.astype(int)))
    processed = pd.read_csv(_release_path('china', 'WHU_Cancer_processed_data.csv'), usecols=['病案号', '入院时间', '出院时间'], dtype='string')
    staydays = {(key(p), a[:10], d[:10]) for p, a, d in processed.itertuples(index=False, name=None) if isinstance(a, str) and isinstance(d, str)}
    paths = pd.read_csv(ROOT / 'clinical_review_2026-10-27/full_index_v1/private/whu_sources.csv', usecols=['source_path'])['source_path'].unique()
    paths = sorted((p for p in paths if Path(p).suffix.lower() == '.json'))
    protocol = _release_resolve_config(json.loads((ROOT / 'revision_2026-10-16/reviewer_priority_analyses_v1/common_transfer/protocol.json').read_text(encoding='utf-8-sig')))
    roster = {f['whu_exact_name']: (i, f) for i, f in enumerate(protocol['features'])}
    rules = protocol['whu_context_rules']
    visits = {}
    counts = Counter()
    all_strict = defaultdict(set)
    all_units = defaultdict(set)
    all_names = defaultdict(set)
    source_members = set()
    native_clocks = defaultdict(set)
    for fi, path in enumerate(paths):
        obj = _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))
        if isinstance(obj, str):
            obj = _release_resolve_config(json.loads(obj))
        for visit in records(obj):
            pid, fields = link(visit)
            if pid not in labmap:
                continue
            counts['source_records'] += 1
            source_members.add(pid)
            adm = unique_clock(fields['入院日期'])
            dis = unique_clock(fields['出院日期'])
            native = str(visit.get('就诊ID', '')).strip()
            eligible = bool(adm and dis and (adm < dis) and native and ((pid, adm.date().isoformat(), dis.date().isoformat()) in staydays))
            if eligible:
                vk = (pid, native, adm, dis)
                native_clocks[pid, native].add((adm, dis))
                if vk not in visits:
                    visits[vk] = {'demo': [], 'labs': set()}
                visits[vk]['demo'].append(demographics(visit))
                counts['source_records_with_matched_stay_clock'] += 1
            else:
                counts['source_records_without_matched_precise_stay_clock'] += 1
            for report in records(visit.get('检验列表')):
                context = ' '.join((str(report.get(f, '')) for f in ['检验名称', '检验类型']))
                reject = bool(re.search(rules['reject_regex'], context))
                outer = clock(report.get('报告时间'))
                rid = str(report.get('检验ID', ''))
                for detail in records(report.get('检验详情')):
                    inner = clock(detail.get('报告时间'))
                    agree = bool(outer and (inner is None or inner.date() == outer.date()))
                    for item in records(detail.get('报告子项')):
                        name = str(item.get('项目名称', '')).strip()
                        if name not in roster:
                            continue
                        j, spec = roster[name]
                        value = numeric(item.get('检查结果'))
                        if value is None:
                            continue
                        unit = str(item.get('单位', '')).strip()
                        iid = str(item.get('项目子项id', ''))
                        ident = (native, rid, iid, str(outer), j, unit, value)
                        all_names[pid, j].add(ident)
                        if unit == spec['whu_unit']:
                            all_units[pid, j].add(ident)
                        valid = unit == spec['whu_unit'] and (not reject) and bool(re.search(rules[spec['whu_context'] + '_accept_regex'], context))
                        if not valid:
                            counts['candidate_items_rejected_unit_or_context'] += 1
                            continue
                        counts['accepted_items_before_deduplication'] += 1
                        all_strict[pid, j].add(ident)
                        if eligible and outer:
                            visits[vk]['labs'].add((rid, iid, j, value, outer, agree))
        if (fi + 1) % 500 == 0:
            print(json.dumps({'phase': 'WHU preparation', 'files': fi + 1}), flush=True)
    candidates = defaultdict(list)
    for vk, v in visits.items():
        pid, native, adm, dis = vk
        if len(native_clocks[pid, native]) == 1:
            candidates[pid].append(vk)
        else:
            counts['excluded_encounter_clock_conflicts'] += 1
    selected = []
    for pid, vks in candidates.items():
        first = min((v[2] for v in vks))
        earliest = [v for v in vks if v[2] == first]
        if len(earliest) == 1:
            selected.append(earliest[0])
        else:
            counts['excluded_first_admission_ties'] += 1
    selected.sort(key=lambda v: v[0])
    n = len(selected)
    matrices = {name: np.full((n, 6), np.nan) for name in ['h0', 'h24', 'h48', 'whole', 'h24_outer_all', 'h48_outer_all', 'h24_date_end', 'h48_date_end']}
    coverage = []
    clockchanges = Counter()
    for i, vk in enumerate(selected):
        v = visits[vk]
        adm, dis = vk[2:]
        demo = np.asarray(v['demo'], float)
        for col in range(2):
            finite = np.unique(demo[:, col][np.isfinite(demo[:, col])])
            value = finite[0] if len(finite) == 1 else np.nan
            for X in matrices.values():
                X[i, col] = value
        for name, X in matrices.items():
            h = 0 if name == 'h0' else 24 if '24' in name else 48 if '48' in name else None
            upper = dis if h is None else min(adm + timedelta(hours=h), dis)
            values = defaultdict(list)
            for rid, iid, j, value, outer, agree in v['labs']:
                if not agree and (not name.endswith('outer_all')):
                    continue
                effective = outer.replace(hour=23, minute=59, second=59, microsecond=999999) if name.endswith('date_end') else outer
                if adm <= effective <= upper:
                    values[j].append(value)
            for j, values_j in values.items():
                X[i, j + 2] = np.median(values_j)
        clockchanges['selected_accepted_deduplicated_items'] += len(v['labs'])
        clockchanges['selected_items_with_inner_outer_date_difference'] += sum((not r[-1] for r in v['labs']))
    y = np.array([labmap[v[0]] for v in selected], int)
    for name, X in matrices.items():
        coverage.append({'scope': name, 'patients': n, 'positive_labels': int(y.sum()), 'any_lab': int(np.isfinite(X[:, 2:]).any(axis=1).sum()), 'all_four_labs': int(np.isfinite(X[:, 2:]).all(axis=1).sum()), 'age_available': int(np.isfinite(X[:, 0]).sum()), 'sex_available': int(np.isfinite(X[:, 1]).sum())})
    np.savez_compressed(PRIVATE / 'whu_windows.npz', y=y, groups=np.arange(n), **matrices)
    pd.DataFrame({'patient_key': [v[0] for v in selected], 'native_encounter': [v[1] for v in selected], 'admission': [v[2] for v in selected], 'discharge': [v[3] for v in selected]}).to_csv(PRIVATE / 'whu_selected_linkage.csv', index=False)
    members = sorted(source_members)
    wide = {}
    for name, values in [('verified', all_strict), ('unit_only', all_units), ('name_only', all_names)]:
        X = np.full((len(members), 4), np.nan)
        for i, pid in enumerate(members):
            for j in range(4):
                if values[pid, j]:
                    X[i, j] = np.median([r[-1] for r in values[pid, j]])
        wide[name] = X
    np.savez_compressed(PRIVATE / 'whu_assay_sensitivity.npz', y=np.array([labmap[p] for p in members], int), groups=np.arange(len(members)), **wide)
    affected = ~np.isclose(wide['verified'], wide['name_only'], equal_nan=True)
    unit_affected = ~np.isclose(wide['unit_only'], wide['name_only'], equal_nan=True)
    save('whu_preparation.json', {'counts': dict(counts), 'source_patients': len(members), 'selected_patients': n, 'selected_positive_labels': int(y.sum()), 'clock_counts': dict(clockchanges), 'coverage': coverage, 'assay_sensitivity': {'patients': len(members), 'positive_labels': int(sum((labmap[p] for p in members))), 'patients_with_any_changed_feature': int(affected.any(axis=1).sum()), 'changed_cells_by_feature': affected.sum(axis=0).tolist(), 'unit_only_changed_patients': int(unit_affected.any(axis=1).sum()), 'unit_only_changed_cells_by_feature': unit_affected.sum(axis=0).tolist()}, 'source_paths': [_release_path('njm', 'WUH'), _release_path('china', 'WHU_Cancer_processed_data.csv'), _release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv')], 'clock_and_outcome': 'Caps use report availability; original patient labels retained, event timing not inferred'})
    pd.DataFrame(coverage).to_csv(AGG / 'whu_window_coverage.csv', index=False)
    print(json.dumps({'complete': 'WHU preparation', 'patients': n, 'positive_labels': int(y.sum()), 'coverage': coverage, 'mapping_affected_patients': int(affected.any(axis=1).sum())}), flush=True)
if __name__ == '__main__':
    main()
