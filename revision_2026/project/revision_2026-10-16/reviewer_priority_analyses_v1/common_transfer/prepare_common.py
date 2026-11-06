"""Extract a limited name/unit-reviewed laboratory subset without source writes."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
PREVIOUS = PROJECT / 'revision_2026-10-15/original_v5_independent_validation_v1'
RAW_WH = Path(_release_path('njm', 'WUH'))
RAW_MI = Path(_release_path('mimic', ''))
OLD_UNITS = PROJECT / 'revision_2026-10-12/njm_same_source_completion_2026-10-12/labs'
LABELS = Path(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'))
PAPER = Path(_release_path('pipeline', 'Manuscript/v13/paper_predictions_9946.csv'))
EXPECTED = {}

def sha(path):
    return None

def identity(path):
    p = Path(path)
    return {'path': p.resolve().as_posix(), 'bytes': p.stat().st_size}

def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=True, indent=2, allow_nan=False)

def rows(value):
    return [value] if isinstance(value, dict) else [x for x in value if isinstance(x, dict)] if isinstance(value, list) else []

def normalize(value):
    if not isinstance(value, (str, int, float)) or pd.isna(value):
        return None
    s = re.sub('\\.0+$', '', str(value).strip())
    return s.lstrip('0') or '0' if s.isdigit() else s or None

def patient(visit):
    keys = set()
    for block in rows(visit.get('电子病历菜单及详情')):
        for section, payload in block.items():
            if section in {'个人基本信息', '住院基本信息'}:
                for entry in rows(payload):
                    if '病案号' in entry:
                        keys.add(normalize(entry['病案号']))
    keys.discard(None)
    return next(iter(keys)) if len(keys) == 1 else None

def numeric(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip()
    if not re.fullmatch('[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?', text):
        return None
    number = float(text)
    return number if np.isfinite(number) else None

def frozen(cohort):
    with np.load(PREVIOUS / f'private/prepared/{cohort}.npz', allow_pickle=False) as z:
        y, groups = (z['y'].copy(), z['groups'].copy())
    with np.load(PREVIOUS / f'private/runs/{cohort}_original/patient_roles.npz', allow_pickle=False) as z:
        candidates = [z[k] for k in z.files if z[k].shape == (5, len(y))]
        assert len(candidates) == 1
        roles = candidates[0].copy()
    assert np.all((roles == 0).sum(axis=0) == 1)
    for role in roles:
        group_sets = [set(groups[role == r]) for r in range(4)]
        assert all((not group_sets[a].intersection(group_sets[b]) for a in range(4) for b in range(a + 1, 4)))
    return (y, groups, roles)

def aggregate(values, n, features):
    X = np.full((n, len(features)), np.nan, dtype=np.float64)
    for (i, j), observed in values.items():
        X[i, j] = np.median(observed)
    return X

def prepare_whu(protocol):
    features = protocol['features']
    y, groups, roles = frozen('WHU')
    labels = pd.read_csv(LABELS, usecols=['Center', '病案号', 'MACE'])
    labels = labels[labels.Center.eq('WHU')].reset_index(drop=True)
    assert len(labels) == 2562 and np.array_equal(labels.MACE.astype(int), y)
    keys = [normalize(v) for v in labels['病案号']]
    assert len(set(keys)) == len(keys) and None not in keys
    member = {key: i for i, key in enumerate(keys)}
    expected = _release_resolve_config(json.loads((OLD_UNITS / 'private/source_index.json').read_text(encoding='utf-8-sig')))
    files = sorted(RAW_WH.glob('*.json'))
    assert len(files) == len(expected) == 37384
    source_metadata = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    roster = {f['whu_exact_name']: (j, f) for j, f in enumerate(features)}
    values, counts = (defaultdict(list), Counter())
    per_feature = [Counter() for _ in features]
    source_members = set()
    corpus = hashlib.sha256()
    rules = protocol['whu_context_rules']
    for ordinal, (path, entry) in enumerate(zip(files, expected), 1):
        assert path.name == entry['basename'] and ordinal - 1 == entry['ordinal']
        data = path.read_bytes()
        digest = hashlib.sha256(data).digest()
        corpus.update(digest)
        counts['source_files'] += 1
        try:
            obj = _release_resolve_config(json.loads(data.decode('utf-8-sig')))
            if isinstance(obj, str):
                obj = _release_resolve_config(json.loads(obj))
        except (ValueError, UnicodeError):
            counts['unparseable_files'] += 1
            continue
        for visit in rows(obj):
            key = patient(visit)
            if key not in member:
                continue
            i = member[key]
            source_members.add(i)
            counts['member_source_records'] += 1
            for report in rows(visit.get('检验列表')):
                context = ' '.join((str(report.get(field, '')) for field in ('检验名称', '检验类型')))
                rejected = bool(re.search(rules['reject_regex'], context))
                for detail in rows(report.get('检验详情')):
                    for item in rows(detail.get('报告子项')):
                        found = roster.get(item.get('项目名称'))
                        if found is None:
                            continue
                        j, spec = found
                        c = per_feature[j]
                        c['exact_name_occurrences'] += 1
                        if str(item.get('单位', '')).strip() != spec['whu_unit']:
                            c['rejected_unit'] += 1
                            continue
                        if rejected or not re.search(rules[spec['whu_context'] + '_accept_regex'], context):
                            c['rejected_or_unknown_report_context'] += 1
                            continue
                        value = numeric(item.get('检查结果'))
                        if value is None:
                            c['rejected_nonnumeric_or_nonfinite'] += 1
                            continue
                        values[i, j].append(value)
                        c['accepted_numeric_occurrences'] += 1
        if ordinal % 5000 == 0:
            print(json.dumps({'phase': 'WHU_source_scan', 'files_completed': ordinal}), flush=True)
    X = aggregate(values, len(y), features)
    assert len(source_members) == 2258
    assert source_metadata == [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return (X, y, groups, roles, {'counts': dict(counts), 'source_member_patients': len(source_members), 'every_raw_file_matches_frozen_index': True, 'raw_names_sizes_mtimes_unchanged_after_scan': True, 'per_feature': [{'feature': f['name'], **dict(c)} for f, c in zip(features, per_feature)]})

def prepare_mimic(protocol):
    features = protocol['features']
    y, groups, roles = frozen('MIMIC')
    paper = pd.read_csv(PAPER, usecols=['HADM_ID', 'SUBJECT_ID', 'MACE'])
    assert len(paper) == 9946 and paper.HADM_ID.is_unique and np.array_equal(paper.MACE, y)
    codes, _ = pd.factorize(paper.SUBJECT_ID, sort=True)
    assert np.array_equal(codes, groups)
    member = {int(h): i for i, h in enumerate(paper.HADM_ID)}
    dictionary = pd.read_csv(RAW_MI / 'D_LABITEMS.csv.gz')
    roster = {}
    for j, spec in enumerate(features):
        item = dictionary[dictionary.ITEMID.eq(spec['mimic_itemid'])]
        assert len(item) == 1 and item.iloc[0].LABEL == spec['mimic_label']
        assert item.iloc[0].FLUID == 'Blood' and item.iloc[0].CATEGORY == spec['mimic_category']
        roster[spec['mimic_itemid']] = (j, spec)
    values, per_feature, counts = (defaultdict(list), [Counter() for _ in features], Counter())
    cols = ['HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM', 'VALUEUOM']
    for chunk in pd.read_csv(RAW_MI / 'LABEVENTS.csv.gz', usecols=cols, chunksize=500000, low_memory=False):
        counts['raw_records_scanned'] += len(chunk)
        subset = chunk[chunk.ITEMID.isin(roster) & chunk.HADM_ID.isin(member)]
        for hadm, itemid, charttime, valuenum, unit in subset[cols].itertuples(index=False, name=None):
            j, spec = roster[int(itemid)]
            c = per_feature[j]
            c['exact_itemid_cohort_occurrences'] += 1
            if not isinstance(unit, str) or unit.strip() != spec['mimic_unit']:
                c['rejected_unit'] += 1
                continue
            value = numeric(valuenum)
            if value is None:
                c['rejected_nonnumeric_or_nonfinite'] += 1
                continue
            values[member[int(hadm)], j].append(value * spec['mimic_factor_to_g_per_l'])
            c['accepted_numeric_occurrences'] += 1
            c['charttime_missing'] += int(pd.isna(charttime))
        if counts['raw_records_scanned'] % 5000000 == 0:
            print(json.dumps({'phase': 'MIMIC_source_scan', 'records_completed': counts['raw_records_scanned']}), flush=True)
    assert counts['raw_records_scanned'] == 27854055
    X = aggregate(values, len(y), features)
    return (X, y, groups, roles, {'counts': dict(counts), 'per_feature': [{'feature': f['name'], **dict(c)} for f, c in zip(features, per_feature)]})

def main():
    protocol = _release_resolve_config(json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8-sig')))
    dest = ROOT / 'private'
    assert not dest.exists() or not any(dest.iterdir()), 'Refusing to overwrite prepared matrices'
    for folder in (dest, ROOT / 'aggregate'):
        folder.mkdir(parents=True, exist_ok=True)
    paths = list(EXPECTED) + [ROOT / 'protocol.json', Path(__file__), RAW_MI / 'D_LABITEMS.csv.gz', RAW_MI / 'LABEVENTS.csv.gz', PROJECT / 'revision_2026-10-12/analysis/a2_sch_archive_unit_headers_v5.json', PROJECT / 'revision_2026-10-12/analysis/a2_sch_header_unit_scope_v1.json']
    before = [identity(p) for p in paths]
    for item in before:
        if Path(item['path']) in EXPECTED:
            pass
    report, availability = ({}, [])
    for centre, function in (('WHU', prepare_whu), ('MIMIC', prepare_mimic)):
        X, y, groups, roles, detail = function(protocol)
        assert np.all(np.isfinite(X).any(axis=0)), 'Prespecified feature has no accepted observation; stop before any fitting'
        with (dest / f'{centre}.npz').open('xb') as f:
            np.savez_compressed(f, X=X, y=y, groups=groups, roles=roles, feature_names=np.asarray([s['name'] for s in protocol['features']]))
        detail.update({'observations': len(y), 'patients': len(np.unique(groups)), 'positive_labels': int(y.sum()), 'all_four_missing': int(np.isnan(X).all(axis=1).sum()), 'matrix': identity(dest / f'{centre}.npz'), 'original_label_and_group_order_exact': True})
        report[centre] = detail
        for j, spec in enumerate(protocol['features']):
            finite = np.isfinite(X[:, j])
            availability.append({'centre': centre, 'feature': spec['name'], 'unit': spec['output_unit'], 'observations': len(y), 'measured_observations': int(finite.sum()), 'measured_patients': int(np.unique(groups[finite]).size), 'missing_fraction': float((~finite).mean())})
    after = [identity(p) for p in paths]
    assert before == after, 'Input changed during source extraction'
    report.update({'execution_complete': True, 'completed_utc': datetime.now(timezone.utc).isoformat(), 'input_identities_before': before, 'input_identities_after': after, 'all_frozen_sources_unchanged': True, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}, 'feature_gate': 'Four predeclared laboratory names, report/dictionary contexts and literal physical units; methods/platform not certified equivalent; no source claim about SCH units'})
    write_json(ROOT / 'aggregate/preparation.json', report)
    with (ROOT / 'aggregate/feature_availability.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(availability[0]))
        writer.writeheader()
        writer.writerows(availability)
    print(json.dumps({'phase': 'preparation_complete', 'WHU_observations': 2562, 'MIMIC_observations': 9946, 'feature_count': 4, 'source_identities_unchanged': True}), flush=True)
if __name__ == '__main__':
    main()
