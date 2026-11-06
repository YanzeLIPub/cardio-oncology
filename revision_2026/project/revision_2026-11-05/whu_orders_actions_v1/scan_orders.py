"""Inspect WHU source orders and action timestamps; export aggregates only."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import json
import platform
import re
import sys
import time
import pandas as pd
ROOT = Path(_release_path('project', ''))
RAW = Path(_release_path('njm', 'WUH'))
OUT = Path(__file__).resolve().parent
MEMBERS = set()
STAY_DAYS = set()
DRUGS = {'antiplatelet_names': '阿司匹林|氯吡格雷|替格瑞洛', 'anticoagulant_names': '肝素|依诺肝素|利伐沙班|阿哌沙班|华法林|达比加群|磺达肝癸', 'statin_names': '阿托伐他汀|瑞舒伐他汀|辛伐他汀|普伐他汀|匹伐他汀|氟伐他汀', 'beta_blocker_names': '美托洛尔|比索洛尔|卡维地洛|艾司洛尔|普萘洛尔', 'renin_angiotensin_drug_names': '卡托普利|依那普利|培哚普利|贝那普利|雷米普利|赖诺普利|缬沙坦|厄贝沙坦|氯沙坦|坎地沙坦|替米沙坦|奥美沙坦|沙库巴曲', 'diuretic_names': '呋塞米|托拉塞米|螺内酯|氢氯噻嗪|布美他尼', 'nitrate_names': '硝酸甘油|硝酸异山梨酯|单硝酸异山梨酯', 'platinum_drug_names': '顺铂|卡铂|奥沙利铂|奈达铂|洛铂', 'taxane_drug_names': '紫杉醇|多西他赛', 'anthracycline_names': '多柔比星|表柔比星|吡柔比星|柔红霉素|阿霉素', 'other_cytotoxic_names': '环磷酰胺|异环磷酰胺|氟尿嘧啶|卡培他滨|吉西他滨|培美曲塞', 'her2_drug_names': '曲妥珠单抗|帕妥珠单抗', 'checkpoint_inhibitor_names': '帕博利珠单抗|纳武利尤单抗|信迪利单抗|卡瑞利珠单抗|特瑞普利单抗|替雷利珠单抗|度伐利尤单抗|阿替利珠单抗'}
ACTIONS = {'ecg_order': '心电图', 'cardiac_monitoring_order': '心电监护', 'echocardiography_order': '超声心动|心脏彩超|心脏彩色', 'consultation_order': '会诊', 'cardiology_order_text': '心内科|心血管内科', 'oxygen_order': '吸氧|氧气吸入', 'radiotherapy_order_text': '放射治疗|放疗'}
PATTERNS = {k: re.compile(v) for k, v in {**DRUGS, **ACTIONS}.items()}
ACTION_TIME = re.compile('(?:医嘱|开单|处方|给药|服药|用药|执行|手术|操作).*(?:时间|日期)|(?:开始|结束|停止|撤销|取消)(?:时间|日期)')
NONVALUES = {'', '-', '--', '/', 'null', 'none', 'nan', '未填', '未填写'}

def records(value):
    if isinstance(value, str) and value.lstrip().startswith(('[', '{')):
        try:
            value = _release_resolve_config(json.loads(value))
        except ValueError:
            return []
    if isinstance(value, dict):
        return [value]
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []

def key(value):
    if value is None:
        return ''
    s = re.sub('\\.0+$', '', str(value).strip())
    return s.lstrip('0') or '0' if s.isdigit() else s

def filled(value):
    if value is None or isinstance(value, (dict, list)):
        return False
    return str(value).strip().lower() not in NONVALUES

def clock(value):
    if not filled(value):
        return None
    s = re.sub('(\\d{4})年(\\d{1,2})月(\\d{1,2})日', '\\1-\\2-\\3', str(value).strip()).replace('/', '-')
    match = re.fullmatch('(\\d{4})-(\\d{1,2})-(\\d{1,2})(?:[ T](\\d{1,2}):(\\d{2})(?::(\\d{2})(?:\\.(\\d+))?)?)?', s)
    if not match:
        return None
    parts = match.groups()
    try:
        return datetime(*[int(x or 0) for x in parts[:6]], microsecond=int((parts[6] or '').ljust(6, '0')[:6] or 0))
    except ValueError:
        return None

def precise(value):
    return clock(value) if ':' in str(value) else None

def link(visit):
    values = defaultdict(set)
    for block in records(visit.get('电子病历菜单及详情')):
        for section, payload in block.items():
            if section in ['个人基本信息', '住院基本信息']:
                for row in records(payload):
                    for k in ['病案号', '入院日期', '出院日期']:
                        if filled(row.get(k)):
                            values[k].add(str(row[k]))
    ids = {key(v) for v in values['病案号']}
    pid = next(iter(ids)) if len(ids) == 1 else ''
    clocks = []
    for k in ['入院日期', '出院日期']:
        vals = [clock(v) for v in values[k]]
        exact = {precise(v) for v in values[k] if precise(v) is not None}
        chosen = next(iter(exact)) if len(exact) == 1 else None
        clocks.append(chosen if chosen and vals and all((v is not None and v.date() == chosen.date() for v in vals)) else None)
    return (pid, *clocks)

def initialize(members, staydays):
    global MEMBERS, STAY_DAYS
    MEMBERS = members
    STAY_DAYS = staydays

def walk_fields(obj):
    stack = [(obj, 'record')]
    while stack:
        node, context = stack.pop()
        if isinstance(node, dict):
            yield (node, context)
            for k, v in node.items():
                if k == '医嘱列表':
                    continue
                if isinstance(v, (dict, list)):
                    stack.append((v, k))
                elif isinstance(v, str) and v.lstrip().startswith(('[', '{')):
                    try:
                        stack.append((_release_resolve_config(json.loads(v)), k))
                    except ValueError:
                        pass
        elif isinstance(node, list):
            if node and all((isinstance(v, dict) and len(v) == 1 for v in node)):
                combined = {k: v for row in node for k, v in row.items()}
                yield (combined, '__combined_form__')
            stack.extend(((v, context) for v in node if isinstance(v, (dict, list))))

def inspect_file(path):
    c = Counter()
    fields = defaultdict(Counter)
    people = defaultdict(set)
    visits = defaultdict(set)
    timefields = defaultdict(Counter)
    operation_pairs = Counter()
    top = Counter()
    groups = defaultdict(Counter)
    try:
        obj = _release_resolve_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))
        if isinstance(obj, str):
            obj = _release_resolve_config(json.loads(obj))
    except (ValueError, UnicodeError):
        return {'counts': {'unparsed_files': 1}}
    c['parsed_files'] = 1
    file_cohort = False
    for visit in records(obj):
        c['all_visit_occurrences'] += 1
        top.update(visit.keys())
        pid, adm, dis = link(visit)
        cohort = pid in MEMBERS
        scopes = ['all'] + (['cohort'] if cohort else [])
        native = key(visit.get('就诊ID'))
        stay = key(visit.get('住院流水号'))
        vk = (pid, native) if pid and native else None
        matched = bool(cohort and adm and dis and (adm < dis) and ((pid, adm.date().isoformat(), dis.date().isoformat()) in STAY_DAYS))
        if cohort:
            file_cohort = True
            c['cohort_visit_occurrences'] += 1
            people['source'].add(pid)
            if vk:
                visits['source'].add(vk)
            if matched:
                people['matched_stay_clock'].add(pid)
                c['matched_stay_clock_visit_occurrences'] += 1
        raw_orders = visit.get('医嘱列表')
        orders = records(raw_orders)
        for scope in scopes:
            c[scope + '_visits_with_order_list_field'] += int('医嘱列表' in visit)
            c[scope + '_visits_with_nonempty_orders'] += bool(orders)
            c[scope + '_order_row_occurrences'] += len(orders)
        if cohort and orders:
            people['any_order'].add(pid)
            if vk:
                visits['any_order'].add(vk)
        for order in orders:
            start = clock(order.get('开单时间'))
            end = clock(order.get('结束时间'))
            name = str(order.get('医嘱名称', ''))
            ownstay = key(order.get('住院流水号'))
            for scope in scopes:
                fields[scope + '_present'].update(order.keys())
                fields[scope + '_nonempty'].update((k for k, v in order.items() if filled(v)))
                c[scope + '_orders_with_name'] += filled(order.get('医嘱名称'))
                c[scope + '_orders_with_start_date'] += start is not None
                c[scope + '_orders_with_start_clock'] += start is not None and ':' in str(order.get('开单时间'))
                c[scope + '_orders_with_end_date'] += end is not None
                c[scope + '_orders_with_end_clock'] += end is not None and ':' in str(order.get('结束时间'))
                c[scope + '_orders_with_start_and_end'] += bool(start and end)
                if start and end:
                    c[scope + '_end_before_start'] += end < start
                    c[scope + '_end_equals_start'] += end == start
                if ownstay and stay:
                    c[scope + '_order_visit_stay_id_compared'] += 1
                    c[scope + '_order_visit_stay_id_equal'] += ownstay == stay
                if ownstay and native:
                    c[scope + '_order_native_encounter_id_compared'] += 1
                    c[scope + '_order_native_encounter_id_equal'] += ownstay == native
                c[scope + '_order_stay_id_zero'] += ownstay == '0'
                c[scope + '_outer_stay_id_zero'] += stay == '0'
            hit = [group for group, pat in PATTERNS.items() if pat.search(name)]
            if cohort:
                if start:
                    people['any_order_start_time'].add(pid)
                if start and end:
                    people['any_order_start_and_end'].add(pid)
                drughit = any((group in DRUGS for group in hit))
                if drughit:
                    people['listed_drug_name'].add(pid)
                    c['cohort_listed_drug_name_orders'] += 1
                    if start:
                        people['listed_drug_start_time'].add(pid)
                        c['cohort_listed_drug_start_time_orders'] += 1
                    if start and end:
                        people['listed_drug_both_times'].add(pid)
                for group in hit:
                    groups[group]['source_order_occurrences'] += 1
                    people['group:' + group].add(pid)
                    groups[group]['with_start_clock'] += bool(start and ':' in str(order.get('开单时间')))
                    groups[group]['with_end_clock'] += bool(end and ':' in str(order.get('结束时间')))
                    if start and end:
                        groups[group]['end_before_start'] += end < start
                if matched and start:
                    c['cohort_orders_in_matched_clock_visits'] += 1
                    if start < adm:
                        c['cohort_orders_before_linked_admission'] += 1
                    elif start > dis:
                        c['cohort_orders_after_linked_discharge'] += 1
                    else:
                        c['cohort_orders_within_linked_stay'] += 1
                        people['order_within_matched_stay'].add(pid)
                        if ownstay and ownstay == stay:
                            c['cohort_orders_within_stay_and_same_stay_id'] += 1
                            people['order_within_stay_and_same_stay_id'].add(pid)
                            if drughit:
                                c['cohort_listed_drug_within_stay_and_same_stay_id'] += 1
                                people['listed_drug_within_stay_and_same_stay_id'].add(pid)
                        for h in [0, 24, 48]:
                            if start <= adm + timedelta(hours=h):
                                c[f'cohort_orders_within_{h}h'] += 1
                                people[f'order_within_{h}h'].add(pid)
                                if drughit:
                                    c[f'cohort_listed_drug_orders_within_{h}h'] += 1
                                    people[f'listed_drug_within_{h}h'].add(pid)
                                for group in hit:
                                    groups[group][f'within_{h}h_occurrences'] += 1
                                    people[f'group:{group}:h{h}'].add(pid)
            for scope in scopes:
                freq = str(order.get('频次', '')).strip()
                c[scope + '_frequency_numeric_code'] += bool(re.fullmatch('\\d+(?:\\.\\d+)?', freq))
                dose = str(order.get('单次剂量（单位）', '')).strip()
                c[scope + '_dose_numeric_only'] += bool(re.fullmatch('[+-]?\\d+(?:\\.\\d+)?', dose))
                c[scope + '_dose_numeric_and_unit_text'] += bool(re.match('^[+-]?\\d+(?:\\.\\d+)?\\s*[^\\d.\\s]', dose))
        if cohort:
            for node, context in walk_fields(visit.get('电子病历菜单及详情')):
                for k, v in [] if context == '__combined_form__' else node.items():
                    if len(k) < 45 and ACTION_TIME.search(k):
                        rec = timefields[k]
                        rec['field_occurrences'] += 1
                        rec['nonempty'] += filled(v)
                        dt = clock(v)
                        if dt:
                            rec['parseable_date_occurrences'] += 1
                            rec['precise_clock_occurrences'] += ':' in str(v)
                            people['timefield:' + k].add(pid)
                        elif filled(v) and re.fullmatch('\\d{1,2}:\\d{2}(?::\\d{2})?', str(v).strip()):
                            rec['clock_only_occurrences'] += 1
                            people['timefield_clock_only:' + k].add(pid)
                    if k in ['出院医嘱', '入院前用药史', '既往用药史', '治疗经过', '治疗意见', '出院带药(口服)'] and filled(v):
                        c['cohort_narrative_field:' + k] += 1
                        people['narrative:' + k].add(pid)
                for k, v in node.items():
                    if not re.fullmatch('(?:手术(?:&操作|及操作)?|操作)(?:开始|结束)?(?:日期时间|日期|时间)\\d*', k):
                        continue
                    dt = clock(v)
                    if not dt:
                        continue
                    suffix = re.search('(\\d+)$', k)
                    suffix = suffix.group(1) if suffix else ''
                    candidates = [a for a in ['手术名称', '操作名称', '手术及操作名称', '手术&操作名称'] if filled(node.get(a + suffix))]
                    if candidates:
                        operation_pairs[k] += 1
                        people['operation_name_time_pair'].add(pid)
                surgery_day = clock(node.get('手术日期'))
                if surgery_day:
                    for field in ['手术开始时间', '手术结束时间']:
                        value = str(node.get(field, '')).strip()
                        if re.fullmatch('\\d{1,2}:\\d{2}(?::\\d{2})?', value):
                            operation_pairs['same_form_date_plus_' + field] += 1
                            people['same_form_date_plus_' + field].add(pid)
    c['cohort_linked_files'] = int(file_cohort)
    return {'counts': dict(c), 'fields': {k: dict(v) for k, v in fields.items()}, 'people': {k: list(v) for k, v in people.items()}, 'visits': {k: list(v) for k, v in visits.items()}, 'timefields': {k: dict(v) for k, v in timefields.items()}, 'operation_pairs': dict(operation_pairs), 'top': dict(top), 'groups': {k: dict(v) for k, v in groups.items()}}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    started = time.monotonic()
    cohort = pd.read_csv(_release_path('china', 'WHU_Cancer_processed_data.csv'), usecols=['病案号', '入院时间', '出院时间'], dtype='string')
    members = set(cohort['病案号'].dropna().map(key))
    ad = pd.to_datetime(cohort['入院时间'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')
    di = pd.to_datetime(cohort['出院时间'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')
    staydays = set(zip(cohort['病案号'].map(key), ad, di))
    cohort_only = '--cohort-only' in sys.argv
    if cohort_only:
        index = pd.read_csv(ROOT / 'clinical_review_2026-10-27/full_index_v1/private/whu_sources.csv', usecols=['source_path'])
        paths = sorted({Path(v) for v in index.source_path.dropna()})
    else:
        paths = sorted(RAW.rglob('*.json'))
    counts = Counter()
    fields = defaultdict(Counter)
    people = defaultdict(set)
    visits = defaultdict(set)
    timefields = defaultdict(Counter)
    pairs = Counter()
    top = Counter()
    groups = defaultdict(Counter)
    print(json.dumps({'phase': 'cohort time recheck' if cohort_only else 'start', 'json_files': len(paths), 'original_patients': len(members), 'workers': 4}), flush=True)
    with ProcessPoolExecutor(max_workers=4, initializer=initialize, initargs=(members, staydays)) as pool:
        for i, result in enumerate(pool.map(inspect_file, paths, chunksize=12), 1):
            counts.update(result['counts'])
            for keyname, target in [('fields', fields), ('timefields', timefields), ('groups', groups)]:
                for k, v in result.get(keyname, {}).items():
                    target[k].update(v)
            for k, v in result.get('people', {}).items():
                people[k].update(v)
            for k, v in result.get('visits', {}).items():
                visits[k].update((tuple(x) for x in v))
            pairs.update(result.get('operation_pairs', {}))
            top.update(result.get('top', {}))
            if i % 2000 == 0:
                progress = {'phase': 'scan', 'files_read': i, 'total_files': len(paths), 'cohort_patients_with_orders': len(people['any_order']), 'elapsed_seconds': round(time.monotonic() - started, 1)}
                (OUT / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps(progress), flush=True)
    report = {'date': '2026-11-05', 'source_directory': RAW.as_posix(), 'files_enumerated': len(paths), 'original_cohort_patients': len(members), 'counts': dict(counts), 'cohort_patient_coverage': {k: len(v) for k, v in people.items()}, 'cohort_distinct_native_visit_coverage': {k: len(v) for k, v in visits.items()}, 'order_field_coverage': {k: dict(v) for k, v in fields.items()}, 'top_level_schema': dict(top), 'cohort_action_time_fields': {k: dict(v) for k, v in timefields.items()}, 'cohort_operation_name_time_pairs': dict(pairs), 'cohort_predefined_name_matches': {k: dict(v) for k, v in groups.items()}, 'name_match_dictionary': {**DRUGS, **ACTIONS}, 'python': platform.python_version(), 'pandas': pd.__version__, 'elapsed_seconds': round(time.monotonic() - started, 1), 'counting': 'Rows are source occurrences, not deduplicated prescriptions or administrations. Patient coverage is distinct by original cohort key. Drug groups are predefined name matches, not a comprehensive medication terminology mapping.', 'timing': "Chinese and ISO date formats are parsed. Open-order and end-order fields are evaluated as named; no assumption that they equal first administration, clinical availability, or MACE onset. Relative windows concern any matched stay, not each patient's earliest stay, and require precise source stay clocks matching processed admission/discharge days. Native order/stay identifier agreement is reported separately.", 'scope': 'All JSON files: top-level order inventory and times. Original cohort: linked stay windows plus nested clinical action/operation time fields. No clinical narrative adjudication, model retraining or patient rows exported.'}
    output_name = 'orders_cohort_time_recheck.json' if cohort_only else 'orders_audit.json'
    (OUT / output_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'phase': 'complete', 'files': len(paths), 'cohort_patients_with_orders': len(people['any_order']), 'cohort_order_rows': counts['cohort_order_row_occurrences'], 'seconds': report['elapsed_seconds']}), flush=True)
if __name__ == '__main__':
    main()
