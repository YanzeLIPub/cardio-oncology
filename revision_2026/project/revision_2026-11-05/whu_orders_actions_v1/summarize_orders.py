"""Summarize the completed source inventory without individual clinical rows."""
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import json
HERE = Path(__file__).resolve().parent
o = _release_resolve_config(json.loads((HERE / 'orders_audit.json').read_text(encoding='utf-8')))
c = o['counts']
p = o['cohort_patient_coverage']
g = o['cohort_predefined_name_matches']
labels = {'antiplatelet_names': '阿司匹林/氯吡格雷/替格瑞洛名称', 'anticoagulant_names': '肝素、口服抗凝药等名称', 'statin_names': '他汀名称', 'beta_blocker_names': '美托洛尔等名称', 'renin_angiotensin_drug_names': '普利/沙坦/沙库巴曲等名称', 'diuretic_names': '呋塞米/托拉塞米/螺内酯等名称', 'nitrate_names': '硝酸酯名称', 'platinum_drug_names': '铂类药名', 'taxane_drug_names': '紫杉醇/多西他赛名称', 'anthracycline_names': '蒽环类药名', 'other_cytotoxic_names': '预设其他细胞毒药名', 'her2_drug_names': '曲妥珠单抗/帕妥珠单抗名称', 'checkpoint_inhibitor_names': '预设免疫检查点抑制剂药名', 'ecg_order': '心电图', 'cardiac_monitoring_order': '心电监护', 'echocardiography_order': '超声心动图/心脏彩超', 'consultation_order': '会诊', 'cardiology_order_text': '心内科/心血管内科文字', 'oxygen_order': '吸氧', 'radiotherapy_order_text': '放疗文字'}
lines = ['WHU原始JSON中的用药、医嘱和时间信息', '2026年11月5日', '', '结论：存在明确的结构化临床医嘱和相应时间，也存在手术/操作名称与时间字段。原始资料不能再概括为“没有临床行动”。', '上一轮“没有实际实施某种检查、转诊或治疗”的措辞不准确：本批资料记录了常规诊疗医嘱。尚未验证的是模型评分触发的行动及其效果，不能把这与病历是否记录临床行动混为一谈。', '', '读取范围', f"{_release_path('njm', 'WUH 共遍历')}{o['files_enumerated']:,}个JSON文件；成功解析{c['parsed_files']:,}个，未解析{c.get('unparsed_files', 0):,}个。", f"成功解析文件含{c['all_visit_occurrences']:,}条源就诊记录，{c['all_order_row_occurrences']:,}条医嘱源记录。", f"论文WHU队列为{o['original_cohort_patients']:,}人；本次关联{p.get('source', 0):,}人、{c.get('cohort_visit_occurrences', 0):,}条源就诊记录。", '', '论文队列的覆盖', f"有非空医嘱列表：{p.get('any_order', 0):,}人；医嘱源记录{c.get('cohort_order_row_occurrences', 0):,}条。", f"有至少一条可解析开单时间：{p.get('any_order_start_time', 0):,}人；相应医嘱{c.get('cohort_orders_with_start_date', 0):,}条。", f"有至少一条同时具有开始和结束时间的医嘱：{p.get('any_order_start_and_end', 0):,}人。", f"命中本轮预设具体药名：{p.get('listed_drug_name', 0):,}人、{c.get('cohort_listed_drug_name_orders', 0):,}条医嘱；其中有开单时间{c.get('cohort_listed_drug_start_time_orders', 0):,}条。", f"手术/操作名称与可解析日期在同一表单或记录对象中共同出现：{p.get('operation_name_time_pair', 0):,}人。", '', '结构化医嘱字段（论文队列中实际非空源记录数）']
for name, n in o['order_field_coverage'].get('cohort_nonempty', {}).items():
    lines.append(f'{name}：{n:,}')
lines += ['', '开单及结束时间', f"开单时间精确至时分秒/时分：{c.get('cohort_orders_with_start_clock', 0):,}条；结束时间具有时分信息：{c.get('cohort_orders_with_end_clock', 0):,}条。", f"同时有开始和结束时间：{c.get('cohort_orders_with_start_and_end', 0):,}条；结束早于开单：{c.get('cohort_end_before_start', 0):,}条；两者相同：{c.get('cohort_end_equals_start', 0):,}条。", f"医嘱和外层就诊住院流水号均有值：{c.get('cohort_order_visit_stay_id_compared', 0):,}条，其中一致{c.get('cohort_order_visit_stay_id_equal', 0):,}条。", '内外层流水号没有全部一致，应核对字段关联规则后用于住院级分析；不能把全部源医嘱直接当成已验证属于外层同一次住院。', '字段“开单时间”可描述医嘱开立；“结束时间”可描述该字段记录的终止。医嘱执行记录ID是编号，不能独自证明首次实际给药时刻；本轮不把开单与给药等同。', '', '与已匹配住院时间的关系', f"源入院/出院精确时间与处理表住院日期相符的患者：{p.get('matched_stay_clock', 0):,}人；其中有住院内开单记录{p.get('order_within_matched_stay', 0):,}人。", f"在这些源就诊记录内，有开单时间的医嘱共{c.get('cohort_orders_in_matched_clock_visits', 0):,}条；开单位于住院期间{c.get('cohort_orders_within_linked_stay', 0):,}条，早于该住院{c.get('cohort_orders_before_linked_admission', 0):,}条，晚于该住院{c.get('cohort_orders_after_linked_discharge', 0):,}条。"]
for h in [0, 24, 48]:
    lines.append(f"入院至{h}小时内开单：{p.get(f'order_within_{h}h', 0):,}人、{c.get(f'cohort_orders_within_{h}h', 0):,}条；其中预设药名命中{p.get(f'listed_drug_within_{h}h', 0):,}人、{c.get(f'cohort_listed_drug_orders_within_{h}h', 0):,}条。")
lines += [f"同时要求医嘱与外层住院流水号一致、开单位于该住院日期内：{p.get('order_within_stay_and_same_stay_id', 0):,}人、{c.get('cohort_orders_within_stay_and_same_stay_id', 0):,}条；其中预设药名命中{p.get('listed_drug_within_stay_and_same_stay_id', 0):,}人。", '窗口覆盖按患者任一已匹配住院统计，并未筛选每人首次住院。以上为来源记录层面的时间覆盖，不是新的预测模型或事件发生时间。医嘱条数未按医嘱/执行编号去重；患者数按论文队列病案键去重。', '', '已找到的具体药物及行动名称组（组间允许重叠）', '名称组｜患者数｜医嘱源记录数｜有开单时刻的记录数']
for k, label in labels.items():
    if k in g:
        lines.append(f"{label}｜{p.get('group:' + k, 0):,}｜{g[k]['source_order_occurrences']:,}｜{g[k].get('with_start_clock', 0):,}")
lines += ['这些是预设药名/行动文字的字面匹配，不覆盖全部药品，不能把未命中当作未用药。检查/监护/会诊属于医嘱内容，因此全部医嘱条数不等于药物处方数。', '', '医嘱列表之外的手术/操作时间字段（只列实际可解析的字段）', '字段｜有效日期源记录数｜其中含时分｜患者数']
for k, stats in sorted(o['cohort_action_time_fields'].items()):
    if stats.get('parseable_date_occurrences', 0):
        lines.append(f"{k}｜{stats['parseable_date_occurrences']:,}｜{stats.get('precise_clock_occurrences', 0):,}｜{p.get('timefield:' + k, 0):,}")
lines += [f"另有{p.get('same_form_date_plus_手术开始时间', 0):,}名患者的同一表单同时记录手术日期与单独的开始时分；{p.get('same_form_date_plus_手术结束时间', 0):,}名患者同时有日期与结束时分。日期和时钟分列，不能因为单个字段不是完整日期时间而漏计。", '手术名称与日期共同出现可支持进一步整理手术/操作暴露；具体计划、实施和完成状态需按相应文书含义区分。本轮没有将所有日期字段视为已完成操作。', '', '对返修的意义', '1. 可以整理入院前、入院后0/24/48小时的已开立医嘱及药物暴露；需要先统一药名、频次和剂量，按原医嘱编号处理重复记录。', '2. 可以补充用药和治疗因素的覆盖情况，并在合适的分析中评估这些因素与模型表现的关系；原104项共同输入和WHU处理表没有医嘱/药物字段，不能说原模型已经调整这些用药因素。', '3. 可以把化验报告时间与医嘱开立时间联结，描述检查、用药和诊疗过程的先后顺序。', '4. 仍需区分常规诊疗医嘱与由模型触发的干预。已有行动记录本身不证明模型改变了决策或改善了结局；医嘱开立时间也不能替代MACE发生时间。', '', '本轮未修改原数据、原标签、模型或投稿文件。详细汇总见orders_audit.json，完整扫描入口为scan_orders.py。所有报告仅含字段名、来源路径和汇总数量。']
(HERE / 'WHU医嘱及时间信息核查.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
assert c['parsed_files'] + c.get('unparsed_files', 0) == o['files_enumerated']
assert p.get('any_order', 0) <= p['source'] <= o['original_cohort_patients']
assert c['cohort_orders_with_start_clock'] <= c['cohort_orders_with_start_date'] <= c['cohort_order_row_occurrences']
assert c['cohort_orders_within_linked_stay'] + c['cohort_orders_before_linked_admission'] + c['cohort_orders_after_linked_discharge'] == c['cohort_orders_in_matched_clock_visits']
assert c.get('cohort_orders_within_0h', 0) <= c.get('cohort_orders_within_24h', 0) <= c.get('cohort_orders_within_48h', 0) <= c['cohort_orders_within_linked_stay']
(HERE / 'verification.json').write_text(json.dumps({'aggregate_totals': 'consistent', 'patient_coverage': 'within original denominator', 'time_partitions': 'consistent', 'unit': 'source order occurrences; unique original patients for coverage', 'models_trained': 0, 'patient_rows_exported': False}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'report': 'WHU医嘱及时间信息核查.txt', 'checks': 'passed'}, ensure_ascii=False))
