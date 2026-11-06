"""Fixed-score recorded-label sensitivity; all exported data are aggregate.

No historical modules or fitted models are imported/deserialized. Raw keys are
used in memory for exact joins and never written. Original files are read-only.
"""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
PRIOR = ROOT / 'revision_2026-10-20/priority_validation_v1/label_contract'
VALIDATION = ROOT / 'revision_2026-10-15/original_v5_independent_validation_v1'
CONTRACT_PATH = PRIOR / 'label_contract.json'
PROTOCOL_PATH = OUT / 'component_protocol.json'

def digest(path):
    return None

def identity(path):
    return {'path': path.as_posix(), 'bytes': path.stat().st_size}

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def metric_setup(score, targets):
    order = np.argsort(-score, kind='stable')
    s = score[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    return (order, ends, targets[order].astype(float))

def weighted_metrics(setup, weights):
    order, ends, targets = setup
    w = weights[order, None]
    tp = np.cumsum(w * targets, axis=0)[ends]
    fp = np.cumsum(w * (1 - targets), axis=0)[ends]
    pos, neg = (tp[-1], fp[-1])
    dtp = np.diff(np.vstack((np.zeros((1, tp.shape[1])), tp)), axis=0)
    dfp = np.diff(np.vstack((np.zeros((1, tp.shape[1])), fp)), axis=0)
    previous_tp = tp - dtp
    auc = np.full(tp.shape[1], np.nan)
    ap = np.full(tp.shape[1], np.nan)
    valid = (pos > 0) & (neg > 0)
    auc[valid] = (dfp * (tp + previous_tp) / 2).sum(axis=0)[valid] / (pos * neg)[valid]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    ap[valid] = (dtp * precision).sum(axis=0)[valid] / pos[valid]
    return np.stack((auc, ap), axis=1)

def synthetic_tests():
    checks = []
    score = np.array([0.2, 0.9, 0.9, 0.1, 0.5, 0.5, 0.5, 0.8])
    y = np.array([[0, 1], [1, 0], [0, 1], [0, 0], [1, 1], [1, 0], [0, 0], [1, 1]])
    groups = np.array([0, 1, 1, 2, 3, 3, 3, 4])
    setup = metric_setup(score, y)
    rng = np.random.default_rng(121)
    for i in range(40):
        multiplicities = np.bincount(rng.integers(0, 5, 5), minlength=5)
        weights = multiplicities[groups]
        actual = weighted_metrics(setup, weights)
        expanded = np.repeat(np.arange(len(score)), weights)
        for j in range(2):
            if len(np.unique(y[expanded, j])) != 2:
                assert np.isnan(actual[j]).all()
            else:
                expected = [roc_auc_score(y[expanded, j], score[expanded]), average_precision_score(y[expanded, j], score[expanded])]
                assert np.allclose(actual[j], expected, atol=1e-12)
        checks.append({'name': f'weighted_ties_cluster_expansion_{i}', 'passed': True})
    constant = metric_setup(np.ones(4), np.array([[0], [1], [0], [1]]))
    assert np.allclose(weighted_metrics(constant, np.ones(4)), [[0.5, 0.5]])
    checks.append({'name': 'constant_score_auc_and_ap', 'passed': True})
    assert np.isnan(weighted_metrics(setup, np.zeros(8))).all()
    checks.append({'name': 'empty_weight_not_estimable', 'passed': True})
    assert 'MI' not in 'MI'.lower() and 'mi' in 'MI'.lower()
    checks.append({'name': 'original_china_case_difference_preserved', 'passed': True})
    return checks

def verify_roles(pred, role_file):
    roles = np.load(role_file, allow_pickle=False)
    assert np.array_equal(roles['groups'], pred['groups'])
    assert np.array_equal(roles['y'], pred['y'])
    r = roles['roles']
    assert r.shape == (5, len(pred['y'])) and set(np.unique(r)) == {0, 1, 2, 3}
    assert np.all((r == 0).sum(axis=0) == 1)
    assert np.array_equal(r.argmin(axis=0) + 1, pred['fold'])
    for k in range(5):
        sets = [set(pred['groups'][r[k] == j]) for j in range(4)]
        assert all((not sets[i].intersection(sets[j]) for i in range(4) for j in range(i + 1, 4)))
    return {'five_outer_folds': True, 'each_admission_test_once': True, 'patient_overlap_between_roles': 0, 'patient_groups_and_labels_exact': True, 'fold_assignments_exact': True}

def mimic_analysis(contract, paths, protocol):
    pred = np.load(paths['independent_predictions'], allow_pickle=False)
    prepared = np.load(paths['independent_prepared'], allow_pickle=False)
    paper = pd.read_csv(paths['mimic_paper_predictions'], usecols=['HADM_ID', 'SUBJECT_ID', 'MACE'])
    assert paper.HADM_ID.is_unique
    assert np.array_equal(paper.MACE.to_numpy(), pred['y'])
    groups = pd.factorize(paper.SUBJECT_ID, sort=True)[0]
    assert np.array_equal(groups, pred['groups'])
    assert np.array_equal(prepared['groups'], groups) and np.array_equal(prepared['y'], pred['y'])
    score = pred[protocol['probability_column']].astype(float)
    assert score.shape == (len(paper),) and np.isfinite(score).all() and ((0 <= score) & (score <= 1)).all()
    assert (len(paper), len(np.unique(groups)), int(pred['y'].sum())) == (9946, 6783, 4462)
    role_checks = verify_roles(pred, paths['independent_roles'])
    admissions = pd.read_csv(paths['mimic_raw_admissions'], usecols=['HADM_ID', 'SUBJECT_ID', 'HOSPITAL_EXPIRE_FLAG'])
    admissions = admissions.set_index('HADM_ID').loc[paper.HADM_ID]
    assert np.array_equal(admissions.SUBJECT_ID.to_numpy(), paper.SUBJECT_ID.to_numpy())
    assert (admissions.HOSPITAL_EXPIRE_FLAG == 0).all()
    diagnoses = pd.read_csv(paths['mimic_raw_diagnoses'], usecols=['HADM_ID', 'ICD9_CODE'], dtype={'ICD9_CODE': str})
    diagnoses = diagnoses[diagnoses.HADM_ID.isin(paper.HADM_ID)].copy()
    code = diagnoses.ICD9_CODE.fillna('').str.replace('.', '', regex=False).str.upper()
    prefixes = contract['rules']['MIMIC']['component_prefixes']
    components = {}
    for name, terms in prefixes.items():
        positive_hids = set(diagnoses.loc[code.str.startswith(tuple(terms)), 'HADM_ID'])
        components[name] = paper.HADM_ID.isin(positive_hids).to_numpy()
    union = np.logical_or.reduce(list(components.values()))
    assert np.array_equal(union, pred['y'].astype(bool))
    assert not (components['Unstable_Angina'] & ~components['MI']).any()
    cached = pd.read_csv(paths['mimic_cached_labels'], usecols=['HADM_ID', 'MACE_Types'], keep_default_na=False).set_index('HADM_ID')
    cached = cached.loc[paper.HADM_ID, 'MACE_Types']
    expected_types = [set(v.split('|')) - {'None'} for v in cached]
    actual_types = [{k for k, v in components.items() if v[i]} for i in range(len(paper))]
    assert actual_types == expected_types
    reduced = np.logical_or.reduce([v for k, v in components.items() if k != 'Cardiomyopathy'])
    target_map = {'original_union': union, **components, 'union_without_cardiomyopathy_specific_contribution': reduced}
    assert list(target_map) == protocol['analysis_targets']
    y = np.column_stack(list(target_map.values())).astype(int)
    setup = metric_setup(score, y)
    point = weighted_metrics(setup, np.ones(len(score)))
    for j in range(y.shape[1]):
        assert np.allclose(point[j], [roc_auc_score(y[:, j], score), average_precision_score(y[:, j], score)], atol=1e-12)
    b = protocol['bootstrap_replicates']
    bootstrap = np.empty((b, y.shape[1], 2))
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    for i in range(b):
        multiplicities = np.bincount(rng.integers(0, 6783, 6783), minlength=6783)
        bootstrap[i] = weighted_metrics(setup, multiplicities[groups])
    rows = []
    for j, name in enumerate(target_map):
        npos = int(y[:, j].sum())
        result = {'target': name, 'admissions': len(paper), 'patient_clusters': 6783, 'positive_admissions': npos, 'negative_admissions': len(paper) - npos, 'positive_patient_clusters': int(len(np.unique(groups[y[:, j] == 1]))), 'admission_prevalence': npos / len(paper), 'model': 'MIMIC_original_primary_raw_fixed_composite_score'}
        for m, metric in enumerate(['ROC_AUC', 'average_precision']):
            finite = np.isfinite(bootstrap[:, j, m])
            low, high = np.quantile(bootstrap[finite, j, m], [0.025, 0.975])
            result.update({metric: float(point[j, m]), metric + '_lower_95': float(low), metric + '_upper_95': float(high), metric + '_valid_bootstraps': int(finite.sum())})
        rows.append(result)
    pd.DataFrame(rows).to_csv(OUT / 'Table_F3_fixed_score_components.csv', index=False)
    bitset = np.column_stack(list(components.values())).astype(int) @ 2 ** np.arange(6)
    combinations = []
    for mask in sorted(np.unique(bitset)):
        selected = bitset == mask
        combinations.append({'component_set': '|'.join((k for j, k in enumerate(components) if mask & 2 ** j)) or 'none', 'admissions': int(selected.sum()), 'patient_clusters_with_at_least_one_such_admission': int(len(np.unique(groups[selected])))})
    assert sum((row['admissions'] for row in combinations)) == len(paper)
    pd.DataFrame(combinations).to_csv(OUT / 'mimic_nonoverlapping_component_sets.csv', index=False)
    overlap = [{'component_a': a, 'component_b': b, 'admissions_with_both': int((av & bv).sum())} for a, av in components.items() for b, bv in components.items()]
    pd.DataFrame(overlap).to_csv(OUT / 'mimic_component_overlap.csv', index=False)
    checks = {'source_rows': len(paper), 'patient_clusters': 6783, 'source_label_positives': int(union.sum()), 'raw_to_paper_row_join_exact': True, 'subject_to_cluster_factorization_exact': True, 'raw_original_components_equal_cached_components': True, 'raw_union_equals_source_OOF_labels': True, 'prepared_labels_and_groups_exact': True, 'all_source_admissions_survived': True, 'UA_subset_of_MI': True, 'roles': role_checks, 'weighted_metric_point_estimates_equal_sklearn': True, 'cardiomyopathy_only_admissions_reclassified': int((union & ~reduced).sum()), 'rows_retained_in_reduced_union': len(paper), 'exclusive_component_sets_partition_rows': True}
    save('mimic_component_results.json', {'protocol': identity(PROTOCOL_PATH), 'checks': checks, 'results': rows, 'death_group': {'status': 'not_estimable', 'reason': 'Source analysis excludes all inpatient deaths; not evidence for cardiovascular-death classification'}, 'scope': 'Fixed composite-score discrimination of recorded code groups; not component-specific risk or clinical adjudication', 'interval_scope': protocol['bootstrap_interval'], 'no_refitting': True, 'no_patient_level_output': True})
    return checks

def chinese_composition(contract, paths):
    aggregate = {}
    rows_out, pairs_out, cues_out = ([], [], [])
    patient_labels = pd.read_csv(paths['china_patient_labels'], usecols=['病案号', 'Center', 'MACE'], dtype={'病案号': str})
    cues = {'negation_or_exclusion_text': ['未见', '否认', '排除', '除外', '未发现'], 'uncertainty_text': ['疑似', '考虑', '待排', '可能'], 'history_text': ['既往', '陈旧']}
    for center, key, textcol in [('SCH', 'sch_joined_diagnostic_cache', '所有诊断'), ('WHU', 'whu_cancer_records', '主要诊断')]:
        terms = contract['rules'][center]['keywords']
        row_counts = np.zeros(len(terms), dtype=int)
        fragments, cue_fragments = ([], [])
        nrows = missing = 0
        for chunk in pd.read_csv(paths[key], usecols=['病案号', textcol], dtype={'病案号': str}, chunksize=100000, low_memory=False):
            assert chunk['病案号'].notna().all()
            text = chunk[textcol].fillna('').astype(str).str.lower()
            nrows += len(chunk)
            missing += int(chunk[textcol].isna().sum())
            matches = np.column_stack([text.str.contains(term.lower() if center == 'SCH' else term, regex=False).to_numpy() for term in terms])
            row_counts += matches.sum(axis=0)
            hit = matches.any(axis=1)
            frame = pd.DataFrame(matches, columns=terms)
            frame.index = chunk['病案号'].to_numpy()
            fragments.append(frame.groupby(level=0).max())
            cue_frame = pd.DataFrame({name: hit & np.logical_or.reduce([text.str.contains(t, regex=False).to_numpy() for t in tokens]) for name, tokens in cues.items()}, index=chunk['病案号'].to_numpy())
            cue_fragments.append(cue_frame.groupby(level=0).max())
        hits = pd.concat(fragments).groupby(level=0).max()
        context = pd.concat(cue_fragments).groupby(level=0).max()
        any_hit = hits.any(axis=1)
        expected = patient_labels.loc[patient_labels.Center == center].set_index('病案号').MACE
        assert set(hits.index) == set(expected.index) and np.array_equal(any_hit.loc[expected.index], expected.astype(bool))
        matrix = hits.to_numpy(dtype=int)
        for j, term in enumerate(terms):
            rows_out.append({'center': center, 'source_keyword': term, 'actual_match_literal_after_text_lowercase': term.lower() if center == 'SCH' else term, 'source_record_rows': nrows, 'patient_denominator': len(hits), 'record_rows_with_hit': int(row_counts[j]), 'patients_with_hit': int(matrix[:, j].sum()), 'patients_positive_exclusively_via_this_keyword': int(((matrix[:, j] == 1) & (matrix.sum(axis=1) == 1)).sum())})
            for k, other in enumerate(terms):
                pairs_out.append({'center': center, 'keyword_a': term, 'keyword_b': other, 'patients_with_both': int((matrix[:, j] * matrix[:, k]).sum())})
        for name, tokens in cues.items():
            cues_out.append({'center': center, 'literal_context_cue_family': name, 'cues': '|'.join(tokens), 'patients_with_same_record_keyword_and_cue': int(context[name].sum()), 'clinical_negation_uncertainty_or_history_adjudicated': False})
        aggregate[center] = {'source_records': nrows, 'patients': len(hits), 'positive_record_rule_patients': int(any_hit.sum()), 'source_records_missing_diagnostic_text': missing, 'reproduced_original_patient_labels': True, 'keywords': terms, 'keyword_count': len(terms), 'patients_with_one_matching_keyword': int((matrix.sum(axis=1) == 1).sum()), 'patients_with_multiple_matching_keywords': int((matrix.sum(axis=1) > 1).sum()), 'boundary': 'Literal source-rule composition, not adjudicated components; same-record context cues do not establish clinical negation or history'}
    pd.DataFrame(rows_out).to_csv(OUT / 'china_keyword_composition.csv', index=False)
    pd.DataFrame(pairs_out).to_csv(OUT / 'china_keyword_overlap.csv', index=False)
    pd.DataFrame(cues_out).to_csv(OUT / 'china_context_cues.csv', index=False)
    save('china_keyword_composition.json', aggregate)
    return aggregate

def main():
    protocol = _release_resolve_config(json.loads(PROTOCOL_PATH.read_text(encoding='utf-8')))
    contract = _release_resolve_config(json.loads(CONTRACT_PATH.read_text(encoding='utf-8')))
    paths = {k: Path(v['path']) for k, v in contract['inputs'].items() if k in ['sch_joined_diagnostic_cache', 'whu_cancer_records', 'china_patient_labels', 'mimic_raw_diagnoses', 'mimic_raw_admissions', 'mimic_cached_labels', 'mimic_paper_predictions']}
    paths.update({'independent_predictions': VALIDATION / 'private/runs/MIMIC_original/predictions.npz', 'independent_roles': VALIDATION / 'private/runs/MIMIC_original/patient_roles.npz', 'independent_prepared': VALIDATION / 'private/prepared/MIMIC.npz', 'independent_protocol': VALIDATION / 'protocol.json', 'independent_source': _release_source_path(VALIDATION / 'run_validation.py'), 'label_contract': CONTRACT_PATH, 'protocol': PROTOCOL_PATH})
    for k in ['sch_diagnoses', 'china_labels', 'mimic_labels']:
        paths[k] = Path(contract['sources'][k]['path'])
    before = {k: identity(p) for k, p in paths.items()}
    for k, meta in contract['inputs'].items():
        if k in before:
            assert before[k] == meta, k
    for k in ['sch_diagnoses', 'china_labels', 'mimic_labels']:
        assert before[k] == contract['sources'][k], k
    save('component_input_manifest.json', {'created_before_new_performance': True, 'created_at': datetime.now().astimezone().isoformat(), 'inputs': before, 'component_prefixes': contract['rules']['MIMIC']['component_prefixes'], 'exact_original_rule_lines': {'mimic_labels': [52, 75], 'sch_diagnoses': [119, 129], 'china_labels': [81, 84]}, 'script': identity(Path(__file__)), 'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__}})
    tests = synthetic_tests()
    mimic = mimic_analysis(contract, paths, protocol)
    china = chinese_composition(contract, paths)
    after = {k: identity(p) for k, p in paths.items()}
    assert before == after
    save('component_analysis_qa.json', {'schema': 'component_analysis_qa_v1', 'all_passed': True, 'synthetic_tests': tests, 'synthetic_test_count': len(tests), 'mimic_checks': mimic, 'china_label_reproduction': {c: v['reproduced_original_patient_labels'] for c, v in china.items()}, 'source_inputs_unchanged': True, 'patient_level_outputs_written': False, 'models_trained_or_loaded': False, 'clinical_records_adjudicated': 0})
    outputs = [p for p in OUT.iterdir() if p.is_file() and p.suffix in {'.csv', '.json', '.py'} and (p.name != 'component_run_manifest.json')]
    save('component_run_manifest.json', {'completed_at': datetime.now().astimezone().isoformat(), 'status': 'completed_recorded_label_analysis_not_clinical_validation', 'inputs_unchanged': before == after, 'outputs': {p.name: identity(p) for p in outputs}})
    print(json.dumps({'status': 'complete', 'MIMIC_admissions': 9946, 'MIMIC_patients': 6783, 'targets': len(protocol['analysis_targets']), 'synthetic_tests_passed': len(tests)}, ensure_ascii=False))
if __name__ == '__main__':
    main()
