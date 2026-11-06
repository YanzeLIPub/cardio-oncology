"""Audit historical transfer caches and bootstrap only verified fixed predictions."""
from __future__ import annotations
from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import zipfile
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent
CHINA = Path(_release_path('china', 'outputs/ensemble_model/real_model_predictions_v2.csv'))
COPY = Path(_release_path('pipeline', 'data/real_model_predictions_v2.csv'))
LABELS = Path(_release_path('china', 'outputs/survival_analysis_patient_level/patient_level_data.csv'))
PRED = Path(_release_path('pipeline', ''))
PAPER = PRED / 'Manuscript/v13/paper_predictions_9946.csv'
KEY = '病案号'
PROB_COLS = ['pred_from_sch', 'pred_from_whu']
SCOPES = [Path(_release_path('china', '')), PRED, Path(_release_path('workspace', ''))]
SKIP = {'.venv', '.git', 'node_modules', '__pycache__', 'NPJ_DM', 'mimic-iii-clinical-database-1.4'}

def identity(path: Path) -> dict:
    p = Path(path)
    return {'path': p.as_posix(), 'bytes': p.stat().st_size}

def dump(path: Path, value: dict) -> None:
    assert path.parent == ROOT
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')

def prepare_auc(y: np.ndarray, scores: np.ndarray):
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    return (order, starts, y[order].astype(float))

def weighted_auc(prepared, weights: np.ndarray) -> float:
    order, starts, y = prepared
    w = weights[order]
    pos = np.add.reduceat(w * y, starts)
    neg = np.add.reduceat(w * (1.0 - y), starts)
    npos, nneg = (pos.sum(), neg.sum())
    if npos == 0 or nneg == 0:
        return float('nan')
    return float(np.dot(pos, np.cumsum(neg) - 0.5 * neg) / (npos * nneg))

def self_tests() -> dict:
    rng = np.random.default_rng(9173)
    errors = []
    for _ in range(30):
        y = rng.integers(0, 2, 40)
        s = rng.integers(0, 6, 40) / 5
        groups = np.repeat(np.arange(10), 4)
        multiplicity = rng.integers(0, 5, 10)
        w = multiplicity[groups].astype(float)
        got = weighted_auc(prepare_auc(y, s), w)
        expected = roc_auc_score(y, s, sample_weight=w)
        expanded = np.repeat(np.arange(len(y)), w.astype(int))
        literal = roc_auc_score(y[expanded], s[expanded])
        errors.extend([abs(got - expected), abs(got - literal)])
    tied = weighted_auc(prepare_auc(np.array([0, 1]), np.array([0.5, 0.5])), np.ones(2))
    assert max(errors) < 1e-12 and tied == 0.5
    return {'passed': True, 'random_cluster_cases': 30, 'max_absolute_auc_error': max(errors), 'all_tied_auc': tied, 'checks': ['sklearn sample_weight', 'literal repeated rows', 'half-credit ties']}

def load_china() -> tuple[pd.DataFrame, dict]:
    cols = [KEY, 'Center', 'MACE'] + PROB_COLS
    a = pd.read_csv(CHINA, usecols=cols, dtype={KEY: 'string'})
    b = pd.read_csv(COPY, usecols=cols, dtype={KEY: 'string'})
    labels = pd.read_csv(LABELS, usecols=[KEY, 'Center', 'MACE'], dtype={KEY: 'string'})
    for frame in (a, b, labels):
        assert not frame[[KEY, 'Center', 'MACE']].isna().any().any()
        assert not frame.duplicated(['Center', KEY]).any()
        assert set(frame.MACE.unique()) <= {0, 1}
    aa = a.set_index(['Center', KEY]).sort_index()
    bb = b.set_index(['Center', KEY]).sort_index()
    ll = labels.set_index(['Center', KEY]).sort_index()
    assert aa.index.equals(bb.index) and aa.index.equals(ll.index)
    assert np.array_equal(aa.MACE.to_numpy(), bb.MACE.to_numpy())
    assert np.array_equal(aa.MACE.to_numpy(), ll.MACE.to_numpy())
    differences = {}
    for c in PROB_COLS:
        x, z = (aa[c].to_numpy(), bb[c].to_numpy())
        assert np.array_equal(np.isnan(x), np.isnan(z))
        differences[c] = float(np.nanmax(np.abs(x - z)))
        assert differences[c] == 0
    return (a, {'n': len(a), 'copy_keys_and_labels_equal': True, 'original_label_source_equal': True, 'probability_max_abs_difference': differences, 'cross_centre_patient_key_overlap': len(set(a.loc[a.Center.eq('SCH'), KEY]) & set(a.loc[a.Center.eq('WHU'), KEY]))})

def bootstrap_china(a: pd.DataFrame, protocol: dict) -> tuple[list[dict], pd.DataFrame]:
    results, replicates = ([], {})
    for target_spec in protocol['targets'][:2]:
        source, target = (target_spec['source'], target_spec['target'])
        col = 'pred_from_' + source.lower()
        d = a.loc[a.Center.eq(target)].sort_values(KEY, kind='stable')
        y, p = (d.MACE.to_numpy(dtype=int), d[col].to_numpy(dtype=float))
        groups, codes = np.unique(d[KEY].to_numpy(dtype=str), return_inverse=True)
        assert len(groups) == len(d) and (not np.isnan(p).any())
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        expected_n, expected_pos = {'WHU': (2562, 281), 'SCH': (38509, 1323)}[target]
        assert len(d) == expected_n and int(y.sum()) == expected_pos
        point = float(roc_auc_score(y, p))
        assert round(point, target_spec['reported_decimals']) == target_spec['reported_auc']
        prepared = prepare_auc(y, p)
        assert abs(weighted_auc(prepared, np.ones(len(y))) - point) < 1e-12
        rng = np.random.default_rng(target_spec['seed'])
        draws = np.empty(protocol['bootstrap']['draws'])
        for i in range(len(draws)):
            multiplicity = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            draws[i] = weighted_auc(prepared, multiplicity[codes])
        valid = draws[np.isfinite(draws)]
        ci = np.quantile(valid, [0.025, 0.975], method='linear')
        name = source + '_to_' + target
        replicates[name] = draws
        results.append({'source': source, 'target': target, 'status': 'recovered_and_bootstrapped', 'probability_column': col, 'target_unit': 'patient', 'target_rows': len(d), 'target_patients': len(groups), 'positive_labels': int(y.sum()), 'auc': point, 'ci_low': float(ci[0]), 'ci_high': float(ci[1]), 'reported_auc': target_spec['reported_auc'], 'matches_reported_precision': True, 'seed': target_spec['seed'], 'bootstrap_draws': len(draws), 'valid_draws': len(valid), 'degenerate_draws': int((~np.isfinite(draws)).sum()), 'unique_scores': len(np.unique(p)), 'interval_scope': 'conditional_fixed_historical_predictions'})
    return (results, pd.DataFrame({'draw': np.arange(1, protocol['bootstrap']['draws'] + 1), **replicates}))

def legacy_candidate_check() -> dict:
    s_path = PRED / 'MIMIC-3/results/separate_transfer_predictions.csv'
    m_path = PRED / 'MIMIC-3/results/mimic_to_sch_whu_preds.csv'
    s = pd.read_csv(s_path)
    paper = pd.read_csv(PAPER, usecols=['HADM_ID', 'MACE', 'SUBJECT_ID'])
    z = s.merge(paper, on='HADM_ID', validate='one_to_one')
    assert len(z) == 9946 and (z.mace_label == z.MACE).all()
    m = pd.read_csv(m_path)
    return {'candidate_identities': [identity(s_path), identity(m_path)], 'separate_transfer': {'all_rows': len(s), 'original_9946_overlap': len(z), 'label_mismatches': int((z.mace_label != z.MACE).sum()), 'auc_all_11622': {c: float(roc_auc_score(s.mace_label, s[c])) for c in s if c.startswith('prob_')}, 'auc_original_9946': {c: float(roc_auc_score(z.MACE, z[c])) for c in s if c.startswith('prob_')}, 'decision': 'Not the historical v13 9946 prediction vectors; neither raw nor calibrated scores reproduce the requested AUCs.'}, 'mimic_to_china': [{'target': k, 'n': len(g), 'positive_labels': int(g.MACE.sum()), 'auc': float(roc_auc_score(g.MACE, g.pred)), 'patient_key_present': False, 'decision': 'Legacy model predictions do not reproduce the requested v13 historical AUC.'} for k, g in m.groupby('Source')], 'paper_9946': {'rows': len(paper), 'patients': int(paper.SUBJECT_ID.nunique()), 'positive_labels': int(paper.MACE.sum()), 'columns': pd.read_csv(PAPER, nrows=0).columns.tolist()}}

def bounded_search() -> dict:
    candidates, archives = ([], [])
    for root in SCOPES:
        for dirname, dirs, names in os.walk(root):
            dirs[:] = sorted((d for d in dirs if d not in SKIP))
            for name in sorted(names):
                path = Path(dirname) / name
                low = name.lower()
                if path.suffix.lower() == '.zip':
                    matches = []
                    try:
                        with zipfile.ZipFile(path) as archive:
                            for member in archive.infolist():
                                basename = Path(member.filename).name.lower()
                                if any((t in basename for t in ('predict', 'transfer', 'ckpt'))) and Path(basename).suffix in ('.csv', '.npz', '.npy', '.pkl', '.joblib'):
                                    matches.append({'entry': member.filename, 'bytes': member.file_size})
                        archives.append({**identity(path), 'candidate_entries': matches})
                    except (OSError, zipfile.BadZipFile) as e:
                        archives.append({'path': path.as_posix(), 'error_type': type(e).__name__})
                if not any((t in low for t in ('predict', 'transfer', 'ckpt'))):
                    continue
                if path.suffix.lower() == '.csv':
                    try:
                        columns = pd.read_csv(path, nrows=0).columns.tolist()
                        candidates.append({**identity(path), 'kind': 'csv_header_only', 'columns': columns})
                    except (OSError, UnicodeError, pd.errors.ParserError) as e:
                        candidates.append({'path': path.as_posix(), 'error_type': type(e).__name__})
                elif path.suffix.lower() == '.npz':
                    try:
                        with np.load(path, allow_pickle=False) as z:
                            fields = []
                            for key in z.files:
                                try:
                                    fields.append({'key': key, 'shape': list(z[key].shape), 'dtype': str(z[key].dtype)})
                                except ValueError:
                                    fields.append({'key': key, 'object_array_not_loaded': True})
                            candidates.append({**identity(path), 'kind': 'npz_metadata_only', 'fields': fields})
                    except (OSError, ValueError) as e:
                        candidates.append({'path': path.as_posix(), 'error_type': type(e).__name__})
    return {'roots': [p.as_posix() for p in SCOPES], 'excluded_directory_names': sorted(SKIP), 'selection': 'Basename contains predict, transfer or ckpt; CSV headers and NPZ array metadata inspected. ZIP entry inventories searched with the same tokens. No patient rows printed and no source modules imported.', 'candidate_count': len(candidates), 'candidates': candidates, 'archives': archives, 'scope_limit': 'This is a bounded search of the listed workspaces and archives, not proof that no historical probability vector exists elsewhere.'}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    protocol_path = ROOT / 'protocol.json'
    protocol = _release_resolve_config(json.loads(protocol_path.read_text(encoding='utf-8')))
    paths = [CHINA, COPY, LABELS, PAPER, protocol_path, Path(__file__).resolve(), Path(_release_path('china', 'optionB_parallel_analysis.py')), _release_source_path(PRED / 'Manuscript/v13/compute_v13_metrics.py'), _release_source_path(PRED / 'Manuscript/v13/compute_v13_fast.py'), PRED / 'Manuscript/v13/transfer_metrics_9946.csv']
    before = [identity(p) for p in paths]
    tests = self_tests()
    china, label_check = load_china()
    results, draws = bootstrap_china(china, protocol)
    after = [identity(p) for p in paths]
    assert before == after
    if args.verify:
        previous = _release_resolve_config(json.loads((ROOT / 'summary.json').read_text(encoding='utf-8')))
        recorded_draws = pd.read_csv(ROOT / 'bootstrap_replicates.csv', float_precision='round_trip')
        assert np.array_equal(draws.to_numpy(), recorded_draws.to_numpy(), equal_nan=True)
        assert results == previous['directions'][:2]
        assert before == previous['input_identities']
        dump(ROOT / 'verification.json', {'passed': True, 'scope': 'Independent rerun of this deterministic script; no new model fitting.', 'draws_identical': True, 'summary_identical': True, 'input_identities_unchanged': True, 'self_tests': tests, 'code': identity(Path(__file__).resolve()), 'verified_outputs': [identity(ROOT / n) for n in ('summary.json', 'bootstrap_replicates.csv', 'transfer_auc_intervals.csv', 'search_audit.json')]})
        print(json.dumps({'verification_passed': True, 'directions_recomputed': 2, 'draws_per_direction': 2000}))
        return
    assert not (ROOT / 'summary.json').exists(), 'Use a new output branch instead of overwriting an analysis.'
    search = bounded_search()
    legacy = legacy_candidate_check()
    for spec in protocol['targets'][2:]:
        results.append({'source': spec['source'], 'target': spec['target'], 'reported_auc': spec['reported_auc'], 'status': 'historical_probability_vector_not_recovered', 'auc': None, 'ci_low': None, 'ci_high': None, 'recovery_required': 'Recover the original v13 vector and exact target order/patient keys or serialized historical fitted models plus verified historical inputs. A newly trained model or legacy prediction vector is not a substitute.'})
    dump(ROOT / 'search_audit.json', {'search': search, 'legacy_candidates': legacy, 'code_evidence': [{'path': paths[6].as_posix(), 'lines': '163-178; 294-298; 455-459', 'finding': 'Chinese full-source ensemble transfer vectors are calculated and written as pred_from_sch/pred_from_whu.'}, {'path': paths[7].as_posix(), 'lines': '530-575; 587-611; 616-623', 'finding': 'v13 transfer returns proba, but caller assigns the vector to underscore and saves only rounded aggregate AUC rows.'}, {'path': paths[8].as_posix(), 'lines': '112-129; 263-266; 302-309', 'finding': 'Fast transfer returns only AUC; saved checkpoint is not a transfer probability cache.'}]})
    draws.to_csv(ROOT / 'bootstrap_replicates.csv', index=False, float_format='%.17g')
    pd.DataFrame(results).to_csv(ROOT / 'transfer_auc_intervals.csv', index=False, float_format='%.17g')
    summary = {'analysis': protocol['analysis'], 'protocol_identity': identity(protocol_path), 'input_identities': before, 'source_files_unchanged': before == after, 'runtime': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__}, 'self_tests': tests, 'label_and_copy_verification': label_check, 'directions': results, 'limitations': protocol['limitations'] + [protocol['bootstrap']['conditioning']], 'outputs': [identity(ROOT / n) for n in ('bootstrap_replicates.csv', 'transfer_auc_intervals.csv', 'search_audit.json')]}
    dump(ROOT / 'summary.json', summary)
    print(json.dumps({'recovered_directions': results[:2], 'unrecovered_directions': 4, 'candidate_files_audited': search['candidate_count'], 'archives_audited': len(search['archives'])}, indent=2))
if __name__ == '__main__':
    main()
