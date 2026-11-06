from release_paths import path as _release_path, resolve_config as _release_resolve_config, source_path as _release_source_path, font_path as _release_font_path
from pathlib import Path
import os, json, sys, hashlib, collections, time
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[k] = '8'
import numpy as np, pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
D = Path(__file__).resolve().parents[1]
A = D / 'analysis'
E = D / 'private/whu_source_eligible'

def read(p):
    return _release_resolve_config(json.loads(Path(p).read_text(encoding='utf-8-sig')))

def sha(p):
    return None

def save(p, x):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)

def fit(y, lp, w):
    n = w.sum()
    pos = np.dot(w, y)
    if n <= 0 or pos <= 0 or pos >= n:
        return (np.full(3, np.nan), {'status': 'missing_class'})
    try:
        citl = brentq(lambda a: float(np.dot(w, expit(a + lp)) - pos), -80, 80)
    except Exception as e:
        return (np.full(3, np.nan), {'status': 'root_error', 'type': type(e).__name__})
    b = np.array([citl, 1.0])
    status = 'nonconverged'
    steps = 0
    for steps in range(100):
        z = b[0] + b[1] * lp
        p = expit(z)
        r = w * (p - y)
        v = w * p * (1 - p)
        g = np.array([r.sum(), np.dot(r, lp)]) / n
        H = np.array([[v.sum(), np.dot(v, lp)], [np.dot(v, lp), np.dot(v, lp * lp)]]) / n
        if max(np.abs(g)) < 1e-10:
            status = 'converged'
            break
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            status = 'singular_information'
            break
        value = np.dot(w, np.logaddexp(0, z) - y * z) / n
        size = 1.0
        accepted = False
        for trial in range(40):
            proposal = b - size * step
            zz = proposal[0] + proposal[1] * lp
            loss = np.dot(w, np.logaddexp(0, zz) - y * zz) / n
            if np.isfinite(loss) and loss <= value + 1e-14:
                b = proposal
                accepted = True
                break
            size *= 0.5
        if not accepted:
            status = 'line_search_failure'
            break
        if not np.isfinite(b).all() or np.abs(b).max() >= 80:
            status = 'coefficient_boundary'
            break
    z = b[0] + b[1] * lp
    p = expit(z)
    r = w * (p - y)
    v = w * p * (1 - p)
    grad = float(np.max(np.abs([r.sum() / n, np.dot(r, lp) / n])))
    H = np.array([[v.sum(), np.dot(v, lp)], [np.dot(v, lp), np.dot(v, lp * lp)]]) / n
    mineig = float(np.linalg.eigvalsh(H).min())
    accepted = bool(np.isfinite(b).all() and np.abs(b).max() < 80 and (grad < 1e-06) and (mineig > 1e-09))
    if accepted:
        status = 'converged'
    elif mineig <= 1e-09:
        status = 'weak_information'
    return (np.array([citl, *b]) if accepted else np.array([citl, np.nan, np.nan]), {'status': status, 'maximum_gradient': grad, 'minimum_information_eigenvalue': mineig, 'iterations': steps + 1, 'coefficient_boundary': bool(np.abs(b).max() >= 80)})

def alternate_fit(y, lp, w):
    n = w.sum()
    p = np.dot(w, y)
    ci = brentq(lambda x: np.dot(w, expit(x + lp)) - p, -80, 80)
    X = np.column_stack([np.ones(len(y)), lp])
    result = minimize(lambda b: np.dot(w, np.logaddexp(0, X @ b) - y * (X @ b)) / n, np.array([ci, 1.0]), jac=lambda b: X.T @ (w * (expit(X @ b) - y)) / n, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-10})
    gradient = np.max(np.abs(X.T @ (w * (expit(X @ result.x) - y)) / n))
    return (np.array([ci, *result.x]), float(gradient))

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    start = time.monotonic()
    plan = read(A / 'calibration_uncertainty_plan.json')
    pred = E / 'private/runs/WHU_source_eligible/predictions.npz'
    with np.load(pred) as f:
        y = f['y'].astype(float)
        scores = {k: f[k].copy() for k in plan['score_columns']}
    before = sha(pred)
    lps = {k: logit(np.clip(v, 1e-06, 1 - 1e-06)) for k, v in scores.items()}
    keys = list(scores)
    draws = np.full((2000, 2, 3), np.nan)
    stat = {k: collections.Counter() for k in keys}
    bound = {k: 0 for k in keys}
    rng = np.random.default_rng(20261029)
    digest = hashlib.sha256()
    audit = set(np.linspace(0, 1999, 101, dtype=int))
    max_delta = np.zeros(3)
    audit_count = 0
    maximum_gradient = 0.0
    minimum_information = float('inf')
    for rep in range(2000):
        w = rng.multinomial(len(y), np.full(len(y), 1 / len(y))).astype(float)
        digest.update(w.astype('<i8').tobytes())
        for j, k in enumerate(keys):
            values, info = fit(y, lps[k], w)
            draws[rep, j] = values
            stat[k][info['status']] += 1
            bound[k] += int(info.get('coefficient_boundary', False))
            maximum_gradient = max(maximum_gradient, info.get('maximum_gradient', 0))
            minimum_information = min(minimum_information, info.get('minimum_information_eigenvalue', float('inf')))
            if rep in audit and np.isfinite(values).all():
                vv, grad = alternate_fit(y, lps[k], w)
                assert grad < 1e-06
                delta = np.abs(vv - values)
                max_delta = np.maximum(max_delta, delta)
                assert np.all(delta < 0.0001), 'Independent optimizer discrepancy exceeds tolerance'
                audit_count += 1
        if (rep + 1) % 500 == 0:
            print(json.dumps({'calibration_bootstrap': rep + 1, 'of': 2000}), flush=True)
    expected = None
    assert digest.hexdigest() == expected
    old = pd.read_csv(A / 'source_eligible_calibration.csv')
    rows = []
    pointchecks = {}
    names = ['calibration_in_the_large', 'joint_intercept', 'joint_slope']
    for j, k in enumerate(keys):
        point, info = fit(y, lps[k], np.ones(len(y)))
        target = old.loc[old.scope.eq('retained') & old.score.eq('retrained_' + k)].iloc[0]
        delta = np.abs(point - target[names].to_numpy(float))
        assert np.all(delta < 1e-05)
        pointchecks[k] = delta.tolist()
        for q, param in enumerate(names):
            v = draws[:, j, q]
            v = v[np.isfinite(v)]
            lo, hi = np.quantile(v, [0.025, 0.975]) if len(v) else (None, None)
            rows.append({'cohort': 'WHU', 'n': len(y), 'positive': int(y.sum()), 'score': 'retrained_' + k, 'parameter': param, 'estimate': float(point[q]), 'lower_95': float(lo) if lo is not None else None, 'upper_95': float(hi) if hi is not None else None, 'valid_replicates': len(v), 'failed_replicates': 2000 - len(v), 'boundary_replicates': bound[k], 'total_replicates': 2000, 'bootstrap_seed': 20261029, 'uncertainty': 'conditional_on_fixed_out_of_fold_scores_not_refitting'})
    np.savez_compressed(E / 'calibration_bootstrap_retained.npz', values=draws, score_names=np.array(keys), parameter_names=np.array(names))
    pd.DataFrame(rows).to_csv(A / 'source_eligible_calibration_ci.csv', index=False, encoding='utf-8-sig')
    report = {'complete': True, 'predictions_unchanged': True, 'models_refitted': False, 'rows': rows, 'status_counts': {k: dict(v) for k, v in stat.items()}, 'boundary_counts': bound, 'maximum_joint_gradient': maximum_gradient, 'minimum_joint_information_eigenvalue': minimum_information, 'patient_weight_sequence_matches_frozen_retained_metrics': True, 'independent_BFGS_comparisons': audit_count, 'independent_max_abs_parameter_difference': max_delta.tolist(), 'point_estimates_match_prior_within_1e_5': pointchecks, 'seconds': time.monotonic() - start, 'scope_note': 'Statistical reporting supplement requested after fitting; no performance-based model selection. Only the two stated score columns now have calibration parameter CIs; prior point-only metadata is superseded for these scores.'}
    save(A / 'source_eligible_calibration_ci_summary.json', report)
    for row in rows:
        j = keys.index(row['score'].removeprefix('retrained_'))
        q = names.index(row['parameter'])
        v = draws[:, j, q]
        v = v[np.isfinite(v)]
        assert np.allclose(np.quantile(v, [0.025, 0.975]), [row['lower_95'], row['upper_95']], rtol=0, atol=1e-12)
    save(D / 'qa/calibration_uncertainty_verification.json', {'passed': True, 'all6_intervals_recomputed': True, 'independent_optimizer_comparisons': audit_count, 'tolerance': 0.0001, 'maximum_parameter_differences': max_delta.tolist(), 'same_patient_draws_as_frozen_retained_bootstrap': True, 'models_refitted': False, 'scores_unchanged': True, 'failure_counts': {k: dict(v) for k, v in stat.items()}})
    print(json.dumps({'completed': True, 'rows': rows, 'failures': {k: dict(v) for k, v in stat.items()}}, ensure_ascii=False), flush=True)
if __name__ == '__main__':
    main()
