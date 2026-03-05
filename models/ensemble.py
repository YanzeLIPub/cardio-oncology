#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ensemble model: XGBoost + Random Forest (50:50 average).

For Lancet Digital Health cardio-oncology MACE prediction study.
Supports within-cohort 5-fold CV and cross-cohort transfer evaluation.

Usage:
    from models.ensemble import EnsembleModel
    model = EnsembleModel()
    oof_probs = model.fit_cv(X_train, y_train)
    test_probs = model.predict(X_test)
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class EnsembleModel:
    """
    XGBoost + Random Forest 50:50 ensemble.

    Hyperparameters:
        XGBoost:      max_depth=5, lr=0.05, subsample=0.8, colsample_bytree=0.8
        RandomForest: n_estimators=300, max_depth=8, min_samples_leaf=5
        Both:         scale_pos_weight / class_weight='balanced'
    """

    def __init__(self,
                 xgb_rounds: int = 300,
                 xgb_early_stop: int = 20,
                 n_folds: int = 5,
                 random_state: int = 42):
        self.xgb_rounds = xgb_rounds
        self.xgb_early_stop = xgb_early_stop
        self.n_folds = n_folds
        self.random_state = random_state
        self._xgb_models: list = []
        self._rf_models: list = []

    # ── Hyperparameters ───────────────────────────────────────
    def _xgb_params(self, pos_weight: float) -> dict:
        return {
            'objective':        'binary:logistic',
            'eval_metric':      'auc',
            'max_depth':        5,
            'learning_rate':    0.05,
            'subsample':        0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': pos_weight,
            'seed':             self.random_state,
            'nthread':          -1,
            'verbosity':        0,
        }

    def _rf_params(self, pos_weight: float) -> dict:
        # RF uses class_weight instead of pos_weight
        return {
            'n_estimators':    300,
            'max_depth':       8,
            'min_samples_leaf': 5,
            'class_weight':    'balanced',
            'random_state':    self.random_state,
            'n_jobs':          -1,
        }

    # ── 5-fold cross-validation on training set ───────────────
    def fit_cv(self, X: np.ndarray, y: np.ndarray,
               verbose: bool = True) -> tuple[np.ndarray, list[float]]:
        """
        Fit ensemble via 5-fold stratified CV.

        Returns
        -------
        oof_probs : np.ndarray, shape (n_samples,)
            Out-of-fold predicted probabilities (within-CV).
        fold_aucs : list[float]
            Per-fold AUC values.
        """
        pos_weight = float((y == 0).sum()) / max(1, (y == 1).sum())
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.random_state)
        oof_probs  = np.zeros(len(y))
        fold_aucs  = []
        self._xgb_models = []
        self._rf_models  = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # XGBoost
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval   = xgb.DMatrix(X_val, label=y_val)
            xgb_m  = xgb.train(
                self._xgb_params(pos_weight), dtrain,
                num_boost_round=self.xgb_rounds,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=self.xgb_early_stop,
                verbose_eval=False,
            )
            xgb_prob = xgb_m.predict(dval)
            self._xgb_models.append(xgb_m)

            # Random Forest
            rf_m = RandomForestClassifier(**self._rf_params(pos_weight))
            rf_m.fit(X_tr, y_tr)
            rf_prob = rf_m.predict_proba(X_val)[:, 1]
            self._rf_models.append(rf_m)

            # 50:50 ensemble
            ens_prob = 0.5 * xgb_prob + 0.5 * rf_prob
            oof_probs[val_idx] = ens_prob
            auc = roc_auc_score(y_val, ens_prob)
            fold_aucs.append(auc)
            if verbose:
                print(f"  Fold {fold}: AUC={auc:.4f}  "
                      f"[XGB={roc_auc_score(y_val, xgb_prob):.4f}, "
                      f"RF={roc_auc_score(y_val, rf_prob):.4f}]")

        cv_auc = roc_auc_score(y, oof_probs)
        if verbose:
            print(f"\n  CV AUC (OOF): {cv_auc:.4f} ± {np.std(fold_aucs):.4f}")
        return oof_probs, fold_aucs

    # ── Fit on full data (for cross-cohort transfer) ──────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleModel':
        """Fit a single ensemble on the full training set."""
        pos_weight = float((y == 0).sum()) / max(1, (y == 1).sum())

        # XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        self._xgb_full = xgb.train(
            self._xgb_params(pos_weight), dtrain,
            num_boost_round=self.xgb_rounds,
            verbose_eval=False,
        )

        # Random Forest
        self._rf_full = RandomForestClassifier(**self._rf_params(pos_weight))
        self._rf_full.fit(X, y)
        return self

    # ── Predict ───────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Average predictions from all CV fold models (if available),
        else use the single full-fit model.
        """
        if self._xgb_models:
            # Average across all folds
            xgb_preds = np.mean(
                [m.predict(xgb.DMatrix(X)) for m in self._xgb_models], axis=0)
            rf_preds  = np.mean(
                [m.predict_proba(X)[:, 1] for m in self._rf_models], axis=0)
        else:
            xgb_preds = self._xgb_full.predict(xgb.DMatrix(X))
            rf_preds  = self._rf_full.predict_proba(X)[:, 1]
        return 0.5 * xgb_preds + 0.5 * rf_preds

    # ── Feature importance ────────────────────────────────────
    def feature_importances(self, feature_names: list[str],
                            top_n: int = 25) -> 'dict':
        """
        Return normalised XGBoost feature importances (gain) from the
        full-trained model.
        """
        if not hasattr(self, '_xgb_full'):
            raise RuntimeError("Call fit() before feature_importances()")
        scores = self._xgb_full.get_score(importance_type='gain')
        total  = sum(scores.values())
        normed = {k: v / total for k, v in sorted(
            scores.items(), key=lambda x: -x[1])}
        # Map f0, f1, … → actual names if available
        named  = {}
        for k, v in normed.items():
            if k.startswith('f') and k[1:].isdigit():
                idx = int(k[1:])
                name = feature_names[idx] if idx < len(feature_names) else k
            else:
                name = k
            named[name] = round(v, 6)
        return dict(list(named.items())[:top_n])
