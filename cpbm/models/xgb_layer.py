import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from typing import Optional, Dict, Tuple
import joblib


class XGBLayer:

    def __init__(
        self,
        n_estimators int = 300,
        max_depth int = 5,
        learning_rate float = 0.08,
        subsample float = 0.8,
        colsample_bytree float = 0.8,
        early_stopping_rounds int = 25,
        random_state int = 42,
        calibrate bool = True,
        n_cv_folds int = 5,
    )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.calibrate = calibrate
        self.n_cv_folds = n_cv_folds
        self.model_ Optional[xgb.XGBClassifier] = None
        self.calibrated_model_ Optional[CalibratedClassifierCV] = None
        self.cv_auc_scores_ list = []
        self.feature_importances_ Optional[np.ndarray] = None
        self.feature_names_ Optional[list] = None

    def _build_model(self, scale_pos_weight float) - xgb.XGBClassifier
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            eval_metric=auc,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )

    def fit(
        self,
        X_train np.ndarray,
        y_train np.ndarray,
        X_val np.ndarray,
        y_val np.ndarray,
        feature_names Optional[list] = None,
    ) - XGBLayer
        self.feature_names_ = feature_names
        spw = float((y_train == 0).sum())  float((y_train == 1).sum() + 1e-9)
        self.model_ = self._build_model(spw)
        self.model_.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.feature_importances_ = self.model_.feature_importances_
        if self.calibrate
            base = self._build_model(spw)
            self.calibrated_model_ = CalibratedClassifierCV(base, method=isotonic, cv=3)
            self.calibrated_model_.fit(X_train, y_train)
        return self

    def cross_validate(
        self,
        X np.ndarray,
        y np.ndarray,
    ) - Dict[str, float]
        skf = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state)
        aucs = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y))
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            spw = float((y_tr == 0).sum())  float((y_tr == 1).sum() + 1e-9)
            m = self._build_model(spw)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            proba = m.predict_proba(X_va)[, 1]
            aucs.append(roc_auc_score(y_va, proba))
        self.cv_auc_scores_ = aucs
        return {mean_auc float(np.mean(aucs)), std_auc float(np.std(aucs)), fold_aucs aucs}

    def predict_proba(self, X np.ndarray) - np.ndarray
        if self.calibrate and self.calibrated_model_ is not None
            return self.calibrated_model_.predict_proba(X)[, 1]
        if self.model_ is None
            raise RuntimeError(Call fit first.)
        return self.model_.predict_proba(X)[, 1]

    def get_feature_importance_df(self) - pd.DataFrame
        if self.feature_importances_ is None
            raise RuntimeError(Call fit first.)
        names = self.feature_names_ or [ff{i} for i in range(len(self.feature_importances_))]
        df = pd.DataFrame({feature names, importance self.feature_importances_})
        return df.sort_values(importance, ascending=False).reset_index(drop=True)

    def save(self, path str) - None
        joblib.dump(self, path)

    @classmethod
    def load(cls, path str) - XGBLayer
        return joblib.load(path)
