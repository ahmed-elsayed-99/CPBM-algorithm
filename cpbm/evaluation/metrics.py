import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from typing import Dict, Optional


class CPBMEvaluator:

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    def full_report(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "CPBM",
    ) -> Dict:
        y_pred = (y_proba >= self.threshold).astype(int)
        report = classification_report(y_true, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        prec, rec, _ = precision_recall_curve(y_true, y_proba)

        return {
            "model": model_name,
            "auc_roc": float(roc_auc_score(y_true, y_proba)),
            "auc_pr": float(average_precision_score(y_true, y_proba)),
            "accuracy": float(report["accuracy"]),
            "precision": float(report.get("1", {}).get("precision", 0.0)),
            "recall": float(report.get("1", {}).get("recall", 0.0)),
            "f1": float(report.get("1", {}).get("f1-score", 0.0)),
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
            "classification_report": report,
        }

    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        rows = []
        for name, proba in predictions.items():
            r = self.full_report(y_true, proba, name)
            rows.append({
                "Model": name,
                "AUC-ROC": round(r["auc_roc"], 4),
                "AUC-PR": round(r["auc_pr"], 4),
                "Accuracy": round(r["accuracy"], 4),
                "Precision": round(r["precision"], 4),
                "Recall": round(r["recall"], 4),
                "F1": round(r["f1"], 4),
            })
        return pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)

    def calibration_report(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        return pd.DataFrame({
            "mean_predicted_probability": mean_pred.round(4),
            "fraction_of_positives": fraction_pos.round(4),
            "calibration_error": (mean_pred - fraction_pos).round(4),
        })

    def stratum_report(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        strata: np.ndarray,
    ) -> pd.DataFrame:
        rows = []
        for s in np.unique(strata):
            mask = strata == s
            yt, yp = y_true[mask], y_proba[mask]
            if len(np.unique(yt)) < 2:
                continue
            rows.append({
                "stratum": s,
                "n": int(mask.sum()),
                "purchase_rate": round(float(yt.mean()), 4),
                "mean_proba": round(float(yp.mean()), 4),
                "auc": round(float(roc_auc_score(yt, yp)), 4),
            })
        return pd.DataFrame(rows)