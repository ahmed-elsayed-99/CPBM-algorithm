import numpy as np
import pandas as pd
from typing import Dict, Optional


class NPICalculator:

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.npi_history_: Dict[str, list] = {}

    def compute(
        self,
        category_spend: float,
        mean_velocity: float,
        resistance: float,
        mean_exposure: float,
    ) -> float:
        npi = (category_spend * mean_velocity) / (
            resistance * mean_exposure + self.eps
        )
        return float(npi)

    def compute_from_dataframe(
        self,
        df: pd.DataFrame,
        target_category: str,
        exposure_col: str = "marketing_exposure",
        spend_col: str = "amount",
        transaction_col: str = "txn_id",
        new_product_col: Optional[str] = "is_new_product",
    ) -> float:
        cat_df = df[df["category"] == target_category] if "category" in df.columns else df

        M = float(cat_df[spend_col].sum())
        V = float(cat_df.groupby("ind_id")[transaction_col].count().mean())

        if new_product_col and new_product_col in cat_df.columns:
            trial_rate = float(cat_df[new_product_col].mean())
        else:
            trial_rate = 0.05
        R = 1.0 - trial_rate

        E = float(df[exposure_col].mean()) if exposure_col in df.columns else 1.0

        return self.compute(M, V, R, E)

    def compute_timeseries(
        self,
        df: pd.DataFrame,
        target_category: str,
        period_col: str = "period",
        geo_col: str = "geo_code",
    ) -> pd.DataFrame:
        records = []
        for (geo, period), group in df.groupby([geo_col, period_col]):
            npi = self.compute_from_dataframe(group, target_category)
            records.append({"geo_code": geo, "period": period, "npi": npi})
        return pd.DataFrame(records)

    def interpret(self, npi: float, reference_npi: Optional[float] = None) -> str:
        if reference_npi is not None:
            ratio = npi / (reference_npi + 1e-12)
            if ratio > 1.5:
                return "high_momentum_resistant"
            elif ratio > 0.8:
                return "moderate_inertia"
            else:
                return "low_momentum_receptive"
        if npi > 1000:
            return "high_momentum_resistant"
        elif npi > 100:
            return "moderate_inertia"
        else:
            return "low_momentum_receptive"