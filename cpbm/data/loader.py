import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


class CPBMDataLoader:

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url
        self.engine = create_engine(db_url) if db_url else None

    def load_transactions_csv(
        self,
        path: str,
        ind_col: str = "ind_id",
        amount_col: str = "amount",
        time_col: str = "txn_timestamp",
        category_col: str = "category",
        stratum_col: Optional[str] = "stratum",
    ) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=[time_col])
        df = df.rename(columns={
            ind_col: "ind_id",
            amount_col: "amount",
            time_col: "txn_timestamp",
            category_col: "category",
        })
        if stratum_col and stratum_col in df.columns:
            df = df.rename(columns={stratum_col: "stratum"})
        df["txn_timestamp"] = pd.to_datetime(df["txn_timestamp"])
        df = df.sort_values(["ind_id", "txn_timestamp"]).reset_index(drop=True)
        min_date = df["txn_timestamp"].min()
        df["day"] = (df["txn_timestamp"] - min_date).dt.days
        return df

    def load_transactions_db(
        self,
        target_category: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if self.engine is None:
            raise RuntimeError("Provide db_url to use database loading.")
        query = text("""
            SELECT t.ind_id, t.amount, t.txn_timestamp,
                   p.category, p.price_tier, i.stratum, i.geo_code
            FROM transactions t
            JOIN products p ON t.product_id = p.product_id
            JOIN individuals i ON t.ind_id = i.ind_id
            WHERE p.category = :cat
              AND t.txn_timestamp BETWEEN :start AND :end
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                "cat": target_category, "start": start_date, "end": end_date
            })
        return df

    def compute_total_spend_per_individual(self, df: pd.DataFrame) -> pd.DataFrame:
        total = df.groupby("ind_id")["amount"].sum().rename("total_spend").reset_index()
        return df.merge(total, on="ind_id", how="left")

    def build_label_vector(
        self,
        df: pd.DataFrame,
        target_category: str,
        forecast_horizon_days: int = 30,
        split_date: Optional[str] = None,
    ) -> pd.Series:
        if split_date is None:
            max_day = df["day"].max()
            split_day = max_day - forecast_horizon_days
        else:
            split_day = (
                pd.to_datetime(split_date) - pd.to_datetime(df["txn_timestamp"].min())
            ).days

        future = df[(df["day"] > split_day) & (df["category"] == target_category)]
        buyers = set(future["ind_id"].unique())
        all_inds = df["ind_id"].unique()
        return pd.Series(
            {ind: int(ind in buyers) for ind in all_inds}, name="label"
        )

    def load_market_signals(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["signal_date"])
        return df.sort_values("signal_date").reset_index(drop=True)

    def load_social_network(self, path: str) -> list:
        df = pd.read_csv(path)
        return list(zip(df["node_i"], df["node_j"], df["strength"]))