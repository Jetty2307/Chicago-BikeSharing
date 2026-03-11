import pandas as pd
import numpy as np
from typing import Tuple


def _clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().replace([np.inf, -np.inf], np.nan).dropna()


def fetch_features_xgboost(df: pd.DataFrame, interval: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = _clean_training_frame(df)
    y = df["rides"]
    X = df[[
        "rideable_type",
        "year",
        interval,
        "season",
        f"rides_2{interval}s_ago",
        f"rides_last{interval}",
    ]]
    return X, y


def fetch_features_gam(df: pd.DataFrame, interval: str) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    df = _clean_training_frame(df)
    feature_names = ["rideable_type", "year", interval, "season"]  # order matters
    X = df[feature_names].to_numpy()
    y = df["rides"].to_numpy()
    return X, y, feature_names




