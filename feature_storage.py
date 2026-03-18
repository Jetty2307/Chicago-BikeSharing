import pandas as pd
import numpy as np
# from typing import Any, Tuple
from typing import Any, Tuple, List, Dict
from weather import df_weather

exogenous_features = {"month": [],
                      "week": ["max_temp", "avg_temp", "min_temp", "total_rain", "total_snow"]}

def _clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().replace([np.inf, -np.inf], np.nan).dropna()


def fetch_features_xgboost(df: pd.DataFrame, interval: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = _clean_training_frame(df)
    y = df["rides"]
    feature_names = ["rideable_type",
        "year",
        interval,
        "season",
        f"rides_2{interval}s_ago",
        f"rides_last{interval}"] + exogenous_features[interval]
    X = df[feature_names]
    return X, y, feature_names


def fetch_features_gam(df: pd.DataFrame, interval: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = _clean_training_frame(df)
    feature_names = ["rideable_type", "year", interval, "season"] + exogenous_features[interval]
    X = df[feature_names].to_numpy()
    y = df["rides"].to_numpy()
    return X, y, feature_names


def _season_from_timestamp(ts: pd.Timestamp) -> int:
    return (ts.month - 1) // 3


def _build_weekly_state(last_2row: pd.Series, last_row: pd.Series, add_interval) -> Dict[str, Any]:
    next_week_start = pd.Timestamp(add_interval(last_row["year_week"]))

    return {
        "rideable_type": last_row["rideable_type"],
        "current_year": next_week_start.year,
        "current_interval": int(next_week_start.isocalendar().week),
        "current_season": _season_from_timestamp(next_week_start),
        "rides_2ago": last_2row["rides"],
        "rides_last": last_row["rides"],
        "current_year_interval": next_week_start.strftime("%Y-%m-%d"),
    }


def initialize_forecast_state(df: pd.DataFrame, interval: str, period: int, add_interval) -> Dict[str, Any]:
    last_2row = df.iloc[-2]
    last_row = df.iloc[-1]

    if interval == "week":
        return _build_weekly_state(last_2row=last_2row, last_row=last_row, add_interval=add_interval)

    current_interval = last_row[interval] + 1
    if current_interval % (period / 4) == 1:
        current_season = last_row["season"] + 1
    else:
        current_season = last_row["season"]

    if current_interval > period:
        current_interval = 1
        current_year = last_row["year"] + 1
        current_season = 0
    else:
        current_year = last_row["year"]

    return {
        "rideable_type": df["rideable_type"].iloc[0],
        "current_year": current_year,
        "current_interval": current_interval,
        "current_season": current_season,
        "rides_2ago": last_2row["rides"],
        "rides_last": last_row["rides"],
        "current_year_interval": add_interval(last_row[f"year_{interval}"]),
    }


def generate_next_vector(state: Dict[str, Any], interval: str) -> pd.DataFrame:
    return pd.DataFrame({
        "rideable_type": [state["rideable_type"]],
        "year": [state["current_year"]],
        interval: [state["current_interval"]],
        "season": [state["current_season"]],
        f"rides_2{interval}s_ago": [state["rides_2ago"]],
        f"rides_last{interval}": [state["rides_last"]],
    })


def update_forecast_state(
    state: Dict[str, Any],
    predicted_rides: float,
    interval: str,
    period: int,
    add_interval,
) -> Dict[str, Any]:
    next_state = state.copy()
    next_state["rides_2ago"] = state["rides_last"]
    next_state["rides_last"] = predicted_rides

    if interval == "week":
        next_week_start = pd.Timestamp(add_interval(state["current_year_interval"]))
        next_state["current_year_interval"] = next_week_start.strftime("%Y-%m-%d")
        next_state["current_year"] = next_week_start.year
        next_state["current_interval"] = int(next_week_start.isocalendar().week)
        next_state["current_season"] = _season_from_timestamp(next_week_start)
        return next_state

    next_state["current_interval"] += 1

    if next_state["current_interval"] % (period / 4) == 1:
        next_state["current_season"] += 1

    if next_state["current_interval"] > period:
        next_state["current_interval"] = 1
        next_state["current_year"] += 1
        next_state["current_season"] = 0

    next_state["current_year_interval"] = add_interval(state["current_year_interval"])
    return next_state
