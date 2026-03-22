import pandas as pd
import numpy as np
import os
# from typing import Any, Tuple
from typing import Any, Tuple, List, Dict
from dotenv import load_dotenv
from weather_api import load_forecast_weather, load_historical_weather

load_dotenv()

today = pd.Timestamp.today(tz="America/Chicago").tz_localize(None).normalize()

model_features = {
    "week": {
        "xgboost": ["rideable_type", "year", "week", "season", "rides_2weeks_ago", "rides_lastweek",
                    "max_temp", "avg_temp", "min_temp", "total_rain", "total_snow"],
        "GAM": ["rideable_type", "year", "week", "season", "avg_temp", "total_rain", "total_snow"],
    },
    "month": {
        "xgboost": ["rideable_type", "year", "month", "season", "rides_2months_ago", "rides_lastmonth"],
        "GAM": ["rideable_type", "year", "month", "season"],
    },
}

def _clean_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().replace([np.inf, -np.inf], np.nan).dropna()


def get_weather_forecast_df() -> pd.DataFrame:
    week_file = os.environ["WEEK_FILE"]
    df_week = pd.read_csv(week_file, sep="\t")
    start_date = pd.to_datetime(df_week["year_week"]).min().strftime("%Y-%m-%d")
    end_date = pd.to_datetime(df_week["year_week"]).max().strftime("%Y-%m-%d")
    df_weather = load_historical_weather(start_date, end_date)

    end_ts = pd.Timestamp(end_date).normalize()

    if end_ts > today:
        raise ValueError(f"Invalid end date: end date in the future {end_date}")

    daily_parts = []

    if end_ts < today:
        daily_parts.append(
            load_historical_weather(
                (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            )
        )

    daily_parts.append(load_forecast_weather(14))
    df_forecast = pd.concat(daily_parts, axis=0, ignore_index=True)

    df_forecast["weather_date"] = pd.to_datetime(df_forecast["weather_date"])
    df_forecast["year_week"] = (
        df_forecast["weather_date"]
        - pd.to_timedelta(df_forecast["weather_date"].dt.weekday, unit="D")
    ).dt.strftime("%Y-%m-%d")

    df_forecast_weekly = (
        df_forecast.groupby("year_week", as_index=False)
        .agg(
            max_temp=("temperature_2m_mean", "max"),
            avg_temp=("temperature_2m_mean", "mean"),
            min_temp=("temperature_2m_mean", "min"),
            total_rain=("rain_sum", "sum"),
            total_snow=("snowfall_sum", "sum"),
        )
    )

    df_weather_copy = df_weather.copy()
    df_weather_copy["weather_date"] = pd.to_datetime(df_weather_copy["weather_date"])
    df_weather_copy["year_week"] = (
        df_weather_copy["weather_date"]
        - pd.to_timedelta(df_weather_copy["weather_date"].dt.weekday, unit="D")
    ).dt.strftime("%Y-%m-%d")

    weekly_avg = (
        df_weather_copy.groupby("year_week", as_index=False)
        .agg(
            max_temp=("temperature_2m_mean", "max"),
            avg_temp=("temperature_2m_mean", "mean"),
            min_temp=("temperature_2m_mean", "min"),
            total_rain=("rain_sum", "sum"),
            total_snow=("snowfall_sum", "sum"),
        )
    )
    weekly_avg["week"] = pd.to_datetime(weekly_avg["year_week"]).dt.isocalendar().week.astype(int)
    weekly_avg = weekly_avg.groupby("week", as_index=False)[
        ["max_temp", "avg_temp", "min_temp", "total_rain", "total_snow"]
    ].mean()

    future_weeks = pd.date_range(
        end_ts - pd.Timedelta(days=end_ts.weekday()) + pd.Timedelta(days=7),
        periods=104,
        freq="7D",
    )

    full_weeks = pd.DataFrame({"year_week": future_weeks.strftime("%Y-%m-%d")})
    full_weeks["week"] = pd.to_datetime(full_weeks["year_week"]).dt.isocalendar().week.astype(int)

    df_forecast_weekly = full_weeks.merge(df_forecast_weekly, on="year_week", how="left")
    df_forecast_weekly = df_forecast_weekly.merge(weekly_avg, on="week", how="left", suffixes=("", "_avg"))

    for col in ["max_temp", "avg_temp", "min_temp", "total_rain", "total_snow"]:
        df_forecast_weekly[col] = df_forecast_weekly[col].fillna(df_forecast_weekly[f"{col}_avg"])
        df_forecast_weekly[col] = df_forecast_weekly[col].ffill()
        df_forecast_weekly[col] = df_forecast_weekly[col].fillna(weekly_avg[col].mean())

    return df_forecast_weekly[["year_week", "max_temp", "avg_temp", "min_temp", "total_rain", "total_snow"]]


def fetch_features_xgboost(df: pd.DataFrame, interval: str, model_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = _clean_training_frame(df)
    y = df["rides"]
    feature_names = model_features[interval][model_name]
    X = df[feature_names]
    return X, y, feature_names


def fetch_features_gam(df: pd.DataFrame, interval: str, model_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = _clean_training_frame(df)

    # order matters for GAM since we need to align with the model's expected feature order

    feature_names = model_features[interval][model_name]
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
    # rideable type fixed

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
        f"year_{interval}": [state["current_year_interval"]],
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


def attach_weekly_weather_features(
    forecast_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    model: str,
    interval: str,
) -> pd.DataFrame:

    weather_cols = [column for column in model_features[interval][model] if column in weather_df.columns]

    weather_week = (
        weather_df[["year_week"] + weather_cols]
        .drop_duplicates(subset=["year_week"])
        .copy()
    )

    return forecast_df.merge(
        weather_week,
        left_on="time",
        right_on="year_week",
        how="left",
    ).drop(columns=["year_week"])
