import os
import pandas as pd
from datetime import datetime
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

today = datetime.today().strftime("%Y_%m_%d")

from sqlalchemy import create_engine


ENGINE_URL = os.getenv(
    "DATABASE_URL"
)

if not ENGINE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(ENGINE_URL)

MART_SCHEMA = os.getenv("DBT_MART_SCHEMA", "analytics")

DAY_QUERY = f"""
SELECT
    year_day,
    rideable_type,
    year,
    month,
    season,
    day_of_year,
    day_of_week,
    is_weekend,
    rides,
    rides_lastday,
    temp,
    total_rain,
    total_snow,
    is_snow
FROM {MART_SCHEMA}.mart_rides_daily
ORDER BY year_day, rideable_type
"""

WEEK_QUERY = f"""
SELECT
    year_week,
    rideable_type,
    year,
    week,
    season,
    rides,
    rides_lastweek,
    rides_2weeks_ago,
    max_temp,
    avg_temp,
    min_temp,
    total_rain,
    total_snow
FROM {MART_SCHEMA}.mart_rides_weekly
ORDER BY year_week, rideable_type
"""

MONTH_QUERY = f"""
SELECT
    year_month,
    rideable_type,
    year,
    month,
    season,
    rides,
    rides_lastmonth,
    rides_2months_ago
FROM {MART_SCHEMA}.mart_rides_monthly
ORDER BY year, month, rideable_type
"""

def fetch_df(query: str, engine=engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def build_interval_table() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_day = fetch_df(DAY_QUERY)
    df_week = fetch_df(WEEK_QUERY)
    df_month = fetch_df(MONTH_QUERY)

    return df_day, df_week, df_month


def validate_freshness(frames, latest_rides: pd.DataFrame) -> None:
    """Reject stale marts before overwriting any exported files."""
    if latest_rides.empty:
        raise ValueError("No supported rides found in the source table")
    for row in latest_rides.itertuples(index=False):
        latest = pd.Timestamp(row.latest_day)
        expected = (latest.normalize(), latest.to_period("W").start_time,
                    latest.to_period("M").start_time)
        for df, column, period, label in zip(
            frames, ("year_day", "year_week", "year_month"), expected,
            ("daily", "weekly", "monthly"),
        ):
            actual = pd.to_datetime(df.loc[df["rideable_type"] == row.rideable_type, column]).max()
            if pd.isna(actual) or actual != period:
                raise ValueError(
                    f"Stale {label} mart for rideable_type={row.rideable_type}: "
                    f"latest={actual}, expected={period}. Run dbt run and dbt test before export."
                )


LATEST_RIDES_QUERY = f"""
SELECT CASE rideable_type WHEN 'classic_bike' THEN 1 ELSE 2 END AS rideable_type,
       max(started_at)::date AS latest_day
FROM {os.getenv('DBT_RAW_SCHEMA', 'public')}.merged
WHERE rideable_type IN ('classic_bike', 'electric_bike')
GROUP BY rideable_type
"""


def export_tsv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    df_day, df_week, df_month = build_interval_table()
    validate_freshness((df_day, df_week, df_month), fetch_df(LATEST_RIDES_QUERY))

    export_tsv(df_day, os.environ["DAY_FILE"])
    export_tsv(df_week, os.environ["WEEK_FILE"])
    export_tsv(df_month, os.environ["MONTH_FILE"])
