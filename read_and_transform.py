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

WEEK_QUERY = f"""
SELECT
    year_week,
    rideable_type,
    year,
    week,
    season,
    rides,
    rides_lastweek,
    rides_2weeks_ago
FROM {MART_SCHEMA}.mart_rides_weekly
ORDER BY year, week, rideable_type
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


def build_week_month() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_week = fetch_df(WEEK_QUERY)
    df_month = fetch_df(MONTH_QUERY)

    return df_week, df_month


def export_tsv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    df_week, df_month = build_week_month()

    export_tsv(df_week, os.environ["WEEK_FILE"])
    export_tsv(df_month, os.environ["MONTH_FILE"])
