import requests
import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

ENGINE_URL = os.getenv(
    "DATABASE_URL"
)

if not ENGINE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(ENGINE_URL)

STAGING_SCHEMA = os.getenv("DBT_STAGING_SCHEMA", "staging")
RAW_SCHEMA = os.getenv("DBT_RAW_SCHEMA", "public")
WEATHER_TABLE = "weather_daily"

FIRST_LAST_DAY = f"""
SELECT
    to_char(min(started_at), 'YYYY-MM-DD') AS first_day,
    to_char(max(started_at), 'YYYY-MM-DD') AS last_day
FROM {STAGING_SCHEMA}.stg_divvy_rides
"""

with engine.connect() as conn:
    values = pd.read_sql(FIRST_LAST_DAY, conn)
    start_date = values["first_day"][0]
    end_date = values["last_day"][0]


url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 41.8781,  # Chicago's coordinates
    "longitude": -87.6298,
    "start_date": start_date,
    "end_date": end_date,
    "daily": [
        "temperature_2m_mean",  # 2 meters above sea level
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "weather_code",
    ],
    "timezone": "America/Chicago",
}

resp = requests.get(url, params=params, timeout=30)
resp.raise_for_status()
data = resp.json()

df_weather = pd.DataFrame(data["daily"])
df_weather = df_weather.rename(columns={"time": "weather_date"})
df_weather["weather_date"] = pd.to_datetime(df_weather["weather_date"]).dt.date

CREATE_WEATHER_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_SCHEMA}.{WEATHER_TABLE} (
    weather_date DATE PRIMARY KEY,
    temperature_2m_mean DOUBLE PRECISION,
    precipitation_sum DOUBLE PRECISION,
    rain_sum DOUBLE PRECISION,
    snowfall_sum DOUBLE PRECISION,
    weather_code INTEGER
)
"""

INSERT_WEATHER_SQL = f"""
INSERT INTO {RAW_SCHEMA}.{WEATHER_TABLE} (
    weather_date,
    temperature_2m_mean,
    precipitation_sum,
    rain_sum,
    snowfall_sum,
    weather_code
)
VALUES (
    :weather_date,
    :temperature_2m_mean,
    :precipitation_sum,
    :rain_sum,
    :snowfall_sum,
    :weather_code
)
ON CONFLICT (weather_date) DO UPDATE SET
    temperature_2m_mean = EXCLUDED.temperature_2m_mean,
    precipitation_sum = EXCLUDED.precipitation_sum,
    rain_sum = EXCLUDED.rain_sum,
    snowfall_sum = EXCLUDED.snowfall_sum,
    weather_code = EXCLUDED.weather_code
"""

with engine.begin() as conn:
    conn.execute(text(CREATE_WEATHER_TABLE_SQL))
    conn.execute(text(INSERT_WEATHER_SQL), df_weather.to_dict(orient="records"))

print(df_weather.head())
print(
    f"Loaded {len(df_weather)} rows into {RAW_SCHEMA}.{WEATHER_TABLE} "
    f"for {start_date}..{end_date}"
)
