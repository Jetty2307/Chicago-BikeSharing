import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

today = datetime.today().strftime("%Y_%m_%d")

ENGINE_URL = os.getenv(
    "DATABASE_URL"
)

if not ENGINE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(ENGINE_URL)

STAGING_SCHEMA = os.getenv("DBT_STAGING_SCHEMA", "staging")

FIRST_LAST_DAY = f"""
SELECT
    to_char(min(started_at), 'YYYY-MM-DD') AS first_day,
    to_char(max(started_at), 'YYYY-MM-DD') AS last_day
FROM {STAGING_SCHEMA}.stg_divvy_rides
"""

with engine.connect() as conn:
    values = pd.read_sql(FIRST_LAST_DAY, conn)
    start_date = values['first_day'][0]
    first_date = values['last_day'][0]


url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 41.8781, # Chicago's coordinates
    "longitude": -87.6298,
    "start_date": start_date,
    "end_date": first_date,
    "daily": [
        "temperature_2m_mean", # 2 meters above sea level
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

# print(data["daily"].keys())
# print(data["daily"]["time"][:3])
# print(data["daily"]["temperature_2m_mean"][:3])
# print(data["daily"]["rain_sum"][:3])
# print(data["daily"]["snowfall_sum"][:3])
# print(data["daily"]["weather_code"][:3])

# print(df_weather.head())