import pandas as pd
import requests


def load_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 41.8781,
        "longitude": -87.6298,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
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
    return df_weather


def load_forecast_weather(forecast_days: int) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 41.8781,
        "longitude": -87.6298,
        "daily": [
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "weather_code",
        ],
        "forecast_days": forecast_days,
        "timezone": "America/Chicago",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    df_weather = pd.DataFrame(data["daily"])
    df_weather = df_weather.rename(columns={"time": "weather_date"})
    df_weather["weather_date"] = pd.to_datetime(df_weather["weather_date"]).dt.date
    return df_weather
