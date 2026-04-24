#backend.py

# from typing import Annotated
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
# from models import train_all_models, trained_models, training_status
from dataframes_loader import load_dataframe
from intervals import build_interval_mapping
from model_loader import load_xgb, load_gam, load_sarima, training_status
from sarima_service import build_sarima_series, forecast_with_fitted_sarima
from features_inference import (
    generate_next_vector,
    get_weekly_weather_forecast_df,
    initialize_forecast_state,
    update_forecast_state,
)

import pandas as pd
import logging
import os

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the request model

app = FastAPI()
port = int(os.getenv("APP_PORT", 8003))

df_day = load_dataframe(os.environ["DAY_FILE"])
df_week = load_dataframe(os.environ["WEEK_FILE"])
df_month = load_dataframe(os.environ["MONTH_FILE"])

class ForecastRequest(BaseModel):
    steps: int # Number of months to forecast
    interval: str # weekly or monthly
    include_retro: bool # whether to include retro forecast
    lookback: int # how long back to consider for retro forecast
    forecast_date: Optional[str] = None # used for daily pseudo-forecast

@app.get("/")
def get_training_status():
    return training_status

@app.on_event("startup")
def load_models():
    global MODELS
    MODELS = {
        'day': {
            "xgboost": load_xgb("day")[0],
            "GAM": load_gam("day")[0],
            "sarima": None,
        },
        'week': {
            "xgboost": load_xgb("week")[0],
            "GAM": load_gam("week")[0],
            "sarima": load_sarima("week")[0],
        },
        'month': {
            "xgboost": load_xgb("month")[0],
            "GAM": load_gam("month")[0],
            "sarima": load_sarima("month")[0],
        }
    }

    global DESCRIPTIONS
    DESCRIPTIONS = {
        'day': {
            "xgboost": load_xgb("day")[1],
            "GAM": load_gam("day")[1],
            "sarima": None,
        },
        'week': {
            "xgboost": load_xgb("week")[1],
            "GAM": load_gam("week")[1],
            "sarima": load_sarima("week")[1],
        },
        'month': {
            "xgboost": load_xgb("month")[1],
            "GAM": load_gam("month")[1],
            "sarima": load_sarima("month")[1],
        }
    }
    global df_forecast_weekly
    df_forecast_weekly = get_weekly_weather_forecast_df()

'''@app.on_event("startup")
def startup_event():
    print(">>> STARTUP EVENT TRIGGERED <<<")
    logger.debug(f"Training models")
    train_all_models(df_week, df_month)
    print(">>> TRAINING COMPLETED<<<")
    return {"Models training in process"} '''


def make_prediction(timeframe, steps, regr_model, rideable_type, model_key):
    df = timeframe.dataframe[timeframe.dataframe.rideable_type == rideable_type].reset_index(drop=True)
    forecast_data = []

    state = initialize_forecast_state(
        df=df,
        interval=timeframe.name,
        period=timeframe.period,
        add_interval=timeframe.add_interval,
    )

    for _ in range(steps):
        date_features = generate_next_vector(state=state, interval=timeframe.name)
        input_features = date_features

        if timeframe.uses_weather:
            weather_features = df_forecast_weekly.loc[
                df_forecast_weekly["year_week"] == state["current_year_interval"]
            ]
            if weather_features.empty:
                raise ValueError(f"Weekly weather forecast not found for {state['current_year_interval']}")
            input_features = date_features.merge(weather_features, on="year_week", how="left")

        input_features = input_features[timeframe.feature_columns(model_key)]
        predicted_ride_id_count = regr_model.predict(input_features)[0]

        forecast_data.append({
            'rideable_type': state["rideable_type"],
            'time': state["current_year_interval"],
            'year': state["current_year"],
            timeframe.name: state["current_interval"],
            'season': state["current_season"],
            f'rides_2{timeframe.name}s_ago': state["rides_2ago"],
            f'rides_last{timeframe.name}': state["rides_last"],
            'rides': predicted_ride_id_count
        })

        state = update_forecast_state(
            state=state,
            predicted_rides=predicted_ride_id_count,
            interval=timeframe.name,
            period=timeframe.period,
            add_interval=timeframe.add_interval,
        )

    return pd.DataFrame(forecast_data)


interval_mapping = build_interval_mapping({
    "day": df_day,
    "week": df_week,
    "month": df_month,
})


def get_daily_forecast_dates():
    timeframe = interval_mapping["day"]
    unique_days = sorted(
        pd.to_datetime(timeframe.dataframe["year_day"]).dt.strftime(timeframe.date_format).unique()
    )
    if not unique_days:
        return []

    window_size = min(timeframe.validation_offset, len(unique_days))
    candidate_days = unique_days[-window_size:]
    available_set = set(unique_days)
    return [
        day for day in candidate_days
        if (pd.Timestamp(day) - timeframe.offset).strftime(timeframe.date_format) in available_set
    ]

def merge_columns(forecast_classic, forecast_electric, interval):

    forecast_full = pd.DataFrame()

    # Iterate through columns
    for col in forecast_classic.columns:
        if col in ['rides', f'rides_last{interval}', f'rides_2{interval}s_ago']:
            # Sum numeric columns
            forecast_full[col] = forecast_classic[col] + forecast_electric[col]
        else:
            forecast_full[col] = forecast_classic[col]

    return forecast_full


@app.get("/daily_forecast_dates")
def daily_forecast_dates():
    return {"dates": get_daily_forecast_dates()}

def to_final_pd(d1, path_nonindexed, description):

    forecast = pd.DataFrame(data=d1)
    clean_description = description if description is not None else ""

    return {
        "historical": path_nonindexed.to_dict(orient='records'),
        "forecast": forecast.to_dict(orient='records'),
        "description": clean_description # Convert forecast to list for JSON serialization
    }

def forecast_rides_daily_pseudo(request: ForecastRequest, model, timeframe, description, model_key):
    if not request.forecast_date:
        raise HTTPException(status_code=400, detail="forecast_date is required for daily pseudo-forecast")
    if request.steps != 1:
        raise HTTPException(status_code=400, detail="Daily pseudo-forecast supports only steps=1")

    try:
        target_day = pd.Timestamp(request.forecast_date).strftime(timeframe.date_format)
    except ValueError:
        raise HTTPException(status_code=400, detail="forecast_date must use YYYY-MM-DD format")

    if target_day not in get_daily_forecast_dates():
        raise HTTPException(status_code=400, detail=f"forecast_date is outside the allowed pseudo-forecast window: {target_day}")

    target_rows = timeframe.dataframe[timeframe.dataframe["year_day"] == target_day]

    if target_rows.empty:
        raise HTTPException(status_code=400, detail=f"forecast date not found in daily data: {target_day}")

    forecast_parts = []
    for rideable_type in sorted(target_rows["rideable_type"].unique()):

        target_for_type = target_rows[target_rows["rideable_type"] == rideable_type]
        if target_for_type.empty:
            raise HTTPException(
                status_code=400,
                detail=f"target date {target_day} has no row for rideable_type={rideable_type}",
            )

        input_features = target_for_type.iloc[[0]].copy()

        feature_frame = input_features[timeframe.feature_columns(model_key)]
        model_input = feature_frame.to_numpy() if model_key == "GAM" else feature_frame
        predicted_rides = model.predict(model_input)[0]

        forecast_parts.append({
            "rideable_type": rideable_type,
            "rides": predicted_rides,
        })

    predicted_total = int(round(sum(part["rides"] for part in forecast_parts)))
    actual_total = int(target_rows["rides"].sum())
    historical = (
        timeframe.dataframe
        .groupby("year_day")["rides"]
        .sum()
        .reset_index()
    )

    clean_description = description if description is not None else ""
    return {
        "historical": historical.to_dict(orient="records"),
        "forecast": [{"year_day": target_day, "rides": predicted_total}],
        "actual": [{"year_day": target_day, "rides": actual_total}],
        "description": clean_description,
    }

def forecast_rides_regressive(request: ForecastRequest, model, timeframe, description, model_key):

    path = timeframe.dataframe.groupby(f'year_{timeframe.name}')["rides"].sum()
    path_nonindexed = path.reset_index()

    try:
        fulldata = model
    except Exception as e:
        logger.error(f"Error fitting model: {e}")
        raise HTTPException(status_code=500, detail=f"Error fitting model: {str(e)}")

    forecast_1_tot = make_prediction(steps=request.steps, timeframe=timeframe,
                                     regr_model=fulldata, rideable_type=1, model_key=model_key)
    forecast_2_tot = make_prediction(steps=request.steps, timeframe=timeframe,
                                     regr_model=fulldata, rideable_type=2, model_key=model_key)
    forecast_tot = merge_columns(forecast_1_tot, forecast_2_tot, interval=timeframe.name)
    forecast_tot = forecast_tot[['time','rides']]

    d1 = {f'year_{timeframe.name}': forecast_tot['time'], 'rides': forecast_tot['rides'].astype(int)}

    return(to_final_pd(d1, path_nonindexed, description))

def _forecast_with_trained_model(request: ForecastRequest, model_key: str):
    timeframe = interval_mapping.get(request.interval)
    if not timeframe:
        raise HTTPException(status_code=400, detail=f"Invalid interval: {request.interval}")

    if training_status["status"] != "ready":
        raise HTTPException(status_code=503, detail="Model is still training")

    try:
        model = MODELS[request.interval][model_key]
    except KeyError:
        raise HTTPException(
            status_code=500,
            detail=f"Model '{model_key}' for interval '{request.interval}' not found"
        )

    try:
        description = DESCRIPTIONS[request.interval][model_key]
    except:
        description = "No performance description available"

    if request.interval == "day":
        return forecast_rides_daily_pseudo(request, model, timeframe, description, model_key)

    return forecast_rides_regressive(request, model, timeframe, description, model_key)


@app.post("/forecast_bikes_sarima")
def forecast_rides_sarima(request: ForecastRequest):
    timeframe = interval_mapping.get(request.interval)
    if not timeframe:
        raise HTTPException(status_code=400, detail=f"Invalid interval: {request.interval}")

    if training_status["status"] != "ready":
        raise HTTPException(status_code=503, detail="Model is still training")

    try:
        model = MODELS[request.interval]["sarima"]
        if model is None:
            raise HTTPException(status_code=400, detail=f"SARIMA is not available for interval: {request.interval}")
        description = DESCRIPTIONS[request.interval]["sarima"]
        historical, _ = build_sarima_series(
            dataframe=timeframe.dataframe,
            interval=request.interval,
            timeframe=timeframe,
        )
        forecast = forecast_with_fitted_sarima(model, steps=request.steps)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail="Error generating forecast.")

    d1 = {
        f'year_{request.interval}': forecast.index.strftime(timeframe.date_format),
        'rides': forecast.values.round().astype(int)
    }

    return to_final_pd(d1, historical, description=description)

@app.post("/forecast_bikes_xgboost")
def forecast_rides_xgboost(request: ForecastRequest):
    return _forecast_with_trained_model(request, "xgboost")

@app.post("/forecast_bikes_gam")
def forecast_rides_GAM(request: ForecastRequest):
    return _forecast_with_trained_model(request, "GAM")

if __name__ == "__main__":
    training_status["status"] = "ready"
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=False)
