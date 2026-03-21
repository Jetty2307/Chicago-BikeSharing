#backend.py

# from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from models import train_all_models, trained_models, training_status
from dataframes_loader import load_dataframe
from model_loader import load_xgb, load_gam, training_status
from feature_storage import (
    exogenous_features,
    generate_next_vector,
    get_weather_forecast_df,
    initialize_forecast_state,
    update_forecast_state,
)
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from dateutil.relativedelta import relativedelta


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

# Define the request model

app = FastAPI()
port = int(os.getenv("APP_PORT", 8003))

df_week = load_dataframe(os.environ["WEEK_FILE"])
df_month = load_dataframe(os.environ["MONTH_FILE"])

class ForecastRequest(BaseModel):
    steps: int # Number of months to forecast
    interval: str # weekly or monthly
    include_retro: bool # whether to include retro forecast
    lookback: int # how long back to consider for retro forecast

@app.get("/")
def get_training_status():
    return training_status

@app.on_event("startup")
def load_models():
    global MODELS
    MODELS = {
        'week': {
            "xgboost": load_xgb("week")[0],
            "GAM": load_gam("week")[0]
        },
        'month': {
            "xgboost": load_xgb("month")[0],
            "GAM": load_gam("month")[0]
        }
    }

    global DESCRIPTIONS
    DESCRIPTIONS = {
        'week': {
            "xgboost": load_xgb("week")[1],
            "GAM": load_gam("week")[1]
        },
        'month': {
            "xgboost": load_xgb("month")[1],
            "GAM": load_gam("month")[1]
        }
    }
    global df_forecast_weekly
    df_forecast_weekly = get_weather_forecast_df()

'''@app.on_event("startup")
def startup_event():
    print(">>> STARTUP EVENT TRIGGERED <<<")
    logger.debug(f"Training models")
    train_all_models(df_week, df_month)
    print(">>> TRAINING COMPLETED<<<")
    return {"Models training in process"} '''

class Interval_prop:
    def __init__(self, dataframe, period, offset, date_format, sarima_freq):
        self.dataframe = dataframe
        self.period = period
        self.offset = offset
        self.date_format = date_format
        self.sarima_freq = sarima_freq

    def add_interval(self, date_str):
        date_obj = datetime.strptime(date_str, self.date_format)
        new_date = date_obj + self.offset
        return new_date.strftime(self.date_format)
    def make_prediction(self, steps, interval, regr_model, rideable_type, model_key):

        df = self.dataframe[self.dataframe.rideable_type == rideable_type].reset_index(drop=True)
        forecast_data = []

        # Recursive forecasting starts from the next-step feature state.
        state = initialize_forecast_state(
            df=df,
            interval=interval,
            period=self.period,
            add_interval=self.add_interval,
        )

        for _ in range(steps):
            date_features = generate_next_vector(state=state, interval=interval)
            feature_columns = [
                "rideable_type",
                "year",
                interval,
                "season",
                f"rides_2{interval}s_ago",
                f"rides_last{interval}",
            ] + exogenous_features[interval][model_key]

            if interval == "week":
                weather_features = df_forecast_weekly.loc[
                    df_forecast_weekly["year_week"] == state["current_year_interval"]
                ]
                if weather_features.empty:
                    raise ValueError(f"Weekly weather forecast not found for {state['current_year_interval']}")

                input_features = date_features.merge(weather_features, on="year_week", how="left")
            else:
                input_features = date_features

            input_features = input_features[feature_columns]

            prediction_input = input_features.to_numpy() if model_key == "GAM" else input_features
            predicted_ride_id_count = regr_model.predict(prediction_input)[0]

            forecast_data.append({
                'rideable_type': state["rideable_type"],
                'time' : state["current_year_interval"],
                'year': state["current_year"],
                interval : state["current_interval"],
                'season': state["current_season"],
                f'rides_2{interval}s_ago': state["rides_2ago"],
                f'rides_last{interval}': state["rides_last"],
                'rides': predicted_ride_id_count
            })

            state = update_forecast_state(
                state=state,
                predicted_rides=predicted_ride_id_count,
                interval=interval,
                period=self.period,
                add_interval=self.add_interval,
            )

        return pd.DataFrame(forecast_data)


month = Interval_prop(
    dataframe=df_month,
    period=12,
    offset=relativedelta(months=1),
    date_format="%Y-%m",
    sarima_freq="MS"
)

week = Interval_prop(
    dataframe=df_week,
    period=52,
    offset=relativedelta(weeks=1),
    date_format="%Y-%m-%d",
    sarima_freq="W-MON"
)

interval_mapping = {
    "week": week,
    "month": month}

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

def to_final_pd(d1, path_nonindexed, description):

    forecast = pd.DataFrame(data=d1)
    clean_description = description if description is not None else ""

    return {
        "historical": path_nonindexed.to_dict(orient='records'),
        "forecast": forecast.to_dict(orient='records'),
        "description": clean_description # Convert forecast to list for JSON serialization
    }

def forecast_rides_regressive(request: ForecastRequest, model, timeframe, description, model_key):

    path = timeframe.dataframe.groupby(f'year_{request.interval}')["rides"].sum()
    path_nonindexed = path.reset_index()

    try:
        fulldata = model
    except Exception as e:
        logger.error(f"Error fitting model: {e}")
        raise HTTPException(status_code=500, detail=f"Error fitting model: {str(e)}")

    forecast_1_tot = timeframe.make_prediction(steps=request.steps,
                                                    interval=request.interval, regr_model=fulldata, rideable_type=1, model_key=model_key)
    forecast_2_tot = timeframe.make_prediction(steps=request.steps,
                                                    interval=request.interval, regr_model=fulldata, rideable_type=2, model_key=model_key)
    forecast_tot = merge_columns(forecast_1_tot, forecast_2_tot, interval=request.interval)
    forecast_tot = forecast_tot[['time','rides']]

    d1 = {f'year_{request.interval}': forecast_tot['time'], 'rides': forecast_tot['rides'].astype(int)}

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

    return forecast_rides_regressive(request, model, timeframe, description, model_key)


@app.post("/forecast_bikes_sarima")
def forecast_rides_sarima(request: ForecastRequest):
    # Prepare and fit SARIMAX model
    timeframe = interval_mapping.get(request.interval)
    if not timeframe:
        raise ValueError(f"Invalid interval: {request.interval}")

    path_df = timeframe.dataframe.groupby(f'year_{request.interval}')["rides"].sum().reset_index()

    path_df[f'year_{request.interval}'] = pd.to_datetime(
        path_df[f'year_{request.interval}'],
        format=timeframe.date_format
    )

    path_nonindexed = path_df.copy()
    path_nonindexed[f'year_{request.interval}'] = (
                    path_nonindexed[f'year_{request.interval}']
                    .dt.strftime(timeframe.date_format)
    )

    path = (
        path_df
        .set_index(f'year_{request.interval}')["rides"]
        .asfreq(timeframe.sarima_freq)
    )

    y = np.log1p(path)

    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 0, 1, timeframe.period),
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    results = model.fit(disp=False)


    # Forecast future rides
    try:
        forecast_log = results.get_forecast(steps=request.steps).predicted_mean
        forecast = np.expm1(forecast_log).clip(lower=0)
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail="Error generating forecast.")

    d1 = {f'year_{request.interval}': forecast.index.strftime(timeframe.date_format),
          'rides': forecast.values.round().astype(int)}

    return (to_final_pd(d1, path_nonindexed, description = None))

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
