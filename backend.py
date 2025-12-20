#backend.py

# from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from models import train_all_models, trained_models, training_status
import logging
import os

from datetime import datetime
from dateutil.relativedelta import relativedelta


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

# Define the request model

app = FastAPI()

try:
    # Load the data from /Users/victor/Desktop/DS/Chicago-BikeSharing/
    file_week = "df_week_test_sql.tsv"
    file_month = "df_month_test_sql.tsv"
    logger.debug(f"Looking for files")
    if not os.path.exists(file_week) or not os.path.exists(file_month):
        raise HTTPException(status_code=404, detail="Data file not found.")

    df_week = pd.read_csv(file_week, sep='\t', index_col=False).reset_index(drop=True)
    df_month = pd.read_csv(file_month, sep='\t', index_col=False).reset_index(drop=True)

    if 'Unnamed: 0' in df_week.columns:  # Remove unintended index column
        df_week = df_week.drop(columns=['Unnamed: 0'])

    if 'Unnamed: 0' in df_month.columns:  # Remove unintended index column
        df_month = df_month.drop(columns=['Unnamed: 0'])

    if df_week.empty or df_month.empty:
        raise HTTPException(status_code=400, detail="No data found in the file.")


except HTTPException as e:
    logger.error(f"HTTP Exception: {e.detail}")
    raise e
except Exception as e:
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="Internal Server Error")

class ForecastRequest(BaseModel):
    steps: int # Number of months to forecast
    interval: str # weekly or monthly

@app.get("/")
def get_training_status():
    return training_status

'''@app.on_event("startup")
def startup_event():
    print(">>> STARTUP EVENT TRIGGERED <<<")
    logger.debug(f"Training models")
    train_all_models(df_week, df_month)
    print(">>> TRAINING COMPLETED<<<")
    return {"Models training in process"} '''

class interval_prop:
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
    def make_prediction(self, steps, interval, grid, rideable_type):

        df = self.dataframe[self.dataframe.rideable_type == rideable_type].reset_index(drop=True)

        last_2row = df.iloc[-2]
        last_row = df.iloc[-1]
        forecast_data = []

        current_interval = last_row[interval] + 1

        if current_interval % (self.period/4) == 1:
            current_season = last_row['season'] + 1
        else:
            current_season = last_row['season']

        if current_interval > self.period:  # Increment the year if the month exceeds 12
            current_interval = 1
            current_year = last_row['year'] + 1
            current_season = 0
        else:
            current_year = last_row['year']

        current_ride_id_count = last_2row['rides']  # Use as starting value for ride_id_count_lastmonth
        current_ride_id_count_plusinterval = last_row['rides']

        current_year_interval = self.add_interval(last_row[f'year_{interval}'])

        for _ in range(steps):
            # Prepare the input features for prediction
            input_features = pd.DataFrame({
                'rideable_type': [df['rideable_type'][0]],
                'year': [current_year],
                interval: [current_interval],
                'season': [current_season],
                f'rides_2{interval}s_ago': [current_ride_id_count],
                f'rides_last{interval}': [current_ride_id_count_plusinterval]
            })

            # Predict the ride_id_count for the current month
            predicted_ride_id_count = grid.predict(input_features)[0]  # Extract the prediction value

            # Append the forecasted data
            forecast_data.append({
                'rideable_type': df['rideable_type'][0],
                'time' : current_year_interval,
                'year': current_year,
                interval : current_interval,
                'season': current_season,
                f'rides_2{interval}s_ago': current_ride_id_count,
                f'rides_last{interval}': current_ride_id_count_plusinterval,
                'rides': predicted_ride_id_count
            })

            # Update the values for the next iteration

            current_ride_id_count = current_ride_id_count_plusinterval
            current_ride_id_count_plusinterval= predicted_ride_id_count  # Use the current prediction for the next month's lastmonth value

            current_interval += 1

            if current_interval % (self.period/4) == 1:
                current_season += 1

            if current_interval > self.period:  # Increment the year if the month exceeds 12
                current_interval = 1
                current_year += 1
                current_season = 0

            current_year_interval = self.add_interval(current_year_interval)

        # Create a DataFrame from the forecasted data
        forecast_df = pd.DataFrame(forecast_data)

        # Display the result

        return forecast_df


month = interval_prop(
    dataframe=df_month,
    period=12,
    offset=relativedelta(months=1),
    date_format="%Y-%m",
    sarima_freq="MS"
)

week = interval_prop(
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

def to_final_pd(d1, path_nonindexed):

    forecast = pd.DataFrame(data=d1)
    return {
        "historical": path_nonindexed.to_dict(orient='records'),
        "forecast": forecast.to_dict(orient='records')  # Convert forecast to list for JSON serialization
    }

def forecast_rides_regressive(request: ForecastRequest, model, timeframe):

    path = timeframe.dataframe.groupby(f'year_{request.interval}')["rides"].sum()
    path_nonindexed = path.reset_index()

    try:
        fulldata = model
    except Exception as e:
        logger.error(f"Error fitting model: {e}")
        raise HTTPException(status_code=500, detail=f"Error fitting model: {str(e)}")

    forecast_1_tot = timeframe.make_prediction(steps=request.steps,
                                                    interval=request.interval, grid=fulldata, rideable_type=1)
    forecast_2_tot = timeframe.make_prediction(steps=request.steps,
                                                    interval=request.interval, grid=fulldata, rideable_type=2)
    forecast_tot = merge_columns(forecast_1_tot, forecast_2_tot, interval=request.interval)
    forecast_tot = forecast_tot[['time','rides']]

    d1 = {f'year_{request.interval}': forecast_tot['time'], 'rides': forecast_tot['rides'].astype(int)}

    return(to_final_pd(d1, path_nonindexed))

def _forecast_with_trained_model(request: ForecastRequest, model_key: str):
    timeframe = interval_mapping.get(request.interval)
    if not timeframe:
        raise HTTPException(status_code=400, detail=f"Invalid interval: {request.interval}")

    if training_status["status"] != "ready":
        raise HTTPException(status_code=503, detail="Model is still training")

    try:
        model = trained_models[request.interval][model_key]
    except KeyError:
        raise HTTPException(
            status_code=500,
            detail=f"Model '{model_key}' for interval '{request.interval}' not found"
        )

    return forecast_rides_regressive(request, model, timeframe)


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

    d1 = {f'year_{request.interval}': forecast.index.strftime(timeframe.date_format), 'rides': forecast.values.round().astype(int)}

    return (to_final_pd(d1, path_nonindexed))

@app.post("/forecast_bikes_xgboost")
def forecast_rides_xgboost(request: ForecastRequest):
    return _forecast_with_trained_model(request, "xgboost")

@app.post("/forecast_bikes_gam")
def forecast_rides_GAM(request: ForecastRequest):
    return _forecast_with_trained_model(request, "GAM")

if __name__ == "__main__":
    print(">>> STARTUP EVENT TRIGGERED <<<")
    train_all_models(df_week, df_month)
    print(">>> TRAINING COMPLETED<<<")
    training_status["status"] = "ready"
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8003, reload=False)
