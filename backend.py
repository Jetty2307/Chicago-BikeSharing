#backend.py

# from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pygam import GAM, s, f, LogisticGAM
import logging
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
#from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from datetime import datetime
from dateutil.relativedelta import relativedelta


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#from sklearn.model_selection import train_test_split, GridSearchCV
#from xgboost import XGBClassifier, XGBRegressor

import pandas as pd
import numpy as np

app = FastAPI()


try:
    # Load the data from /Users/victor/Desktop/DS/Chicago-BikeSharing/
    file_week = "df_week.tsv"
    file_month = "df_month.tsv"
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
# Define the request model

class ForecastRequest(BaseModel):
    steps: int # Number of months to forecast
    interval: str # weekly or monthly
@app.get("/")
def root():
    return {"message": "Chicago Bike Sharing forecast API"}

class interval_prop:
    def __init__(self, dataframe, period, offset, date_format):
        self.dataframe = dataframe
        self.period = period
        self.offset = offset
        self.date_format = date_format

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
    date_format="%Y-%m"
)

week = interval_prop(
    dataframe=df_week,
    period=52,
    offset=relativedelta(weeks=1),
    date_format="%Y-%m-%d"
)

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

def fit_xgboost(df, interval):


    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Columns: {df.columns}")
    y = df['rides']
    #X = pd.get_dummies(df.drop(columns=['year_month', 'ride_id_count']), drop_first=True)
    # X = df.drop(columns=[f'year_{interval}', 'rides'], axis=1)
    X = df[['rideable_type', 'year', f'{interval}', 'season', f'rides_2{interval}s_ago', f'rides_last{interval}']]
    logger.debug(f"Target (y) shape: {y.shape}, Features (X) shape: {X.shape}")
    logger.debug(f"Feature preview:\n{X.head()}")
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True,
                                                          random_state=1)
    param_grid = {'max_depth': [3, 4, 5],
                  'n_estimators': [50, 100, 200, 300],
                  'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
                  }

    #rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)

    #grid = XGBRegressor(random_state=0, max_depth=3, n_estimators=100, learning_rate=0.3,
    #                          objective='count:poisson')
    base_score = y_train.mean()


    #grid = GridSearchCV(CustomXGBRegressor(base_score=base_score, random_state=1, objective='count:poisson'),
    #                    param_grid, refit=True, n_jobs=-1)

    grid = GridSearchCV(
        XGBRegressor(random_state=1, objective='count:poisson', base_score=base_score),
        param_grid,
        refit=True,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    #print(grid.best_params_)
    #grid_pred = grid.predict(X_valid)
    #print("Mean absolute error: %s" % mean_absolute_error(y_valid, grid_pred))
    #print("RMSE: %s" % math.sqrt(mean_squared_error(y_valid, grid_pred)))
    #print("R2 score: %s" % r2_score(y_valid, grid_pred))
    return grid

def fit_GAM(df, interval):
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Columns: {df.columns}")
    y = df['rides']
    # X = df.drop(columns=[f'year_{interval}', 'rides'], axis=1)
    X = df[['rideable_type', 'year', f'{interval}', 'season', f'rides_2{interval}s_ago', f'rides_last{interval}']]
    logger.debug(f"Target (y) shape: {y.shape}, Features (X) shape: {X.shape}")
    logger.debug(f"Feature preview:\n{X.head()}")
    model = GAM().fit(X, y)
    return model

def to_final_pd(d1, path_nonindexed):

    forecast = pd.DataFrame(data=d1)
    return {
        "historical": path_nonindexed.to_dict(orient='records'),
        "forecast": forecast.to_dict(orient='records')  # Convert forecast to list for JSON serialization
    }

def forecast_rides_regressive(request: ForecastRequest, model):
    if request.interval == "week":
        timeframe = week
    elif request.interval == "month":
        timeframe = month

    df_full = timeframe.dataframe

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


@app.post("/forecast_bikes_sarima")
def forecast_rides_sarima(request: ForecastRequest):
    # Prepare and fit SARIMAX model
    if request.interval == "week":
        timeframe = week

    elif request.interval == "month":
        timeframe = month

    path = timeframe.dataframe.groupby(f'year_{request.interval}')["rides"].sum()
    path_nonindexed = path.reset_index()

    model = SARIMAX(path, order=(1, 1, 1), seasonal_order=(1, 1, 1, timeframe.period))

    results = model.fit(disp=False)

    # Forecast future rides
    try:
        predicted = results.get_forecast(steps=request.steps)
        forecasted = predicted.predicted_mean
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail="Error generating forecast.")

    #forecasted.index = pd.to_datetime(forecasted.index)  # Convert to DateTimeIndex
    d1 = {f'year_{request.interval}': forecasted.index.astype(str), 'rides': forecasted.values.astype(int)}

    return (to_final_pd(d1, path_nonindexed))

@app.post("/forecast_bikes_xgboost")
def forecast_rides_xgboost(request: ForecastRequest):
    if request.interval == "week":
        timeframe = week
    elif request.interval == "month":
        timeframe = month

    model = fit_xgboost(timeframe.dataframe, request.interval)
    return forecast_rides_regressive(request, model)

@app.post("/forecast_bikes_gam")
def forecast_rides_GAM(request: ForecastRequest):
    if request.interval == "week":
        timeframe = week
    elif request.interval == "month":
        timeframe = month

    model = fit_GAM(timeframe.dataframe, request.interval)
    return forecast_rides_regressive(request, model)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8003, reload=True)
