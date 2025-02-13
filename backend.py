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
#from xgboost import XGBClassifier, XGBRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#from sklearn.model_selection import train_test_split, GridSearchCV
#from xgboost import XGBClassifier, XGBRegressor

import pandas as pd
import numpy as np

app = FastAPI()

class CustomXGBRegressor(XGBRegressor, BaseEstimator):
    def __sklearn_tags__(self):
        """Manually define sklearn tags to avoid attribute errors."""
        return {
            "non_deterministic": True,
            "requires_positive_X": False,
            "requires_positive_y": False,
            "X_types": ["2darray"],
            "poor_score": False,
            "no_validation": False
        }

try:
    # Load the data from /Users/victor/Desktop/DS/Chicago-BikeSharing/
    file_weekly = "total_rides_weekly.tsv"
    file_monthly = "total_rides_monthly.tsv"
    file_path2 = "df_full.tsv"
    logger.debug(f"Looking for files")
    if not os.path.exists(file_weekly) or not os.path.exists(file_monthly) or not os.path.exists(file_path2):
        raise HTTPException(status_code=404, detail="Data file not found.")

    total_rides_weekly = pd.read_csv(file_weekly, sep='\t')
    total_rides_monthly = pd.read_csv(file_monthly, sep='\t')
    df_full = pd.read_csv(file_path2, sep='\t', index_col=False)
    df_full = df_full.reset_index(drop=True)
    if 'Unnamed: 0' in df_full.columns:  # Remove unintended index column
        df_full = df_full.drop(columns=['Unnamed: 0'])
    # logger.debug(f"df_full columns: {df_full.columns}")

    if total_rides_weekly.empty or total_rides_monthly.empty or df_full.empty:
        raise HTTPException(status_code=400, detail="No data found in the file.")

    #total_rides = total_rides[['year_month', 'rides']]
    #total_rides_indexed = total_rides.set_index(['year_month'])

    total_rides_weekly = total_rides_weekly[['week', 'rides']]
    total_rides_weekly_indexed = total_rides_weekly.set_index(['week'])

    total_rides_monthly = total_rides_monthly[['month', 'rides']]
    total_rides_monthly_indexed = total_rides_monthly.set_index(['month'])

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

def make_prediction(df, grid, steps):
    df = df.reset_index(drop=True)
    last_2row = df.iloc[-2]
    last_row = df.iloc[-1]
    # Initialize the new DataFrame for predictions
    forecast_data = []

    # Extract initial values

    # Generate predictions for the next 12 months

    current_month = last_row['month'] + 1

    if current_month % 3 == 1:
        current_season = last_row['season'] + 1
    else:
        current_season = last_row['season']

    if current_month > 12:  # Increment the year if the month exceeds 12
        current_month = 1
        current_year = last_row['year'] + 1
        current_season = 0
    else:
        current_year = last_row['year']

    current_ride_id_count = last_2row['ride_id_count']  # Use as starting value for ride_id_count_lastmonth
    current_ride_id_count_plusmonth = last_row['ride_id_count']
    for _ in range(steps):
        # Prepare the input features for prediction
        input_features = pd.DataFrame({
            'rideable_type': [df['rideable_type'][0]],
            'year': [current_year],
            'month': [current_month],
            'season': [current_season],
            'ride_id_count_2month_ago': [current_ride_id_count],
            'ride_id_count_lastmonth': [current_ride_id_count_plusmonth]
        })

        # Predict the ride_id_count for the current month
        predicted_ride_id_count = grid.predict(input_features)[0]  # Extract the prediction value

        # Append the forecasted data
        forecast_data.append({
            'rideable_type': df['rideable_type'][0],
            'year': current_year,
            'month': current_month,
            'season': current_season,
            'ride_id_count_2month_ago': current_ride_id_count,
            'ride_id_count_lastmonth': current_ride_id_count_plusmonth,
            'ride_id_count': predicted_ride_id_count
        })

        # Update the values for the next iteration
        # rideable_type = 1 if rideable_type == 2 else 2
        current_ride_id_count = current_ride_id_count_plusmonth
        current_ride_id_count_plusmonth = predicted_ride_id_count  # Use the current prediction for the next month's lastmonth value

        current_month += 1

        if current_month % 3 == 1:
            current_season += 1

        if current_month > 12:  # Increment the year if the month exceeds 12
            current_month = 1
            current_year += 1
            current_season = 0

    # Create a DataFrame from the forecasted data
    forecast_df = pd.DataFrame(forecast_data)

    # Display the result

    return forecast_df

def make_time_column(forecast):
    forecast['time'] = forecast['year'].astype(int).astype(str) + '-' + forecast['month'].astype(int).astype(str).str.zfill(2)
def merge_columns(forecast_classic, forecast_electric):
    make_time_column(forecast_classic)
    make_time_column(forecast_electric)

    forecast_full = pd.DataFrame()

    # Iterate through columns
    for col in forecast_classic.columns:
        if col in ['ride_id_count', 'ride_id_count_lastmonth', 'ride_id_count_2month_ago']:
            # Sum numeric columns

            forecast_full[col] = forecast_classic[col] + forecast_electric[col]
        else:
            forecast_full[col] = forecast_classic[col]

    return forecast_full

def fit_xgboost(df):
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Columns: {df.columns}")
    y = df['ride_id_count']
    #X = pd.get_dummies(df.drop(columns=['year_month', 'ride_id_count']), drop_first=True)
    X = df.drop(columns=['year_month', 'ride_id_count'], axis=1)
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


    grid = GridSearchCV(CustomXGBRegressor(base_score=base_score, random_state=1, objective='count:poisson'), param_grid, refit=True, n_jobs=-1)
    grid.fit(X_train, y_train)
    #print(grid.best_params_)
    #grid_pred = grid.predict(X_valid)
    #print("Mean absolute error: %s" % mean_absolute_error(y_valid, grid_pred))
    #print("RMSE: %s" % math.sqrt(mean_squared_error(y_valid, grid_pred)))
    #print("R2 score: %s" % r2_score(y_valid, grid_pred))
    return grid

def fit_GAM(df):
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Columns: {df.columns}")
    y = df['ride_id_count']
    X = df.drop(columns=['year_month', 'ride_id_count'], axis=1)
    logger.debug(f"Target (y) shape: {y.shape}, Features (X) shape: {X.shape}")
    logger.debug(f"Feature preview:\n{X.head()}")
    model = GAM().fit(X, y)
    return model

def make_prediction(df, grid, steps):
    df = df.reset_index(drop=True)
    last_2row = df.iloc[-2]
    last_row = df.iloc[-1]
    # Initialize the new DataFrame for predictions
    forecast_data = []

    # Extract initial values
    # Generate predictions for the next 12 months

    current_month = last_row['month'] + 1

    if current_month % 3 == 1:
        current_season = last_row['season'] + 1
    else:
        current_season = last_row['season']

    if current_month > 12:  # Increment the year if the month exceeds 12
        current_month = 1
        current_year = last_row['year'] + 1
        current_season = 0
    else:
        current_year = last_row['year']

    current_ride_id_count = last_2row['ride_id_count']  # Use as starting value for ride_id_count_lastmonth
    current_ride_id_count_plusmonth = last_row['ride_id_count']
    for _ in range(steps):
        # Prepare the input features for prediction
        input_features = pd.DataFrame({
            'rideable_type': [df['rideable_type'][0]],
            'year': [current_year],
            'month': [current_month],
            'season': [current_season],
            'ride_id_count_2month_ago': [current_ride_id_count],
            'ride_id_count_lastmonth': [current_ride_id_count_plusmonth]
        })

        # Predict the ride_id_count for the current month
        predicted_ride_id_count = grid.predict(input_features)[0]  # Extract the prediction value

        # Append the forecasted data
        forecast_data.append({
            'rideable_type': df['rideable_type'][0],
            'year': current_year,
            'month': current_month,
            'season': current_season,
            'ride_id_count_2month_ago': current_ride_id_count,
            'ride_id_count_lastmonth': current_ride_id_count_plusmonth,
            'ride_id_count': predicted_ride_id_count
        })

        # Update the values for the next iteration
        # rideable_type = 1 if rideable_type == 2 else 2
        current_ride_id_count = current_ride_id_count_plusmonth
        current_ride_id_count_plusmonth = predicted_ride_id_count  # Use the current prediction for the next month's lastmonth value

        current_month += 1

        if current_month % 3 == 1:
            current_season += 1

        if current_month > 12:  # Increment the year if the month exceeds 12
            current_month = 1
            current_year += 1
            current_season = 0

    # Create a DataFrame from the forecasted data
    forecast_df = pd.DataFrame(forecast_data)

    # Display the result

    return forecast_df

def make_time_column(forecast):
    forecast['time'] = forecast['year'].astype(int).astype(str) + '-' + forecast['month'].astype(int).astype(str).str.zfill(2)

def merge_columns(forecast_classic, forecast_electric):
    make_time_column(forecast_classic)
    make_time_column(forecast_electric)

    forecast_full = pd.DataFrame()

    # Iterate through columns
    for col in forecast_classic.columns:
        if col in ['ride_id_count', 'ride_id_count_lastmonth', 'ride_id_count_2month_ago']:
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

@app.post("/forecast_bikes_sarima")
def forecast_rides_sarima(request: ForecastRequest):
    # Prepare and fit SARIMAX model
    if request.interval == "week":
        path = total_rides_weekly_indexed
        path_nonindexed = total_rides_weekly
        model = SARIMAX(path['rides'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 52))
    elif request.interval == "month":
        path = total_rides_monthly_indexed
        path_nonindexed = total_rides_monthly
        model = SARIMAX(path['rides'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

    results = model.fit(disp=False)

    # Forecast future rides
    try:
        predicted = results.get_forecast(steps=request.steps)
        forecasted = predicted.predicted_mean
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail="Error generating forecast.")

    forecasted.index = pd.to_datetime(forecasted.index)  # Convert to DateTimeIndex
    d1 = {request.interval: forecasted.index.astype(str), 'rides': forecasted.values.astype(int)}

    return (to_final_pd(d1, path_nonindexed))

@app.post("/forecast_bikes_xgboost")
def forecast_rides_xgboost(request: ForecastRequest):
    try:
        fulldata_xgboost = fit_xgboost(df_full)
    except Exception as e:
        logger.error(f"Error fitting XGBoost model: {e}")
        raise HTTPException(status_code=500, detail=f"Error fitting XGBoost model: {str(e)}")

    forecast_1_xgboost = make_prediction(df_full[df_full.rideable_type == 1], fulldata_xgboost, steps=request.steps)
    forecast_2_xgboost = make_prediction(df_full[df_full.rideable_type == 2], fulldata_xgboost, steps=request.steps)
    forecast_xgboost = merge_columns(forecast_1_xgboost, forecast_2_xgboost)
    forecast_xgboost = forecast_xgboost[['time','ride_id_count']]

    d1 = {request.interval: forecast_xgboost['time'], 'rides': forecast_xgboost['ride_id_count'].astype(int)}

    return(to_final_pd(d1))

@app.post("/forecast_bikes_gam")
def forecast_rides_gam(request: ForecastRequest):
    try:
        fulldata_GAM = fit_GAM(df_full)
    except Exception as e:
        logger.error(f"Error fitting GAM model: {e}")
        raise HTTPException(status_code=500, detail=f"Error fitting GAM model: {str(e)}")

    forecast_1_GAM = make_prediction(df_full[df_full.rideable_type == 1], fulldata_GAM, steps=request.steps)
    forecast_2_GAM = make_prediction(df_full[df_full.rideable_type == 2], fulldata_GAM, steps=request.steps)
    forecast_GAM = merge_columns(forecast_1_GAM, forecast_2_GAM)
    forecast_GAM = forecast_GAM[['time','ride_id_count']]

    d1 = {request.interval: forecast_GAM['time'], 'rides': forecast_GAM['ride_id_count'].astype(int)}
    path_nonindexed = total_rides_monthly

    return(to_final_pd(d1, path_nonindexed))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8003, reload=True)
