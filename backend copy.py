#backend.py

from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
#from sklearn.model_selection import train_test_split, GridSearchCV
#from xgboost import XGBClassifier, XGBRegressor

import pandas as pd

app = FastAPI()

# Define the request model

class ForecastRequest(BaseModel):
    periods: int # Number of months to forecast

@app.get("/")
def root():
    return {"message": "Chicago Bike Sharing forecast API"}

 #@app.post("/files/")
 #async def create_file(file: Annotated[bytes, File()]):
 #   return {"file_size": len(file)}


 #@app.post("/uploadfile/")
 #async def create_upload_file(file: UploadFile):
#    return {"filename": "df.tsv"}


@app.post("/forecast_bikes")
def forecast_rides(request: ForecastRequest):

    # Fetch data from the saved folder
    # Create a new column that represents the 'year-month' combination
        # Load the data

    df = pd.read_csv("df.tsv", sep='\t')
    if df.empty:
        return {"error": "No data found"}

    df['year_month'] = df['started_at'].dt.to_period('M')
    total_rides = df.groupby(['year_month']).size().dropna()
    d = {'year_month': total_rides.index, 'rides': total_rides.values}
    orig_data = pd.DataFrame(data=d)

    # Prepare data for SARIMAX
    model = SARIMAX(orig_data['rides'], order=(1,1,1), seasonal_order=(1,1,1,12))
    results = model.fit(disp=False)

    # Make future predictions
    predicted = results.get_forecast(steps=request.steps)
    forecast = predicted.predicted_mean

    # Return forecast and historical data
    return {
        "historical" : orig_data.to_dict(orient='records'),
        "forecast" : forecast.to_dict(orient='records')
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8001, reload=True)
