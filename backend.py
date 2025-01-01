#backend.py

from typing import Annotated

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#from sklearn.model_selection import train_test_split, GridSearchCV
#from xgboost import XGBClassifier, XGBRegressor

import pandas as pd

app = FastAPI()

# Define the request model

class ForecastRequest(BaseModel):
    steps: int # Number of months to forecast

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
    try:
        # Load the data
        #file_path = "total_rides.tsv"
        file_path = "/Users/victor/Desktop/DS/Chicago-BikeSharing/orig_data.tsv"
        logger.debug(f"Looking for file at: {file_path}")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Data file not found.")

        orig_data = pd.read_csv(file_path, sep='\t')
        if orig_data.empty:
            raise HTTPException(status_code=400, detail="No data found in the file.")

        orig_data = orig_data[['year_month','rides']]
        orig_data_indexed = orig_data.set_index(['year_month'])
        # Prepare and fit SARIMAX model
        try:
            model = SARIMAX(orig_data_indexed['rides'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            results = model.fit(disp=False)
        except Exception as e:
            logger.error(f"Error fitting SARIMAX model: {e}")
            raise HTTPException(status_code=500, detail="Error fitting SARIMAX model.")

        # Forecast future rides
        try:
            predicted = results.get_forecast(steps=request.steps)
            forecasted = predicted.predicted_mean

        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            raise HTTPException(status_code=500, detail="Error generating forecast.")

        d1 = {'year_month': forecasted.index.to_period('M').astype(str), 'rides': forecasted.values.astype(int)}
        forecast = pd.DataFrame(data=d1)

        # Return response
        return {
            "historical": orig_data.to_dict(orient='records'),
            "forecast": forecast.to_dict(orient='records') # Convert forecast to list for JSON serialization
        }

    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8003, reload=True)
