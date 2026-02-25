from fastapi import FastAPI, File, UploadFile, HTTPException
import os
import pandas as pd
import logging

def load_dataframe(filename: str) -> pd.DataFrame:

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    try:

        logger.debug(f"Looking for file")
        if not os.path.exists(filename):
            raise HTTPException(status_code=404, detail=f"Data file {filename} not found ")

        df = pd.read_csv(filename, sep='\t', index_col=False).reset_index(drop=True)

        if 'Unnamed: 0' in df.columns:  # Remove unintended index column
            df = df.drop(columns=['Unnamed: 0'])

        if df.empty:
            raise HTTPException(status_code=400, detail=f"No data found in the file {filename}.")

    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

    return df