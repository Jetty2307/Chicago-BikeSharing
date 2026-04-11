import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SarimaForecastResult:
    historical: pd.DataFrame
    forecast: pd.Series
    fitted_model: Any


def build_sarima_series(dataframe: pd.DataFrame, interval: str, timeframe) -> Tuple[pd.DataFrame, pd.Series]:
    path_df = dataframe.groupby(f"year_{interval}")["rides"].sum().reset_index()

    path_df[f"year_{interval}"] = pd.to_datetime(
        path_df[f"year_{interval}"],
        format=timeframe.date_format,
    )

    historical = path_df.copy()
    historical[f"year_{interval}"] = historical[f"year_{interval}"].dt.strftime(timeframe.date_format)

    series = (
        path_df
        .set_index(f"year_{interval}")["rides"]
        .asfreq(timeframe.sarima_freq)
    )

    return historical, series


def fit_sarima_model(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
):
    if seasonal_order is None:
        raise ValueError("seasonal_order must be provided for SARIMA")

    y = np.log1p(series)

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    return model.fit(disp=False)


def forecast_with_fitted_sarima(results, steps: int) -> pd.Series:
    forecast_log = results.get_forecast(steps=steps).predicted_mean
    return np.expm1(forecast_log).clip(lower=0)


def fit_and_forecast_sarima(
    dataframe: pd.DataFrame,
    interval: str,
    timeframe,
    steps: int,
    order: Tuple[int, int, int] = (1, 1, 1),
):
    historical, series = build_sarima_series(
        dataframe=dataframe,
        interval=interval,
        timeframe=timeframe,
    )

    seasonal_order = (1, 0, 1, timeframe.period)
    fitted_model = fit_sarima_model(
        series=series,
        order=order,
        seasonal_order=seasonal_order,
    )

    try:
        forecast = forecast_with_fitted_sarima(fitted_model, steps=steps)
    except Exception as exc:
        logger.error("Error generating SARIMA forecast: %s", exc)
        raise

    return SarimaForecastResult(
        historical=historical,
        forecast=forecast,
        fitted_model=fitted_model,
    )
