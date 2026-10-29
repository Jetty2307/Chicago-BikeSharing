import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sarima_service import forecast_with_fitted_sarima, update_sarima_observations


@pytest.mark.parametrize(
    'freq,start,length,old_length,expected',
    [('MS', '2025-01-01', 21, 13, '2026-10-01'),
     ('W-MON', '2025-01-06', 91, 56, '2026-10-05')],
)
def test_forecast_uses_current_observations(freq, start, length, old_length, expected):
    index = pd.date_range(start, periods=length, freq=freq)
    series = pd.Series(100 + np.arange(length) * 2, index=index, name='rides')
    model = SARIMAX(np.log1p(series.iloc[:old_length]), order=(1, 0, 0), trend='c')
    fitted = model.filter([0.1, 0.9, 0.2])

    updated = update_sarima_observations(fitted, series)
    forecast = forecast_with_fitted_sarima(updated, steps=3)

    assert forecast.index[0] == pd.Timestamp(expected)
    assert len(forecast) == 3
    assert np.isfinite(forecast).all()
    np.testing.assert_array_equal(updated.params, fitted.params)
    assert updated.nobs == len(series)
    assert fitted.nobs == old_length
    pd.testing.assert_series_equal(updated.model.data.orig_endog, np.log1p(series))
    assert update_sarima_observations(updated, series) is updated

    corrected = series.copy()
    corrected.iloc[-1] *= 2
    revised = update_sarima_observations(updated, corrected)
    assert not np.allclose(forecast_with_fitted_sarima(revised, 3), forecast)
