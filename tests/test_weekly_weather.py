import numpy as np
import pandas as pd
import pytest
import features_inference as features


def daily_weather(dates, temperature=10.0, rain=2.0):
    return pd.DataFrame({
        'weather_date': dates,
        'temperature_2m_mean': temperature,
        'rain_sum': rain,
        'snowfall_sum': 0.0,
    })


@pytest.mark.parametrize('weekday', range(7))
def test_weather_switches_to_seasonal_at_full_week_boundary(monkeypatch, tmp_path, weekday):
    today = pd.Timestamp('2026-10-26') + pd.Timedelta(days=weekday)
    week_file = tmp_path / 'weekly.tsv'
    pd.DataFrame({'year_week': ['2024-01-01', '2026-09-28']}).to_csv(
        week_file, sep='\t', index=False,
    )
    monkeypatch.setenv('WEEK_FILE', str(week_file))
    monkeypatch.setattr(features, 'today', today)
    monkeypatch.setattr(features, 'load_historical_weather',
                        lambda start, end: daily_weather(pd.date_range(start, end)))
    calls = []

    def forecast(days):
        calls.append(days)
        return daily_weather(pd.date_range(today, periods=days), temperature=100.0, rain=5.0)

    monkeypatch.setattr(features, 'load_forecast_weather', forecast)
    result = features.get_weekly_weather_forecast_df().set_index('year_week')
    assert calls == [16]
    assert len(result) == 104
    assert not result.isna().any().any()

    final_day = today + pd.Timedelta(days=15)
    final_sunday = final_day - pd.Timedelta(days=(final_day.weekday() + 1) % 7)
    final_monday = final_sunday - pd.Timedelta(days=6)
    final_week = result.loc[final_monday.strftime('%Y-%m-%d')]
    assert final_week['avg_temp'] == 100.0
    assert final_week['total_rain'] == 35.0

    next_monday = final_sunday + pd.Timedelta(days=1)
    seasonal = result.loc[next_monday.strftime('%Y-%m-%d')]
    assert seasonal['avg_temp'] == 10.0
    assert seasonal['total_rain'] == 14.0

    # The current week uses past observations plus forecast, all seven days.
    current_monday = today - pd.Timedelta(days=weekday)
    mixed = result.loc[current_monday.strftime('%Y-%m-%d')]
    assert mixed['total_rain'] == weekday * 2.0 + (7 - weekday) * 5.0
    assert mixed['avg_temp'] == pytest.approx((weekday * 10 + (7 - weekday) * 100) / 7)


def test_weather_requires_seven_distinct_complete_days():
    daily = daily_weather(pd.date_range('2026-10-26', periods=14))
    daily = pd.concat([daily, daily.iloc[[0]]], ignore_index=True)
    daily.loc[8, 'rain_sum'] = np.nan
    result = features._aggregate_complete_weather_weeks(daily)
    assert result['year_week'].tolist() == ['2026-10-26']
    assert result['total_rain'].tolist() == [14.0]
