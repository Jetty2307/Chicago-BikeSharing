select
    weather_date::date as date,
    date_part('year', weather_date)::int as year,
    date_part('month', weather_date)::int as month,
    date_part('week', weather_date)::int as week,
    to_char(weather_date, 'YYYY-MM') as year_month,
    to_char(date_trunc('week', weather_date), 'YYYY-MM-DD') as year_week,
    temperature_2m_mean,
    precipitation_sum,
    rain_sum,
    snowfall_sum,
    weather_code
from {{ source('raw', 'weather_daily') }}

