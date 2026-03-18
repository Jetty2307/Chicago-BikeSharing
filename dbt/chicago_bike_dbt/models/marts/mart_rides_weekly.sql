with grouped as (
    select
        year_week,
        rideable_type_code,
        mode() within group (order by year) as year,
        mode() within group (order by week) as week,
        mode() within group (order by season) as season,
        count(*) as rides
    from {{ ref('stg_divvy_rides') }}
    group by 1, 2
),
pre as (
    select
        *,
        lag(rides, 1) over (
            partition by rideable_type_code
            order by year_week
        ) as rides_lastweek,
        lag(rides, 2) over (
            partition by rideable_type_code
            order by year_week
        ) as rides_2weeks_ago
    from grouped
),

weather_week as
    (select
        year_week,
        max(temperature_2m_mean) as max_temp,
        avg(temperature_2m_mean) as avg_temp,
        min(temperature_2m_mean) as min_temp,
        sum(rain_sum) as total_rain,
        sum(snowfall_sum) as total_snow
    from {{ ref('stg_weather_daily') }}
    group by year_week
    )
select
    year_week,
    rideable_type_code as rideable_type,
    year,
    week,
    season,
    rides,
    rides_lastweek,
    rides_2weeks_ago,
    max_temp,
    round(avg_temp::numeric,1) as avg_temp,
    min_temp,
    round(total_rain::numeric,1) as total_rain,
    round(total_snow::numeric,1) as total_snow
from pre
left join weather_week using (year_week)
where rides_lastweek is not null
  and rides_2weeks_ago is not null

order by year_week, rideable_type
