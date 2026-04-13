with grouped as (
    select
        date,
        rideable_type_code,
        year,
        month,
        season,
        day_of_year,
        day_of_week,
        count(*) as rides
    from {{ ref('stg_divvy_rides') }}
    group by 1, 2, 3, 4, 5
),

pre as (
    select
        *,
        lag(rides, 1) over (
            partition by rideable_type_code
            order by year, month, date
        ) as rides_lastday

    from grouped
),

weather_day as
    (select
        date,
        temperature_2m_mean as temp,
        rain_sum as total_rain,
        snowfall_sum as total_snow
    from {{ ref('stg_weather_daily') }}
    )

select
    date,
    rideable_type_code as rideable_type,
    year,
    month,
    season,
    day_of_year,
    day_of_week,
    rides,
    rides_lastday,
    temp,
    total_rain,
    total_snow
from pre
left join weather_week using (date)
where rides_lastday is not null
order by date, rideable_type

