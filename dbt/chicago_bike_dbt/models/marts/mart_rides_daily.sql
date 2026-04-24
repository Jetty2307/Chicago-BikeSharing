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
    group by 1, 2, 3, 4, 5, 6, 7
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
    date as year_day,
    rideable_type_code as rideable_type,
    year,
    month,
    season,
    day_of_year,
    day_of_week,
    case when day_of_week in (6, 7) then 1 else 0 end as is_weekend,
    rides,
    rides_lastday,
    temp,
    total_rain,
    total_snow,
    case when total_snow > 0 then 1 else 0 end as is_snow
from pre
left join weather_day using (date)
where rides_lastday is not null
and date >= '2021-01-01'
order by year_day, rideable_type

