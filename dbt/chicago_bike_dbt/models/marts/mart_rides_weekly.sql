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
)
select
    year_week,
    rideable_type_code as rideable_type,
    year,
    week,
    season,
    rides,
    rides_lastweek,
    rides_2weeks_ago
from pre
where rides_lastweek is not null
  and rides_2weeks_ago is not null

order by year_week, rideable_type
