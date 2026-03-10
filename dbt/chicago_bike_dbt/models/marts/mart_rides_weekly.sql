with grouped as (
    select
        year_week,
        rideable_type_code,
        year,
        week,
        season,
        count(*) as rides
    from {{ ref('stg_divvy_rides') }}
    group by 1, 2, 3, 4, 5
),
pre as (
    select
        *,
        lag(rides, 1) over (
            partition by rideable_type_code
            order by year, week
        ) as rides_lastweek,
        lag(rides, 2) over (
            partition by rideable_type_code
            order by year, week
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
