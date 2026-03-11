with grouped as (
    select
        year_month,
        rideable_type_code,
        year,
        month,
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
            order by year, month
        ) as rides_lastmonth,
        lag(rides, 2) over (
            partition by rideable_type_code
            order by year, month
        ) as rides_2months_ago
    from grouped
)
select
    year_month,
    rideable_type_code as rideable_type,
    year,
    month,
    season,
    rides,
    rides_lastmonth,
    rides_2months_ago
from pre
where rides_lastmonth is not null
  and rides_2months_ago is not null
