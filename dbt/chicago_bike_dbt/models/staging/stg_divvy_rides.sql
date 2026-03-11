with src as (
    select
        cast(ride_id as text) as ride_id,
        cast(rideable_type as text) as rideable_type,
        cast(started_at as timestamp) as started_at
    from {{ source('raw', 'merged') }}
),
typed as (
    select
        ride_id,
        rideable_type,
        started_at,
        date_part('year', started_at)::int as year,
        date_part('month', started_at)::int as month,
        date_part('week', started_at)::int as week,
        to_char(started_at, 'YYYY-MM') as year_month,
        to_char(date_trunc('week', started_at), 'YYYY-MM-DD') as year_week,
        ((date_part('month', started_at)::int - 1) / 3) as season,
        case
            when rideable_type = 'classic_bike' then 1
            when rideable_type = 'electric_bike' then 2
            else null
        end as rideable_type_code
    from src
)
select *
from typed
where rideable_type_code in (1, 2)
