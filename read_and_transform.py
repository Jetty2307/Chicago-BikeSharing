import os
import pandas as pd
import numpy as np
from datetime import datetime

today = datetime.today().strftime("%Y_%m_%d")

from sqlalchemy import create_engine
from sqlalchemy import text

engine = create_engine('postgresql+psycopg2://victor@localhost:5432/my_csv_db')

with engine.connect() as conn:
    with open('/Users/victor/airflow/dags/templates/last_one.sql', "r") as file:
        query = file.read()
        conn.execute(text(query))

cte_block = \
''' WITH cte as
(SELECT ride_id,
          date_part('year', started_at) as year,
          date_part('month', started_at) as month,
          TO_CHAR(started_at, 'YYYY-MM') as year_month,
          TO_CHAR(date_trunc('week', started_at), 'YYYY-MM-DD') as year_week,
          date_part('week', started_at) as week,
          CAST(date_part('month', started_at) - 1 as INTEGER) / 3 as season,
          CASE WHEN rideable_type = 'classic_bike' then 1
               WHEN rideable_type = 'electric_bike' then 2
               ELSE null
          END as rideable_type  
  from merged 
),
'''

main_query_month = \
'''
grouped as
(SELECT year_month,
       rideable_type,
       year,
       month,
       season,
       count(*) as rides
  from cte
  where rideable_type = 1 OR rideable_type = 2
  GROUP BY 1,2,3,4,5 
),

pre as
(SELECT *,
       LAG(rides, 1) OVER (PARTITION BY rideable_type ORDER BY year, month) as rides_lastmonth,
       LAG(rides, 2) OVER (PARTITION BY rideable_type ORDER BY year, month) as rides_2months_ago
from grouped          
)

SELECT *
from pre
where rides is not null
and rides_lastmonth is not null
and rides_2months_ago is not null
ORDER BY year, month, rideable_type
'''

main_query_week = \
'''
grouped as
(SELECT year_week,
       rideable_type,
       year,
       month,
       week,
       season,
       count(*) as rides
  from cte
  where rideable_type = 1 OR rideable_type = 2
  GROUP BY 1,2,3,4,5,6
),

pre as 
(SELECT *,
       LAG(rides, 1) OVER (PARTITION BY rideable_type ORDER BY year, month, week) as rides_lastweek,
       LAG(rides, 2) OVER (PARTITION BY rideable_type ORDER BY year, month, week) as rides_2weeks_ago
from grouped              
)
 
SELECT * 
from pre
where rides is not null
and rides_lastweek is not null
and rides_2weeks_ago is not null
ORDER BY year, month, week, rideable_type
'''

final_query_month = cte_block + main_query_month
final_query_week = cte_block + main_query_week

df_week = pd.read_sql(final_query_week, con=engine)
df_month = pd.read_sql(final_query_month, con=engine)

df_week.to_csv('df_week_test_sql.tsv', sep='\t')
df_month.to_csv('df_month_test_sql.tsv', sep='\t')