import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple
from dotenv import load_dotenv
load_dotenv()

today = datetime.today().strftime("%Y_%m_%d")

from sqlalchemy import create_engine
from sqlalchemy import text

# engine = create_engine('postgresql+psycopg2://victor@localhost:5432/my_csv_db')

ENGINE_URL = os.getenv(
    "DATABASE_URL"
)

if not ENGINE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(ENGINE_URL)

# with engine.connect() as conn:
#     with open('/Users/victor/airflow/dags/templates/last_one.sql', "r") as file:
#         query = file.read()
#         conn.execute(text(query))

# def load_sql(path: str) -> str:
#     with open(path, "r", encoding="utf-8") as f:
#         return f.read()

def load_last_one(sql_path: str, engine=engine) -> None:
    with engine.connect() as conn:
        with open(sql_path, "r", encoding="utf-8") as f:
            query = f.read()
        conn.execute(text(query))
def fetch_df(query: str, engine=engine) -> pd.DataFrame:
    with engine.connect() as conn:
        conn.execute(text(query))
        return pd.read_sql(query, conn)

def build_queries() -> Tuple[str, str]:
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
           week,
           season,
           count(*) as rides
      from cte
      where rideable_type IN (1,2)
      GROUP BY 1,2,3,4,5
    ),
    
    pre as 
    (SELECT *,
           LAG(rides, 1) OVER (PARTITION BY rideable_type ORDER BY year, week) as rides_lastweek,
           LAG(rides, 2) OVER (PARTITION BY rideable_type ORDER BY year, week) as rides_2weeks_ago
    from grouped              
    )
     
    SELECT * 
    from pre
    where rides is not null
    and rides_lastweek is not null
    and rides_2weeks_ago is not null
    ORDER BY year, week, rideable_type
    '''

    final_query_month = cte_block + main_query_month
    final_query_week = cte_block + main_query_week

    return final_query_week, final_query_month


def build_week_month() -> Tuple[pd.DataFrame, pd.DataFrame]:
    final_query_week, final_query_month = build_queries()

    df_week = fetch_df(final_query_week)
    df_month = fetch_df(final_query_month)

    return df_week, df_month

def export_tsv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    SQL_PATH = os.environ["SQL_PATH"]

    load_last_one(SQL_PATH)

    df_week, df_month = build_week_month()

    export_tsv(df_week, os.environ["WEEK_FILE"])
    export_tsv(df_month, os.environ["MONTH_FILE"])