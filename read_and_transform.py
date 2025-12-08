import os
import pandas as pd
import numpy as np

directory = "./extracted_files"

# dataframes_number = {}
# dataframes_divvy = {}
# i = 0
# j = 0
# for root, dirs, files in os.walk(directory):
#     for filename in files:
#         file_path = os.path.join(root, filename)
#         #print(f"File: {file_path}")
#         if filename.endswith('tripdata.csv'): #or filename.startswith('Divvy_Trips'):
#             #print(filename)
#             try:
#                 df = pd.read_csv(file_path)
#                 dataframes_number[i] = df
#                 i += 1
#             except:
#                 pass
#
# def merge_dfs(dataframes):
#     merged = dataframes[0]
#     for i in range(1,len(dataframes)):
#         merged = pd.concat([merged,dataframes[i]], axis=0, ignore_index=True)
#     return merged
#
# merged_number = merge_dfs(dataframes_number)
# merged_number.to_csv('df_merged.tsv', sep='\t')
# df = merged_number

from sqlalchemy import create_engine
from sqlalchemy import text

engine = create_engine('postgresql+psycopg2://victor@localhost:5432/my_csv_db')

with engine.connect() as conn:
    with open('dags/templates/union_all.sql', "r") as file:
        query = file.read()
        conn.execute(text(query))

df = pd.read_sql("SELECT ride_id, rideable_type, started_at FROM merged", con=engine)

print(df.head())

df['started_at'] = pd.to_datetime(df['started_at'])
df['week_temp'] = df['started_at'].dt.to_period('W').astype(str)
df['year_week'] = [df['week_temp'][i].split('/')[0] for i in range(len(df)) ]


df['year'] = df['started_at'].dt.year
df['year_month'] = df['started_at'].dt.to_period('M')
df['week'] = df['started_at'].dt.strftime('%W')
df['month'] = df['started_at'].dt.month
df['season'] = (df['month']-1)//3

df = df[df.rideable_type != 'electric_scooter']
df.loc[df['rideable_type'] == 'docked_bike', 'rideable_type'] = 'classic_bike'

df['rideable_type'] = df['rideable_type'].replace({'classic_bike': 1, 'electric_bike': 2})
df['rideable_type'] = df['rideable_type'].astype(int)

def make_grouped(df,period):
    df_grouped = df.groupby([f'year_{period}','rideable_type']).agg(
    rides =('ride_id', 'count'),  # Count of 'ride_id'
    year = ('year','max'),
    month = ('month','max'),
    week = ('week','max'),
    season = ('season','max')
    ).reset_index()

    df1 = split_tables(df_grouped[df_grouped.rideable_type == 1], period)
    df2 = split_tables(df_grouped[df_grouped.rideable_type == 2], period)

    df_full = pd.concat([df1, df2], ignore_index=True)
    df_full = df_full.sort_values(by=f'year_{period}').reset_index(drop=True)

    return df_full

def split_tables(df, period):
    df[f'rides_2{period}s_ago'] = df['rides'].shift(2)
    df[f'rides_last{period}'] = df['rides'].shift(1)
    df = df.iloc[2:]
    return df

df_week = make_grouped(df,'week')
df_month = make_grouped(df,'month')

df_week.to_csv('df_week_exper.tsv', sep='\t')
df_month.to_csv('df_month_exper.tsv', sep='\t')