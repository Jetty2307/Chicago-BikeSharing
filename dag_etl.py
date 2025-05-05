from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import pandas as pd
from sqlalchemy import create_engine

csv_dir = '/Users/victor/Desktop/DS/Chicago-BikeSharing/extracted_files'
db_url = 'postgresql+psycopg2://victor@localhost:5432/my_csv_db'
ending = 'tripdata.csv'

def load_csvs():
    engine = create_engine(db_url)

    for file in os.listdir(csv_dir):
        if file.endswith(ending):
            table_name = file.replace(ending,'').lower()
            df = pd.read_csv(os.path.join(csv_dir, file))
            df.to_sql(table_name, engine, if_exists='replace', index=False)

def generate_union_sql():
    files = [f for f in os.listdir(csv_dir) if f.endswith(ending)]
    table_names = [f.replace(ending, '').lower() for f in files]

    union_sql = "CREATE TABLE IF NOT EXISTS merged AS\n"
    union_sql += "\nUNION ALL\n".join(
        [f"SELECT * FROM {table}" for table in table_names]
    ) + ";"

    with open('/tmp/union_all.sql', 'w') as f:
        f.write(union_sql)

default_args = {'start_date' : datetime(2025, 4, 30)}

with DAG('make_tables_union',
    schedule_interval = None,
    default_args=default_args,
    description="Load csv tables and my one SQL table",
    catchup=False) as dag:

    load_csvs = PythonOperator(
        task_id='load_csvs',
        python_callable=load_csvs,
    )

    prepare_union_sql = PythonOperator(
        task_id='prepare_union_sql',
        python_callable=generate_union_sql,
    )

    union_all = PostgresOperator(
        task_id='merge_tables',
        postgres_conn_id='my_postgres_conn',
        sql="/Users/victor/airflow/union_all.sql"
    )

    load_csvs >> prepare_union_sql >> union_all