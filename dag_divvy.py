from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner' : 'airflow',
    'start_date' : datetime(2024, 12, 4),
    'retries' : 1,
}

with DAG(
    dag_id='monthly_divvy_download',
    default_args=default_args,
    description='Load divvy_download.py monthly',
    schedule_interval='@monthly',
    catchup=False,
    tags=['divvy','monthly'],
) as dag:

    run_download_script = BashOperator(
        task_id='run_divvy_download',
        bash_command='source /Users/victor/miniconda3/bin/activate base && python /Users/victor/Desktop/DS/Chicago-BikeSharing/divvy_download.py'
    )

    run_download_script