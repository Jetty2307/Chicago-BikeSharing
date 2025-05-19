from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner' : 'airflow',
    'start_date' : datetime(2025, 4, 19),
    'retries' : 1,
}

with DAG(
    dag_id='monthly_divvy_download',
    default_args=default_args,
    description='Load data from divvy and transform them monthly',
    schedule_interval='@monthly',
    catchup=False,
    tags=['divvy','monthly'],
) as dag:

    download = BashOperator(
        task_id='run_download',
        bash_command='cd /Users/victor/Desktop/DS/Chicago-BikeSharing && source /Users/victor/miniconda3/bin/activate base && python divvy_download.py'
    )
	
    process = BashOperator(
        task_id='run_transform',
        bash_command='cd /Users/victor/Desktop/DS/Chicago-BikeSharing && source /Users/victor/miniconda3/bin/activate base && python read_and_transform.py'
    )

    download >> process