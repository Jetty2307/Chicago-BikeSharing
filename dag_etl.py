from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable, TaskInstance

from datetime import datetime
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

today = datetime.today().strftime("%Y_%m_%d")

project_dir = os.environ["PROJECT_DIR"]
conda_path = os.environ["CONDA_PATH"]
conda_env = os.environ["CONDA_ENV"]

csv_dir = os.environ["CSV_DIR"]
db_url = os.environ["DB_URL"]
ending = os.environ["ENDING"]
sql_path_merge = os.environ["SQL_PATH_MERGE"]
sql_path_chunk = os.environ["SQL_PATH_CHUNK"]


# csv_dir = '/Users/victor/Desktop/DS/Chicago-BikeSharing/extracted_files'
# db_url = 'postgresql+psycopg2://victor@localhost:5432/my_csv_db'
# ending = '-tripdata.csv'
# sql_path_merge = '/Users/victor/airflow/dags/templates/union_all.sql'
# sql_path_chunk = f'/Users/victor/airflow/dags/templates/last_one.sql'

def fetch_new_files():
    from zipfile import ZipFile
    import os
    import re
    import requests, io

    url = 'https://divvy-tripdata.s3.amazonaws.com'

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve XML page: {response.status_code}")
        exit()

    zip_urls = re.findall(r'([\w\-_]+\.zip)', response.text)

    download_dir = os.environ["ZIP_DIR"]
    extracted_dir = os.environ["EXTRACTED_DIR"]

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extracted_dir, exist_ok=True)

    new_files = []

    for zip_filename in zip_urls:
        local_zip_path = os.path.join(download_dir, zip_filename)

        if os.path.exists(local_zip_path):
            print(f"Already exists, skipping : {zip_filename}")
            continue

        full_url = f"{url}/{zip_filename}"

        print(f"Downloading: {full_url}")
        zip_response = requests.get(full_url)
        if zip_response.status_code != 200:
            print(f"Failed to download: {full_url}")
            continue

        with open(local_zip_path, 'wb') as f:
            f.write(zip_response.content)

        target_csv_name = zip_filename.replace('.zip', '.csv')
        target_csv_path = os.path.join(extracted_dir, target_csv_name)
            # zip_filename = os.path.join(download_dir, zip_url.split('/')[-1])

        with ZipFile(local_zip_path, 'r') as zip_ref:
            zip_contents = zip_ref.namelist()

            csv_files = [f for f in zip_contents if f.endswith('.csv')]

            if not csv_files:
                print(f"No CSV found in {zip_filename}")
                continue

            original_csv_name = csv_files[0]

            print(f"Extracting and renaming: {original_csv_name} → {target_csv_name}")

            with zip_ref.open(original_csv_name) as source_file:
                with open(target_csv_path, 'wb') as target_file:
                    target_file.write(source_file.read())

        new_files.append(target_csv_name)


    return new_files


def load_csvs(**context):

    engine = create_engine(db_url)
    ti: TaskInstance = context['ti']
    new_files = ti.xcom_pull(task_ids='fetch_new_files')

    print('hi')
    print(new_files)
    #for file in os.listdir(csv_dir):
    for file in new_files:
        if file.endswith(ending):
            table_name = file.replace(ending,'').lower()
            df = pd.read_csv(os.path.join(csv_dir, file),
                )
            df.to_sql(table_name, engine, if_exists='replace', index=False)

def generate_union_sqls(**context):
    #files = [f for f in os.listdir(csv_dir) if f.endswith(ending)]
    ti: TaskInstance = context['ti']
    new_files = ti.xcom_pull(task_ids='fetch_new_files')

    files = [f for f in new_files if f.endswith(ending)]
    table_names = [f.replace(ending, '').lower() for f in files]

    if not table_names:
        print("No tables to include in SQL. Skipping SQL generation.")
        raise AirflowSkipException("No tables available, skipping SQL generation.")

    insert_sql = "INSERT INTO merged (\n" \
                 "ride_id, rideable_type, started_at\n" \
                 ")\n"
    insert_sql += "\nUNION ALL\n".join(
        [f'SELECT '
         f'ride_id::TEXT, '
         f'rideable_type::TEXT, '
         f'started_at::TIMESTAMP '
         f'FROM "{table}"' for table in table_names]
    ) + ";"

    union_sql = "CREATE TABLE IF NOT EXISTS last_one AS\n"
    union_sql += "\nUNION ALL\n".join(
        [f'SELECT '
         f'ride_id::TEXT, '
         f'rideable_type::TEXT, '
         f'started_at::TIMESTAMP '
         f'FROM "{table}"' for table in table_names]
    ) + ";"

    with open(sql_path_merge, 'w') as f:
        f.write(insert_sql)

    with open(sql_path_chunk, 'w') as f:
        f.write(union_sql)

# def get_sql_path(**context):
#     from datetime import datetime
#     today = datetime.today().strftime('%Y_%m_%d')
#     return f'templates/loaded_{today}.sql'

default_args = {'start_date' : datetime(2025, 12, 10)}

with DAG('full_etl',
    schedule_interval = '@monthly',
    default_args=default_args,
    description="Load data from divvy and transform them monthly",
    catchup=False,
    tags=['divvy','monthly']
) as dag:

    fetch_new_files = PythonOperator(
        task_id='fetch_new_files',
        python_callable=fetch_new_files,
    )

    load_csvs = PythonOperator(
        task_id='load_csvs',
        python_callable=load_csvs,
        provide_context=True
    )

    prepare_sql = PythonOperator(
        task_id='prepare_sql',
        python_callable=generate_union_sqls,
        provide_context=True
    )

    # get_sql_task = PythonOperator(
    #     task_id='get_sql_path',
    #     python_callable=get_sql_path,
    # )

    make_union_sql = PostgresOperator(
        task_id='insert_into_tables',
        postgres_conn_id='my_postgres_conn',
        sql=sql_path_merge
    )

    make_new_sql = PostgresOperator(
        task_id='create_new_table',
        postgres_conn_id='my_postgres_conn',
        sql=sql_path_chunk
    )

    transform = BashOperator(
        task_id='run_transform',
        bash_command=f""""
                        cd {project_dir} &&
                        source {conda_path} {conda_env} &&
                        python read_and_transform.py
                        """
    )

    train_models = BashOperator(
        task_id='run_training',
        bash_command=f""""
                        cd {project_dir} &&
                        source {conda_path} {conda_env} &&
                        python train_and_register.py
                        """
    )

    fetch_new_files >> load_csvs >> prepare_sql >> make_union_sql >> make_new_sql >> transform >> train_models

