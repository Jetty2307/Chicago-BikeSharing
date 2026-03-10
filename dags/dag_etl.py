from __future__ import annotations

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.trigger_rule import TriggerRule

from datetime import datetime
from pathlib import Path
import os
import shutil
import re
import requests
from zipfile import ZipFile

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Environment / Paths
# ----------------------------
PROJECT_DIR = os.environ["PROJECT_DIR"]
CONDA_PATH = os.environ["CONDA_PATH"]
CONDA_ENV = os.environ["CONDA_ENV"]

DB_URL = os.environ["DB_URL"]

# Final dirs (committed artifacts)
ZIP_DIR = Path(os.environ["ZIP_DIR"])
CSV_DIR = Path(os.environ["CSV_DIR"])

# Staging root (temporary per-run workspace)
STAGING_ROOT = ZIP_DIR / "_staging"

ENDING = os.environ.get("ENDING", "-tripdata.csv")

# SQL template file paths (relative to your DAG folder, used by PostgresOperator)
SQL_PATH_MERGE = os.environ["SQL_PATH_MERGE"]
SQL_PATH_CHUNK = os.environ["SQL_PATH_CHUNK"]
DAG_DIR = Path(os.environ["DAG_DIR"])

DIVVY_S3_URL = os.environ.get("DIVVY_S3_URL", "https://divvy-tripdata.s3.amazonaws.com")
DBT_PROJECT_DIR = f"{PROJECT_DIR}/dbt/chicago_bike_dbt"


# ----------------------------
# Tasks
# ----------------------------
def fetch_new_files(**context) -> dict:
    """
    Downloads NEW zip files into staging per-run directory, extracts CSV into staging,
    and returns payload:
      {
        "run_tag": "...",
        "staging_zip_dir": "...",
        "staging_csv_dir": "...",
        "zip_files": [...],
        "csv_files": [...],
      }

    "Already processed" is determined by presence of ZIP in final ZIP_DIR (committed state).
    """
    run_tag = (
        str(context["run_id"])
        .replace(":", "_")
        .replace("+", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )

    staging_zip_dir = STAGING_ROOT / run_tag / "zips"
    staging_csv_dir = STAGING_ROOT / run_tag / "csvs"
    staging_zip_dir.mkdir(parents=True, exist_ok=True)
    staging_csv_dir.mkdir(parents=True, exist_ok=True)

    resp = requests.get(DIVVY_S3_URL, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to retrieve listing page: {resp.status_code}")

    zip_filenames = re.findall(r"([\w\-_]+\.zip)", resp.text)
    if not zip_filenames:
        return {
            "run_tag": run_tag,
            "staging_zip_dir": str(staging_zip_dir),
            "staging_csv_dir": str(staging_csv_dir),
            "zip_files": [],
            "csv_files": [],
        }

    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    new_zip_files: list[str] = []
    new_csv_files: list[str] = []

    for zip_name in zip_filenames:
        # If ZIP already exists in final dir => already committed => skip
        if (ZIP_DIR / zip_name).exists():
            continue

        zip_url = f"{DIVVY_S3_URL}/{zip_name}"
        print(f"Downloading: {zip_url}")
        zr = requests.get(zip_url, timeout=180)
        if zr.status_code != 200:
            print(f"Failed to download {zip_name}: {zr.status_code}")
            continue

        staging_zip_path = staging_zip_dir / zip_name
        staging_zip_path.write_bytes(zr.content)
        new_zip_files.append(zip_name)

        # Extract first CSV and rename to "<zip>.csv" into staging csv dir
        target_csv_name = zip_name.replace(".zip", ".csv")
        target_csv_path = staging_csv_dir / target_csv_name

        try:
            with ZipFile(staging_zip_path, "r") as zf:
                csv_inside = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_inside:
                    print(f"No CSV found in {zip_name}. Skipping extraction.")
                    continue

                original_csv_name = csv_inside[0]
                print(f"Extracting: {original_csv_name} -> {target_csv_name}")

                with zf.open(original_csv_name) as src, open(target_csv_path, "wb") as dst:
                    dst.write(src.read())

            new_csv_files.append(target_csv_name)

        except Exception as e:
            print(f"Extraction failed for {zip_name}: {e}")
            raise

    payload = {
        "run_tag": run_tag,
        "staging_zip_dir": str(staging_zip_dir),
        "staging_csv_dir": str(staging_csv_dir),
        "zip_files": new_zip_files,
        "csv_files": new_csv_files,
    }
    print(f"New ZIP files: {len(new_zip_files)}")
    print(f"New CSV files: {len(new_csv_files)}")
    return payload


def load_csvs(**context):
    """
    Loads newly extracted CSVs from staging into Postgres (one table per file).
    """
    engine = create_engine(DB_URL)
    ti = context["ti"]
    payload = ti.xcom_pull(task_ids="fetch_new_files") or {}

    csv_files = payload.get("csv_files", [])
    if not csv_files:
        raise AirflowSkipException("No new CSV files to load.")

    staging_csv_dir = Path(payload["staging_csv_dir"])

    for fname in csv_files:
        if not fname.endswith(ENDING):
            continue

        table_name = fname.replace(ENDING, "").lower()
        csv_path = staging_csv_dir / fname
        print(f"Loading {csv_path} -> table {table_name}")

        df = pd.read_csv(csv_path)
        df.to_sql(table_name, engine, if_exists="replace", index=False)


def generate_union_sqls(**context):
    """
    Generates two SQL files based on newly loaded table names:
    - SQL_PATH_MERGE: INSERT INTO merged ... UNION ALL SELECT ... FROM each table
    - SQL_PATH_CHUNK: CREATE TABLE IF NOT EXISTS last_one AS ... UNION ALL SELECT ...
    """
    ti = context["ti"]
    payload = ti.xcom_pull(task_ids="fetch_new_files") or {}

    csv_files = payload.get("csv_files", [])
    files = [f for f in csv_files if f.endswith(ENDING)]
    table_names = [f.replace(ENDING, "").lower() for f in files]

    if not table_names:
        raise AirflowSkipException("No tables to include in SQL generation.")

    insert_sql = (
        "INSERT INTO merged (ride_id, rideable_type, started_at)\n"
        + "\nUNION ALL\n".join(
            [
                'SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP '
                f'FROM "{t}"'
                for t in table_names
            ]
        )
        + ";"
    )

    union_sql = (
        "CREATE TABLE IF NOT EXISTS last_one AS\n"
        + "\nUNION ALL\n".join(
            [
                'SELECT ride_id::TEXT, rideable_type::TEXT, started_at::TIMESTAMP '
                f'FROM "{t}"'
                for t in table_names
            ]
        )
        + ";"
    )

    (DAG_DIR / SQL_PATH_MERGE).write_text(insert_sql)
    (DAG_DIR / SQL_PATH_CHUNK).write_text(union_sql)

    print(f"Wrote SQL: {DAG_DIR / SQL_PATH_MERGE}")
    print(f"Wrote SQL: {DAG_DIR / SQL_PATH_CHUNK}")


def commit_staging(**context):
    """
    Moves staged ZIP/CSV into final dirs (commit) only when all upstream succeeded.
    """
    ti = context["ti"]
    payload = ti.xcom_pull(task_ids="fetch_new_files") or {}

    staging_zip_dir = Path(payload["staging_zip_dir"])
    staging_csv_dir = Path(payload["staging_csv_dir"])
    zip_files = payload.get("zip_files", []) or []
    csv_files = payload.get("csv_files", []) or []

    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    for z in zip_files:
        src = staging_zip_dir / z
        dst = ZIP_DIR / z
        if src.exists():
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))

    for c in csv_files:
        src = staging_csv_dir / c
        dst = CSV_DIR / c
        if src.exists():
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))

    # optional marker
    run_root = STAGING_ROOT / payload["run_tag"]
    (run_root / ".committed").write_text("ok")
    print("Commit done.")


def cleanup_staging(**context):
    """
    Always removes staging directory for this run (success or failure).
    """
    ti = context["ti"]
    payload = ti.xcom_pull(task_ids="fetch_new_files") or {}
    run_tag = payload.get("run_tag")
    if not run_tag:
        return

    run_root = STAGING_ROOT / run_tag
    if run_root.exists():
        shutil.rmtree(run_root, ignore_errors=True)
        print(f"Removed staging: {run_root}")


# ----------------------------
# DAG
# ----------------------------
default_args = {"start_date": datetime(2025, 12, 10)}

with DAG(
    dag_id="full_etl",
    schedule_interval="@monthly",
    default_args=default_args,
    description="Load data from Divvy and transform them monthly (staging + commit)",
    catchup=False,
    tags=["divvy", "monthly"],
) as dag:

    t_fetch_new_files = PythonOperator(
        task_id="fetch_new_files",
        python_callable=fetch_new_files,
    )

    t_load_csvs = PythonOperator(
        task_id="load_csvs",
        python_callable=load_csvs,
        provide_context=True,
    )

    t_prepare_sql = PythonOperator(
        task_id="prepare_sql",
        python_callable=generate_union_sqls,
        provide_context=True,
    )

    t_insert_into_merged = PostgresOperator(
        task_id="insert_into_tables",
        postgres_conn_id="my_postgres_conn",
        sql=SQL_PATH_MERGE,
    )

    t_create_last_one = PostgresOperator(
        task_id="create_new_table",
        postgres_conn_id="my_postgres_conn",
        sql=SQL_PATH_CHUNK,
    )

    t_transform = BashOperator(
        task_id="run_transform",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"source {CONDA_PATH} {CONDA_ENV} && "
            f"python read_and_transform.py"
        ),
    )

    t_dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            f"source {CONDA_PATH} {CONDA_ENV} && "
            "dbt run"
        ),
    )

    t_dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=(
            f"cd {DBT_PROJECT_DIR} && "
            f"source {CONDA_PATH} {CONDA_ENV} && "
            "dbt test"
        ),
    )

    t_train_models = BashOperator(
        task_id="run_training",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"source {CONDA_PATH} {CONDA_ENV} && "
            f"python train_and_register.py"
        ),
    )

    t_commit = PythonOperator(
        task_id="commit_staging",
        python_callable=commit_staging,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    t_cleanup = PythonOperator(
        task_id="cleanup_staging",
        python_callable=cleanup_staging,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Main pipeline
    (
        t_fetch_new_files
        >> t_load_csvs
        >> t_prepare_sql
        >> t_insert_into_merged
        >> t_create_last_one
        >> t_dbt_run
        >> t_dbt_test
        >> t_transform
        >> t_train_models
        >> t_commit
        >> t_cleanup
    )

    # Ensure cleanup runs even if failure happens before commit
    [
        t_load_csvs,
        t_prepare_sql,
        t_insert_into_merged,
        t_create_last_one,
        t_dbt_run,
        t_dbt_test,
        t_transform,
        t_train_models,
        t_commit,
    ] >> t_cleanup
