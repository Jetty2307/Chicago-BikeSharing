from mlflow.tracking import MlflowClient
import mlflow.xgboost
import joblib

def load_xgb(interval: str):
    # Load the XGBoost model from MLflow model registry

    client = MlflowClient()
    model_name = f"xgboost_{interval}"

    versions = client.get_latest_versions(model_name)
    latest = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{model_name}/{latest.version}"

    run_id = latest.run_id

    run = client.get_run(run_id)
    tags = run.data.tags

    # if versions:
    #     previous_run_id = versions[0].run_id
    #     previous_metrics = client.get_run(previous_run_id).data.metrics
    #     prev_rmse = previous_metrics.get("rmse")

    # *** Load penultimate version for stability during updates **
    # versions = client.search_model_versions(f"name='{model_name}'")
    # sorted_versions = sorted(versions, key=lambda v: int(v.version))
    # print(len(sorted_versions) )
    # penultimate = sorted_versions[-2] if len(sorted_versions) >= 2 else None
    # model_uri = f"models:/{model_name}/{penultimate.version}"

    return mlflow.xgboost.load_model(model_uri), tags.get('performance_description')

def load_gam(interval: str):
    # Load the GAM model from MLflow model registry

    client = MlflowClient()
    model_name = f"gam_{interval}"

    versions = client.get_latest_versions(model_name)
    latest = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{model_name}/{latest.version}"

    local_path = mlflow.artifacts.download_artifacts(model_uri)

    run_id = latest.run_id

    run = client.get_run(run_id)
    tags = run.data.tags

    return joblib.load(f"{local_path}/gam_model_{interval}.pkl"), tags.get('performance_description')

training_status = {"status": "ready"}
