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

    return mlflow.xgboost.load_model(model_uri)

def load_gam(interval: str):
    # Load the GAM model from MLflow model registry

    client = MlflowClient()
    model_name = f"gam_{interval}"

    versions = client.get_latest_versions(model_name)
    latest = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/{model_name}/{latest.version}"

    local_path = mlflow.artifacts.download_artifacts(model_uri)

    return joblib.load(f"{local_path}/gam_model_{interval}.pkl")
