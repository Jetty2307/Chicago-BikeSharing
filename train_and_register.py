import logging
from itertools import product
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.statsmodels
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from pygam import GAM, s, f, LogisticGAM, PoissonGAM
from dataframes_loader import load_dataframe
from intervals import get_interval_spec, build_interval_mapping
from ai_agent import make_summary, registry_decision
from feature_storage import fetch_features_xgboost, fetch_features_gam
from sarima_service import build_sarima_series, fit_sarima_model, forecast_with_fitted_sarima
import os
import joblib
import tempfile

from dotenv import load_dotenv
load_dotenv()
# from statsmodels.tsa.statespace.sarimax import SARIMAX

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

training_status = {"status": "not_started"}
trained_models = {}

mlflow.set_experiment("bikes_rides_forecasting")

PRIMARY_METRIC = "val_rmse"
MAX_REL_DEGRADATION = 0.05
LOWER_IS_BETTER = True


def log_shap_xgb(best_model, X_background, X_explain, feature_names, prefix = 'shap'):
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_explain, check_additivity=False)

    with tempfile.TemporaryDirectory() as td:
        plt.figure()
        shap.summary_plot(shap_values,
                          X_explain,
                          feature_names=feature_names,
                          show=False)
        path1 = os.path.join(td, f"{prefix}_summary.png")
        plt.tight_layout()
        plt.savefig(path1, dpi=160)
        plt.close()
        mlflow.log_artifact(path1, artifact_path='explainability')

        plt.figure()
        shap.summary_plot(shap_values,
                          X_explain,
                          feature_names=feature_names,
                          plot_type='bar',
                          show=False)
        path2 = os.path.join(td, f"{prefix}_bar.png")
        plt.tight_layout()
        plt.savefig(path2, dpi=160)
        plt.close()
        mlflow.log_artifact(path2, artifact_path='explainability')

def log_gam_term_plots(gam_model, feature_names, X_train, prefix="gam"):
    import numpy as np
    import tempfile
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as td:

        base = np.median(X_train, axis=0)

        for i, term in enumerate(gam_model.terms):

            if term.isintercept:
                continue

            plt.figure()

            feature_idx = term.feature

            if term.n_splines == 0:
                categories = np.unique(X_train[:, feature_idx])

                XX = np.tile(base, (len(categories), 1))
                XX[:, feature_idx] = categories

                pdep = gam_model.partial_dependence(term=i, X=XX)

                plt.bar(categories, pdep)
                plt.xlabel(feature_names[feature_idx])

            else:
                grid = np.linspace(
                    X_train[:, feature_idx].min(),
                    X_train[:, feature_idx].max(),
                    100
                )

                XX = np.tile(base, (len(grid), 1))
                XX[:, feature_idx] = grid

                pdep, conf = gam_model.partial_dependence(
                    term=i, X=XX, width=0.95
                )

                plt.plot(grid, pdep)
                plt.plot(grid, conf[:, 0], linestyle="--")
                plt.plot(grid, conf[:, 1], linestyle="--")
                plt.xlabel(feature_names[feature_idx])

            plt.title(f"{prefix}_term_{i}")
            plt.tight_layout()

            path = os.path.join(td, f"{prefix}_term_{i}.png")
            plt.savefig(path, dpi=160)
            plt.close()

            mlflow.log_artifact(path, artifact_path="explainability")

def _get_baseline_metric(model_name: str, metric_key: str = PRIMARY_METRIC):
    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None, None, None
        latest = max(versions, key=lambda v: int(v.version))
        run = client.get_run(latest.run_id)
        val = run.data.metrics.get(metric_key)
        return val, latest.run_id, latest.version
    except Exception:
        return None, None, None

def quality_gate_rmse(model_name: str, new_rmse: float, max_rel_degradation: float = MAX_REL_DEGRADATION):
    baseline_rmse, baseline_run_id, baseline_version = _get_baseline_metric(model_name, PRIMARY_METRIC)
    if baseline_rmse is None:
        return True, None, baseline_run_id, baseline_version

    threshold = baseline_rmse * (1.0 + max_rel_degradation)
    passed = new_rmse <= threshold
    return passed, baseline_rmse, baseline_run_id, baseline_version

def train_all_models(dataframes):
    print("TRAINING STARTED")
    global trained_models
    training_status["status"] = "in_progress"
    interval_mapping = build_interval_mapping(dataframes)
    trained_models = {
        interval_name: {
            "xgboost": fit_xgboost(spec.dataframe, interval_name),
            "GAM": fit_GAM(spec.dataframe, interval_name),
            "sarima": None if interval_name == "day" else train_sarima_model(spec.dataframe, interval_name),
        }
        for interval_name, spec in interval_mapping.items()
    }

    training_status["status"] = "ready"


def compute_sarima_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)

    denom = np.abs(y_true).sum()
    wmape = float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else np.nan

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "wmape": wmape,
    }


def split_sarima_series(series, interval_spec):
    valid_size = interval_spec.period
    offset_size = interval_spec.validation_offset

    if len(series) <= valid_size + offset_size:
        raise ValueError(
            "Not enough data for SARIMA validation: "
            f"len(series)={len(series)}, valid_size={valid_size}, offset_size={offset_size}"
        )

    valid_start = len(series) - valid_size - offset_size
    valid_end = len(series) - offset_size if offset_size else len(series)

    train = series.iloc[:valid_start]
    valid = series.iloc[valid_start:valid_end]
    return train, valid


def split_regression_frame(X, y, interval_spec):
    valid_rows = interval_spec.period * interval_spec.rows_per_period
    offset_rows = interval_spec.validation_offset * interval_spec.rows_per_period

    if len(X) <= valid_rows + offset_rows:
        raise ValueError(
            "Not enough data for regression validation: "
            f"len(X)={len(X)}, valid_rows={valid_rows}, offset_rows={offset_rows}"
        )

    valid_start = len(X) - valid_rows - offset_rows
    valid_end = len(X) - offset_rows if offset_rows else len(X)

    X_train, y_train = X[:valid_start], y[:valid_start]
    X_valid, y_valid = X[valid_start:valid_end], y[valid_start:valid_end]
    return X_train, X_valid, y_train, y_valid


def iter_sarima_configs(period):
    for p, d, q, P, D, Q in product([0, 1], repeat=6):
        if p == q == P == Q == 0:
            continue

        yield {
            "order": (p, d, q),
            "seasonal_order": (P, D, Q, period),
        }


def apply_registry_decision(model_name, model_params):
    description = make_summary(model_params)
    mlflow.set_tag("performance_description", str(description))

    decision = registry_decision(model_params=model_params, description=description)
    mlflow.set_tag("ai_registry_decision", decision["decision"].lower())
    mlflow.set_tag("ai_registry_reason", decision["reason"])

    if decision["decision"] == "STOP":
        logger.warning(f"[AI REGISTRY STOP] {model_name}: {decision['reason']}")
        return None, description, decision

    return True, description, decision

def train_sarima_model(df, interval):
    interval_spec = get_interval_spec(interval)
    _, series = build_sarima_series(
        dataframe=df,
        interval=interval,
        timeframe=interval_spec,
    )

    train, valid = split_sarima_series(series, interval_spec)
    model_name = f"sarima_{interval}"

    with mlflow.start_run(run_name=model_name) as run:
        best_order = None
        best_seasonal_order = None
        best_metrics = None
        successful_configs = 0
        tested_configs = 0

        for config in iter_sarima_configs(interval_spec.period):
            tested_configs += 1
            order = config["order"]
            seasonal_order = config["seasonal_order"]

            try:
                fitted_valid_model = fit_sarima_model(
                    series=train,
                    order=order,
                    seasonal_order=seasonal_order,
                )

                valid_forecast = forecast_with_fitted_sarima(
                    fitted_valid_model,
                    steps=len(valid),
                )
                valid_forecast.index = valid.index
                metrics = compute_sarima_metrics(valid, valid_forecast)
                successful_configs += 1
            except Exception as exc:
                logger.warning(
                    f"[SARIMA SKIP] {model_name}: order={order}, seasonal_order={seasonal_order}, error={exc}"
                )
                continue

            if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
                best_order = order
                best_seasonal_order = seasonal_order
                best_metrics = metrics

        if best_metrics is None:
            raise ValueError(f"No valid SARIMA configuration found for {model_name}")

        mlflow.log_param("model_type", "sarima")
        mlflow.log_param("interval", interval)
        mlflow.log_param("order", str(best_order))
        mlflow.log_param("seasonal_order", str(best_seasonal_order))
        mlflow.log_param("sample_size", len(train))
        mlflow.log_param("tested_configs", tested_configs)
        mlflow.log_param("successful_configs", successful_configs)

        mlflow.log_metric("val_rmse", best_metrics["rmse"])
        mlflow.log_metric("val_mae", best_metrics["mae"])
        mlflow.log_metric("val_wmape", best_metrics["wmape"])

        passed, baseline_rmse, baseline_run_id, baseline_version = quality_gate_rmse(model_name, best_metrics["rmse"])
        mlflow.set_tag("quality_gate_passed", str(passed).lower())
        if baseline_rmse is not None:
            mlflow.log_metric("baseline_rmse", float(baseline_rmse))
            mlflow.set_tag("baseline_run_id", str(baseline_run_id))
            mlflow.set_tag("baseline_version", str(baseline_version))
        mlflow.set_tag("max_rel_degradation", str(MAX_REL_DEGRADATION))

        if not passed:
            logger.warning(f"[GATE FAIL] {model_name}: val_rmse={best_metrics['rmse']:.4f} worse than baseline={baseline_rmse:.4f}")
            return None

        model_params = {
            "model_type": "sarima",
            "interval": interval,
            "sample size": len(train),
            "validation size": len(valid),
            "order": str(best_order),
            "seasonal_order": str(best_seasonal_order),
            "tested configs": tested_configs,
            "successful configs": successful_configs,
            "rmse": best_metrics["rmse"],
            "mae": best_metrics["mae"],
            "wmape": best_metrics["wmape"],
        }

        approved, description, decision = apply_registry_decision(model_name, model_params)
        if not approved:
            return None

        final_model = fit_sarima_model(
            series=series,
            order=best_order,
            seasonal_order=best_seasonal_order,
        )

        mlflow.statsmodels.log_model(final_model, artifact_path="model")
        run_id = run.info.run_id

    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name,
    )

    return final_model

def fit_xgboost(df, interval):
    # logger.debug(f"Input DataFrame shape: {df.shape}")
    # logger.debug(f"Columns: {df.columns}")
    interval_spec = get_interval_spec(interval)

    X, y, feature_names = fetch_features_xgboost(df, interval, model_name="xgboost")
    logger.debug(f"Features: {feature_names}")

    X_train, X_valid, y_train, y_valid = split_regression_frame(X, y, interval_spec)
    X_valid, y_valid = interval_spec.trim_validation(X_valid, y_valid)

    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.03, 0.05, 0.1],
        'min_child_weight': [1, 3, 5, 7],
    }

    grid = GridSearchCV(
        XGBRegressor(random_state=1, objective='count:poisson'),
        param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        refit=True,
        n_jobs=-1
    )

    model_name = f"xgboost_{interval}"

    with mlflow.start_run(run_name=f"xgboost_{interval}") as run:

        model = grid.fit(X_train, y_train)
        best_model = model.best_estimator_

        y_pred = best_model.predict(X_valid)
        val_rmse = mean_squared_error(y_valid, y_pred, squared=False)
        val_r2 = r2_score(y_valid, y_pred)

        print(f"XGBoost {interval} - Best Params: {model.best_params_}, Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}")

        mlflow.log_params(model.best_params_)
        mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("val_r2", float(val_r2))

        passed, baseline_rmse, baseline_run_id, baseline_version = quality_gate_rmse(model_name, float(val_rmse))
        mlflow.set_tag("quality_gate_passed", str(passed).lower())
        if baseline_rmse is not None:
            mlflow.log_metric("baseline_rmse", float(baseline_rmse))
            mlflow.set_tag("baseline_run_id", str(baseline_run_id))
            mlflow.set_tag("baseline_version", str(baseline_version))
        mlflow.set_tag("max_rel_degradation", str(MAX_REL_DEGRADATION))

        if not passed:
            logger.warning(f"[GATE FAIL] {model_name}: val_rmse={val_rmse:.4f} worse than baseline={baseline_rmse:.4f}")
            return None

        model_params = {'model_type': 'xgboost',
                        'interval': interval,
                        'sample size': len(X_train),
                        'features size': X_train.shape[1],
                        'mean': np.mean(y_train),
                        'max_depth': best_model.max_depth,
                        'n_estimators': best_model.n_estimators,
                        'learning_rate': best_model.learning_rate,
                        'rmse': val_rmse,
                        'r2': val_r2}

        approved, description, decision = apply_registry_decision(model_name, model_params)
        if not approved:
            return None

        mlflow.xgboost.log_model(best_model, artifact_path="model")

        run_id = run.info.run_id

        if passed:
            X_bg = X_train.sample(min(len(X_train), 200), random_state=1)
            X_exp= X_valid.sample(min(len(X_valid), 200), random_state=1)

            log_shap_xgb(
                best_model=best_model,
                X_background=X_bg,
                X_explain=X_exp,
                feature_names=X.columns.tolist(),
                prefix=f"xgboost_{interval}",

            )

    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=f"xgboost_{interval}"
    )

    return best_model

def fit_GAM(df, interval):
    # logger.debug(f"Input DataFrame shape: {df.shape}")
    # logger.debug(f"Columns: {df.columns}")
    interval_spec = get_interval_spec(interval)

    X, y, feature_names = fetch_features_gam(df, interval, model_name="GAM")
    logger.debug(f"Features: {feature_names}")
    # "rideable_type", "year", interval, "season", "avg_temp", "total_rain", "total_snow"

    X_train, X_valid, y_train, y_valid = split_regression_frame(X, y, interval_spec)
    X_valid, y_valid = interval_spec.trim_validation(X_valid, y_valid)
    model = interval_spec.build_gam().gridsearch(X_train, y_train)

    y_pred = model.predict(X_valid)
    val_r2 = r2_score(y_valid, y_pred)
    val_rmse = mean_squared_error(y_valid, y_pred, squared=False)

    print(f"GAM {interval} - Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}")

    model_name = f"gam_{interval}"

    with mlflow.start_run(run_name=f"gam_{interval}") as run:
        mlflow.log_param("interval", interval)
        mlflow.log_param("model_type", "GAM")
        mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("val_r2", float(val_r2))

        passed, baseline_rmse, baseline_run_id, baseline_version = quality_gate_rmse(model_name, float(val_rmse))
        mlflow.set_tag("quality_gate_passed", str(passed).lower())
        if baseline_rmse is not None:
            mlflow.log_metric("baseline_val_rmse", float(baseline_rmse))
            mlflow.set_tag("baseline_run_id", str(baseline_run_id))
            mlflow.set_tag("baseline_version", str(baseline_version))
        mlflow.set_tag("max_rel_degradation", str(MAX_REL_DEGRADATION))

        passed = True

        if not passed:
            logger.warning(f"[GATE FAIL] {model_name}: val_rmse={val_rmse:.4f} \
                           worse than baseline={baseline_rmse:.4f}")
            return None

        model_params = {'model_type': 'Generalized Additive Model',
                        'interval': interval,
                        'sample size': len(X_train),
                        'features size': len(model.terms),
                        'mean': np.mean(y_train),
                        'rmse': val_rmse,
                        'r2': val_r2}

        approved, description, decision = apply_registry_decision(model_name, model_params)
        if not approved:
            return None

        if passed:
            log_gam_term_plots(
                gam_model=model,
                feature_names=feature_names,
                X_train=X_train,
                prefix=f"gam_{interval}",
            )


        model_path = f"gam_model_{interval}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path='model')
        os.remove(model_path)

        run_id = run.info.run_id

    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=f"gam_{interval}"
    )

    return model

df_day = load_dataframe(os.environ["DAY_FILE"])
df_week = load_dataframe(os.environ["WEEK_FILE"])
df_month = load_dataframe(os.environ["MONTH_FILE"])

print(">>> STARTUP EVENT TRIGGERED <<<")
train_all_models({
    "day": df_day,
    "week": df_week,
    "month": df_month,
})
print(">>> TRAINING COMPLETED<<<")
