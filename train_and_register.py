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
from model_policy import decide_promotion, REFRESH_MONTHS, MAX_REFRESH_DEGRADATION
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from pygam import GAM, s, f, LogisticGAM, PoissonGAM
from dataframes_loader import load_dataframe
from intervals import get_interval_spec, build_interval_mapping
from features_inference import fetch_features_xgboost, fetch_features_gam, prepare_gam_features
from sarima_service import (
    build_sarima_series,
    compare_sarima_models,
    fit_sarima_model,
    forecast_with_fitted_sarima,
)
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
            by_idx = getattr(term, "by", None)
            label = feature_names[feature_idx]
            if by_idx is not None:
                label += "_by_" + feature_names[by_idx]

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
                if by_idx is not None:
                    XX[:, by_idx] = 1

                pdep, conf = gam_model.partial_dependence(
                    term=i, X=XX, width=0.95
                )

                plt.plot(grid, pdep)
                plt.plot(grid, conf[:, 0], linestyle="--")
                plt.plot(grid, conf[:, 1], linestyle="--")
                plt.xlabel(feature_names[feature_idx])

            plt.title(f"{prefix}_{label}")
            plt.tight_layout()

            path = os.path.join(td, f"{prefix}_term_{i}.png")
            plt.savefig(path, dpi=160)
            plt.close()

            mlflow.log_artifact(path, artifact_path="explainability")

def _get_baseline_info(model_name):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v.version))
    run = client.get_run(latest.run_id)
    return {"run_id": latest.run_id, "version": latest.version,
            "uri": f"models:/{model_name}/{latest.version}", "tags": run.data.tags}


def log_data_boundaries(df, interval, train_index, valid_index):
    dates = pd.to_datetime(df.loc[train_index, f"year_{interval}"])
    validation = pd.to_datetime(df.loc[valid_index, f"year_{interval}"])
    tags = {"train_end_date": dates.max().date().isoformat(),
            "validation_start_date": validation.min().date().isoformat(),
            "validation_end_date": validation.max().date().isoformat(),
            "data_end_date": pd.to_datetime(df[f"year_{interval}"]).max().date().isoformat()}
    mlflow.set_tags(tags)
    return tags


def quality_gate_rmse(model_name, new_rmse, boundaries, baseline, baseline_rmse):
    baseline_end = None
    if baseline:
        baseline_end = baseline["tags"].get("data_end_date")
        if not baseline_end:
            # Legacy runs have no reliable data cutoff. Anchor the six-month clock
            # once, without pretending that the training cutoff is known.
            baseline_end = baseline["tags"].get("refresh_anchor_data_end")
            if not baseline_end:
                baseline_end = boundaries["data_end_date"]
                MlflowClient().set_tag(baseline["run_id"], "refresh_anchor_data_end", baseline_end)
            mlflow.set_tag("baseline_date_source", "legacy_refresh_anchor")
        mlflow.set_tags({"baseline_run_id": baseline["run_id"],
                         "baseline_version": str(baseline["version"]),
                         "baseline_data_end_date": baseline_end})
        mlflow.log_metric("baseline_rmse", baseline_rmse)
    decision = decide_promotion(new_rmse, baseline_rmse, boundaries["data_end_date"], baseline_end)
    mlflow.set_tags({"quality_gate_passed": str(decision.passed).lower(),
                     "promotion_reason": decision.reason,
                     "refresh_required": str(decision.refresh_required).lower(),
                     "refresh_months": str(REFRESH_MONTHS),
                     "max_refresh_degradation": str(MAX_REFRESH_DEGRADATION)})
    if decision.refresh_required and not decision.passed:
        logger.warning("Freshness update blocked by quality floor: %s", model_name)
    return decision


def regression_baseline_score(model_name, baseline, X_valid, y_valid, boundaries):
    if baseline is None:
        return None
    train_end = baseline["tags"].get("train_end_date")
    if train_end and pd.Timestamp(train_end) >= pd.Timestamp(boundaries["validation_start_date"]):
        raise ValueError(f"Baseline training overlaps validation: {model_name}")
    if model_name.startswith("xgboost_"):
        model = mlflow.xgboost.load_model(baseline["uri"])
        prediction = model.predict(X_valid)
    else:
        path = mlflow.artifacts.download_artifacts(baseline["uri"])
        interval = model_name[len("gam_"):]
        model = joblib.load(f"{path}/gam_model_{interval}.pkl")
        frame = pd.DataFrame(X_valid, columns=get_interval_spec(interval).feature_columns("GAM"))
        prediction = model.predict(prepare_gam_features(frame, interval, model).to_numpy())
    return float(mean_squared_error(y_valid, prediction, squared=False))


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


def apply_registry_decision(model_name, model_params, allow_veto=True):
    from ai_agent import make_summary, registry_decision

    description = make_summary(model_params)
    mlflow.set_tag("performance_description", str(description))

    decision = registry_decision(model_params=model_params, description=description)
    mlflow.set_tag("ai_registry_decision", decision["decision"].lower())
    mlflow.set_tag("ai_registry_reason", decision["reason"])

    mlflow.set_tag("ai_registry_veto_applied", str(allow_veto and decision["decision"] == "STOP").lower())

    if allow_veto and decision["decision"] == "STOP":
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
        best_validation_model = None
        successful_configs = 0
        tested_configs = 0

        configs = list(iter_sarima_configs(interval_spec.period))
        for config in configs:
            tested_configs += 1
            order = config["order"]
            seasonal_order = config["seasonal_order"]
            logger.info("%s: configuration %d/%d, order=%s, seasonal_order=%s",
                        model_name, tested_configs, len(configs), order, seasonal_order)

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
                best_validation_model = fitted_valid_model

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

        boundaries = {
            "train_end_date": train.index.max().date().isoformat(),
            "validation_start_date": valid.index.min().date().isoformat(),
            "validation_end_date": valid.index.max().date().isoformat(),
            "data_end_date": series.index.max().date().isoformat(),
        }
        mlflow.set_tags(boundaries)
        baseline = _get_baseline_info(model_name)
        baseline_rmse = None
        comparison_rmse = best_metrics["rmse"]
        if baseline:
            old = mlflow.statsmodels.load_model(baseline["uri"])
            comparison = compare_sarima_models(
                candidate=best_validation_model, baseline=old, series=series, validation=valid,
            )
            baseline_rmse = comparison.baseline_rmse
            comparison_rmse = comparison.candidate_rmse
            baseline["tags"].setdefault("data_end_date", comparison.baseline_data_end.date().isoformat())
            mlflow.set_tag("comparison_start_date", comparison.comparison_start.date().isoformat())
        mlflow.log_metric("comparison_rmse", comparison_rmse)
        promotion = quality_gate_rmse(model_name, comparison_rmse, boundaries, baseline, baseline_rmse)
        passed = promotion.passed

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

        approved, description, decision = apply_registry_decision(
            model_name, model_params, allow_veto=not promotion.refresh_required,
        )
        if not approved:
            return None

        final_model = fit_sarima_model(
            series=series,
            order=best_order,
            seasonal_order=best_seasonal_order,
        )

        mlflow.set_tag("train_end_date", series.index.max().date().isoformat())
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

        boundaries = log_data_boundaries(df, interval, X_train.index, X_valid.index)
        baseline = _get_baseline_info(model_name)
        baseline_rmse = regression_baseline_score(model_name, baseline, X_valid, y_valid, boundaries)
        promotion = quality_gate_rmse(model_name, float(val_rmse), boundaries, baseline, baseline_rmse)
        passed = promotion.passed

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

        approved, description, decision = apply_registry_decision(
            model_name, model_params, allow_veto=not promotion.refresh_required,
        )
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

    cleaned_df = df.copy().replace([np.inf, -np.inf], np.nan).dropna()
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
        mlflow.log_param("gam_year_trend", "by_rideable_type")
        mlflow.log_param("gam_feature_names", ",".join(feature_names))
        mlflow.log_metric("val_rmse", float(val_rmse))
        mlflow.log_metric("val_r2", float(val_r2))

        boundaries = log_data_boundaries(
            cleaned_df, interval, cleaned_df.index[:len(X_train)],
            cleaned_df.index[len(X_train):len(X_train) + len(X_valid)],
        )
        baseline = _get_baseline_info(model_name)
        baseline_rmse = regression_baseline_score(model_name, baseline, X_valid, y_valid, boundaries)
        promotion = quality_gate_rmse(model_name, float(val_rmse), boundaries, baseline, baseline_rmse)
        passed = promotion.passed

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

        approved, description, decision = apply_registry_decision(
            model_name, model_params, allow_veto=not promotion.refresh_required,
        )
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

if __name__ == "__main__":
    mlflow.set_experiment("bikes_rides_forecasting")
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
