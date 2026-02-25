import logging
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from pygam import GAM, s, f, LogisticGAM, PoissonGAM
from dataframes_loader import load_dataframe
import os
import joblib
import tempfile
# from statsmodels.tsa.statespace.sarimax import SARIMAX

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

def train_all_models(week, month):
    print("TRAINING STARTED")
    global trained_models
    training_status["status"] = "in_progress"
    trained_models = {
        'week': {
            "xgboost": fit_xgboost(week, "week"),
            "GAM": fit_GAM(week, "week")
        },
        'month': {
            "xgboost": fit_xgboost(month, "month"),
            "GAM": fit_GAM(month, "month")
        }
    }

    training_status["status"] = "ready"

def fit_xgboost(df, interval):
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Columns: {df.columns}")
    y = df['rides']

    X = df[['rideable_type', 'year', f'{interval}', 'season',
            f'rides_2{interval}s_ago', f'rides_last{interval}']]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.95,
                        test_size=0.05, shuffle=False, random_state=1)

    param_grid = {'max_depth': [3, 4, 5],
                  'n_estimators': [50, 100, 200, 300],
                  'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
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
    logger.debug(f"Input DataFrame shape: {df.shape}")
    logger.debug(f"Columns: {df.columns}")
    feature_cols = [
        'rideable_type',
        'year',
        interval,
        'season',
        f'rides_2{interval}s_ago',
        f'rides_last{interval}'
    ]

    data = df[feature_cols + ['rides']].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    X = data[feature_cols].to_numpy()
    y = data["rides"].to_numpy()

    split_idx = int(len(X) * 0.95)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_valid, y_valid = X[split_idx:], y[split_idx:]

    # model = GAM().fit(X, y)

    model = PoissonGAM(
        f(0) +
        s(1) +
        s(2, basis='cp') +
        f(3)
        # s(4) +
        # s(5)
    ).gridsearch(X_train, y_train)

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

        if not passed:
            logger.warning(f"[GATE FAIL] {model_name}: val_rmse={val_rmse:.4f} worse than baseline={baseline_rmse:.4f}")
            return None

        if passed:
            log_gam_term_plots(
                gam_model=model,
                feature_names=feature_cols,
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

df_week = load_dataframe("df_week_test_sql.tsv")
df_month = load_dataframe("df_month_test_sql.tsv")

print(">>> STARTUP EVENT TRIGGERED <<<")
train_all_models(df_week, df_month)
print(">>> TRAINING COMPLETED<<<")

