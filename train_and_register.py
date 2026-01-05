import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from pygam import GAM, s, f, LogisticGAM, PoissonGAM
from dataframes_loader import load_dataframe
import os
import joblib
# from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

training_status = {"status": "not_started"}
trained_models = {}

mlflow.set_experiment("bikes_rides_forecasting")

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

    X = df[['rideable_type', 'year', f'{interval}', 'season', f'rides_2{interval}s_ago', f'rides_last{interval}']]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, shuffle=True,
                                                          random_state=1)
    param_grid = {'max_depth': [3, 4, 5],
                  'n_estimators': [50, 100, 200, 300],
                  'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
                  }

    base_score = y_train.mean()

    grid = GridSearchCV(
        XGBRegressor(random_state=1, objective='count:poisson', base_score=base_score),
        param_grid,
        refit=True,
        n_jobs=-1
    )

    with mlflow.start_run(run_name=f"xgboost_{interval}") as run:

        model = grid.fit(X_train, y_train)
        best_model = model.best_estimator_
        y_pred = best_model.predict(X_valid)

        mlflow.log_params(model.best_params_)
        mlflow.log_metric("val_rmse", mean_squared_error(y_valid, y_pred, squared=False))
        mlflow.log_metric("val_r2", r2_score(y_valid, y_pred))

        mlflow.xgboost.log_model(best_model, artifact_path="model")

    run_id = run.info.run_id
    mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=f"xgboost_{interval}"
    )

    return model

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

    # model = GAM().fit(X, y)

    model = PoissonGAM(
        f(0) +
        s(1) +
        s(2, basis='cp') +
        f(3)
        # s(4) +
        # s(5)
    ).gridsearch(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)

    with mlflow.start_run(run_name=f"gam_{interval}") as run:
        mlflow.log_param("interval", interval)
        mlflow.log_param("model_type", "GAM")
        mlflow.log_metric("train_r2", r2)
        mlflow.log_metric("train_rmse", rmse)

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

