import asyncio
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from pygam import GAM, s, f, LogisticGAM
# from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

training_status = {"status": "not_started"}
trained_models = {}

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
    #logger.debug(f"Input DataFrame shape: {df.shape}")
    #logger.debug(f"Columns: {df.columns}")
    y = df['rides']

    X = df[['rideable_type', 'year', f'{interval}', 'season', f'rides_2{interval}s_ago', f'rides_last{interval}']]
    #logger.debug(f"Target (y) shape: {y.shape}, Features (X) shape: {X.shape}")
    #logger.debug(f"Feature preview:\n{X.head()}")
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

    model = grid.fit(X_train, y_train)

    return model

def fit_GAM(df, interval):
    #logger.debug(f"Input DataFrame shape: {df.shape}")
    #logger.debug(f"Columns: {df.columns}")
    y = df['rides']
    # X = df.drop(columns=[f'year_{interval}', 'rides'], axis=1)
    X = df[['rideable_type', 'year', f'{interval}', 'season', f'rides_2{interval}s_ago', f'rides_last{interval}']]
    #logger.debug(f"Target (y) shape: {y.shape}, Features (X) shape: {X.shape}")
    #logger.debug(f"Feature preview:\n{X.head()}")
    model = GAM().fit(X, y)
    return model

training_status = {"status": "in_progress"}

