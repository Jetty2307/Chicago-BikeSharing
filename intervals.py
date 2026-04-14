from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Dict, List

from dateutil.relativedelta import relativedelta
from pygam import PoissonGAM, f, s


MODEL_FEATURES = {
    "week": {
        "xgboost": ["rideable_type", "year", "week", "season", "rides_2weeks_ago", "rides_lastweek",
                    "max_temp", "avg_temp", "min_temp", "total_rain", "total_snow"],
        "GAM": ["rideable_type", "year", "week", "season", "avg_temp", "total_rain", "total_snow"],
    },
    "month": {
        "xgboost": ["rideable_type", "year", "month", "season", "rides_2months_ago", "rides_lastmonth"],
        "GAM": ["rideable_type", "year", "month", "season"],
    },
    "day": {
        "xgboost": ["rideable_type", "year", "month", "season", "day_of_year", "day_of_week","temp", "total_rain",
    "total_snow"],
        "GAM": ["rideable_type", "year", "season", "day_of_year", "day_of_week","temp", "total_rain", "total_snow"],
    },
}

GAM_TERM_BUILDERS: Dict[str, Callable[[int], Any]] = {
    "rideable_type": lambda idx: f(idx),
    "year": lambda idx: s(idx),
    "week": lambda idx: s(idx, basis="cp"),
    "month": lambda idx: s(idx, basis="cp"),
    "day_of_year": lambda idx: s(idx, basis="cp"),
    "day_of_week": lambda idx: f(idx),
    "season": lambda idx: f(idx),
    "temp": lambda idx: s(idx),
    "avg_temp": lambda idx: s(idx),
    "total_rain": lambda idx: s(idx),
    "total_snow": lambda idx: s(idx),
}


@dataclass(frozen=True)
class IntervalSpec:
    name: str
    period: int
    offset: relativedelta
    date_format: str
    sarima_freq: str
    model_features: Dict[str, List[str]]
    uses_weather: bool = False
    validation_trim: int = 0
    dataframe: Any = None

    def with_dataframe(self, dataframe: Any) -> "IntervalSpec":
        return replace(self, dataframe=dataframe)

    def add_interval(self, date_str: str) -> str:
        date_obj = datetime.strptime(date_str, self.date_format)
        return (date_obj + self.offset).strftime(self.date_format)

    def feature_columns(self, model_key: str) -> List[str]:
        return self.model_features[model_key]

    def trim_validation(self, X_valid, y_valid):
        if self.validation_trim:
            return X_valid[:-self.validation_trim], y_valid[:-self.validation_trim]
        return X_valid, y_valid

    def build_gam(self):
        gam_features = self.feature_columns("GAM")
        missing_features = [feature for feature in gam_features if feature not in GAM_TERM_BUILDERS]
        if missing_features:
            raise ValueError(f"Missing GAM term builders for features: {missing_features}")

        terms = None
        for idx, feature in enumerate(gam_features):
            term = GAM_TERM_BUILDERS[feature](idx)
            terms = term if terms is None else terms + term

        return PoissonGAM(terms)


INTERVAL_SPECS = {
    "week": IntervalSpec(
        name="week",
        period=52,
        offset=relativedelta(weeks=1),
        date_format="%Y-%m-%d",
        sarima_freq="W-MON",
        model_features=MODEL_FEATURES["week"],
        uses_weather=True,
        validation_trim=1,
    ),
    "month": IntervalSpec(
        name="month",
        period=12,
        offset=relativedelta(months=1),
        date_format="%Y-%m",
        sarima_freq="MS",
        model_features=MODEL_FEATURES["month"],
        uses_weather=False,
        validation_trim=0,
    ),
    "day": IntervalSpec(
        name="day",
        period=365,
        offset=relativedelta(days=1),
        date_format="%Y-%m-%d",
        sarima_freq="D",
        model_features=MODEL_FEATURES["day"],
        uses_weather=True,
        validation_trim=1,
    ),
}


def get_interval_spec(name: str, dataframe: Any = None) -> IntervalSpec:
    spec = INTERVAL_SPECS[name]
    return spec.with_dataframe(dataframe) if dataframe is not None else spec


def build_interval_mapping(dataframes: Dict[str, Any]) -> Dict[str, IntervalSpec]:
    return {
        name: get_interval_spec(name, dataframe=dataframe)
        for name, dataframe in dataframes.items()
    }
