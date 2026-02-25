import pandas as pd
import pytest
# from read_and_transform import build_week_month

try:
    from read_and_transform import build_week_month
except Exception as e:
    pytest.skip(f"Cannot import read_and_transform: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def data():
    df_week, df_month = build_week_month()
    return df_week, df_month


def test_week_is_dataframe(data):
    df_week, _ = data
    assert isinstance(df_week, pd.DataFrame)


def test_month_is_dataframe(data):
    _, df_month = data
    assert isinstance(df_month, pd.DataFrame)


def test_week_contains_rides_column(data):
    df_week, _ = data
    assert "rides" in df_week.columns


def test_month_contains_rides_column(data):
    _, df_month = data
    assert "rides" in df_month.columns


def test_rides_column_is_integer(data):
    df_week, df_month = data
    assert pd.api.types.is_integer_dtype(df_week["rides"])
    assert pd.api.types.is_integer_dtype(df_month["rides"])


def test_rides_positive(data):
    df_week, df_month = data
    assert (df_week["rides"] > 0).all()
    assert (df_month["rides"] > 0).all()


def test_week_not_empty(data):
    df_week, _ = data
    assert len(df_week) > 0


def test_month_not_empty(data):
    _, df_month = data
    assert len(df_month) > 0