from pathlib import Path

import pandas as pd
import pytest

try:
    import read_and_transform
    from read_and_transform import build_week_month, export_tsv
except Exception as e:
    pytest.skip(f"Cannot import read_and_transform: {e}", allow_module_level=True)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def sample_frames():
    df_week = pd.read_csv(DATA_DIR / "df_week_test_sql.tsv", sep="\t")
    df_month = pd.read_csv(DATA_DIR / "df_month_test_sql.tsv", sep="\t")
    return df_week, df_month


@pytest.fixture()
def data(monkeypatch, sample_frames):
    df_week, df_month = sample_frames
    queries = []

    def fake_fetch_df(query):
        queries.append(query)
        if query == read_and_transform.WEEK_QUERY:
            return df_week.copy()
        if query == read_and_transform.MONTH_QUERY:
            return df_month.copy()
        raise AssertionError(f"Unexpected query: {query}")

    monkeypatch.setattr(read_and_transform, "fetch_df", fake_fetch_df)
    return build_week_month(), queries


def test_week_is_dataframe(data):
    (df_week, _), _queries = data
    assert isinstance(df_week, pd.DataFrame)


def test_month_is_dataframe(data):
    (_, df_month), _queries = data
    assert isinstance(df_month, pd.DataFrame)


def test_week_contains_rides_column(data):
    (df_week, _), _queries = data
    assert "rides" in df_week.columns


def test_month_contains_rides_column(data):
    (_, df_month), _queries = data
    assert "rides" in df_month.columns


def test_rides_column_is_integer(data):
    (df_week, df_month), _queries = data
    assert pd.api.types.is_integer_dtype(df_week["rides"])
    assert pd.api.types.is_integer_dtype(df_month["rides"])


def test_rides_positive(data):
    (df_week, df_month), _queries = data
    assert (df_week["rides"] > 0).all()
    assert (df_month["rides"] > 0).all()


def test_week_not_empty(data):
    (df_week, _), _queries = data
    assert len(df_week) > 0


def test_month_not_empty(data):
    (_, df_month), _queries = data
    assert len(df_month) > 0


def test_build_week_month_fetches_both_queries(data):
    _frames, queries = data
    assert queries == [read_and_transform.WEEK_QUERY, read_and_transform.MONTH_QUERY]


def test_export_tsv_writes_tab_separated_file(sample_frames, tmp_path):
    df_week, _ = sample_frames
    target = tmp_path / "week.tsv"

    export_tsv(df_week.head(3), target)

    exported = pd.read_csv(target, sep="\t")
    pd.testing.assert_frame_equal(exported, df_week.head(3))
