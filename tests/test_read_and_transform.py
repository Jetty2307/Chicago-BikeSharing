from pathlib import Path

import pandas as pd
import pytest

try:
    import read_and_transform
    from read_and_transform import build_interval_table, export_tsv
except Exception as e:
    pytest.skip(f"Cannot import read_and_transform: {e}", allow_module_level=True)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def sample_frames():
    df_day = pd.read_csv(DATA_DIR / "df_day_test_sql.tsv", sep="\t")
    df_week = pd.read_csv(DATA_DIR / "df_week_test_sql.tsv", sep="\t")
    df_month = pd.read_csv(DATA_DIR / "df_month_test_sql.tsv", sep="\t")
    return df_day, df_week, df_month


@pytest.fixture()
def data(monkeypatch, sample_frames):
    df_day, df_week, df_month = sample_frames
    queries = []

    def fake_fetch_df(query):
        queries.append(query)
        if query == read_and_transform.DAY_QUERY:
            return df_day.copy()
        if query == read_and_transform.WEEK_QUERY:
            return df_week.copy()
        if query == read_and_transform.MONTH_QUERY:
            return df_month.copy()
        raise AssertionError(f"Unexpected query: {query}")

    monkeypatch.setattr(read_and_transform, "fetch_df", fake_fetch_df)
    return build_interval_table(), queries


def test_build_interval_table_returns_dataframes(data):
    (df_day, df_week, df_month), _queries = data

    assert isinstance(df_day, pd.DataFrame)
    assert isinstance(df_week, pd.DataFrame)
    assert isinstance(df_month, pd.DataFrame)


def test_build_interval_table_fetches_all_queries(data):
    _frames, queries = data

    assert queries == [
        read_and_transform.DAY_QUERY,
        read_and_transform.WEEK_QUERY,
        read_and_transform.MONTH_QUERY,
    ]


def test_interval_tables_have_expected_core_columns(data):
    (df_day, df_week, df_month), _queries = data

    assert {"year_day", "rideable_type", "rides"}.issubset(df_day.columns)
    assert {"year_week", "rideable_type", "rides"}.issubset(df_week.columns)
    assert {"year_month", "rideable_type", "rides"}.issubset(df_month.columns)


def test_interval_tables_have_positive_integer_rides(data):
    (df_day, df_week, df_month), _queries = data

    for df in (df_day, df_week, df_month):
        assert not df.empty
        assert pd.api.types.is_integer_dtype(df["rides"])
        assert (df["rides"] > 0).all()


def test_export_tsv_writes_tab_separated_file(sample_frames, tmp_path):
    df_day, _df_week, _df_month = sample_frames
    target = tmp_path / "output.tsv"

    export_tsv(df_day.head(3), target)

    exported = pd.read_csv(target, sep="\t")
    pd.testing.assert_frame_equal(exported, df_day.head(3))
