import pandas as pd
import pytest

try:
    import read_and_transform
    from read_and_transform import build_interval_table, export_tsv
except Exception as e:
    pytest.skip(f"Cannot import read_and_transform: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def sample_frames():
    df_day = pd.DataFrame(
        {
            "year_day": ["2021-01-02", "2021-01-02", "2021-01-03", "2021-01-03"],
            "rideable_type": [1, 2, 1, 2],
            "year": [2021, 2021, 2021, 2021],
            "month": [1, 1, 1, 1],
            "season": [0, 0, 0, 0],
            "day_of_year": [2, 2, 3, 3],
            "day_of_week": [6, 6, 7, 7],
            "is_weekend": [1, 1, 1, 1],
            "rides": [120, 95, 140, 110],
            "rides_lastday": [100, 90, 120, 95],
            "temp": [-2.5, -2.5, 0.0, 0.0],
            "total_rain": [0.0, 0.0, 1.2, 1.2],
            "total_snow": [0.5, 0.5, 0.0, 0.0],
            "is_snow": [1, 1, 0, 0],
        }
    )
    df_week = pd.DataFrame(
        {
            "year_week": ["2021-01-04", "2021-01-04", "2021-01-11", "2021-01-11"],
            "rideable_type": [1, 2, 1, 2],
            "year": [2021, 2021, 2021, 2021],
            "week": [1, 1, 2, 2],
            "season": [0, 0, 0, 0],
            "rides": [800, 650, 900, 700],
            "rides_lastweek": [750, 610, 800, 650],
            "rides_2weeks_ago": [700, 580, 750, 610],
            "max_temp": [2.0, 2.0, 5.0, 5.0],
            "avg_temp": [-1.0, -1.0, 1.5, 1.5],
            "min_temp": [-5.0, -5.0, -2.0, -2.0],
            "total_rain": [4.2, 4.2, 2.8, 2.8],
            "total_snow": [1.5, 1.5, 0.0, 0.0],
        }
    )
    df_month = pd.DataFrame(
        {
            "year_month": ["2021-01", "2021-01", "2021-02", "2021-02"],
            "rideable_type": [1, 2, 1, 2],
            "year": [2021, 2021, 2021, 2021],
            "month": [1, 1, 2, 2],
            "season": [0, 0, 0, 0],
            "rides": [3200, 2800, 4100, 3600],
            "rides_lastmonth": [3000, 2600, 3200, 2800],
            "rides_2months_ago": [2800, 2400, 3000, 2600],
        }
    )
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
