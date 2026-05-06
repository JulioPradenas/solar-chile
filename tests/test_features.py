import pandas as pd
from pathlib import Path

MODEL_CSV = Path("data/processed/solar_chile_model.csv")
EXPECTED_COLS = [
    "system_size_kw", "tilt", "azimuth", "losses",
    "tilt_deviation", "azimuth_deviation", "latitude",
    "solrad_annual", "ac_annual", "city", "region",
]


def load():
    return pd.read_csv(MODEL_CSV)


def test_expected_columns_present():
    df = load()
    assert set(EXPECTED_COLS) == set(df.columns)


def test_no_nan_values():
    df = load()
    assert df.isnull().sum().sum() == 0


def test_row_count():
    df = load()
    assert len(df) == 3000


def test_tilt_deviation_range():
    df = load()
    assert df["tilt_deviation"].min() >= -30
    assert df["tilt_deviation"].max() <= 30


def test_azimuth_deviation_range():
    df = load()
    assert df["azimuth_deviation"].min() >= -90
    assert df["azimuth_deviation"].max() <= 90


def test_system_size_within_bounds():
    df = load()
    assert df["system_size_kw"].min() >= 2
    assert df["system_size_kw"].max() <= 15
