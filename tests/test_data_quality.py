import pandas as pd
from pathlib import Path

CLEAN_CSV = Path("data/processed/solar_chile_model.csv")
EXPECTED_CITIES = {"Antofagasta", "Santiago", "Rancagua", "Concepción", "Temuco", "Punta Arenas"}
EXPECTED_COLS = {
    "system_size_kw", "tilt", "azimuth", "losses",
    "tilt_deviation", "azimuth_deviation", "latitude",
    "solrad_annual", "ac_annual", "city", "region",
}


def validate_dataset(df: pd.DataFrame) -> list[str]:
    errors = []
    if df.isnull().any().any():
        errors.append("nulls encontrados")
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        errors.append(f"columnas faltantes: {missing}")
    if "city" in df.columns:
        unknown = set(df["city"].unique()) - EXPECTED_CITIES
        if unknown:
            errors.append(f"ciudades desconocidas: {unknown}")
    if "solrad_annual" in df.columns:
        if df["solrad_annual"].min() < 2 or df["solrad_annual"].max() > 9:
            errors.append("solrad_annual fuera de rango Chile")
    if "ac_annual" in df.columns:
        if df["ac_annual"].min() < 0:
            errors.append("ac_annual negativo")
    return errors


def test_quality_gate_passes_on_clean_data():
    df = pd.read_csv(CLEAN_CSV)
    assert validate_dataset(df) == []


def test_quality_gate_catches_nulls():
    df = pd.read_csv(CLEAN_CSV).copy()
    df.loc[0, "solrad_annual"] = None
    errors = validate_dataset(df)
    assert any("null" in e for e in errors)


def test_quality_gate_catches_unknown_city():
    df = pd.read_csv(CLEAN_CSV).copy()
    df.loc[0, "city"] = "Atlantida"
    errors = validate_dataset(df)
    assert any("ciudades" in e for e in errors)


def test_quality_gate_catches_negative_energy():
    df = pd.read_csv(CLEAN_CSV).copy()
    df.loc[0, "ac_annual"] = -500
    errors = validate_dataset(df)
    assert any("negativo" in e for e in errors)
