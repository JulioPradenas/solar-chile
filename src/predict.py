"""
Función de predicción de producción solar y proyección financiera para Chile.

Uso:
    from src.predict import get_prediction
    result = get_prediction(city='Santiago', system_size_kw=5.0, tilt=33.0, azimuth=0.0)
"""

import pickle
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import (
    CHILE_CITIES_LIST,
    ELECTRICITY_RATE_CLP,
    RATE_ESCALATION,
    INSTALL_COST_PER_KWP,
)

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "src" / "models" / "gb_energy_model.pkl"
DB_PATH = BASE_DIR / "data" / "raw" / "nrel_chile.db"


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _get_weather(city: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = pd.read_sql(
            "SELECT * FROM nrel_chile WHERE city = ?", conn, params=(city,)
        )
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def _annual_savings(annual_kwh: float, years: int = 10) -> list[float]:
    savings = []
    rate = ELECTRICITY_RATE_CLP
    for _ in range(years):
        savings.append(annual_kwh * rate)
        rate *= 1 + RATE_ESCALATION
    return savings


def get_prediction(
    city: str,
    system_size_kw: float,
    tilt: float,
    azimuth: float = 0.0,
) -> dict | None:
    """
    Retorna predicción de producción y proyección financiera.

    Args:
        city: nombre de ciudad (debe existir en nrel_chile.db)
        system_size_kw: tamaño del sistema en kWp
        tilt: ángulo de inclinación del panel (grados)
        azimuth: orientación (0=norte, 90=este, 180=sur, 270=oeste)

    Returns:
        dict con annual_energy_kwh, install_cost_clp, savings_10y, payback_years
        o None si la ciudad no está en la base de datos.
    """
    weather = _get_weather(city)
    if weather is None:
        return None

    latitude = weather["latitude"]
    tilt_deviation = tilt - abs(latitude)
    azimuth_deviation = ((azimuth + 180) % 360) - 180

    features = pd.DataFrame([{
        "system_size_kw":    system_size_kw,
        "tilt":              tilt,
        "azimuth":           azimuth,
        "losses":            14.0,
        "tilt_deviation":    tilt_deviation,
        "azimuth_deviation": azimuth_deviation,
        "latitude":          latitude,
        "solrad_annual":     weather["solrad_annual"],
    }])

    model = _load_model()
    annual_energy_kwh = float(model.predict(features)[0])

    install_cost_clp = system_size_kw * INSTALL_COST_PER_KWP
    savings = _annual_savings(annual_energy_kwh, years=10)
    cumulative_10y = sum(savings)
    payback_years = None
    cumulative = 0.0
    for i, s in enumerate(savings, start=1):
        cumulative += s
        if cumulative >= install_cost_clp:
            payback_years = i
            break

    return {
        "city": city,
        "system_size_kw": system_size_kw,
        "annual_energy_kwh": round(annual_energy_kwh, 1),
        "install_cost_clp": round(install_cost_clp),
        "annual_savings_clp": round(savings[0]),
        "savings_10y_clp": round(cumulative_10y),
        "payback_years": payback_years,
        "optimal_tilt_deg": round(abs(latitude), 1),
    }
