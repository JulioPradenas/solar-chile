"""
Descarga datos de producción solar via PVWatts v8 API (NREL) para ciudades chilenas.
Usa dataset=intl que cubre Sudamérica.
Requiere: API key gratuita en https://developer.nrel.gov/signup/

Uso:
    python src/fetch_nrel.py --api-key TU_API_KEY
    python src/fetch_nrel.py --api-key TU_API_KEY --city Santiago
"""

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import CHILE_CITIES

DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

PVWATTS_URL = "https://developer.nrel.gov/api/pvwatts/v8.json"

# Parámetros fijos para todas las llamadas
PVWATTS_FIXED = {
    "dataset":     "intl",
    "array_type":  1,    # fixed roof mount
    "module_type": 0,    # standard
    "losses":      14,   # % pérdidas típicas residencial
}


def fetch_city_baseline(city: str, meta: dict, api_key: str) -> dict | None:
    """
    Una llamada por ciudad con parámetros óptimos para obtener solrad_annual
    (irradiancia real del sitio) que usaremos en el dataset de entrenamiento.
    """
    optimal_tilt = round(abs(meta["lat"]))
    params = {
        "api_key":         api_key,
        "lat":             meta["lat"],
        "lon":             meta["lon"],
        "system_capacity": 5,
        "tilt":            optimal_tilt,
        "azimuth":         0,   # norte en Chile
        **PVWATTS_FIXED,
    }
    try:
        r = requests.get(PVWATTS_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            print(f"  API error en {city}: {data['errors']}")
            return None
        return {
            "city":          city,
            "region":        meta["region"],
            "latitude":      meta["lat"],
            "longitude":     meta["lon"],
            "solrad_annual": data["outputs"]["solrad_annual"],   # kWh/m²/día promedio
            "ac_annual_ref": data["outputs"]["ac_annual"],       # kWh/año para 5kWp óptimo
        }
    except requests.RequestException as e:
        print(f"  Error en {city}: {e}")
        return None


def generate_training_data(baselines: list[dict], rng: np.random.Generator, n_per_city: int = 500) -> pd.DataFrame:
    """
    Genera el dataset de entrenamiento con instalaciones variadas.
    El target ac_annual se calcula con la fórmula física de PVWatts:
        AC = capacity × solrad × 365 × (1 - losses/100) × tilt_factor × (1 - temp_coeff)
    Calibrada con los valores reales de la API.
    """
    rows = []
    for b in baselines:
        lat = b["latitude"]
        optimal_tilt = abs(lat)
        solrad = b["solrad_annual"]

        # Factor de calibración: ajusta la fórmula para que coincida con el valor real de la API
        ac_ref = b["ac_annual_ref"]
        # ac_ref ≈ 5 × solrad × 365 × (1-0.14) × 1.0 × calib
        calib = ac_ref / (5 * solrad * 365 * 0.86)

        system_size = rng.uniform(2, 15, n_per_city)
        tilt = rng.normal(optimal_tilt, 8, n_per_city).clip(0, 70)
        azimuth = rng.normal(0, 25, n_per_city) % 360
        losses = rng.uniform(10, 20, n_per_city)

        # Penalización por inclinación incorrecta (coseno del error)
        tilt_factor = np.cos(np.radians(tilt - optimal_tilt)).clip(0.75, 1.0)

        # Penalización por orientación incorrecta (los paneles deben mirar al norte en Chile)
        azimuth_dev = np.abs(((azimuth + 180) % 360) - 180)  # 0-180, 0=norte
        azimuth_factor = np.cos(np.radians(azimuth_dev * 0.5)).clip(0.7, 1.0)

        ac_annual = (
            system_size
            * solrad * 365
            * (1 - losses / 100)
            * tilt_factor
            * azimuth_factor
            * calib
        )

        for i in range(n_per_city):
            rows.append({
                "city":            b["city"],
                "region":          b["region"],
                "latitude":        lat,
                "longitude":       b["longitude"],
                "system_size_kw":  round(system_size[i], 2),
                "tilt":            round(tilt[i], 1),
                "azimuth":         round(azimuth[i], 1),
                "losses":          round(losses[i], 1),
                "solrad_annual":   solrad,
                "ac_annual":       round(ac_annual[i], 1),
            })
    return pd.DataFrame(rows)


def save_to_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        df.to_sql("nrel_chile", conn, if_exists="replace", index=False)
    print(f"SQLite guardado en {db_path}")


def main(api_key: str, city_filter: str | None = None, n_per_city: int = 500) -> None:
    cities = (
        {city_filter: CHILE_CITIES[city_filter]}
        if city_filter and city_filter in CHILE_CITIES
        else CHILE_CITIES
    )

    baselines = []
    for city, meta in cities.items():
        print(f"Descargando {city}...")
        result = fetch_city_baseline(city, meta, api_key)
        if result:
            baselines.append(result)
            print(f"  solrad={result['solrad_annual']:.2f} kWh/m²/día  "
                  f"ac_ref={result['ac_annual_ref']:.0f} kWh/año (5kWp)")
        time.sleep(1)  # rate limit conservador

    if not baselines:
        print("No se descargaron datos.")
        return

    df_baseline = pd.DataFrame(baselines)
    df_baseline.to_csv(DATA_RAW / "pvwatts_chile_baseline.csv", index=False)
    save_to_sqlite(df_baseline, DATA_RAW / "nrel_chile.db")

    rng = np.random.default_rng(42)
    df_train = generate_training_data(baselines, rng, n_per_city=n_per_city)
    df_train.to_csv(DATA_RAW / "pvwatts_chile_training.csv", index=False)

    print(f"\nDataset generado: {df_train.shape}")
    print(df_baseline[["city", "region", "solrad_annual", "ac_annual_ref"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="NREL API key")
    parser.add_argument("--city", default=None, help="Ciudad específica")
    parser.add_argument("--n", type=int, default=500, help="Instalaciones sintéticas por ciudad (default: 500)")
    args = parser.parse_args()

    main(api_key=args.api_key, city_filter=args.city, n_per_city=args.n)
