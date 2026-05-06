"""
Feature engineering para el modelo de producción solar en Chile.

Todas las features tienen justificación física — no son transformaciones mecánicas.
La función create_features() es el contrato público de este módulo.

Uso:
    from src.features.engineering import create_features
    df_enriched = create_features(df_raw)
"""

import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features físicamente motivadas al DataFrame de instalaciones solares.

    Requiere columnas:
        system_size_kw, tilt, azimuth, losses,
        tilt_deviation, azimuth_deviation, latitude, solrad_annual
    Columna ac_annual es opcional (disponible solo en training data).

    Retorna un nuevo DataFrame con las columnas originales más las features
    enriquecidas. No modifica el DataFrame de entrada.
    """
    out = df.copy()

    # ── Categoría 1: Eficiencia de orientación del panel ─────────────────────
    # Física: la irradiancia que captura un panel es proporcional al coseno del
    # ángulo de incidencia. Un panel perfectamente inclinado captura el 100%.

    # Irradiancia efectiva = solrad × cos(desviación de inclinación respecto al óptimo).
    # Cuanto más se aleja el tilt de |latitud|, menos irradiancia llega al panel.
    out["effective_irradiance"] = (
        out["solrad_annual"] * np.cos(np.radians(out["tilt_deviation"]))
    )

    # Factor de pérdida por azimuth. En hemisferio sur, norte = 0°.
    # cos(0°) = 1.0 (óptimo), cos(90°) = 0.0 (pérdida total lateral).
    out["azimuth_cos_factor"] = np.cos(np.radians(out["azimuth_deviation"]))

    # Factor de orientación combinado: penaliza simultáneamente tilt Y azimuth erróneos.
    # Permite que el modelo capture el efecto multiplicativo de ambos errores.
    out["combined_orientation_factor"] = (
        np.cos(np.radians(out["tilt_deviation"]))
        * np.cos(np.radians(out["azimuth_deviation"]))
    )

    # ── Categoría 2: Capacidad efectiva del sistema ───────────────────────────
    # Física: las pérdidas del sistema (cableado, temperatura, suciedad, inversores)
    # reducen la capacidad nominal declarada. Un sistema de 5 kWp con 14% de pérdidas
    # opera efectivamente como uno de 4.3 kWp.

    # Capacidad real tras deducir pérdidas declaradas del sistema.
    out["loss_adjusted_capacity"] = out["system_size_kw"] * (1 - out["losses"] / 100)

    # Producción teórica máxima: capacidad efectiva × irradiancia disponible × días/año.
    # Es el techo físico de producción dada la ubicación, sin errores de orientación.
    out["theoretical_max_kwh"] = (
        out["loss_adjusted_capacity"] * out["solrad_annual"] * 365
    )

    # Factor de capacidad: fracción del tiempo que el sistema produce a plena capacidad.
    # Solo se calcula si ac_annual está disponible (datos de training con target conocido).
    # Referencia: 8760 horas/año × system_size_kw = producción a pleno rendimiento 24/7.
    if "ac_annual" in out.columns:
        out["capacity_factor"] = out["ac_annual"] / (out["system_size_kw"] * 8760)

    # ── Categoría 3: Geografía y recurso solar ───────────────────────────────
    # Física: la latitud determina el ángulo solar óptimo y la intensidad del recurso.
    # En el hemisferio sur, el tilt óptimo ≈ |latitud| (paneles mirando al norte).

    # Ratio tilt/latitud: 1.0 = instalación con ángulo perfectamente óptimo.
    # Valores > 1.0 o < 1.0 indican sobre o subinclinación respecto al óptimo geográfico.
    out["latitude_tilt_ratio"] = out["tilt"] / out["latitude"].abs()

    # Distancia angular mínima al norte verdadero.
    # Captura la penalización por orientación este/oeste de forma simétrica.
    out["north_distance"] = out["azimuth"].apply(lambda a: min(a, 360 - a))

    # Irradiancia normalizada por latitud: compara el recurso solar de una ciudad
    # respecto a su posición geográfica. Antofagasta tiene mayor solrad/°lat que Temuco.
    out["solrad_per_lat_degree"] = out["solrad_annual"] / out["latitude"].abs()

    # ── Categoría 4: Interacciones multiplicativas ────────────────────────────
    # Física: dos variables juntas explican más que por separado cuando tienen
    # efecto conjunto. Para producción solar, tamaño × irradiancia es la interacción
    # dominante (explica el ~90% de la varianza según feature importances).

    # Interacción principal: tamaño × irradiancia disponible.
    # Es la aproximación más directa a la producción bruta antes de pérdidas y orientación.
    out["size_x_solrad"] = out["system_size_kw"] * out["solrad_annual"]

    # Interacción tamaño × orientación real: producción esperada ajustada por
    # cuán bien está orientado el sistema. Un sistema grande mal orientado puede
    # producir menos que uno pequeño bien orientado.
    out["size_x_orientation"] = out["system_size_kw"] * out["combined_orientation_factor"]

    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    input_path = Path("data/processed/solar_chile_model.csv")
    output_path = Path("data/processed/solar_chile_features.csv")

    df = pd.read_csv(input_path)
    print(f"Input:  {df.shape} — {input_path}")

    df_out = create_features(df)
    new_cols = [c for c in df_out.columns if c not in df.columns]

    print(f"Output: {df_out.shape}")
    print(f"Nuevas features ({len(new_cols)}):")
    for col in new_cols:
        print(f"  {col:35s}  min={df_out[col].min():.3f}  max={df_out[col].max():.3f}")

    df_out.to_csv(output_path, index=False)
    print(f"\nGuardado → {output_path}")
