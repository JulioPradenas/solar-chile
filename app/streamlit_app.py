import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOLAR_ML — Chile",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR = "#f4a261"
CITY_ORDER = ["Antofagasta", "Santiago", "Rancagua", "Concepción", "Temuco", "Punta Arenas"]
CITY_SOLRAD = {
    "Antofagasta": 6.26, "Santiago": 5.28, "Rancagua": 5.27,
    "Concepción": 4.83, "Temuco": 4.81, "Punta Arenas": 3.93,
}
CITY_LAT = {
    "Antofagasta": -23.65, "Santiago": -33.45, "Rancagua": -34.17,
    "Concepción": -36.82, "Temuco": -38.74, "Punta Arenas": -53.16,
}

DEMO_MODE = False  # activado si algún archivo crítico falta

# ── Data loaders con fallback ─────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    global DEMO_MODE
    csv = Path("data/processed/solar_chile_model.csv")
    if csv.exists():
        return pd.read_csv(csv)
    DEMO_MODE = True
    rng = np.random.default_rng(42)
    rows = []
    for city, lat, solrad in [
        ("Antofagasta", -23.65, 6.26), ("Santiago", -33.45, 5.28),
        ("Rancagua", -34.17, 5.27),   ("Concepción", -36.82, 4.83),
        ("Temuco", -38.74, 4.81),     ("Punta Arenas", -53.16, 3.93),
    ]:
        for _ in range(10):
            kw = rng.uniform(2, 15)
            rows.append({
                "city": city, "region": "", "latitude": lat,
                "system_size_kw": round(kw, 2), "tilt": abs(lat),
                "azimuth": 0, "losses": 14, "solrad_annual": solrad,
                "ac_annual": round(kw * solrad * 365 * 0.8, 1),
                "tilt_deviation": 0, "azimuth_deviation": 0,
            })
    return pd.DataFrame(rows)


@st.cache_data
def load_model_results():
    global DEMO_MODE
    p = Path("data/model_results.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    DEMO_MODE = True
    return {
        "models": [
            {"name": "Regresión Lineal",  "r2": 0.9753, "rmse": 810.5, "winner": False},
            {"name": "Random Forest",     "r2": 0.9936, "rmse": 411.1, "winner": False},
            {"name": "Gradient Boosting", "r2": 0.9984, "rmse": 204.9, "winner": True},
        ],
        "feature_importances": {
            "system_size_kw": 0.9072, "solrad_annual": 0.0404,
            "latitude": 0.0394, "losses": 0.0058,
            "azimuth_deviation": 0.0056, "tilt_deviation": 0.0006,
            "azimuth": 0.0005, "tilt": 0.0005,
        },
    }


@st.cache_data
def load_residuals():
    global DEMO_MODE
    pkl = Path("src/models/gb_energy_model.pkl")
    features_json = Path("data/selected_features.json")
    if pkl.exists() and features_json.exists():
        df = load_dataset()
        if "ac_annual" in df.columns:
            from src.features.engineering import create_features
            df_enriched = create_features(df)
            with open(features_json) as f:
                selected = json.load(f)
            if all(c in df_enriched.columns for c in selected):
                X = df_enriched[selected]
                y = df["ac_annual"]
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                with open(pkl, "rb") as f:
                    model = pickle.load(f)
                y_pred = model.predict(X_test)
                return pd.DataFrame({"real": y_test.values, "predicho": y_pred})
    DEMO_MODE = True
    rng = np.random.default_rng(42)
    real = rng.uniform(2000, 28000, 600)
    pred = real + rng.normal(0, 200, 600)
    return pd.DataFrame({"real": real, "predicho": pred})


def _predict(city, kw, tilt, azimuth):
    """Devuelve dict de predicción del modelo o estimación física si el modelo no está."""
    try:
        from src.predict import get_prediction
        result = get_prediction(city, system_size_kw=kw, tilt=tilt, azimuth=azimuth)
        if result:
            return result, False
    except Exception:
        pass
    kwh = round(kw * CITY_SOLRAD.get(city, 5.0) * 365 * 0.8)
    return {"annual_energy_kwh": kwh, "install_cost_clp": int(kw * 900_000),
            "annual_savings_clp": int(kwh * 130), "payback_years": None}, True


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ☀️ SOLAR_ML")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navegación",
    ["Resumen del Proyecto", "Explorar los Datos", "Resultados del Modelo", "Cómo se construyó"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-solar--chile-181717?logo=github)]"
    "(https://github.com/JulioPradenas/solar-chile)"
)

# Cargar datos (activa DEMO_MODE si algo falta)
_df_check = load_dataset()
_mr_check = load_model_results()
_re_check = load_residuals()

if DEMO_MODE:
    st.warning(
        "Modo demo — algún archivo no se encontró. "
        "Clona el repo completo para datos y modelo reales.",
        icon="⚠️",
    )

# ── Página 1: Resumen ─────────────────────────────────────────────────────────
if page == "Resumen del Proyecto":
    st.markdown("# ☀️ SOLAR_ML")
    st.markdown("### Estimador de producción solar fotovoltaica para Chile")
    st.markdown(
        "Pipeline end-to-end que combina datos reales de irradiancia del API de NREL con un modelo "
        "Gradient Boosting para estimar generación anual, ahorro en la cuenta de la luz y período "
        "de recuperación de la inversión en 6 ciudades chilenas."
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instalaciones analizadas", "3,000", "500 por ciudad")
    c2.metric("Features engineered", "8", "tilt & azimuth deviation")
    c3.metric("R² del modelo", "0.9984", "+2.3 pp vs Random Forest")
    c4.metric("Reducción RMSE vs baseline", "−74.7%", "810 → 205 kWh/año")

    st.markdown("---")
    st.subheader("Tech Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Datos**")
        st.markdown("- NREL PVWatts v8 (intl)\n- SQLite\n- pandas 2.0")
    with col2:
        st.markdown("**Modelo**")
        st.markdown("- scikit-learn 1.3\n- Gradient Boosting\n- GridSearchCV (5-fold)")
    with col3:
        st.markdown("**Deploy**")
        st.markdown("- Flask (calculadora)\n- Docker + Compose\n- GitHub Actions CI/CD")

    st.markdown("---")
    st.subheader("Ciudades cubiertas")
    st.dataframe(pd.DataFrame({
        "Ciudad":            CITY_ORDER,
        "Latitud":           ["−23.65°", "−33.45°", "−34.17°", "−36.82°", "−38.74°", "−53.16°"],
        "Irradiancia":       [f"{CITY_SOLRAD[c]} kWh/m²/d" for c in CITY_ORDER],
        "Producción (5 kWp)": ["9,009 kWh", "7,313", "7,297", "6,979", "6,938", "5,950"],
    }), use_container_width=True, hide_index=True)

# ── Página 2: Explorar Datos ──────────────────────────────────────────────────
elif page == "Explorar los Datos":
    st.title("Explorar los Datos")
    df = load_dataset()

    cities = st.sidebar.multiselect("Filtrar ciudades", CITY_ORDER, default=CITY_ORDER)
    if not cities:
        st.warning("Selecciona al menos una ciudad.")
        st.stop()
    df_f = df[df["city"].isin(cities)]

    st.info("**Antofagasta produce un 51% más energía que Punta Arenas** para el mismo sistema.")
    st.success("**El tamaño del sistema explica el 90.7% de la varianza** en producción anual.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de producción por ciudad")
        fig = px.box(
            df_f, x="city", y="ac_annual", color="city",
            category_orders={"city": CITY_ORDER},
            labels={"ac_annual": "Producción anual (kWh)", "city": "Ciudad"},
            color_discrete_sequence=px.colors.sequential.Oranges_r,
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tamaño del sistema vs producción")
        fig2 = px.scatter(
            df_f, x="system_size_kw", y="ac_annual", color="city",
            category_orders={"city": CITY_ORDER},
            labels={"system_size_kw": "Tamaño (kWp)", "ac_annual": "Producción anual (kWh)", "city": "Ciudad"},
            opacity=0.6,
        )
        fig2.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Irradiancia solar por ciudad")
        irr_f = pd.DataFrame({
            "Ciudad": [c for c in CITY_ORDER if c in cities],
            "kWh/m²/día": [CITY_SOLRAD[c] for c in CITY_ORDER if c in cities],
        })
        fig3 = px.bar(
            irr_f, x="Ciudad", y="kWh/m²/día",
            color="kWh/m²/día", color_continuous_scale="Oranges",
        )
        fig3.update_layout(showlegend=False, plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Correlación entre variables")
        num_cols = [c for c in ["system_size_kw", "tilt", "azimuth", "losses",
                    "tilt_deviation", "azimuth_deviation", "latitude",
                    "solrad_annual", "ac_annual"] if c in df_f.columns]
        corr = df_f[num_cols].corr().round(2)
        fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig4.update_layout(height=420)
        st.plotly_chart(fig4, use_container_width=True)

# ── Página 3: Resultados del Modelo ──────────────────────────────────────────
elif page == "Resultados del Modelo":
    st.title("Resultados del Modelo")
    results = load_model_results()

    st.subheader("Benchmark de modelos")
    models_df = pd.DataFrame(results["models"])
    models_df["Ganador"] = models_df["winner"].map({True: "✓", False: ""})
    models_df = models_df.rename(columns={"name": "Modelo", "r2": "R²", "rmse": "RMSE (kWh/año)"})
    st.dataframe(
        models_df[["Modelo", "R²", "RMSE (kWh/año)", "Ganador"]],
        use_container_width=True, hide_index=True,
    )

    with st.expander("¿Por qué ganó Gradient Boosting?"):
        st.markdown("""
- **3,000 filas no justifican deep learning** — GBM converge rápido y sin GPU
- **Captura interacciones no lineales** entre irradiancia y tamaño que la regresión lineal no puede
- **Interpretable**: las feature importances explican qué variables mueven la predicción
- **R² 0.9984 = error de ~205 kWh/año** — menos del 2% de error relativo
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Importancia de features")
        fi = results["feature_importances"]
        fi_df = pd.DataFrame(
            sorted(fi.items(), key=lambda x: x[1]),
            columns=["Feature", "Importancia"],
        )
        fig_fi = px.bar(
            fi_df, x="Importancia", y="Feature", orientation="h",
            color="Importancia", color_continuous_scale="Oranges",
            text=fi_df["Importancia"].apply(lambda v: f"{v:.1%}"),
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="white", xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col2:
        st.subheader("Predicho vs Real (test set)")
        resid = load_residuals()
        max_val = max(resid["real"].max(), resid["predicho"].max())
        fig_res = px.scatter(
            resid, x="real", y="predicho", opacity=0.5,
            labels={"real": "Producción real (kWh/año)", "predicho": "Predicción (kWh/año)"},
            color_discrete_sequence=[COLOR],
        )
        fig_res.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="gray", dash="dash"),
        )
        fig_res.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")
    st.subheader("Predictor interactivo")
    st.markdown("Ingresa los parámetros de tu instalación y el modelo estima la producción.")

    pc1, pc2, pc3, pc4 = st.columns(4)
    city_sel = pc1.selectbox("Ciudad", CITY_ORDER)
    kw_sel   = pc2.slider("Tamaño (kWp)", 2.0, 15.0, 5.0, step=0.5)
    tilt_sel = pc3.slider("Inclinación (°)", 0, 70, 30)
    az_sel   = pc4.slider("Azimuth (°)", 0, 360, 0)

    result, is_demo = _predict(city_sel, kw_sel, float(tilt_sel), float(az_sel))
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Producción anual", f"{result['annual_energy_kwh']:,.0f} kWh")
    r2.metric("Costo instalación", f"${result['install_cost_clp']:,.0f} CLP")
    r3.metric("Ahorro año 1", f"${result['annual_savings_clp']:,.0f} CLP")
    if result.get("payback_years"):
        r4.metric("Recuperación", f"{result['payback_years']} años")
    else:
        r4.metric("Recuperación", "—")
    if is_demo:
        st.caption("Estimación física simplificada — modelo no disponible.")

# ── Página 4: Cómo se construyó ───────────────────────────────────────────────
elif page == "Cómo se construyó":
    st.title("Cómo se construyó")

    st.subheader("Arquitectura del pipeline")
    st.code("""
NREL PVWatts API v8  (dataset=intl, South America)
          │
          ▼
    fetch_nrel.py
    ├── 6 ciudades → baseline de irradiancia real
    └── 3,000 instalaciones sintéticas (500 por ciudad)
          │
          ▼
    01_eda.ipynb  →  exploración de recurso solar por ciudad
          │
          ▼
    02_features.ipynb
    ├── tilt_deviation   = tilt − |latitud|
    └── azimuth_deviation = desviación desde norte verdadero
          │
          ▼
    03_modeling.ipynb → GridSearchCV (LR / RF / GB) → best model
          │
          ▼
    src/models/gb_energy_model.pkl
          ├──► src/predict.py  →  estimación + cálculo financiero
          ├──► app/server.py   →  Flask calculadora (puerto 8080)
          └──► app/streamlit_app.py  →  portfolio showcase (8501)
    """, language="text")

    st.markdown("---")
    st.subheader("Timeline del build")
    for day, desc in [
        ("Día 1", "NREL API + datos base de 6 ciudades chilenas"),
        ("Día 2", "EDA (01_eda.ipynb) + feature engineering (02_features.ipynb)"),
        ("Día 3", "Benchmark LR / RF / GB con GridSearchCV — GBM ganó con R²=0.9984"),
        ("Día 4", "Flask app + predictor financiero (payback, ahorro 10 años)"),
        ("Día 5", "Dockerfile + docker-compose + GitHub Actions CI/CD"),
        ("Día 6", "18 tests en pytest (datos, features, modelo, predict)"),
        ("Día 7", "App Streamlit multi-página para portfolio"),
    ]:
        st.markdown(f"**{day}** — {desc}")

    st.markdown("---")
    st.subheader("Decisiones clave")

    with st.expander("¿Por qué datos sintéticos calibrados y no scraping?"):
        st.markdown(
            "No existe un dataset público de instalaciones reales en Chile. "
            "Se usa la fórmula física de PVWatts con variación realista en tilt y azimuth "
            "(distribución normal centrada en el óptimo), lo que genera 3,000 filas creíbles y trazables."
        )
    with st.expander("¿Por qué Gradient Boosting y no una red neuronal?"):
        st.markdown(
            "3,000 filas no justifican PyTorch ni TensorFlow. GBM llega a R²=0.9984 en segundos, "
            "es interpretable vía feature importances y no requiere GPU."
        )
    with st.expander("¿Por qué Flask + Streamlit y no solo uno?"):
        st.markdown(
            "Flask sirve la calculadora funcional (formulario limpio, sin dependencias pesadas). "
            "Streamlit sirve el portfolio showcase. Ambos usan el mismo get_prediction() — sin duplicación de lógica."
        )

    st.markdown("---")
    st.link_button("Ver código en GitHub", "https://github.com/JulioPradenas/solar-chile")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85rem'>"
    "SOLAR_ML · Julio Pradenas · Datos: NREL PVWatts v8 · Modelo: Gradient Boosting (R²=0.9984)"
    "</div>",
    unsafe_allow_html=True,
)
