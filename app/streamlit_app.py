import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.predict import get_prediction
from src.config import CHILE_CITIES_LIST

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SOLAR_ML — Chile",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR = "#f4a261"
CITY_ORDER = ["Antofagasta", "Santiago", "Rancagua", "Concepción", "Temuco", "Punta Arenas"]

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    return pd.read_csv("data/processed/solar_chile_model.csv")

@st.cache_data
def load_model_results():
    with open("data/model_results.json") as f:
        return json.load(f)

@st.cache_data
def load_residuals():
    df = load_dataset()
    features = ["system_size_kw", "tilt", "azimuth", "losses",
                "tilt_deviation", "azimuth_deviation", "latitude", "solrad_annual"]
    X = df[features]
    y = df["ac_annual"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with open("src/models/gb_energy_model.pkl", "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    return pd.DataFrame({"real": y_test.values, "predicho": y_pred})

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"## ☀️ SOLAR_ML")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navegación",
    ["Resumen del Proyecto", "Explorar los Datos", "Resultados del Modelo", "Cómo se construyó"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-solar--chile-181717?logo=github)](https://github.com/JulioPradenas/solar-chile)"
)

# ── Página 1: Resumen ─────────────────────────────────────────────────────────
if page == "Resumen del Proyecto":
    st.markdown(f"# ☀️ SOLAR_ML")
    st.markdown("### Estimador de producción solar fotovoltaica para Chile")
    st.markdown(
        "Pipeline end-to-end que combina datos reales de irradiancia del API de NREL con un modelo "
        "Gradient Boosting para estimar generación anual, ahorro en la cuenta de la luz y período "
        "de recuperación de la inversión en 6 ciudades chilenas."
    )
    st.markdown("---")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instalaciones analizadas", "3,000", "500 por ciudad")
    c2.metric("Features engineered", "8", "tilt & azimuth deviation")
    c3.metric("R² del modelo", "0.9984", "+2.3 pp vs Random Forest")
    c4.metric("Reducción de error RMSE", "−74.7%", "810 → 205 kWh/año")

    st.markdown("---")
    st.subheader("Tech Stack")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Datos**")
        st.markdown("- NREL PVWatts v8 (intl)")
        st.markdown("- SQLite")
        st.markdown("- pandas 2.0")
    with col2:
        st.markdown("**Modelo**")
        st.markdown("- scikit-learn 1.3")
        st.markdown("- Gradient Boosting")
        st.markdown("- GridSearchCV (5-fold)")
    with col3:
        st.markdown("**Deploy**")
        st.markdown("- Flask (calculadora)")
        st.markdown("- Docker + Compose")
        st.markdown("- GitHub Actions CI/CD")

    st.markdown("---")
    st.subheader("Ciudades cubiertas")
    baseline = {
        "Ciudad":        CITY_ORDER,
        "Latitud":       ["−23.65°", "−33.45°", "−34.17°", "−36.82°", "−38.74°", "−53.16°"],
        "Irradiancia":   ["6.26 kWh/m²/d", "5.28", "5.27", "4.83", "4.81", "3.93"],
        "Producción (5 kWp)": ["9,009 kWh", "7,313", "7,297", "6,979", "6,938", "5,950"],
    }
    st.dataframe(pd.DataFrame(baseline), use_container_width=True, hide_index=True)

# ── Página 2: Explorar Datos ──────────────────────────────────────────────────
elif page == "Explorar los Datos":
    st.title("Explorar los Datos")
    df = load_dataset()

    # Filtro de ciudades
    cities = st.sidebar.multiselect(
        "Filtrar ciudades",
        CITY_ORDER,
        default=CITY_ORDER,
    )
    if not cities:
        st.warning("Selecciona al menos una ciudad.")
        st.stop()
    df_f = df[df["city"].isin(cities)]

    st.info("**Antofagasta produce un 51% más energía que Punta Arenas** para el mismo sistema, gracias a su irradiancia de 6.26 vs 3.93 kWh/m²/día.")
    st.success("**El tamaño del sistema explica el 90.7% de la varianza** en producción anual — el factor dominante del modelo.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de producción por ciudad")
        fig = px.box(
            df_f,
            x="city",
            y="ac_annual",
            color="city",
            category_orders={"city": CITY_ORDER},
            labels={"ac_annual": "Producción anual (kWh)", "city": "Ciudad"},
            color_discrete_sequence=px.colors.sequential.Oranges_r,
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tamaño del sistema vs producción")
        fig2 = px.scatter(
            df_f,
            x="system_size_kw",
            y="ac_annual",
            color="city",
            category_orders={"city": CITY_ORDER},
            labels={"system_size_kw": "Tamaño (kWp)", "ac_annual": "Producción anual (kWh)", "city": "Ciudad"},
            opacity=0.6,
        )
        fig2.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Irradiancia solar por ciudad")
        irr = pd.DataFrame({
            "Ciudad": CITY_ORDER,
            "kWh/m²/día": [6.26, 5.28, 5.27, 4.83, 4.81, 3.93],
        })
        irr_f = irr[irr["Ciudad"].isin(cities)]
        fig3 = px.bar(
            irr_f, x="Ciudad", y="kWh/m²/día",
            color="kWh/m²/día",
            color_continuous_scale="Oranges",
            labels={"kWh/m²/día": "Irradiancia solar (kWh/m²/día)"},
        )
        fig3.update_layout(showlegend=False, plot_bgcolor="white", coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Correlación entre variables")
        num_cols = ["system_size_kw", "tilt", "azimuth", "losses",
                    "tilt_deviation", "azimuth_deviation", "latitude",
                    "solrad_annual", "ac_annual"]
        corr = df_f[num_cols].corr().round(2)
        fig4 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
        )
        fig4.update_layout(height=420)
        st.plotly_chart(fig4, use_container_width=True)

# ── Página 3: Resultados del Modelo ──────────────────────────────────────────
elif page == "Resultados del Modelo":
    st.title("Resultados del Modelo")
    results = load_model_results()

    # Tabla comparativa
    st.subheader("Benchmark de modelos")
    models_df = pd.DataFrame(results["models"])
    models_df["Ganador"] = models_df["winner"].map({True: "✓", False: ""})
    models_df = models_df.rename(columns={"name": "Modelo", "r2": "R²", "rmse": "RMSE (kWh/año)"})
    st.dataframe(
        models_df[["Modelo", "R²", "RMSE (kWh/año)", "Ganador"]],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("¿Por qué ganó Gradient Boosting?"):
        st.markdown("""
- **3,000 filas no justifican deep learning** — GBM converge rápido y sin GPU
- **Captura interacciones no lineales** entre irradiancia y tamaño de sistema que la regresión lineal no puede
- **Interpretable**: las feature importances explican directamente qué variables mueven la predicción
- **R² 0.9984 = error de ~205 kWh/año** sobre producciones de 2,000–28,000 kWh — menos del 2% de error relativo
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
            fi_df, x="Importancia", y="Feature",
            orientation="h",
            color="Importancia",
            color_continuous_scale="Oranges",
            text=fi_df["Importancia"].apply(lambda v: f"{v:.1%}"),
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            plot_bgcolor="white",
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col2:
        st.subheader("Predicho vs Real (test set)")
        resid = load_residuals()
        fig_res = px.scatter(
            resid, x="real", y="predicho",
            labels={"real": "Producción real (kWh/año)", "predicho": "Predicción del modelo (kWh/año)"},
            opacity=0.5,
            color_discrete_sequence=[COLOR],
        )
        max_val = max(resid["real"].max(), resid["predicho"].max())
        fig_res.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="gray", dash="dash"),
        )
        fig_res.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_res, use_container_width=True)

    # Predictor interactivo
    st.markdown("---")
    st.subheader("Predictor interactivo")
    st.markdown("Ingresa los parámetros de tu instalación y el modelo estima la producción.")

    pc1, pc2, pc3, pc4 = st.columns(4)
    city_sel = pc1.selectbox("Ciudad", CITY_ORDER)
    kw_sel = pc2.slider("Tamaño (kWp)", 2.0, 15.0, 5.0, step=0.5)
    tilt_sel = pc3.slider("Inclinación (°)", 0, 70, 30)
    az_sel = pc4.slider("Azimuth (°)", 0, 360, 0)

    result = get_prediction(city_sel, system_size_kw=kw_sel, tilt=float(tilt_sel), azimuth=float(az_sel))
    if result:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Producción anual", f"{result['annual_energy_kwh']:,.0f} kWh")
        r2.metric("Costo instalación", f"${result['install_cost_clp']:,.0f} CLP")
        r3.metric("Ahorro año 1", f"${result['annual_savings_clp']:,.0f} CLP")
        r4.metric("Período de recuperación", f"{result['payback_years']} años")

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
    timeline = [
        ("Día 1", "NREL API + datos base de 6 ciudades chilenas"),
        ("Día 2", "EDA (01_eda.ipynb) + feature engineering (02_features.ipynb)"),
        ("Día 3", "Benchmark LR / RF / GB con GridSearchCV — GBM ganó con R²=0.9984"),
        ("Día 4", "Flask app + predictor financiero (payback, ahorro 10 años)"),
        ("Día 5", "Dockerfile + docker-compose + GitHub Actions CI/CD"),
        ("Día 6", "18 tests en pytest (datos, features, modelo, predict)"),
        ("Día 7", "App Streamlit multi-página para portfolio"),
    ]
    for day, desc in timeline:
        st.markdown(f"**{day}** — {desc}")

    st.markdown("---")
    st.subheader("Decisiones clave")

    with st.expander("¿Por qué datos sintéticos calibrados y no scraping?"):
        st.markdown(
            "No existe un dataset público de instalaciones reales en Chile. "
            "Se usa la fórmula física de PVWatts con variación realista en tilt y azimuth "
            "(distribución normal centrada en el óptimo), lo que genera 3,000 filas creíbles y trazables "
            "sin depender de una fuente que puede cambiar o desaparecer."
        )

    with st.expander("¿Por qué Gradient Boosting y no una red neuronal?"):
        st.markdown(
            "3,000 filas de entrenamiento no justifican el overhead de PyTorch o TensorFlow. "
            "GBM llega a R²=0.9984 en segundos, es interpretable via feature importances, "
            "y no requiere GPU ni tuning de arquitectura."
        )

    with st.expander("¿Por qué Flask + Streamlit y no solo uno?"):
        st.markdown(
            "Flask sirve la calculadora funcional (formulario limpio, sin dependencias pesadas). "
            "Streamlit sirve el portfolio showcase (visualizaciones interactivas, predictor en vivo). "
            "Ambos usan el mismo modelo y la misma función get_prediction() — no hay duplicación de lógica."
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
