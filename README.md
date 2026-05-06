# SOLAR_ML — Estimador de Energía Solar para Chile

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)
![Fuente de datos](https://img.shields.io/badge/Datos-NREL%20PVWatts%20v8-green)

**[Calculadora Flask](http://localhost:8080)** · **[Dashboard Streamlit](https://solar-chile-erbsppcztecexk2tykda4p.streamlit.app/)**

Calculadora de producción solar fotovoltaica para instalaciones residenciales en Chile. Combina datos reales de irradiancia del API de NREL con un modelo de Gradient Boosting para estimar generación anual, ahorro en la cuenta de la luz y período de recuperación de la inversión.

---

## Preguntas de negocio que responde

- ¿Cuántos kWh al año va a producir un sistema solar de X kWp en mi ciudad?
- ¿Cuántos años tarda en pagarse la instalación con la tarifa eléctrica actual?
- ¿Cuánto importa el ángulo de inclinación de los paneles en el hemisferio sur?
- ¿Cuál es la diferencia real de recurso solar entre Antofagasta y Punta Arenas?

---

## Contexto

Chile tiene una de las irradiancias solares más altas del mundo en su zona norte, pero la penetración residencial de paneles solares sigue siendo baja en parte por falta de herramientas accesibles de evaluación. Este proyecto construye un pipeline completo —desde la extracción de datos de irradiancia hasta una app web— para que instaladores y propietarios puedan estimar producción y retorno sin depender de cotizaciones opacas.

El modelo está calibrado geográficamente para las 6 principales ciudades de Chile (de Antofagasta a Punta Arenas), cubriendo el gradiente norte-sur de irradiancia del país.

---

## Arquitectura del pipeline

```
NREL PVWatts API v8  (dataset=intl, South America)
          │
          ▼
    fetch_nrel.py
    ├── 6 ciudades → baseline de irradiancia real
    └── 3,000 instalaciones sintéticas (500 por ciudad)
          │
          ▼
    01_eda.ipynb
    └── exploración de recurso solar por ciudad y latitud
          │
          ▼
    02_features.ipynb
    ├── tilt_deviation   = tilt − |latitud|
    ├── azimuth_deviation = desviación desde norte verdadero
    └── solar_chile_model.csv  (3,000 × 11 features)
          │
          ▼
    src/features/engineering.py
    ├── create_features()  →  +11 features físicas (20 columnas totales)
    └── select_features()  →  10 features no-redundantes (CV² + correlación)
          │
          ▼
    src/train.py  (MLflow tracking + Optuna 30 trials)
          │
          ▼
    src/models/gb_energy_model.pkl  +  data/selected_features.json
          │
          ├──► src/predict.py  →  estimación + cálculo financiero
          │
          ├──► app/server.py   →  http://localhost:8080
          │
          └──► app/streamlit_app.py  →  http://localhost:8501
```

---

## Exploración de datos

Dataset de entrenamiento: **3,000 instalaciones × 11 columnas**, sin valores nulos.
Generado sintéticamente con PVWatts para 6 ciudades (500 instalaciones por ciudad),
con variación realista en tilt y azimuth (distribución normal centrada en el óptimo geográfico).

**Patrones clave:**

- **system_size_kw domina la producción** (r = 0.99 con ac_annual). La feature importance del modelo final confirma esto: 90.7% del poder predictivo viene del tamaño del sistema.
- **Gradiente norte-sur pronunciado**: Antofagasta tiene 1.6× más irradiancia que Punta Arenas (6.26 vs 3.93 kWh/m²/día), lo que equivale a hasta 3,000 kWh/año de diferencia para el mismo sistema.
- **tilt_deviation y azimuth_deviation** están centradas en 0° con colas de ±15° — los instaladores tienden al ángulo óptimo pero con variación suficiente para entrenar el modelo.
- **Alta correlación entre features engineered**: `select_features()` descartó 8 de 20 columnas (p.ej. `theoretical_max_kwh` ≈ `size_x_solrad` con r = 0.99), dejando 11 features no redundantes para el modelo.

---

## Resultados del modelo

Se evaluaron tres algoritmos con 5-fold cross-validation, el mejor se tuneó con Optuna (30 trials) y finalmente se integró feature engineering con selección automática de features. Conjunto de entrenamiento: 2,400 filas; prueba: 600.

| Modelo                                       | R²         | RMSE (kWh/año) |
|----------------------------------------------|------------|----------------|
| Regresión Lineal (8 features base)           | 0.9753     | 810.5          |
| Random Forest                                | 0.9936     | 411.1          |
| GBM (GridSearchCV)                           | 0.9984     | 204.9          |
| GBM (Optuna 30 trials)                       | 0.9992     | 147.7          |
| Regresión Lineal (10 features engineeradas)  | 0.9960     | 326.7          |
| **GBM + Optuna + feature engineering**       | **0.9995** | **113.4**      |

El error final de 113 kWh/año equivale al ~1.5% de la producción típica de un sistema de 5 kWp en Santiago. El feature engineering redujo el RMSE un 23% adicional respecto a Optuna solo.

---

## Cobertura de ciudades

Datos reales de irradiancia obtenidos del API de NREL (dataset internacional, mediciones satelitales).

| Ciudad       | Latitud  | Irradiancia solar     | Producción anual (5 kWp óptimo) |
|--------------|----------|-----------------------|----------------------------------|
| Antofagasta  | −23.65°  | 6.26 kWh/m²/día       | 9,009 kWh                        |
| Santiago     | −33.45°  | 5.28 kWh/m²/día       | 7,313 kWh                        |
| Rancagua     | −34.17°  | 5.27 kWh/m²/día       | 7,297 kWh                        |
| Concepción   | −36.82°  | 4.83 kWh/m²/día       | 6,979 kWh                        |
| Temuco       | −38.74°  | 4.81 kWh/m²/día       | 6,938 kWh                        |
| Punta Arenas | −53.16°  | 3.93 kWh/m²/día       | 5,950 kWh                        |

---

## Stack técnico

| Capa                 | Herramienta              | Por qué                                                       |
|----------------------|--------------------------|---------------------------------------------------------------|
| Fuente de datos      | NREL PVWatts v8 (intl)   | Único API público con datos validados para Chile              |
| Almacenamiento       | SQLite                   | Zero-config, suficiente para 6 ciudades base                  |
| Feature engineering  | pandas + numpy           | 11 features físicamente motivadas + filtro CV² + correlación  |
| ML                   | scikit-learn GBM         | Mejor R² del benchmark sin overhead de deep learning          |
| Experiment tracking  | MLflow                   | 32 runs registrados, comparación visual de trials             |
| Hyperparameter tuning| Optuna                   | Búsqueda bayesiana, 30 trials, mejora 23% vs GridSearchCV     |
| Calculadora web      | Flask                    | App sin estado, sin autenticación — no necesita más           |
| Dashboard portfolio  | Streamlit                | 4 páginas interactivas, funciona sin API key (demo mode)      |
| Contenedores         | Docker + Compose         | Un comando para correr todo: `docker-compose up`              |
| CI/CD                | GitHub Actions           | test + lint (ruff) + docker-build en cada push               |
| Tests                | pytest (18 tests)        | Cobertura: datos → features → modelo → API end-to-end         |
| Notebooks            | Jupyter                  | Pipeline reproducible y auditable paso a paso                 |

---

## Feature engineering

`create_features()` agrega 11 features físicamente motivadas al dataset base (8 columnas).
`select_features()` filtra las 20 resultantes por CV² (baja varianza) y correlación de Pearson
(umbral 0.95), dejando las 10 features que usa el modelo final.

| Feature                  | Categoría       | Justificación física                                               |
|--------------------------|-----------------|---------------------------------------------------------------------|
| `system_size_kw`         | Base            | Domina la producción (r=0.99 con target)                           |
| `tilt`                   | Base            | Ángulo de inclinación del panel                                    |
| `azimuth`                | Base            | Orientación absoluta del panel (0°=norte)                          |
| `losses`                 | Base            | Pérdidas del sistema declaradas (%)                                |
| `tilt_deviation`         | Orientación     | tilt − \|latitud\|: desviación del ángulo óptimo geográfico        |
| `azimuth_deviation`      | Orientación     | Desviación de norte verdadero (0°=óptimo en hemisferio sur)        |
| `latitude`               | Geografía       | Determina ángulo solar y recurso disponible                        |
| `azimuth_cos_factor`     | Orientación     | cos(azimuth_deviation): pérdida continua por orientación errónea   |
| `theoretical_max_kwh`    | Capacidad       | loss_adjusted_capacity × solrad × 365: techo físico de producción  |
| `size_x_orientation`     | Interacción     | system_size_kw × combined_orientation_factor: tamaño × orientación |

**8 features descartadas por `select_features()`:** `solrad_annual` ↔ `latitude` (r=0.98),
`effective_irradiance` ↔ `latitude`, `loss_adjusted_capacity` ↔ `system_size_kw` (r=0.99),
`size_x_solrad` ↔ `theoretical_max_kwh` (r=0.99), entre otras.

---

## Decisiones de diseño

**Datos sintéticos calibrados en lugar de scraping.**
No existe un dataset público de instalaciones reales en Chile. En lugar de inventar datos, se usa la fórmula física de PVWatts con variación realista en tilt y azimuth (distribución normal centrada en el óptimo), lo que genera un dataset creíble y trazable.

**SQLite sobre CSV para la app.**
La app consulta irradiancia base por ciudad en cada predicción. SQLite permite agregar ciudades o métricas sin cambiar el código de la app; un CSV requeriría cargar todo el archivo en memoria.

**Gradient Boosting sobre redes neuronales.**
3,000 filas de entrenamiento no justifican deep learning. GBM llega a R² = 0.9984 en segundos, es interpretable y no requiere GPU.

**Flask sobre FastAPI o Django.**
La app es una calculadora de una sola pantalla sin autenticación ni base de usuarios. Flask con dos rutas (`/` y `/output`) es todo lo que se necesita.

**El primer filtro de varianza eliminaba demasiado — y en silencio.**
La primera versión de `select_features()` usaba varianza absoluta como umbral, lo que eliminaba
features de escala pequeña como `azimuth_cos_factor` (valores entre -1 y 1) aunque fueran
informativamente relevantes. El segundo intento usó varianza relativa al máximo: mismo problema.
La solución fue CV² (coeficiente de variación al cuadrado = var/mean²), invariante a la escala.
Lección: los filtros de varianza genéricos fallan en datasets con features en escalas distintas.

---

## Cálculo financiero

La estimación de retorno usa parámetros del mercado chileno:

| Parámetro              | Valor            |
|------------------------|------------------|
| Tarifa eléctrica base  | 130 CLP/kWh      |
| Escalación anual       | 4%               |
| Costo de instalación   | 900,000 CLP/kWp  |

El período de recuperación se calcula sobre ahorro acumulado a 10 años con inflación de tarifa compuesta.

---

## Inicio rápido

```bash
git clone https://github.com/JulioPradenas/solar-chile
cd solar-chile
pip install -r requirements.txt
```

**Opción A: usar los datos ya generados (sin API key)**

```bash
python src/train.py          # feature engineering + Optuna + guarda pkl
python app/server.py         # → http://localhost:8080
```

**Opción B: Docker (sin instalar dependencias)**

```bash
docker-compose up            # → http://localhost:8080
```

**Opción C: re-fetch desde NREL (requiere API key gratuita)**

```bash
# Registrarse en https://developer.nrel.gov/signup/
python src/fetch_nrel.py --api-key TU_KEY
# Luego: 01_eda.ipynb → 02_features.ipynb → 03_modeling.ipynb
```

---

## Cómo correr cada componente

```bash
# Modelo: entrenar con feature engineering + Optuna
python src/train.py
# → guarda src/models/gb_energy_model.pkl + data/selected_features.json

# Calculadora Flask
python app/server.py
# → http://localhost:8080

# Dashboard Streamlit
streamlit run app/streamlit_app.py
# → http://localhost:8501

# Tests (18 tests en 4 archivos)
pytest tests/ -v

# MLflow UI — ver los 32 runs registrados
mlflow ui --port 5001
# → http://localhost:5001

# Docker
docker-compose up
# → Flask en :8080 + Streamlit en :8501
```

---

## Estructura del proyecto

```
solar-chile/
├── app/
│   ├── server.py              # Flask calculadora — rutas / y /output (puerto 8080)
│   ├── streamlit_app.py       # Dashboard portfolio 4 páginas (puerto 8501)
│   ├── templates/
│   │   ├── index.html
│   │   └── output.html
│   └── static/style.css
├── data/
│   ├── raw/
│   │   ├── pvwatts_chile_baseline.csv   # Irradiancia real — 6 ciudades
│   │   ├── pvwatts_chile_training.csv   # 3,000 instalaciones sintéticas
│   │   └── nrel_chile.db                # SQLite baseline para inferencia
│   ├── processed/
│   │   ├── solar_chile_model.csv        # Dataset base (3,000 × 11)
│   │   └── solar_chile_features.csv     # Dataset enriquecido (3,000 × 22)
│   ├── selected_features.json           # Lista de 10 features usadas por el modelo
│   ├── best_params.json                 # Mejores hiperparámetros Optuna
│   └── model_results.json               # Benchmark de los 4 modelos
├── notebooks/
│   ├── 01_eda.ipynb           # EDA baseline + dataset de entrenamiento
│   ├── 02_features.ipynb      # Engineering de tilt/azimuth deviation
│   └── 03_modeling.ipynb      # Benchmark GridSearchCV
├── src/
│   ├── config.py              # Ciudades, coordenadas, parámetros financieros
│   ├── fetch_nrel.py          # Cliente NREL PVWatts v8
│   ├── predict.py             # get_prediction() — inferencia + ROI
│   ├── train.py               # Pipeline MLflow + Optuna (reentrenar)
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py     # create_features() + select_features()
│   └── models/
│       └── gb_energy_model.pkl
├── tests/
│   ├── test_data_quality.py   # 4 tests — quality gate del dataset
│   ├── test_features.py       # 6 tests — columnas, NaN, rangos
│   ├── test_model.py          # 5 tests — pkl, predicción, gradiente norte-sur
│   └── test_predict.py        # 3 tests — API end-to-end
├── .github/
│   └── workflows/ci.yml       # CI: test + lint (ruff) + docker-build
├── Dockerfile                 # python:3.11-slim, expone puerto 8080
├── docker-compose.yml         # services: web (8080) + streamlit (8501)
├── conftest.py                # sys.path fix para pytest
├── requirements.txt
└── README.md
```
