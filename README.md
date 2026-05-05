# SOLAR_ML — Estimador de Energía Solar para Chile

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey)
![Fuente de datos](https://img.shields.io/badge/Datos-NREL%20PVWatts%20v8-green)

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
    └── solar_chile_model.csv  (3,000 × 13 features)
          │
          ▼
    03_modeling.ipynb
    └── GridSearchCV: LinearRegression / RandomForest / GradientBoosting
          │
          ▼
    src/models/gb_energy_model.pkl
          │
          ├──► src/predict.py  →  estimación + cálculo financiero
          │
          └──► app/server.py   →  http://localhost:8080
```

---

## Resultados del modelo

Se evaluaron tres algoritmos con 5-fold cross-validation. El conjunto de entrenamiento tiene 2,400 filas y el de prueba 600.

| Modelo              | R²         | RMSE (kWh/año) |
|---------------------|------------|----------------|
| Regresión Lineal    | 0.9753     | 810.5          |
| Random Forest       | 0.9936     | 411.1          |
| **Gradient Boosting** | **0.9984** | **204.9**   |

El error de 204 kWh/año equivale al ~2% de la producción típica de un sistema de 5 kWp en Santiago (7,300 kWh/año), margen aceptable para estimaciones residenciales.

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

| Capa           | Herramienta            | Por qué                                                      |
|----------------|------------------------|--------------------------------------------------------------|
| Fuente de datos | NREL PVWatts v8 (intl) | Único API público con datos validados para Chile             |
| Almacenamiento | SQLite                 | Zero-config, suficiente para 6 ciudades base                 |
| ML             | scikit-learn GBM       | Mejor R² del benchmark sin overhead de deep learning         |
| Web            | Flask                  | App sin estado, sin autenticación — no necesita más          |
| Notebooks      | Jupyter                | Pipeline reproducible y auditable paso a paso                |

---

## Feature engineering

Las dos variables más importantes del modelo capturan la desviación del instalador respecto al ángulo óptimo para el hemisferio sur:

- **`tilt_deviation`** = tilt − |latitud|
  Un panel en Santiago instalado a 33° (= latitud) tiene desviación cero. Cada grado de error baja la producción.

- **`azimuth_deviation`** = diferencia normalizada respecto a norte verdadero (0°)
  En el hemisferio sur, los paneles deben mirar al norte. La desviación penaliza producción de forma asimétrica.

Features del modelo: `system_size_kw`, `tilt`, `azimuth`, `losses`, `solrad_annual`, `latitude`, `tilt_deviation`, `azimuth_deviation`.

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
# Entrenar el modelo directamente
jupyter nbconvert --execute notebooks/03_modeling.ipynb \
  --to notebook --output-dir notebooks/

# Levantar la app
python app/server.py
# → http://localhost:8080
```

**Opción B: re-fetch de datos desde NREL (requiere API key gratuita)**

```bash
# Registrarse en https://developer.nrel.gov/signup/
python src/fetch_nrel.py --api-key TU_KEY

# Luego seguir con los notebooks en orden:
# 01_eda.ipynb → 02_features.ipynb → 03_modeling.ipynb
```

---

## Estructura del proyecto

```
SOLAR_ML/
├── app/
│   ├── server.py              # Flask app — rutas / y /output
│   ├── templates/
│   │   ├── index.html         # Formulario de entrada
│   │   └── output.html        # Resultados (6 métricas)
│   └── static/style.css
├── data/
│   ├── raw/
│   │   ├── pvwatts_chile_baseline.csv   # Irradiancia real por ciudad
│   │   ├── pvwatts_chile_training.csv   # 3,000 instalaciones sintéticas
│   │   └── nrel_chile.db                # SQLite con baseline
│   └── processed/
│       └── solar_chile_model.csv        # Dataset con features engineered
├── notebooks/
│   ├── 01_eda.ipynb           # Exploración de recurso solar por ciudad
│   ├── 02_features.ipynb      # Engineering de tilt/azimuth deviation
│   └── 03_modeling.ipynb      # Benchmark y entrenamiento del modelo final
├── src/
│   ├── config.py              # Ciudades, coordenadas, parámetros financieros
│   ├── fetch_nrel.py          # Cliente del API NREL PVWatts v8
│   ├── predict.py             # Carga el modelo y calcula estimación + ROI
│   └── models/
│       └── gb_energy_model.pkl
└── requirements.txt
```
