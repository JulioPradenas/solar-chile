"""
Pipeline de entrenamiento con MLflow + Optuna.

Uso:
    python src/train.py

Luego ver los experimentos:
    mlflow ui --port 5001
    → http://localhost:5001
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineering import create_features, select_features

logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH     = Path("data/processed/solar_chile_model.csv")
MODEL_PATH    = Path("src/models/gb_energy_model.pkl")
PARAMS_PATH   = Path("data/best_params.json")
FEATURES_PATH = Path("data/selected_features.json")
EXPERIMENT    = "solar-chile"
TARGET        = "ac_annual"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df_enriched = create_features(df)
    selected, _ = select_features(
        df_enriched,
        exclude_cols=["city", "region", TARGET, "capacity_factor"],
    )
    with open(FEATURES_PATH, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Features seleccionadas ({len(selected)}): {selected}")
    X = df_enriched[selected]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2":   round(r2_score(y_true, y_pred), 6),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 2),
        "mae":  round(mean_absolute_error(y_true, y_pred), 2),
    }


def run_baseline(X_train, X_test, y_train, y_test):
    print("── Baseline: Regresión Lineal ──────────────────")
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="baseline-linear-regression"):
        model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        model.fit(X_train, y_train)
        test_metrics = regression_metrics(y_test, model.predict(X_test))

        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("cv_r2_mean", round(cv.mean(), 6))
        mlflow.log_metric("cv_r2_std",  round(cv.std(), 6))
        mlflow.log_metrics(test_metrics)
        mlflow.sklearn.log_model(model, "model")

    print(f"  CV R²: {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"  Test  R²={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.1f} kWh")


def run_optuna_gbm(X_train, X_test, y_train, y_test, n_trials: int = 30):
    print(f"── GBM + Optuna ({n_trials} trials) ───────────────────")
    mlflow.set_experiment(EXPERIMENT)

    best_params_store = {}

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 600),
            "max_depth":       trial.suggest_int("max_depth", 2, 8),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(**params, random_state=42)),
        ])
        cv = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
        score = cv.mean()

        with mlflow.start_run(run_name=f"gbm-trial-{trial.number:02d}"):
            mlflow.log_params(params)
            mlflow.log_metric("cv_r2_mean", round(score, 6))
            mlflow.log_metric("cv_r2_std",  round(cv.std(), 6))
            mlflow.log_param("trial_number", trial.number)

        if not best_params_store or score > best_params_store.get("cv_r2_mean", -1):
            best_params_store.update(params)
            best_params_store["cv_r2_mean"] = round(score, 6)

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  Mejor trial #{study.best_trial.number}: CV R²={study.best_value:.4f}")
    print(f"  Params: {best}")

    # Entrenar modelo final con best params sobre todo el train set
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(**best, random_state=42)),
    ])
    final_model.fit(X_train, y_train)
    test_metrics = regression_metrics(y_test, final_model.predict(X_test))

    with mlflow.start_run(run_name="gbm-final-best"):
        mlflow.log_params(best)
        mlflow.log_metric("cv_r2_mean", round(study.best_value, 6))
        mlflow.log_metrics(test_metrics)
        mlflow.sklearn.log_model(final_model, "model")

    print(f"  Test  R²={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.1f} kWh  MAE={test_metrics['mae']:.1f} kWh")

    # Guardar artefactos
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)

    params_out = {**best, "cv_r2_mean": round(study.best_value, 6), **test_metrics}
    with open(PARAMS_PATH, "w") as f:
        json.dump(params_out, f, indent=2)

    print(f"  Modelo guardado → {MODEL_PATH}")
    print(f"  Params guardados → {PARAMS_PATH}")
    return test_metrics


if __name__ == "__main__":
    print(f"Cargando {DATA_PATH} …")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print()

    mlflow.set_tracking_uri("mlruns")

    run_baseline(X_train, X_test, y_train, y_test)
    print()
    run_optuna_gbm(X_train, X_test, y_train, y_test, n_trials=30)

    print()
    print("Experimentos en MLflow:")
    print("  mlflow ui --port 5001")
    print("  → http://localhost:5001")
