import json
import pickle
import pandas as pd
from pathlib import Path

from src.features.engineering import create_features

MODEL_PATH    = Path("src/models/gb_energy_model.pkl")
FEATURES_PATH = Path("data/selected_features.json")

RAW_COLS = ["system_size_kw", "tilt", "azimuth", "losses",
            "tilt_deviation", "azimuth_deviation", "latitude", "solrad_annual"]


def _enrich(row: list) -> pd.DataFrame:
    raw = pd.DataFrame([row], columns=RAW_COLS)
    with open(FEATURES_PATH) as f:
        selected = json.load(f)
    return create_features(raw)[selected]


# azimuth=0 → norte (óptimo en hemisferio sur) → azimuth_deviation=0.0
FEATURES_SANTIAGO     = _enrich([5.0, 33.0,  0.0, 14.0, -0.45, 0.0, -33.45, 5.277])
FEATURES_ANTOFAGASTA  = _enrich([5.0, 23.65, 0.0, 14.0,  0.0,  0.0, -23.65, 6.26])
FEATURES_PUNTA_ARENAS = _enrich([5.0, 53.16, 0.0, 14.0,  0.0,  0.0, -53.16, 3.93])


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_model_file_exists():
    assert MODEL_PATH.exists()
    assert MODEL_PATH.stat().st_size > 10_000


def test_model_loads_and_has_predict():
    model = load_model()
    assert hasattr(model, "predict")


def test_prediction_is_positive_float():
    model = load_model()
    pred = model.predict(FEATURES_SANTIAGO)[0]
    assert isinstance(pred, float)
    assert pred > 0


def test_prediction_in_realistic_range():
    model = load_model()
    pred = model.predict(FEATURES_SANTIAGO)[0]
    assert 1_000 < pred < 40_000


def test_antofagasta_produces_more_than_punta_arenas():
    model = load_model()
    pred_norte = model.predict(FEATURES_ANTOFAGASTA)[0]
    pred_sur = model.predict(FEATURES_PUNTA_ARENAS)[0]
    assert pred_norte > pred_sur
