from src.predict import get_prediction


def test_santiago_prediction_returns_dict():
    result = get_prediction("Santiago", system_size_kw=5.0, tilt=33.0, azimuth=0.0)
    assert result is not None
    assert result["annual_energy_kwh"] > 0
    assert result["payback_years"] > 0


def test_invalid_city_returns_none():
    result = get_prediction("Tokio", system_size_kw=5.0, tilt=30.0, azimuth=0.0)
    assert result is None


def test_all_cities_return_results():
    cities = ["Antofagasta", "Santiago", "Rancagua", "Concepción", "Temuco", "Punta Arenas"]
    for city in cities:
        result = get_prediction(city, system_size_kw=5.0, tilt=30.0, azimuth=0.0)
        assert result is not None, f"Falló para {city}"
