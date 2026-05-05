# Ciudades con cobertura disponible en PVWatts International (NREL)
CHILE_CITIES = {
    "Antofagasta":  {"lat": -23.65, "lon": -70.40, "region": "Antofagasta"},
    "Santiago":     {"lat": -33.45, "lon": -70.67, "region": "Metropolitana"},
    "Rancagua":     {"lat": -34.17, "lon": -70.74, "region": "O'Higgins"},
    "Concepción":   {"lat": -36.82, "lon": -73.05, "region": "Biobío"},
    "Temuco":       {"lat": -38.74, "lon": -72.60, "region": "La Araucanía"},
    "Punta Arenas": {"lat": -53.16, "lon": -70.91, "region": "Magallanes"},
}

CHILE_CITIES_LIST = sorted(CHILE_CITIES.keys())

# Parámetros financieros Chile (actualizar según CNE/mercado)
ELECTRICITY_RATE_CLP = 130       # CLP/kWh — tarifa BT1 promedio
RATE_ESCALATION = 0.04           # escalación anual histórica
INSTALL_COST_PER_KWP = 900_000   # CLP/kWp — costo instalación residencial
