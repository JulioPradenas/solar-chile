"""
Webapp Flask — Calculadora Solar Chile

Correr:
    cd solar-chile
    python app/server.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request
from src.predict import get_prediction
from src.config import CHILE_CITIES_LIST

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html", cities=CHILE_CITIES_LIST)


@app.route("/output", methods=["POST"])
def output():
    city = request.form["city"]
    system_size_kw = float(request.form["system_size_kw"])
    tilt = float(request.form["tilt"])
    azimuth = float(request.form.get("azimuth", 0.0))

    result = get_prediction(city=city, system_size_kw=system_size_kw, tilt=tilt, azimuth=azimuth)

    if result is None:
        return render_template("index.html", cities=CHILE_CITIES_LIST, error=f"Ciudad '{city}' no encontrada.")

    return render_template("output.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
