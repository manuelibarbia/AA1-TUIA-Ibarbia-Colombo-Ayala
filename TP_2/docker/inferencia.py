"""API HTTP para predecir lluvia al día siguiente con el modelo del TP."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd


DIRECTORIO = Path(__file__).resolve().parent
RUTA_MODELO = DIRECTORIO / "modelo" / "red_optimizada.keras"
RUTA_PREPROCESADOR = DIRECTORIO / "modelo" / "preprocesador.joblib"
UMBRAL_LLUVIA = 0.163

COLUMNAS_NUMERICAS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
    "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm",
]
COLUMNAS_DIRECCION = ["WindGustDir", "WindDir9am", "WindDir3pm"]
DIRECCIONES_VIENTO = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
COORDENADAS = {
    "Adelaide": (-34.93, 138.60), "Albany": (-35.03, 117.88),
    "Albury": (-36.07, 146.91), "AliceSprings": (-23.70, 133.88),
    "BadgerysCreek": (-33.88, 150.76), "Ballarat": (-37.56, 143.85),
    "Bendigo": (-36.76, 144.28), "Brisbane": (-27.47, 153.03),
    "Cairns": (-16.92, 145.77), "Canberra": (-35.28, 149.13),
    "Cobar": (-31.50, 145.84), "CoffsHarbour": (-30.30, 153.11),
    "Dartmoor": (-37.92, 141.27), "Darwin": (-12.46, 130.85),
    "GoldCoast": (-28.02, 153.40), "Hobart": (-42.88, 147.33),
    "Katherine": (-14.45, 132.27), "Launceston": (-41.43, 147.14),
    "Melbourne": (-37.81, 144.96), "MelbourneAirport": (-37.67, 144.84),
    "Mildura": (-34.21, 142.13), "Moree": (-29.46, 149.84),
    "MountGambier": (-37.83, 140.78), "MountGinini": (-35.53, 148.77),
    "Newcastle": (-32.93, 151.78), "Nhil": (-36.33, 141.65),
    "NorahHead": (-33.28, 151.57), "NorfolkIsland": (-29.04, 167.95),
    "Nuriootpa": (-34.47, 139.00), "PearceRAAF": (-31.67, 116.02),
    "Penrith": (-33.75, 150.69), "Perth": (-31.95, 115.86),
    "PerthAirport": (-31.94, 115.97), "Portland": (-38.34, 141.60),
    "Richmond": (-37.82, 144.99), "Sale": (-38.11, 147.07),
    "SalmonGums": (-32.98, 121.65), "Sydney": (-33.87, 151.21),
    "SydneyAirport": (-33.95, 151.18), "Townsville": (-19.26, 146.82),
    "Tuggeranong": (-35.42, 149.07), "Uluru": (-25.34, 131.04),
    "WaggaWagga": (-35.12, 147.37), "Walpole": (-34.98, 116.73),
    "Watsonia": (-37.71, 145.08), "Williamtown": (-32.81, 151.84),
    "Witchcliffe": (-34.02, 115.10), "Wollongong": (-34.42, 150.89),
    "Woomera": (-31.20, 136.83),
}

CAMPOS_ENTRADA = {
    "Date", "Location", "RainToday", *COLUMNAS_NUMERICAS, *COLUMNAS_DIRECCION
}
COLUMNAS_MODELO = [
    *COLUMNAS_NUMERICAS,
    "Location_latitud", "Location_longitud", "Date_sin", "Date_cos",
    *[
        f"{columna}_{componente}"
        for columna in COLUMNAS_DIRECCION
        for componente in ("sin", "cos", "faltante")
    ],
    "RainToday",
]


def _valor_numerico(valor, nombre):
    if valor is None or valor == "":
        return np.nan
    try:
        numero = float(valor)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{nombre} debe ser numérico o null.") from error
    if not np.isfinite(numero):
        raise ValueError(f"{nombre} debe ser finito o null.")
    return numero


def construir_entrada(observacion):
    """Replica el feature engineering usado para ajustar el preprocesador."""
    if not isinstance(observacion, dict):
        raise ValueError("El cuerpo debe ser un objeto JSON.")

    campos_objetivo = {"RainTomorrow", "RainfallTomorrow"} & set(observacion)
    if campos_objetivo:
        raise ValueError(
            "No envíe la etiqueta ni datos del día siguiente: "
            + ", ".join(sorted(campos_objetivo))
        )

    faltantes = CAMPOS_ENTRADA - set(observacion)
    desconocidos = set(observacion) - CAMPOS_ENTRADA
    if faltantes:
        raise ValueError(f"Faltan campos: {', '.join(sorted(faltantes))}.")
    if desconocidos:
        raise ValueError(f"Campos desconocidos: {', '.join(sorted(desconocidos))}.")

    try:
        fecha = pd.to_datetime(observacion["Date"], format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Date debe tener formato YYYY-MM-DD.") from error

    ubicacion = observacion["Location"]
    if not isinstance(ubicacion, str) or ubicacion.strip() not in COORDENADAS:
        raise ValueError("Location debe ser una de las estaciones del dataset.")
    ubicacion = ubicacion.strip()

    fila = {
        columna: _valor_numerico(observacion[columna], columna)
        for columna in COLUMNAS_NUMERICAS
    }
    latitud, longitud = COORDENADAS[ubicacion]
    fila["Location_latitud"] = latitud
    fila["Location_longitud"] = longitud

    fecha_objetivo = fecha + pd.Timedelta(days=1)
    angulo_anual = 2 * np.pi * (fecha_objetivo.dayofyear - 1) / 365.25
    fila["Date_sin"] = np.sin(angulo_anual)
    fila["Date_cos"] = np.cos(angulo_anual)

    indices_direccion = {direccion: indice for indice, direccion in enumerate(DIRECCIONES_VIENTO)}
    for columna in COLUMNAS_DIRECCION:
        direccion = observacion[columna]
        if direccion is None or direccion == "":
            indice = None
        elif isinstance(direccion, str) and direccion.strip().upper() in indices_direccion:
            indice = indices_direccion[direccion.strip().upper()]
        else:
            raise ValueError(f"{columna} no es una dirección de viento válida o null.")

        if indice is None:
            fila[f"{columna}_sin"] = 0.0
            fila[f"{columna}_cos"] = 0.0
            fila[f"{columna}_faltante"] = 1
        else:
            angulo = 2 * np.pi * indice / len(DIRECCIONES_VIENTO)
            fila[f"{columna}_sin"] = np.sin(angulo)
            fila[f"{columna}_cos"] = np.cos(angulo)
            fila[f"{columna}_faltante"] = 0

    lluvia_hoy = observacion["RainToday"]
    if lluvia_hoy is not None and lluvia_hoy != "":
        if lluvia_hoy not in ("No", "Yes"):
            raise ValueError("RainToday debe ser No, Yes o null.")
        fila["RainToday"] = lluvia_hoy
    else:
        fila["RainToday"] = np.nan

    return pd.DataFrame([fila], columns=COLUMNAS_MODELO)


def cargar_recursos():
    faltan = [ruta.name for ruta in (RUTA_MODELO, RUTA_PREPROCESADOR) if not ruta.is_file()]
    if faltan:
        raise FileNotFoundError(
            "Faltan los artefactos del modelo: "
            + ", ".join(faltan)
            + ". Incluya los archivos versionados de modelo/ antes de iniciar la API."
        )

    import tensorflow as tf

    modelo = tf.keras.models.load_model(RUTA_MODELO, compile=False)
    preprocesador = joblib.load(RUTA_PREPROCESADOR)
    return modelo, preprocesador


def predecir(observacion, modelo, preprocesador):
    entrada = construir_entrada(observacion)
    transformada = np.asarray(preprocesador.transform(entrada), dtype="float32")
    probabilidad = float(np.asarray(modelo.predict(transformada, verbose=0)).reshape(-1)[0])
    return {
        "probabilidad_yes": round(probabilidad, 6),
        "lluvia_manana": "Yes" if probabilidad >= UMBRAL_LLUVIA else "No",
        "umbral": UMBRAL_LLUVIA,
    }


def crear_handler(modelo, preprocesador):
    class Handler(BaseHTTPRequestHandler):
        def _responder(self, estado, contenido):
            cuerpo = json.dumps(contenido, ensure_ascii=False).encode("utf-8")
            self.send_response(estado)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self):
            if self.path == "/health":
                self._responder(200, {"estado": "listo"})
            else:
                self._responder(404, {"error": "Ruta no encontrada."})

        def do_POST(self):
            if self.path != "/predict":
                self._responder(404, {"error": "Ruta no encontrada."})
                return

            try:
                longitud = int(self.headers.get("Content-Length", "0"))
                if longitud <= 0 or longitud > 65536:
                    self._responder(413, {"error": "El cuerpo debe pesar entre 1 y 65536 bytes."})
                    return
                observacion = json.loads(self.rfile.read(longitud).decode("utf-8"))
                resultado = predecir(observacion, modelo, preprocesador)
                self._responder(200, resultado)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._responder(400, {"error": "El cuerpo debe ser JSON UTF-8 válido."})
            except ValueError as error:
                self._responder(400, {"error": str(error)})
            except Exception:
                logging.exception("Falló la inferencia.")
                self._responder(500, {"error": "No se pudo calcular la predicción."})

        def log_message(self, formato, *argumentos):
            logging.info("%s - %s", self.address_string(), formato % argumentos)

    return Handler


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    modelo, preprocesador = cargar_recursos()
    host = os.environ.get("HOST", "0.0.0.0")
    puerto = int(os.environ.get("PORT", "8000"))
    servidor = ThreadingHTTPServer((host, puerto), crear_handler(modelo, preprocesador))
    logging.info("Servicio de inferencia disponible en %s:%s", host, puerto)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        logging.info("Se detiene el servicio.")
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
