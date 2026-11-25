# Inferencia del modelo de lluvia

Este servicio carga la red neuronal optimizada del ítem 8 y su ColumnTransformer. Recibe las condiciones meteorológicas observadas durante el día y predice RainTomorrow. No usa RainTomorrow ni RainfallTomorrow como entrada.

## Artefactos

Antes de construir la imagen deben existir estos dos archivos dentro de modelo/:

- red_optimizada.keras: pesos de la red elegida.
- preprocesador.joblib: escaladores, imputadores y codificador ajustados solo con entrenamiento.

Se generan desde la celda de exportación al final del ítem 9 del notebook, con los objetos entrenados en memoria. No se deben volver a ajustar al preparar la inferencia. En este checkout todavía no están generados: no había artefactos serializados y no se ejecutó el notebook para conservar sus resultados guardados.

La inferencia fija el umbral en 0.163, elegido con el índice de Youden sobre validación en el ítem 7. La clase positiva es Yes.

## Construir y ejecutar

Desde esta carpeta, después de generar los dos artefactos:

~~~sh
docker build -t aa1-lluvia .
docker run --rm -p 8000:8000 aa1-lluvia
~~~

La imagen contiene únicamente el servicio, sus dependencias y los artefactos; no incluye el dataset ni el notebook. El contenedor escucha en el puerto 8000.

## API

- GET /health: comprueba que el servicio está activo.
- POST /predict: recibe una observación en JSON y devuelve la probabilidad de lluvia para el día siguiente, la clase predicha y el umbral aplicado.

La solicitud incluye Date con formato YYYY-MM-DD, Location, RainToday, las 16 variables numéricas y las tres direcciones del viento. Los faltantes se envían como null; la fecha y la estación son obligatorias. Las ubicaciones y direcciones aceptadas son las del dataset.

~~~sh
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Date": "2017-06-24",
    "Location": "Sydney",
    "MinTemp": 12.4,
    "MaxTemp": 19.8,
    "Rainfall": 0.2,
    "Evaporation": null,
    "Sunshine": 7.1,
    "WindGustDir": "W",
    "WindGustSpeed": 39,
    "WindDir9am": "NW",
    "WindDir3pm": "W",
    "WindSpeed9am": 13,
    "WindSpeed3pm": 20,
    "Humidity9am": 72,
    "Humidity3pm": 61,
    "Pressure9am": 1018.2,
    "Pressure3pm": 1015.6,
    "Cloud9am": 6,
    "Cloud3pm": 7,
    "Temp9am": 15.1,
    "Temp3pm": 18.4,
    "RainToday": "No"
  }'
~~~

Respuesta ilustrativa:

~~~json
{
  "probabilidad_yes": 0.42,
  "lluvia_manana": "Yes",
  "umbral": 0.163
}
~~~

Los valores de respuesta son ilustrativos. La decisión devuelve Yes cuando la probabilidad alcanza o supera el umbral.
