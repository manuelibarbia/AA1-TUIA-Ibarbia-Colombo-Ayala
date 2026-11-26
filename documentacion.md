# Documentación del Trabajo Práctico 2: clasificación <!-- Última modificación: 2026-11-25 -->
## Alcance <!-- Última modificación: 2026-11-25 -->
El TP predice si lloverá al día siguiente en estaciones meteorológicas de Australia y empaqueta el modelo elegido para inferencia con Docker. <!-- Última modificación: 2026-11-25 -->
## Arquitectura <!-- Última modificación: 2026-11-25 -->
- El desarrollo y la evaluación están en TP_2/TP-clasificacion-AA1.ipynb; el corte de evaluación respeta el orden temporal y la búsqueda de hiperparámetros usa datos de entrenamiento. <!-- Última modificación: 2026-11-25 -->
- La inferencia vive en TP_2/docker/inferencia.py y carga red_optimizada.keras con TensorFlow/Keras y preprocesador.joblib con joblib. <!-- Última modificación: 2026-11-25 -->
- El preprocesador aplica el escalado robusto, la imputación y la codificación ajustados solo con entrenamiento; la inferencia replica las variables cíclicas de fecha y viento y la conversión de estación a coordenadas. <!-- Última modificación: 2026-11-25 -->
## Pantallas y punto de entrada <!-- Última modificación: 2026-11-25 -->
- Esta feature es exclusivamente backend y no tiene pantallas frontend. El punto de entrada es la API HTTP del contenedor, en particular POST /predict. <!-- Última modificación: 2026-11-25 -->
- GET /health informa el estado del servicio una vez cargados los artefactos. <!-- Última modificación: 2026-11-25 -->
## Contrato API <!-- Última modificación: 2026-11-25 -->
- POST /predict recibe un objeto JSON con Date (YYYY-MM-DD), Location, RainToday, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am y Temp3pm. <!-- Última modificación: 2026-11-25 -->
- Los 16 valores numéricos y las tres direcciones pueden ser null para activar el preprocesamiento aprendido; Date y Location son obligatorios. RainToday admite Yes, No o null. <!-- Última modificación: 2026-11-25 -->
- RainTomorrow y RainfallTomorrow se rechazan en la entrada porque son la etiqueta o información del futuro. También se rechazan campos desconocidos, fechas fuera de formato y estaciones o direcciones no reconocidas. <!-- Última modificación: 2026-11-25 -->
- La respuesta contiene probabilidad_yes, lluvia_manana (Yes o No) y umbral. El umbral 0,163 se eligió con el índice de Youden sobre validación del ítem 7. <!-- Última modificación: 2026-11-25 -->
- Errores HTTP: 400 para JSON o datos inválidos, 404 para rutas no existentes, 413 para cuerpos vacíos o mayores a 64 KiB y 500 para fallas inesperadas de inferencia. <!-- Última modificación: 2026-11-25 -->
## Datos, seguridad e integraciones <!-- Última modificación: 2026-11-25 -->
- La API no persiste observaciones ni incluye el dataset en la imagen. Los faltantes se representan con null; la validación limita el cuerpo a 64 KiB y el proceso del contenedor corre con un usuario no privilegiado. <!-- Última modificación: 2026-11-25 -->
- Dependencias de inferencia: Python 3.10, NumPy, pandas, scikit-learn, joblib, TensorFlow y Keras; sus versiones están fijadas en TP_2/docker/requirements.txt. <!-- Última modificación: 2026-11-25 -->
- El modelo seleccionado es la red neuronal optimizada del ítem 8, con AP 0,7328 en validación cruzada y 0,736 en prueba según las salidas guardadas del notebook. <!-- Última modificación: 2026-11-25 -->
## Operación y despliegue <!-- Última modificación: 2026-11-25 -->
- El Dockerfile está en TP_2/docker/Dockerfile; construir desde esa carpeta con docker build --progress=plain -t aa1-lluvia . y ejecutar en segundo plano con docker run -d --name aa1-lluvia -p 8000:8000 aa1-lluvia:latest. Verificar GET /health, POST /predict y los logs con docker logs aa1-lluvia. <!-- Última modificación: 2026-11-26 -->
- Los artefactos requeridos están versionados en TP_2/docker/modelo/: red_optimizada.keras (73.883 bytes) y preprocesador.joblib (2.147 bytes). Dockerfile copia ambos y comprueba que no estén vacíos; la celda de exportación permite regenerarlos desde los objetos ya ajustados si hiciera falta. <!-- Última modificación: 2026-11-26 -->
- El servicio escucha en 0.0.0.0:8000 dentro del contenedor; el puerto se configura con PORT y el host con HOST. No se incluye una imagen Docker en el repositorio. <!-- Última modificación: 2026-11-25 -->
