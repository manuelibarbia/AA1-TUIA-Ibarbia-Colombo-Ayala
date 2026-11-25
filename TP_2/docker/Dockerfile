FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY inferencia.py .
COPY modelo/ ./modelo/
RUN test -s modelo/red_optimizada.keras \
    && test -s modelo/preprocesador.joblib \
    && useradd --system --uid 10001 app \
    && chown -R app:app /app

USER app
EXPOSE 8000
CMD ["python", "inferencia.py"]
