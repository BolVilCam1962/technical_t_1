# syntax=docker/dockerfile:1
# Imagen de la app (API + herramientas de ingesta/eval).
# ~2 GB: incluye torch CPU + sentence-transformers para embeddings locales.
# Si usas EMBEDDING_PROVIDER=openai puedes bajar a python:3.11-slim sin torch.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# libgomp1: lo necesitan onnxruntime (chromadb) y tokenizers (transformers).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./

# 1) torch CPU ANTES que el resto: si no, sentence-transformes tira de las
#    ruedas CUDA (~2 GB extra). pip ya ve torch satisfecho en el paso 2.
# 2) requirements-dev incluye requirements (pytest para `docker compose run --rm app pytest`).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements-dev.txt

COPY app ./app
COPY scripts ./scripts
COPY eval ./eval
COPY tests ./tests
COPY pytest.ini ./

# Usuario no root. Crear los directorios aquí fija la propiedad del volumen
# nombrado hf-cache en su primer uso (docker copia imagen -> volumen vacío).
RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 1 worker por defecto: el modelo de embeddings vive en RAM por proceso.
# Para escalar horizontalmente usa réplicas del contenedor (no --workers).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
