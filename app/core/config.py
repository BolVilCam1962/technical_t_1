"""Configuración central de la aplicación (env vars + .env).

Todas las variables se pueden sobreescribir por entorno (ver `.env.example`).
pydantic-settings mapea ``nombre_campo`` -> ``NOMBRE_CAMPO``.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Aplicación ---
    app_name: str = "MTG Rules Chatbot"
    app_version: str = "0.1.0"
    environment: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"

    # --- LLM (orquestador del agente) ---
    llm_provider: Literal["openai", "mock"] = "mock"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    agent_max_iterations: int = 5

    # --- Embeddings ---
    # local: sentence-transformers en el contenedor (sin clave, descarga el modelo la 1ª vez).
    # openai: API compatible con /v1/embeddings (OpenAI, Ollama, vLLM...).
    # fake: embedder determinista sin red ni dependencias — SOLO tests y smoke tests.
    embedding_provider: Literal["local", "openai", "fake"] = "local"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    # Prefijos requeridos por la familia e5 (vacíos para otros modelos).
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    # --- Retrieval (ChromaDB embebido y persistente) ---
    chroma_dir: str = "data/chroma"
    chroma_collection: str = "mtg_rules"
    rules_version: str = "20260417"
    retrieval_top_k: int = 6
    # Distancia coseno máxima para considerar un chunk relevante (calibrar con
    # scripts/eval_retrieval.py). Encima de este umbral la respuesta debe ser honesta.
    retrieval_max_distance: float = 0.95

    # --- Redis (memoria de sesión + cachés) ---
    # "memory://" => almacén en proceso SIN Redis (solo dev/tests, sin persistencia).
    redis_url: str = "redis://localhost:6379/0"
    history_max_messages: int = 20
    history_ttl_seconds: int = 86400
    cards_cache_ttl_seconds: int = 86400

    # --- API pública de cartas ---
    cards_api_base_url: str = "https://api.magicthegathering.io/v1"
    cards_api_timeout_seconds: float = 10.0
    cards_api_page_size: int = 50
    cards_max_results: int = 10

    # --- Observabilidad LLM (Langfuse) ---
    # Activa el drop-in wrapper de OpenAI: cada generación (prompt, tokens,
    # coste, latencia) se registra en Langfuse. Self-host (perfil
    # "observability" de docker-compose) o Langfuse Cloud.
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # Dentro de docker-compose: http://langfuse-web:3000; local: http://localhost:3000
    langfuse_host: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    """Settings como singleton (cacheado). En tests: ``get_settings.cache_clear()``."""
    return Settings()
