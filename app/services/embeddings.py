"""Capa de embeddings con tres implementaciones intercambiables.

- ``local``: sentence-transformers (CPU). Modelo multilingüe por defecto porque
  las preguntas llegan en español y el reglamento está en inglés (ARQUITECTURA.md §2.4).
  ``encode`` es bloqueante (CPU-bound) -> se ejecuta en el threadpool con
  ``asyncio.to_thread`` para no bloquear el event loop de FastAPI.
- ``openai``: API compatible con ``/v1/embeddings`` (OpenAI, Ollama, vLLM...).
- ``fake``: determinista, sin red ni dependencias — solo tests y smoke tests.
"""

import asyncio
import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod

from app.core.config import Settings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Interfaz común: mismo embedder para ingesta y para consulta."""

    name: str = "base"
    dim: int = 0

    async def load(self) -> None:
        """Carga recursos pesados (modelo local). Se llama en el lifespan de FastAPI."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeddings de pasajes (ingesta)."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embedding de una consulta (serving)."""


class LocalSentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, query_prefix: str, passage_prefix: str) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.name = f"local:{model_name}"
        self._model = None

    async def load(self) -> None:
        if self._model is not None:
            return

        def _load():
            # Import diferido: tarda ~10s y solo se necesita si provider=local.
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(self.model_name, device="cpu")

        logger.info("Cargando modelo de embeddings local", extra={"model": self.model_name})
        self._model = await asyncio.to_thread(_load)
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        logger.info("Modelo de embeddings listo", extra={"model": self.model_name, "dim": self.dim})

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._encode([self.passage_prefix + t for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        return (await self._encode([self.query_prefix + text]))[0]

    async def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            await self.load()
        if not texts:
            return []

        def _run():
            return self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
            )

        vectors = await asyncio.to_thread(_run)
        return [v.tolist() for v in vectors]


class OpenAICompatibleEmbedder(BaseEmbedder):
    """Embeddings vía API. Sirve para OpenAI o cualquier endpoint compatible."""

    _BATCH = 64

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.name = f"openai:{model}"
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH):
            batch = texts[i : i + self._BATCH]
            out.extend(await self._embed_batch(batch))
        return out

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed_batch([text]))[0]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(model=self.model, input=batch)
        data = sorted(response.data, key=lambda d: d.index)
        self.dim = len(data[0].embedding) if data else 0
        return [d.embedding for d in data]


class FakeEmbedder(BaseEmbedder):
    """Embedder determinista (bag-of-words hasheado). SOLO tests/smoke:
    cartas con vocabulario compartido quedan cerca en el espacio. No usar para
    calidad de retrieval (para eso, eval_retrieval.py con un modelo real)."""

    name = "fake"
    dim = 128

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def get_embedder(settings: Settings) -> BaseEmbedder:
    """Factoría según settings. El MISMO embedder debe usarse en ingesta y serving."""
    if settings.embedding_provider == "local":
        return LocalSentenceTransformerEmbedder(
            settings.embedding_model,
            settings.embedding_query_prefix,
            settings.embedding_passage_prefix,
        )
    if settings.embedding_provider == "openai":
        api_key = settings.embedding_api_key or settings.llm_api_key
        if not api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai requiere EMBEDDING_API_KEY (o LLM_API_KEY). "
                "Sin clave, usa EMBEDDING_PROVIDER=local."
            )
        return OpenAICompatibleEmbedder(
            settings.embedding_model, api_key, settings.embedding_base_url
        )
    if settings.embedding_provider == "fake":
        return FakeEmbedder()
    raise ValueError(f"Proveedor de embeddings desconocido: {settings.embedding_provider}")
