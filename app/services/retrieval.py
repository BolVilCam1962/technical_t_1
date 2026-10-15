"""Retrieval semántico sobre el reglamento (ChromaDB embebido y persistente).

ARQUITECTURA.md §3-4: cada chunk es una regla/sub-regla numerada con metadatos
(capítulo, número, versión). La query se embeddea con el mismo embedder usado
en la ingesta y se filtra por distancia máxima: por encima del umbral el sistema
debe responder honestamente en lugar de alucinar (§7.4).

Chroma es síncrono (CPU/IO local) -> todas las operaciones van al threadpool
con ``asyncio.to_thread`` para no bloquear el event loop.
"""

import asyncio
import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.services.embeddings import BaseEmbedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievedRule:
    rule_number: str
    title: str
    text: str
    chapter: str | None = None
    chapter_title: str | None = None
    distance: float = 0.0

    @property
    def score(self) -> float:
        """Similitud coseno normalizada a [0, 1]."""
        return round(1.0 - self.distance, 4)


def open_collection(chroma_dir: str, name: str):
    """Abre (o crea) la colección con métrica coseno. Compartido por app e ingesta."""
    import chromadb

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


class RulesRetriever:
    def __init__(self, embedder: BaseEmbedder, settings: Settings) -> None:
        self._embedder = embedder
        self._settings = settings
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            _, self._collection = open_collection(
                self._settings.chroma_dir, self._settings.chroma_collection
            )
        return self._collection

    async def count(self) -> int:
        return await asyncio.to_thread(self.collection.count)

    async def search(self, query: str, k: int | None = None) -> list[RetrievedRule]:
        """Busca las reglas más similares. Devuelve [] si no hay índice o nada relevante."""
        k = k or self._settings.retrieval_top_k
        if await self.count() == 0:
            logger.warning("Índice ChromaDB vacío: ¿se ha ejecutado scripts/ingest_rules.py?")
            return []

        embedding = await self._embedder.embed_query(query)

        def _query():
            return self.collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

        result = await asyncio.to_thread(_query)

        rules: list[RetrievedRule] = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            if dist > self._settings.retrieval_max_distance:
                continue
            rules.append(
                RetrievedRule(
                    rule_number=str(meta.get("rule_number", "")),
                    title=str(meta.get("title", "")),
                    text=doc,
                    chapter=meta.get("chapter"),
                    chapter_title=meta.get("chapter_title"),
                    distance=float(dist),
                )
            )
        return rules
