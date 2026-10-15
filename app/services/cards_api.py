"""Cliente de la API pública de cartas https://docs.magicthegathering.io (R3).

ARQUITECTURA.md §6: la búsqueda de cartas NO es búsqueda semántica — el LLM
construye filtros estructurados (validados con Pydantic en ``tools.py``) y este
cliente los traduce a parámetros de la API externa.

Semántica verificada contra la API real (2026-10):
- ``colors=W,R`` (códigos de letra + coma) = la carta debe tener AMBOS colores.
- ``subtypes=Warrior`` / ``types=Creature`` / ``text=first strike``: match exacto.
- ``cmc`` solo soporta valor exacto -> ``cmc_max`` se filtra en el cliente.
- La API devuelve cada impresión (duplicados por edición) -> deduplicamos por nombre.
- Rate limit ~1000 req/h -> caché en Redis (TTL 24 h) + User-Agent identificable.
"""

import hashlib
import json
import logging

import httpx

from app.core.config import Settings
from app.models.schemas import CardFilters, CardSummary
from app.services.memory import ConversationMemory

logger = logging.getLogger(__name__)


class CardsAPIError(Exception):
    """La API externa no está disponible (timeout, 5xx, red)."""


class MagicTheGatheringAPI:
    def __init__(self, http: httpx.AsyncClient, memory: ConversationMemory, settings: Settings) -> None:
        self._http = http
        self._memory = memory
        self._settings = settings

    async def search(self, filters: CardFilters) -> list[CardSummary]:
        """Busca cartas por filtros. Caché en Redis → API externa → dedupe + filtro CMC."""
        cache_key = self._cache_key(filters)
        cached = await self._memory.cache_get_json(cache_key)
        if cached is not None:
            logger.info("Caché de cartas (hit)", extra={"cache_key": cache_key})
            return [CardSummary.model_validate(card) for card in cached]

        params: dict[str, str | int] = {"pageSize": self._settings.cards_api_page_size}
        if filters.colors:
            params["colors"] = ",".join(filters.colors)
        if filters.types:
            params["types"] = ",".join(filters.types)
        if filters.subtypes:
            params["subtypes"] = ",".join(filters.subtypes)
        if filters.text:
            params["text"] = filters.text

        logger.info("Consulta a API de cartas", extra={"params": params})
        try:
            response = await self._http.get(f"{self._settings.cards_api_base_url}/cards", params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise CardsAPIError(f"magicthegathering.io no responde: {exc}") from exc

        # cmc_max en cliente (la API solo soporta CMC exacto) + dedupe por nombre
        # (la API devuelve una entrada por impresión).
        best: dict[str, CardSummary] = {}
        for card in payload.get("cards", []):
            if filters.cmc_max is not None and float(card.get("cmc") or 0) > filters.cmc_max:
                continue
            summary = self._to_summary(card)
            previous = best.get(summary.name)
            if previous is None or (not previous.image_url and summary.image_url):
                best[summary.name] = summary

        results = list(best.values())[: self._settings.cards_max_results]
        await self._memory.cache_set_json(
            cache_key, [card.model_dump() for card in results], self._settings.cards_cache_ttl_seconds
        )
        logger.info("Resultado de búsqueda de cartas", extra={"count": len(results)})
        return results

    @staticmethod
    def _cache_key(filters: CardFilters) -> str:
        raw = json.dumps(filters.model_dump(exclude_none=True), sort_keys=True)
        return f"cards:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"

    @staticmethod
    def _to_summary(card: dict) -> CardSummary:
        return CardSummary(
            name=str(card.get("name", "")),
            mana_cost=str(card.get("manaCost", "") or ""),
            colors=[str(color) for color in card.get("colors", [])],
            type_line=str(card.get("type", "") or ""),
            subtypes=[str(sub) for sub in card.get("subtypes", [])],
            text=str(card.get("text", "") or ""),
            power=str(card.get("power", "") or ""),
            toughness=str(card.get("toughness", "") or ""),
            image_url=card.get("imageUrl"),
            set_name=str(card.get("setName", "") or ""),
        )
