"""Memoria de conversación y cachés en Redis (ARQUITECTURA.md §5).

- Historial por sesión: ``LIST chat:{session_id}`` con RPUSH + LTRIM (últimos N
  mensajes) + EXPIRE (TTL 24 h). El servicio es stateless: reconstruye el
  contexto en cada petición -> escala horizontalmente sin sticky sessions.
- Caché de respuestas de la API de cartas: ``STRING cards:{hash}`` con TTL.

Degradación elegante: si Redis no responde, se registra un warning y se sigue
atendiendo (sin historial ni caché) antes que fallar la petición.

``redis_url == "memory://"`` usa un almacén en proceso (tests / dev sin Redis).
"""

import json
import logging

from app.core.config import Settings
from app.models.schemas import ChatMessage

logger = logging.getLogger(__name__)


class InMemoryRedis:
    """Réplica mínima de la API de redis.asyncio que usamos (tests/dev).

    Soporta los mismos métodos y semántica de índices negativos que Redis,
    pero sin TTL: es un alias de un dict en proceso.
    """

    def __init__(self) -> None:
        self._data: dict = {}

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None

    async def rpush(self, key: str, *values: str) -> int:
        bucket = self._data.setdefault(key, [])
        bucket.extend(values)
        return len(bucket)

    @staticmethod
    def _slice(length: int, start: int, end: int) -> tuple[int, int]:
        start = max(0, start if start >= 0 else length + start)
        end = end if end >= 0 else length + end
        return start, end

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        bucket = self._data.get(key, [])
        s, e = self._slice(len(bucket), start, end)
        self._data[key] = bucket[s : e + 1] if e >= s else []
        return True

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        bucket = self._data.get(key, [])
        s, e = self._slice(len(bucket), start, end)
        return bucket[s : e + 1] if e >= s else []

    async def expire(self, key: str, ttl: int) -> bool:  # noqa: ARG002 - sin TTL
        return key in self._data

    async def get(self, key: str) -> str | None:
        value = self._data.get(key)
        return value if isinstance(value, str) else None

    async def setex(self, key: str, ttl: int, value: str) -> bool:  # noqa: ARG002
        self._data[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        removed = sum(1 for k in keys if self._data.pop(k, None) is not None)
        return removed


def build_redis_client(redis_url: str):
    """Crea el cliente async de Redis (o el almacén en proceso para 'memory://')."""
    if redis_url.startswith("memory://"):
        return InMemoryRedis()
    import redis.asyncio as aioredis

    return aioredis.from_url(redis_url, decode_responses=True)


class ConversationMemory:
    """Historial de sesión + caché de cartas. Todas las ops son tolerantes a fallos."""

    def __init__(self, client, settings: Settings) -> None:
        self._redis = client
        self._settings = settings

    @staticmethod
    def _history_key(session_id: str) -> str:
        return f"chat:{session_id}"

    # --- Historial -----------------------------------------------------------

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception as exc:  # redis.RedisError, ConnectionError, ...
            logger.warning("Redis no disponible", extra={"error": str(exc)})
            return False

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        try:
            raw = await self._redis.lrange(self._history_key(session_id), 0, -1)
        except Exception as exc:
            logger.warning("Redis degradado: sin historial", extra={"error": str(exc)})
            return []
        messages = []
        for item in raw:
            try:
                messages.append(ChatMessage.model_validate_json(item))
            except Exception:  # registro corrupto: se ignora antes que romper el chat
                logger.warning("Mensaje de historial ilegible", extra={"session_id": session_id})
        return messages

    async def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        key = self._history_key(session_id)
        try:
            await self._redis.rpush(
                key,
                _dump_message(ChatMessage(role="user", content=user_message)),
                _dump_message(ChatMessage(role="assistant", content=assistant_message)),
            )
            await self._redis.ltrim(key, -self._settings.history_max_messages, -1)
            await self._redis.expire(key, self._settings.history_ttl_seconds)
        except Exception as exc:
            logger.warning("Redis degradado: turno no persistido", extra={"error": str(exc)})

    async def clear(self, session_id: str) -> bool:
        key = self._history_key(session_id)
        try:
            return bool(await self._redis.delete(key))
        except Exception as exc:
            logger.warning("Redis degradado: sesión no eliminada", extra={"error": str(exc)})
            return False

    # --- Caché de cartas -----------------------------------------------------

    async def cache_get_json(self, key: str):
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            logger.warning("Redis degradado: fallo de lectura de caché", extra={"error": str(exc)})
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def cache_set_json(self, key: str, value, ttl_seconds: int) -> None:
        try:
            await self._redis.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))
        except Exception as exc:
            logger.warning("Redis degradado: fallo de escritura de caché", extra={"error": str(exc)})


def _dump_message(message: ChatMessage) -> str:
    return json.dumps({"role": message.role, "content": message.content}, ensure_ascii=False)
