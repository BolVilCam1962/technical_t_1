"""Aplicación FastAPI del chatbot de reglas de Magic: The Gathering.

Gestión de procesos asíncronos (ARQUITECTURA.md §3 y §8):

- **Lifespan** (no el deprecado ``@app.on_event``): crea una sola vez los
  clientes compartidos (httpx, Redis async, LLM async, embedder) y los cierra
  al apagar. Evita clientes por petición.
- **CPU-bound al threadpool**: el modelo de embeddings (sentence-transformers)
  y las queries de Chroma son síncronos -> se envuelven en ``asyncio.to_thread``
  para no bloquear el event loop mientras atiende otras peticiones.
- **I/O puramente async**: httpx (API de cartas), redis.asyncio (memoria) y
  AsyncOpenAI (LLM).
- **Tools concurrentes**: si el LLM pide varias herramientas en una iteración,
  se ejecutan en paralelo con ``asyncio.gather`` (ver ``agent/orchestrator.py``).
- El modelo local se **precarga en el arranque** (warm-up) para que la primera
  petición no pague la carga.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.orchestrator import AgentOrchestrator
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.metrics import HTTP_REQUEST_LATENCY, HTTP_REQUESTS_TOTAL
from app.services.cards_api import MagicTheGatheringAPI
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from app.services.memory import ConversationMemory, build_redis_client
from app.services.retrieval import RulesRetriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "Arrancando aplicación",
        extra={
            "version": settings.app_version,
            "environment": settings.environment,
            "llm_provider": settings.llm_provider,
            "embedding_provider": settings.embedding_provider,
        },
    )

    # 1) Embeddings (precarga del modelo local: warm-up en threadpool).
    embedder = get_embedder(settings)
    await embedder.load()

    # 2) Retrieval (ChromaDB persistente; se avisa si el índice está vacío).
    retriever = RulesRetriever(embedder, settings)
    rules_count = await retriever.count()
    if rules_count == 0:
        logger.warning(
            "Índice ChromaDB vacío: ejecuta la ingesta "
            "(docker compose --profile tools run --rm ingest)"
        )
    else:
        logger.info("Índice de reglas listo", extra={"chunks": rules_count})

    # 3) Redis (memoria de sesión + cachés). Degradación elegante si no responde.
    redis_client = build_redis_client(settings.redis_url)
    memory = ConversationMemory(redis_client, settings)
    if not await memory.ping():
        logger.warning("Redis no disponible: el servicio arranca SIN memoria de sesión ni caché")

    # 4) API externa de cartas + LLM (fail-fast si falta configuración crítica).
    http_client = httpx.AsyncClient(
        timeout=settings.cards_api_timeout_seconds,
        headers={"User-Agent": f"mtg-chatbot/{settings.app_version}"},
    )
    cards_api = MagicTheGatheringAPI(http_client, memory, settings)
    llm = get_llm(settings)  # ValueError si openai sin API key
    orchestrator = AgentOrchestrator(llm, retriever, cards_api, memory, settings)

    app.state.settings = settings
    app.state.embedder = embedder
    app.state.retriever = retriever
    app.state.memory = memory
    app.state.cards_api = cards_api
    app.state.llm = llm
    app.state.orchestrator = orchestrator

    yield

    await http_client.aclose()
    await redis_client.aclose()
    logger.info("Aplicación detenida limpiamente")


settings = get_settings()

DESCRIPTION = """
Chatbot de reglas de **Magic: The Gathering** para el call center de nuestro cliente
(demo técnica — ver `ARQUITECTURA.md` para el diseño completo).

**Stack**: FastAPI · ChromaDB (reglamento indexado) · Redis (memoria de sesión y
caché de la API de cartas) · LLM vía API con function calling.

**Requisitos cubiertos**:

| Req | Capacidad | Cómo |
|---|---|---|
| R1 | Dudas de reglas | RAG sobre el *Comprehensive Rules* con **citas por número de regla** |
| R2 | Interacciones entre cartas | Agente multi-tool: recupera las reglas de las habilidades implicadas y razona citándolas |
| R3 | Búsqueda de cartas por descripción | NL → filtros validados (Pydantic) → [magicthegathering.io](https://docs.magicthegathering.io) |
| R4 | (Bonus) Cartas custom | Generación validada con schema |

**Modo demo**: sin `LLM_API_KEY` el servicio arranca con `llm_provider=mock` —
el flujo completo (retrieval + citas) funciona, pero sin razonamiento del LLM.
Configura `LLM_PROVIDER=openai` (+ `LLM_API_KEY`, o cualquier endpoint
compatible: Groq, OpenRouter, Ollama...) para la experiencia completa.
"""

app = FastAPI(
    title=settings.app_name,
    description=DESCRIPTION,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Chat", "description": "Endpoint principal del chatbot."},
        {"name": "Sesiones", "description": "Gestión del historial de conversación (Redis)."},
        {"name": "Operación", "description": "Health check y métricas Prometheus."},
    ],
)

# Demo: CORS abierto para que cualquier frontend del call center pruebe la API.
# En producción: restringir allow_origins al dominio del cliente.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Trace ID + logs estructurados + métricas Prometheus por petición."""
    trace_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.trace_id = trace_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        HTTP_REQUESTS_TOTAL.labels(request.method, path, "500").inc()
        HTTP_REQUEST_LATENCY.labels(path).observe(duration)
        logger.exception(
            "Error no controlado",
            extra={"trace_id": trace_id, "method": request.method, "path": path},
        )
        raise
    duration = time.perf_counter() - started
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    HTTP_REQUESTS_TOTAL.labels(request.method, path, str(response.status_code)).inc()
    HTTP_REQUEST_LATENCY.labels(path).observe(duration)
    response.headers["X-Request-ID"] = trace_id
    logger.info(
        "Petición HTTP",
        extra={
            "trace_id": trace_id,
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "duration_ms": round(duration * 1000, 1),
        },
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Nunca filtra trazas al cliente: devuelve trace_id para buscar el error en logs."""
    trace_id = getattr(request.state, "trace_id", "")
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor", "trace_id": trace_id},
    )


app.include_router(router)
