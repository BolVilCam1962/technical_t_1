"""Endpoints de la API (documentados en OpenAPI/Swagger en /docs).

Convención: los servicios viven en ``app.state`` (creados en el lifespan de
``app.main``) y se acceden vía ``request.app.state`` — el servicio es stateless
y toda la memoria de sesión está en Redis.
"""

import logging

from fastapi import APIRouter, Request, Response
from fastapi.responses import PlainTextResponse, RedirectResponse

from app.core.config import get_settings
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ComponentStatus,
    HealthResponse,
    HistoryResponse,
    SessionDeletedResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirige a la documentación interactiva."""
    return RedirectResponse(url="/docs")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Enviar un mensaje al chatbot",
    description="Endpoint principal: un mensaje dentro de una sesión y la respuesta **justificada**.",
    responses={
        200: {"description": "Respuesta generada (con citas y/o cartas)"},
        422: {"description": "Payload inválido (validación Pydantic)"},
    },
    tags=["Chat"],
)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Procesa un turno de conversación.

    ## Flujo interno
    1. Recupera el historial de la sesión desde **Redis** (últimos N mensajes, TTL 24 h).
    2. El **agente** (bucle de tool-calling, máx. 5 iteraciones) decide qué
       herramientas usar:
       - `search_rules` → búsqueda semántica en **ChromaDB** (reglamento Comprehensive Rules)
       - `search_cards` → API externa magicthegathering.io (con caché Redis)
       - `create_custom_card` → carta custom validada con schema (bonus)
    3. Genera la respuesta final **citando los números de regla** recuperados.
       Las citas se validan contra lo realmente recuperado (anti-alucinación).
    4. Persiste el turno en Redis y devuelve `ChatResponse`.

    ## Justificación
    `citations` contiene las reglas (`rule_number`, `text`, `score`) que
    sustentan la respuesta; `cards` las cartas encontradas o generadas.
    Si no hay base suficiente, el chatbot responde honestamente que no lo sabe.

    ## Ejemplo
    ```bash
    curl -X POST http://localhost:8000/chat \\
      -H 'Content-Type: application/json' \\
      -d '{"session_id": "demo-1", "message": "¿Qué fases hay en un turno de Magic?"}'
    ```

    Con `LLM_PROVIDER=mock` (modo demo sin clave de API) el flujo completo se
    ejecuta igualmente (retrieval + citas) pero sin razonamiento del LLM.
    """
    orchestrator = request.app.state.orchestrator
    trace_id = getattr(request.state, "trace_id", "")
    return await orchestrator.handle(payload.session_id, payload.message, trace_id)


# ---------------------------------------------------------------------------
# Sesiones
# ---------------------------------------------------------------------------


@router.get(
    "/sessions/{session_id}/history",
    response_model=HistoryResponse,
    summary="Historial de una sesión",
    tags=["Sesiones"],
)
async def session_history(session_id: str, request: Request) -> HistoryResponse:
    """Devuelve los turnos de la conversación (de más antiguo a más reciente).

    El historial vive en Redis con TTL: una sesión sin actividad o inexistente
    devuelve una lista vacía (no es un error).
    """
    memory = request.app.state.memory
    messages = await memory.get_history(session_id)
    return HistoryResponse(session_id=session_id, messages=messages)


@router.delete(
    "/sessions/{session_id}",
    response_model=SessionDeletedResponse,
    summary="Eliminar una sesión",
    tags=["Sesiones"],
)
async def delete_session(session_id: str, request: Request) -> SessionDeletedResponse:
    """Borra el historial de la sesión de Redis.

    Útil para tests, RGPD (derecho al olvido del cliente del call center) o para
    reiniciar una conversación.
    """
    memory = request.app.state.memory
    deleted = await memory.clear(session_id)
    return SessionDeletedResponse(session_id=session_id, deleted=deleted)


# ---------------------------------------------------------------------------
# Operación
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check con estado de dependencias",
    tags=["Operación"],
)
async def health(request: Request) -> HealthResponse:
    """Estado del servicio y sus dependencias.

    | Componente | Descripción |
    |---|---|
    | `chroma` | Índice del reglamento accesible y no vacío |
    | `redis` | Conexión a Redis (memoria de sesión y cachés) |
    | `embeddings` | Modelo de embeddings cargado |
    | `llm` | Proveedor LLM configurado (`mock` = modo demo, se marca como degradado) |

    El estado global es `ok` si Chroma y Redis responden; si no, `degraded`.
    No se comprueba la API externa de cartas aquí (añadiría latencia y
    dependencia externa al health check; su degradación se maneja por turno).
    """
    state = request.app.state
    settings = get_settings()

    components: list[ComponentStatus] = []

    try:
        rules_count = await state.retriever.count()
        components.append(
            ComponentStatus(
                name="chroma",
                status="ok" if rules_count > 0 else "degraded",
                detail=f"{rules_count} chunks indexados"
                if rules_count > 0
                else "índice vacío: ejecuta scripts/ingest_rules.py",
            )
        )
    except Exception as exc:
        components.append(ComponentStatus(name="chroma", status="error", detail=str(exc)))

    redis_ok = await state.memory.ping()
    components.append(
        ComponentStatus(
            name="redis",
            status="ok" if redis_ok else "degraded",
            detail=settings.redis_url if redis_ok else "sin conexión: funcionando sin memoria de sesión",
        )
    )

    components.append(
        ComponentStatus(name="embeddings", status="ok", detail=f"{state.embedder.name} (dim={state.embedder.dim})")
    )

    components.append(
        ComponentStatus(
            name="llm",
            status="ok" if settings.llm_provider == "openai" else "degraded",
            detail=f"{settings.llm_provider}"
            + (" — modo demo sin LLM real" if settings.llm_provider == "mock" else ""),
        )
    )

    critical_ok = all(c.status == "ok" for c in components if c.name in ("chroma", "redis"))
    return HealthResponse(
        status="ok" if critical_ok else "degraded",
        version=settings.app_version,
        environment=settings.environment,
        components=components,
    )


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Métricas Prometheus",
    tags=["Operación"],
)
async def metrics() -> Response:
    """Métricas en formato Prometheus (ARQUITECTURA.md §8).

    - ``app_http_requests_total{method,path,status}``
    - ``app_http_request_latency_seconds{path}``
    - ``agent_tool_calls_total{tool}``

    `path` usa la plantilla de la ruta (``/sessions/{session_id}/history``),
    no la URL concreta, para evitar cardinalidad infinita.
    """
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
