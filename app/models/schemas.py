"""Modelos Pydantic (contrato de la API + validación de argumentos de tools).

Estos modelos son la fuente de la documentación OpenAPI/Swagger: cada campo
lleva ``description`` para que /docs sea autoexplicativo.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ColorCode = Literal["W", "U", "B", "R", "G"]

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """Mensaje de una conversación (solo turnos user/assistant)."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"session_id": "demo-1", "message": "¿Qué fases hay en un turno de Magic?"},
                {"session_id": "demo-1", "message": "Busco una carta blanca de coste 2 o menos que sea guerrero"},
                {"session_id": "demo-1", "message": "Mi criatura tiene dañar primero y hace daño; si el rival la destruye después, ¿el daño se aplicó igual?"},
            ]
        }
    )

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Identificador de la conversación. El historial se persiste en Redis "
        "por sesión (TTL 24 h, últimos N mensajes).",
        examples=["demo-1"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Mensaje del usuario (español o inglés). Puede ser una duda de reglas, "
        "una interacción entre cartas, una búsqueda de cartas o una petición de carta custom.",
        examples=["¿Qué fases hay en un turno de Magic?"],
    )


class Citation(BaseModel):
    """Regla del reglamento que sustenta la respuesta (justificación R1/R2)."""

    rule_number: str = Field(
        ...,
        description="Número de regla del Comprehensive Rules (p. ej. '702.7') o término del glosario.",
        examples=["702.7"],
    )
    title: str = Field("", description="Título de la regla o capítulo (p. ej. 'First Strike').")
    text: str = Field("", description="Texto (recortado) de la regla citada.")
    score: float = Field(
        ...,
        description="Similitud semántica del chunk recuperado (1 - distancia coseno).",
        examples=[0.82],
    )


class CardSummary(BaseModel):
    """Carta real (magicthegathering.io) o custom generada, presentada al usuario."""

    name: str = Field(..., description="Nombre de la carta.", examples=["Phantom Warrior"])
    mana_cost: str = Field("", description="Coste de mana en notación MTG.", examples=["{1}{U}{U}"])
    colors: list[ColorCode] = Field(default_factory=list, description="Colores: W/U/B/R/G.")
    type_line: str = Field("", description="Línea de tipo completa.", examples=["Creature — Illusion Warrior"])
    subtypes: list[str] = Field(default_factory=list, description="Subtipos.")
    text: str = Field("", description="Texto de reglas de la carta.")
    power: str = Field("", description="Fuerza (si aplica).")
    toughness: str = Field("", description="Resistencia (si aplica).")
    image_url: str | None = Field(None, description="Imagen oficial (Gatherer), si existe.")
    set_name: str = Field("", description="Edición (para cartas reales).")
    is_custom: bool = Field(False, description="True si es una carta custom generada (bonus R4).")


class ChatResponse(BaseModel):
    """Respuesta del chatbot. `citations` y `cards` son la base de la justificación."""

    session_id: str
    answer: str = Field(..., description="Respuesta en texto, en el idioma del usuario.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Reglas citadas. Se validan contra las reglas realmente recuperadas: "
        "el LLM no puede citar reglas que no estén en el contexto recuperado.",
    )
    cards: list[CardSummary] = Field(default_factory=list, description="Cartas encontradas o generadas.")
    intent: Literal["rules_qa", "card_search", "custom_card", "general"] | None = Field(
        None, description="Intención inferida a partir de las herramientas usadas."
    )
    tools_used: list[str] = Field(default_factory=list, description="Herramientas invocadas por el agente.")
    latency_ms: float = Field(..., description="Latencia total del turno (ms).")
    trace_id: str = Field(..., description="Identificador de traza (también en cabecera X-Request-ID).")
    llm_provider: str = Field(..., description="Proveedor LLM usado ('mock' = modo demo sin LLM real).")


# ---------------------------------------------------------------------------
# Sesiones
# ---------------------------------------------------------------------------


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage] = Field(default_factory=list, description="Turnos, de más antiguo a más reciente.")


class SessionDeletedResponse(BaseModel):
    session_id: str
    deleted: bool = Field(..., description="True si la sesión existía y se eliminó.")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class ComponentStatus(BaseModel):
    name: str
    status: Literal["ok", "degraded", "error"]
    detail: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    components: list[ComponentStatus] = Field(..., description="Estado de cada dependencia.")


# ---------------------------------------------------------------------------
# Validación de argumentos de herramientas (R3 / R4)
# ---------------------------------------------------------------------------


class CardFilters(BaseModel):
    """Filtros estructurados para buscar cartas en magicthegathering.io.

    El LLM construye estos argumentos vía function calling y Pydantic los valida
    ANTES de llamar a la API externa. Los términos van en INGLÉS (guerrero -> Warrior).
    """

    colors: list[ColorCode] | None = Field(
        None, description="Colores de la carta (W=blanco, U=azul, B=negro, R=rojo, G=verde)."
    )
    cmc_max: int | None = Field(
        None, ge=0, le=20,
        description="Coste de mana convertido máximo (incluido). 'inferior a dos' -> 1.",
    )
    subtypes: list[str] | None = Field(
        None, description="Subtipos en inglés, p. ej. ['Warrior'], ['Wizard']."
    )
    types: list[str] | None = Field(
        None, description="Tipos de carta en inglés, p. ej. ['Creature'], ['Instant']."
    )
    text: str | None = Field(
        None, description="Texto que debe aparecer en el texto de reglas de la carta (inglés)."
    )


class CustomCardSpec(BaseModel):
    """Especificación de una carta custom (bonus R4), validada con schema."""

    name: str = Field(..., min_length=1, max_length=80, description="Nombre de la carta.")
    colors: list[ColorCode] = Field(..., min_length=1, description="Colores de la carta.")
    mana_cost: str = Field("", description="Coste de mana, p. ej. '{1}{W}{R}'.")
    type_line: str = Field(..., min_length=3, max_length=120, description="Línea de tipo, p. ej. 'Creature — Human Smuggler'.")
    rules_text: str = Field(..., min_length=3, description="Texto de reglas de la carta.")
    power: str | None = Field(None, description="Fuerza (criaturas).")
    toughness: str | None = Field(None, description="Resistencia (criaturas).")
