"""Orquestador del agente: bucle de tool-calling con respuesta citada.

ARQUITECTURA.md §3: UN agente con tres herramientas cubre los cuatro requisitos
(no hace falta multi-agente en la demo). El bucle:

1. Recupera el historial de Redis y monta el contexto (system + historial + user).
2. Itera (máx. ``agent_max_iterations``): el LLM pide herramientas -> se
   ejecutan **en paralelo** con ``asyncio.gather`` -> resultados al contexto.
3. Respuesta final en JSON ``{"answer", "citations"}``.
4. **Validación de citas** (§7): solo se aceptan citas a reglas realmente
   recuperadas; si el LLM no cita pero recuperó reglas, se adjuntan las mejores
   como referencias. El LLM no puede inventar números de regla.
5. Persiste el turno en Redis y devuelve ``ChatResponse`` con trazabilidad.
"""

import asyncio
import json
import logging
import re
import time

from app.agent.tools import TOOL_SCHEMAS, AgentContext, ToolExecutor
from app.core.config import Settings
from app.models.schemas import CardSummary, ChatResponse, Citation, CustomCardSpec
from app.services.llm import BaseLLM, LLMResponse
from app.services.memory import ConversationMemory
from app.services.retrieval import RetrievedRule

logger = logging.getLogger(__name__)

MAX_CITATIONS = 5

SYSTEM_PROMPT = """Eres el asistente experto en reglas de Magic: The Gathering del call center de nuestro cliente.

PRINCIPIOS (en orden de prioridad):
1. Justifica SIEMPRE tus respuestas: usa las herramientas y cita los números de regla exactos que hayas recuperado (p. ej. "regla 702.7"). Sin base en el reglamento, no respondas dudas de reglas.
2. Nunca inventes reglas ni números de regla. Si las reglas recuperadas no bastan, dilo claramente y recomienda escalar la consulta a un humano.
3. Para buscar cartas por descripción usa search_cards con filtros estructurados y traduce los términos al inglés ("guerrero" -> Warrior, "blanco" -> W, "dañar primero" -> First Strike).
4. Para crear cartas personalizadas usa create_custom_card (y si usa habilidades reales, recupera antes sus definiciones con search_rules y cítales).
5. Responde en el idioma del usuario.
6. Puedes combinar varias herramientas y hacer varias rondas si lo necesitas.

FORMATO DE RESPUESTA FINAL (obligatorio, SOLO JSON válido, sin texto alrededor):
{"answer": "<respuesta en texto>", "citations": [{"rule_number": "<número de regla recuperada>"}]}
Incluye en "citations" únicamente reglas que hayas recuperado con search_rules."""


class AgentOrchestrator:
    def __init__(
        self,
        llm: BaseLLM,
        retriever,
        cards_api,
        memory: ConversationMemory,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._settings = settings
        self._executor = ToolExecutor(retriever, cards_api, settings)

    async def handle(self, session_id: str, user_message: str, trace_id: str) -> ChatResponse:
        started = time.perf_counter()
        history = await self._memory.get_history(session_id)

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": m.role, "content": m.content} for m in history]
        messages.append({"role": "user", "content": user_message})

        context = AgentContext()
        answer, raw_citations = "", None

        for _attempt in range(self._settings.agent_max_iterations):
            response = await self._llm.chat(
                messages,
                tools=TOOL_SCHEMAS,
                # Trazabilidad en Langfuse: cada paso del bucle queda asociado
                # a la sesión y a la traza del turno (session_id + trace_id).
                metadata={"session_id": session_id, "trace_id": trace_id, "attempt": _attempt},
            )

            if response.tool_calls:
                messages.append(_assistant_tool_message(response))
                # Tools independientes -> ejecución concurrente (§ gestión async).
                results = await asyncio.gather(
                    *(self._executor.execute(tc.name, tc.arguments, context) for tc in response.tool_calls)
                )
                for tool_call, result in zip(response.tool_calls, results):
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                    )
                continue

            answer, raw_citations = _parse_final(response.content or "")
            break
        else:
            answer = (
                "No he podido completar el razonamiento en el número máximo de pasos. "
                "Reformula la pregunta o escálala a un agente humano."
            )
            logger.warning("Agente: máximo de iteraciones alcanzado", extra={"trace_id": trace_id})

        citations = build_citations(raw_citations, context.retrieved_rules)
        cards = context.cards + [_custom_to_summary(spec) for spec in context.custom_cards]

        await self._memory.add_turn(session_id, user_message, answer)

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(
            "Turno de chat completado",
            extra={
                "trace_id": trace_id,
                "session_id": session_id,
                "intent": _infer_intent(context),
                "tools_used": context.tools_used,
                "citations": len(citations),
                "latency_ms": latency_ms,
            },
        )
        return ChatResponse(
            session_id=session_id,
            answer=answer,
            citations=citations,
            cards=cards,
            intent=_infer_intent(context),
            tools_used=context.tools_used,
            latency_ms=latency_ms,
            trace_id=trace_id,
            llm_provider=self._settings.llm_provider,
        )


# ---------------------------------------------------------------------------
# Post-procesado de la respuesta (funciones puras, cubiertas por tests)
# ---------------------------------------------------------------------------


def _assistant_tool_message(response: LLMResponse) -> dict:
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in response.tool_calls
        ],
    }


def _parse_final(content: str) -> tuple[str, list | None]:
    """Extrae ``{"answer", "citations"}`` del contenido final del LLM."""
    data = extract_json_object(content)
    if isinstance(data, dict) and data.get("answer"):
        return str(data["answer"]), data.get("citations")
    return content, None


def extract_json_object(text: str):
    """Parsea el primer objeto JSON del texto (tolera fences ```json ...```)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def build_citations(requested, retrieved: list[RetrievedRule]) -> list[Citation]:
    """Valida las citas del LLM contra las reglas realmente recuperadas (§7).

    - Descarta citas a reglas no recuperadas (anti-alucinación).
    - Empareja por número exacto, por prefijo ("505" <-> "505.1") o por sufijo
      de letra ("702.7" <-> "702.7a").
    - Si el LLM no citó nada pero sí recuperó reglas, adjunta las 3 mejores
      como referencias (la respuesta sí se basó en ellas).
    """
    by_number: dict[str, RetrievedRule] = {}
    for rule in sorted(retrieved, key=lambda r: r.distance):
        by_number.setdefault(rule.rule_number, rule)

    citations: list[Citation] = []
    if requested:
        for item in list(requested)[:10]:
            number = item.get("rule_number") if isinstance(item, dict) else item
            if not number:
                continue
            rule = _find_rule(str(number), by_number)
            if rule and rule.rule_number not in {c.rule_number for c in citations}:
                citations.append(_to_citation(rule))

    if not citations and by_number:
        for rule in list(by_number.values())[:3]:
            citations.append(_to_citation(rule))

    return citations[:MAX_CITATIONS]


def _find_rule(number: str, by_number: dict[str, RetrievedRule]) -> RetrievedRule | None:
    """Empareja la cita del LLM con una regla recuperada: número exacto, por
    prefijo ("505" <-> "505.1") o por sufijo de letra ("702.7a" <-> "702.7")."""
    number = number.strip()
    # "702.7a" -> "702.7" (el chunk fusiona los sufijos por letra).
    base = re.sub(r"[a-z]+$", "", number) if "." in number else number
    for probe in (number, base):
        if probe in by_number:
            return by_number[probe]
    for probe in (number, base):
        for candidate, rule in by_number.items():
            if candidate.startswith(probe + ".") or probe.startswith(candidate + "."):
                return rule
            if re.fullmatch(rf"{re.escape(probe)}[a-z]", candidate):
                return rule
    return None


def _to_citation(rule: RetrievedRule) -> Citation:
    return Citation(
        rule_number=rule.rule_number,
        title=rule.title,
        text=rule.text[:600],
        score=rule.score,
    )


def _custom_to_summary(spec: CustomCardSpec) -> CardSummary:
    return CardSummary(
        name=spec.name,
        mana_cost=spec.mana_cost,
        colors=spec.colors,
        type_line=spec.type_line,
        text=spec.rules_text,
        power=spec.power or "",
        toughness=spec.toughness or "",
        is_custom=True,
    )


def _infer_intent(context: AgentContext) -> str:
    used = set(context.tools_used)
    if "search_cards" in used:
        return "card_search"
    if "create_custom_card" in used:
        return "custom_card"
    if "search_rules" in used:
        return "rules_qa"
    return "general"
