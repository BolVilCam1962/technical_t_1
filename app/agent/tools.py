"""Herramientas del agente (function calling) — ARQUITECTURA.md §3 y §6.

Tres herramientas cubren los cuatro requisitos del cliente:

- ``search_rules`` (R1/R2): búsqueda semántica en ChromaDB. Devuelve reglas
  numeradas que el LLM DEBE citar en la respuesta final.
- ``search_cards`` (R3): filtros estructurados validados con Pydantic ->
  magicthegathering.io (con caché Redis).
- ``create_custom_card`` (R4): especificación de carta custom validada con schema.

Cada argumento se valida ANTES de ejecutar la herramienta; si falla, se devuelve
el error como resultado para que el propio LLM corrija en la siguiente iteración
(patón de agente robusto, no excepción al usuario).
"""

import json
import logging
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.core.config import Settings
from app.core.metrics import AGENT_TOOL_CALLS
from app.models.schemas import CardFilters, CardSummary, CustomCardSpec
from app.services.cards_api import CardsAPIError, MagicTheGatheringAPI
from app.services.retrieval import RetrievedRule, RulesRetriever

logger = logging.getLogger(__name__)

# Máximo de caracteres del texto de regla que se devuelve al LLM (control de tokens).
RULE_TEXT_LIMIT = 1200

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_rules",
            "description": (
                "Search the Magic: The Gathering Comprehensive Rules by semantic similarity. "
                "Use it for ANY question about game rules, mechanics, card interactions, turns, "
                "phases, mana, combat, zones or keyword abilities (first strike, haste, flying...). "
                "Returns the most relevant numbered rules with their text. Always cite the rule "
                "numbers you retrieve in the final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search, in English for best results "
                        "(e.g. 'first strike combat damage step order').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_cards",
            "description": (
                "Search REAL Magic cards by structured filters using the public "
                "magicthegathering.io API. Use it when the user describes cards to find. "
                "Translate Spanish terms to English (guerrero->Warrior, blanco->W, "
                "'dañar primero'->'First Strike')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "colors": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["W", "U", "B", "R", "G"]},
                        "description": "Card colors (W white, U blue, B black, R red, G green). "
                        "Multiple colors = the card must have ALL of them.",
                    },
                    "cmc_max": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Maximum converted mana cost, inclusive "
                        "('cost lower than two' -> 1).",
                    },
                    "subtypes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Creature subtypes in English, e.g. ['Warrior'].",
                    },
                    "types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Card types in English, e.g. ['Creature'].",
                    },
                    "text": {
                        "type": "string",
                        "description": "Free text that must appear in the card's rules text (English).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_custom_card",
            "description": (
                "Generate a custom fan-made card specification. Only use it when the user "
                "explicitly asks to create a custom/invented card. If the card uses real keyword "
                "abilities (First Strike, Haste...), also call search_rules first and cite the "
                "keyword definitions in your answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Card name."},
                    "colors": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["W", "U", "B", "R", "G"]},
                        "description": "Card colors.",
                    },
                    "mana_cost": {"type": "string", "description": "Mana cost, e.g. '{1}{W}{R}'."},
                    "type_line": {
                        "type": "string",
                        "description": "Type line, e.g. 'Creature — Human Smuggler'.",
                    },
                    "rules_text": {"type": "string", "description": "Rules text of the card."},
                    "power": {"type": "string", "description": "Power (creatures only)."},
                    "toughness": {"type": "string", "description": "Toughness (creatures only)."},
                },
                "required": ["name", "colors", "type_line", "rules_text"],
            },
        },
    },
]


@dataclass
class AgentContext:
    """Estado acumulado durante un turno: alimenta la respuesta final (citas, cartas, intent)."""

    retrieved_rules: list[RetrievedRule] = field(default_factory=list)
    cards: list[CardSummary] = field(default_factory=list)
    custom_cards: list[CustomCardSpec] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    def note_tool(self, name: str) -> None:
        if name not in self.tools_used:
            self.tools_used.append(name)


class ToolExecutor:
    def __init__(self, retriever: RulesRetriever, cards_api: MagicTheGatheringAPI, settings: Settings) -> None:
        self._retriever = retriever
        self._cards_api = cards_api
        self._settings = settings

    async def execute(self, name: str, arguments: str, context: AgentContext) -> str:
        """Ejecuta una herramienta y devuelve SIEMPRE un string JSON.

        Los errores (argumentos inválidos, API caída) se devuelven como resultado
        para que el LLM pueda reintentar o degradarse con elegancia.
        """
        AGENT_TOOL_CALLS.labels(tool=name).inc()
        context.note_tool(name)
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return _json({"error": "argumentos no son JSON válido", "detail": str(exc)})
        if not isinstance(args, dict):
            return _json({"error": "los argumentos deben ser un objeto JSON"})

        try:
            if name == "search_rules":
                return await self._search_rules(args, context)
            if name == "search_cards":
                return await self._search_cards(args, context)
            if name == "create_custom_card":
                return self._create_custom_card(args, context)
            return _json({"error": f"herramienta desconocida: {name}"})
        except ValidationError as exc:
            return _json({"error": "argumentos inválidos", "detail": exc.errors(include_url=False)[:5]})
        except CardsAPIError as exc:
            return _json({"error": "API de cartas no disponible", "detail": str(exc)})
        except Exception:  # nunca rompe el turno: el LLM ve el error
            logger.exception("Error ejecutando herramienta", extra={"tool": name})
            return _json({"error": "error interno ejecutando la herramienta"})

    # --- Implementaciones ----------------------------------------------------

    async def _search_rules(self, args: dict, context: AgentContext) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return _json({"error": "falta el campo 'query'"})
        rules = await self._retriever.search(query)
        context.retrieved_rules.extend(rules)
        if not rules:
            return _json(
                {"rules": [], "note": "No relevant rules found. Answer honestly or rephrase the query."}
            )
        return _json(
            {
                "rules": [
                    {
                        "rule_number": rule.rule_number,
                        "title": rule.title,
                        "chapter": rule.chapter_title,
                        "text": rule.text[:RULE_TEXT_LIMIT],
                        "score": rule.score,
                    }
                    for rule in rules
                ]
            }
        )

    async def _search_cards(self, args: dict, context: AgentContext) -> str:
        filters = CardFilters.model_validate(args)
        cards = await self._cards_api.search(filters)
        context.cards.extend(cards)
        return _json(
            {
                "cards": [
                    {
                        "name": card.name,
                        "mana_cost": card.mana_cost,
                        "type_line": card.type_line,
                        "text": card.text[:300],
                        "image_url": card.image_url,
                    }
                    for card in cards
                ],
                "count": len(cards),
                "note": "Present these cards to the user and explain which filters were applied.",
            }
        )

    def _create_custom_card(self, args: dict, context: AgentContext) -> str:
        spec = CustomCardSpec.model_validate(args)
        context.custom_cards.append(spec)
        return _json(
            {
                "card": spec.model_dump(),
                "note": "Card specification validated. Present it to the user and cite the rules "
                "of the keyword abilities it uses.",
            }
        )


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)
