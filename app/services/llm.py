"""Clientes LLM intercambiables (ARQUITECTURA.md §2.4).

- ``openai``: cualquier API compatible con ``/v1/chat/completions`` que soporte
  function calling (OpenAI, Groq, OpenRouter, Ollama, vLLM...). Cliente async.
- ``mock``: modo demo SIN clave de API — ejecuta el flujo completo del agente
  (llama a ``search_rules`` y responde citando lo recuperado) marcando
  claramente que no hay LLM real. Sirve para smoke tests, CI y para levantar la
  demo sin coste.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ToolCallData:
    id: str
    name: str
    arguments: str


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCallData] = field(default_factory=list)


class BaseLLM(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> LLMResponse:
        """Un paso de conversación. Devuelve texto final y/o peticiones de tools."""


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, settings: Settings) -> None:
        if settings.langfuse_enabled:
            if not (settings.langfuse_public_key and settings.langfuse_secret_key):
                raise ValueError(
                    "LANGFUSE_ENABLED=true requiere LANGFUSE_PUBLIC_KEY y LANGFUSE_SECRET_KEY "
                    "(crea unas claves de API en la UI de Langfuse; ver README)."
                )
            # Drop-in wrapper de Langfuse: misma API que openai.AsyncOpenAI,
            # pero cada generación queda trazada (prompt, tokens, coste, latencia).
            import os

            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = settings.langfuse_host
            from langfuse.openai import AsyncOpenAI
        else:
            from openai import AsyncOpenAI

        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if metadata:
            # Parámetro extra del wrapper de Langfuse (sesión/trace por turno).
            kwargs["metadata"] = metadata

        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        tool_calls = [
            ToolCallData(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (message.tool_calls or [])
        ]
        return LLMResponse(content=message.content, tool_calls=tool_calls)


class MockLLM(BaseLLM):
    """Simula el bucle del agente de forma determinista:

    1ª llamada -> pide ``search_rules`` con el último mensaje del usuario.
    2ª llamada -> responde en JSON citando las reglas recuperadas.
    """

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        metadata: dict | None = None,  # noqa: ARG002
    ) -> LLMResponse:
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )

        if not tool_messages:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCallData(
                        id="mock-call-1",
                        name="search_rules",
                        arguments=json.dumps({"query": last_user}, ensure_ascii=False),
                    )
                ],
            )

        citations: list[dict] = []
        rules: list[dict] = []
        try:
            payload = json.loads(tool_messages[-1]["content"])
            rules = payload.get("rules", [])
        except Exception:
            rules = []

        if rules:
            top = rules[:2]
            citations = [{"rule_number": rule["rule_number"]} for rule in top]
            body = "\n\n".join(
                f"**Regla {rule['rule_number']}** ({rule.get('title', '')}): {rule['text'][:300]}…"
                for rule in top
            )
            answer = (
                "**[MODO DEMO — sin LLM real]** Configura `LLM_PROVIDER=openai` y "
                f"`LLM_API_KEY` para obtener respuestas razonadas.\n\n"
                f"He recuperado {len(rules)} reglas relevantes del reglamento:\n\n{body}"
            )
        else:
            answer = (
                "**[MODO DEMO — sin LLM real]** No he encontrado reglas relevantes "
                "para esa consulta en el reglamento indexado."
            )

        return LLMResponse(
            content=json.dumps({"answer": answer, "citations": citations}, ensure_ascii=False)
        )


def get_llm(settings: Settings) -> BaseLLM:
    if settings.llm_provider == "openai":
        if not settings.llm_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai requiere LLM_API_KEY. "
                "Sin clave, usa LLM_PROVIDER=mock (modo demo)."
            )
        return OpenAICompatibleLLM(settings)
    if settings.llm_provider == "mock":
        return MockLLM()
    raise ValueError(f"Proveedor LLM desconocido: {settings.llm_provider}")
