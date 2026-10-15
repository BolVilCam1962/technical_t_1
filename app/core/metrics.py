"""Métricas Prometheus expuestas en ``GET /metrics`` (ARQUITECTURA.md §8).

Uso de ``path`` como etiqueta: se usa la *plantilla* de la ruta
(p. ej. ``/sessions/{session_id}/history``) y no la URL concreta, para evitar
cardinalidad infinita.
"""

from prometheus_client import Counter, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    "app_http_requests_total",
    "Peticiones HTTP totales",
    ["method", "path", "status"],
)

HTTP_REQUEST_LATENCY = Histogram(
    "app_http_request_latency_seconds",
    "Latencia de las peticiones HTTP",
    ["path"],
)

AGENT_TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Invocaciones de herramientas del agente",
    ["tool"],
)
