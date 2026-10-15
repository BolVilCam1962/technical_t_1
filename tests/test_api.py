"""Tests de integración de la API (TestClient + mock LLM + fake embedder + Redis
en proceso): el flujo completo /chat -> agente -> retrieval -> citas -> historial."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_openapi_documentation_is_generated(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert spec["info"]["title"]
    assert "/chat" in spec["paths"]
    assert "/sessions/{session_id}/history" in spec["paths"]
    # Los endpoints documentan sus modelos de respuesta (contrato en Swagger).
    assert spec["paths"]["/chat"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]


def test_chat_returns_justified_answer(client):
    response = client.post(
        "/chat",
        json={"session_id": "test-session", "message": "What is the main phase of a turn?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session"
    assert data["llm_provider"] == "mock"
    assert data["intent"] == "rules_qa"
    assert "search_rules" in data["tools_used"]
    assert data["latency_ms"] >= 0
    assert data["trace_id"]
    # Requisito del enunciado: la respuesta debe poder justificarse.
    assert data["citations"], "la respuesta debe llevar citas a reglas"
    assert data["citations"][0]["rule_number"]
    assert data["citations"][0]["score"] > 0


def test_chat_persists_history_in_redis(client):
    client.post("/chat", json={"session_id": "hist-session", "message": "What is haste?"})
    client.post(
        "/chat", json={"session_id": "hist-session", "message": "And what is first strike?"}
    )
    history = client.get("/sessions/hist-session/history").json()
    assert history["session_id"] == "hist-session"
    roles = [message["role"] for message in history["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_delete_session(client):
    client.post("/chat", json={"session_id": "del-session", "message": "What is mana?"})
    assert client.delete("/sessions/del-session").json()["deleted"] is True
    # Borrar una sesión inexistente no es error.
    assert client.delete("/sessions/del-session").json()["deleted"] is False
    assert client.get("/sessions/del-session/history").json()["messages"] == []


def test_history_of_unknown_session_is_empty_not_404(client):
    response = client.get("/sessions/never-seen/history")
    assert response.status_code == 200
    assert response.json()["messages"] == []


def test_chat_validates_payload(client):
    response = client.post("/chat", json={"session_id": "", "message": ""})
    assert response.status_code == 422


def test_health_reports_components(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    components = {component["name"]: component["status"] for component in data["components"]}
    assert components["chroma"] == "ok"
    assert components["redis"] == "ok"
    assert components["embeddings"] == "ok"
    # El mock es modo demo: se reporta como degradado, pero el servicio es "ok".
    assert components["llm"] == "degraded"
    assert data["status"] == "ok"


def test_metrics_prometheus(client):
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "app_http_requests_total" in response.text
    assert "agent_tool_calls_total" in response.text


def test_trace_id_header_and_reuse(client):
    first = client.post(
        "/chat", json={"session_id": "trace-session", "message": "What is mana?"}
    )
    assert first.headers["x-request-id"]
    reused = client.get("/health", headers={"X-Request-ID": "my-trace-123"})
    assert reused.headers["x-request-id"] == "my-trace-123"
