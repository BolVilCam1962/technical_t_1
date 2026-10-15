# technical_t_1

Chatbot de reglas de **Magic: The Gathering** para el call center de nuestro cliente.
Diseño completo en **[ARQUITECTURA.md](ARQUITECTURA.md)**.

**Stack**: FastAPI · ChromaDB · Redis · LLM vía API (function calling).

## Enunciado

Apartado 1 — Implementación Técnica.

Nuestro cliente tiene un call center para resolución de dudas y consejos para el juego de cartas Magic the Gathering. Nos pide construir un chatbot para poder automatizar las respuestas a sus clientes. Para ello nos ha dado un reglamento de juego (El cual se adjunta) y una API para poder disponer de imágenes de cartas y nuevos releases (https://docs.magicthegathering.io/).

Se pide un sistema que sea capaz de:

- Resolver dudas de reglas básicas del juego (¿Qué fases hay en un turno de juego?, ¿Cómo funciona el mana?, etc.)
- Interacciones entre cartas (Mi rapaz del campo de batalla ha hecho daño con su daña primero, si lo cambio con mi ninja de horas tardías ¿Aplico el daño?)
- Búsqueda de cartas según descripción del usuario (Busco una carta de color blanco de coste inferior a dos de mana que sea guerrero)
- (Bonus) Creación de cartas custom (Quiero una carta de Han Solo, blanca-roja que tenga dañar primero)

Se pide:

- Un documento explicando una solución completa productiva (Servicios, tipo de agentes, monitorización etc) → **ARQUITECTURA.md**
- Una solución demo básica respondiendo a los requerimientos → **este backend**

No se evaluará la veracidad de las respuestas del chatbot aunque sí se pide que tengan una base de referencia (i.e. debe ser capaz de **justificar** su respuesta citando reglas).

## Puesta en marcha (Docker)

Prerrequisitos: Docker y Docker Compose v2.

```bash
# 0) Prepara el directorio del índice (evita que Docker lo cree como root)
#    y (opcional) configura variables: cp .env.example .env
mkdir -p data

# 1) Construir la imagen (~2 GB: incluye torch CPU para embeddings locales;
#    la primera vez tarda varios minutos)
docker compose build

# 2) Ingesta del reglamento: PDF -> chunks -> embeddings -> ChromaDB
#    (la PRIMERA vez descarga el modelo multilingüe, ~470 MB)
docker compose --profile tools run --rm ingest

# 3) Levantar los servicios: app (FastAPI) + redis
docker compose up -d
docker compose ps

# 4) Ver logs de la app
docker compose logs -f app
```

**Documentación interactiva**:

| URL | Qué es |
|---|---|
| http://localhost:8000/docs | **Swagger UI** — endpoints, modelos, ejemplos, "Try it out" |
| http://localhost:8000/redoc | ReDoc |
| http://localhost:8000/openapi.json | Especificación OpenAPI cruda |
| http://localhost:8000/health | Health check con estado de dependencias |
| http://localhost:8000/metrics | Métricas Prometheus |

Con el perfil `observability` activo (ver abajo): **Grafana** http://localhost:3001 (admin/admin), **Dozzle** http://localhost:8081, **Langfuse** http://localhost:3000.

### Probar el chatbot

```bash
# R1 — duda de reglas (responde con citas a números de regla)
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo-1", "message": "¿Qué significa dañar primero?"}'

# R2 — interacción entre cartas (justifica con las reglas recuperadas)
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo-1", "message": "Mi criatura con dañar primero ya hizo daño; si el rival la destruye después, ¿el daño se aplicó igual?"}'

# R3 — búsqueda de cartas (NL -> filtros -> magicthegathering.io)
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo-2", "message": "Busco una carta blanca de coste inferior a dos de mana que sea guerrero"}'

# R4 — bonus: carta custom
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "demo-3", "message": "Quiero una carta de Han Solo, blanca-roja, que tenga dañar primero"}'

# Historial de la sesión (memoria en Redis)
curl http://localhost:8000/sessions/demo-1/history

# Borrar sesión
curl -X DELETE http://localhost:8000/sessions/demo-1
```

> **Modo demo sin clave LLM**: por defecto `LLM_PROVIDER=mock` — el flujo
> completo funciona (retrieval semántico + citas reales al reglamento) pero sin
> razonamiento del LLM. Para la experiencia completa:
> ```bash
> LLM_PROVIDER=openai LLM_API_KEY=sk-... LLM_MODEL=gpt-4o-mini docker compose up -d
> ```
> Vale cualquier endpoint compatible (Groq, OpenRouter, Ollama, vLLM...): ajusta
> `LLM_BASE_URL` y `LLM_MODEL`. Con `LLM_PROVIDER=openai` también se habilitan
> R3/R4 end-to-end (la búsqueda de cartas necesita function calling real).

### Calidad y validación

Dos capas de evaluación (ARQUITECTURA.md §7):

```bash
# 1) Retrieval: golden set ES->EN (Recall@5, MRR@10). Exit 1 si Recall@5 < 0.8 (gate de CI).
docker compose --profile tools run --rm eval

# 2) End-to-end del /chat con promptfoo: asserts deterministas de justificación
#    (citas con número de regla válido, resultados de R3/R4 correctos) — funciona
#    en modo demo; con LLM real puedes descomentar los asserts llm-rubric del YAML.
docker compose --profile tools run --rm promptfoo

# 3) DOBLE JUEZ: cada respuesta la evalúan dos modelos distintos con la misma
#    rúbrica; un test solo pasa si ambos consensuan (R1, R2, R3, R4 + honestidad).
#    Requiere LLM real y jueces configurados (JUDGE_1/JUDGE_2 en .env, ver abajo).
docker compose --profile tools run --rm promptfoo \
  npx -y promptfoo@latest eval --config /eval/promptfoojudges.yaml

# Tests unitarios + API (mock LLM, fake embedder, sin red)
docker compose run --rm app pytest -q

# Solo parsear el PDF y ver estadísticas (sin escribir el índice)
docker compose --profile tools run --rm ingest \
  python scripts/ingest_rules.py --pdf /app/rules/MagicCompRules.pdf --stats-only
```

### Observabilidad (opcional): métricas, logs y trazas de LLM

```bash
docker compose --profile observability up -d
```

| Servicio | URL | Qué ves |
|---|---|---|
| **Grafana** (+ Prometheus) | http://localhost:3001 (admin/admin) | Dashboard provisionado: requests/s por ruta, p95 de `/chat`, llamadas por tool, errores 5xx |
| **Dozzle** | http://localhost:8081 | Logs JSON de todos los contenedores en vivo |
| **Langfuse** (self-hosted v3) | http://localhost:3000 | **Trazas de LLM**: cada generación con prompt, tools, tokens y coste, por sesión |

**Activar las trazas de Langfuse en la app** (el stack self-hosted son ~7 contenedores; alternativa: Langfuse Cloud gratis):

1. Levanta el perfil y regístrate en http://localhost:3000 → Settings → crea **claves de API** de la organización.
2. Ponlas en `.env` y activa el flag:

   ```env
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=http://langfuse-web:3000   # red interna de compose (Cloud: https://cloud.langfuse.com)
   ```

3. `docker compose up -d` — cada llamada al LLM queda trazada (el wrapper drop-in
   de Langfuse envuelve `AsyncOpenAI`; las trazas llevan `session_id` y `trace_id`).

### Doble juez LLM (`eval/promptfoojudges.yaml`)

Los asserts `llm-rubric` admiten un `provider` propio por aserción: cada test
evalúa la respuesta con **dos jueces y la misma rúbrica** (anchor YAML), y
`threshold: 0.99` exige el consenso de ambos (bájalo a `0.5` para "al menos
uno"). Defaults en `.env`:

```env
JUDGE_1=openai:gpt-4o-mini      # juez económico
JUDGE_2=openai:gpt-4o           # juez potente
# Máximo valor (cruzar familias):
# JUDGE_2=anthropic:claude-sonnet-4-5    + ANTHROPIC_API_KEY=sk-ant-...
# JUDGE_2=groq:llama-3.3-70b-versatile  + GROQ_API_KEY=gsk_...
```

La tabla de resultados de promptfoo muestra el veredicto de cada juez por
separado: si divergen, ese test es candidato a revisión manual (la divergencia
entre jueces es señal tan valiosa como el propio consenso).

### Operación de los servicios

```bash
# Redis: inspeccionar el estado real
docker compose exec redis redis-cli DBSIZE            # claves
docker compose exec redis redis-cli KEYS 'chat:*'     # sesiones activas
docker compose exec redis redis-cli LRANGE chat:demo-1 0 -1   # historial crudo

# Reiniciar la app (necesario tras re-ingestar el reglamento:
# el proceso mantiene el índice de Chroma en memoria)
docker compose --profile tools run --rm ingest && docker compose restart app

# Parar todo (conserva el índice en ./data y el modelo en el volumen hf-cache)
docker compose down

# Borrar también los volúmenes (caché del modelo y datos de redis)
docker compose down -v

# Regenerar el índice desde cero
rm -rf data/chroma && docker compose --profile tools run --rm ingest
```

### Desarrollo sin Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
# Sin Redis local: REDIS_URL=memory:// (todo funciona, sin persistencia)
REDIS_URL=memory:// EMBEDDING_PROVIDER=local LLM_PROVIDER=mock \
  uvicorn app.main:app --reload
pytest -q
```

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/chat` | Turno de conversación → respuesta + `citations[]` + `cards[]` |
| `GET` | `/sessions/{id}/history` | Historial de la sesión (Redis, TTL 24 h) |
| `DELETE` | `/sessions/{id}` | Borrar sesión (tests / RGPD) |
| `GET` | `/health` | Estado de Chroma, Redis, embeddings y LLM |
| `GET` | `/metrics` | Métricas Prometheus |
| `GET` | `/docs`, `/redoc` | Documentación interactiva |

Todos los endpoints están documentados en Swagger (descripción, flujo interno,
ejemplos, modelos de request/response y códigos de error). Cada respuesta
incluye la cabecera `X-Request-ID` (trace id propagado a los logs).

## Estructura del repo

```
app/
├── main.py                # FastAPI: lifespan, middleware de observabilidad, OpenAPI
├── api/routes.py          # Endpoints documentados (Swagger)
├── core/config.py         # Settings (env vars / .env)
├── core/logging.py        # Logs JSON estructurados
├── core/metrics.py        # Contadores/histograms Prometheus
├── models/schemas.py      # Contrato Pydantic (valida también los args de las tools)
├── agent/orchestrator.py  # Bucle tool-calling + validación de citas anti-alucinación
├── agent/tools.py         # search_rules / search_cards / create_custom_card
├── services/retrieval.py  # ChromaDB (cosine + umbral de distancia)
├── services/embeddings.py # local (e5 multilingüe) / openai / fake (tests)
├── services/memory.py     # Redis: historial de sesión + cachés (degradación elegante)
├── services/cards_api.py  # Cliente magicthegathering.io (caché + dedupe)
├── services/llm.py        # OpenAI-compatible / mock (modo demo)
└── ingest/rules_parser.py # Chunking estructural del reglamento (regla numerada)
scripts/ingest_rules.py    # PDF -> chunks -> embeddings -> Chroma (one-shot)
scripts/eval_retrieval.py  # Golden set: Recall@5, MRR@10, gate de CI
eval/golden_set.jsonl      # Preguntas ES -> regla esperada EN
eval/promptfooconfig.yaml  # Evaluación end-to-end /chat (promptfoo: justificación + calidad)
observability/             # Prometheus + Grafana provisionados (perfil "observability")
tests/                     # Parser, citas, API end-to-end (sin red ni claves)
```

## Gestión de procesos asíncronos

El backend es async end-to-end y evita bloquear el event loop:

- **Lifespan** (no `@app.on_event`): crea una sola vez los clientes compartidos
  (httpx, `redis.asyncio`, `AsyncOpenAI`, embedder) y los cierra al apagar.
- **CPU-bound al threadpool**: los embeddings (sentence-transformers) y las
  queries de Chroma son síncronos → `asyncio.to_thread`. El modelo se
  **precarga en el arranque** para no pagar la carga en la primera petición.
- **I/O async**: API de cartas (httpx), Redis y LLM.
- **Tools concurrentes**: si el LLM pide varias herramientas en una iteración,
  se ejecutan en paralelo con `asyncio.gather`.
- **1 worker de uvicorn** por contenedor: el modelo de embeddings vive en RAM
  por proceso; se escala con réplicas del contenedor, no con `--workers`.

## Troubleshooting

| Síntoma | Causa/solución |
|---|---|
| `/health` marca `chroma: degraded` | Índice vacío → ejecuta la ingesta (paso 2) |
| Respuestas `[MODO DEMO]` | `LLM_PROVIDER=mock` (sin clave). Configura `LLM_PROVIDER=openai` + `LLM_API_KEY` |
| La app no ve las reglas tras re-ingestar | `docker compose restart app` (el proceso cachea el índice) |
| Error de permisos en `./data` | Docker lo creó como root → `sudo rm -rf data && mkdir data` |
| Primera ingesta muy lenta | Está descargando el modelo (~470 MB); queda cacheado en el volumen `hf-cache` |
| Búsqueda de cartas falla | Rate limit de magicthegathering.io (1000/h) o red; hay caché Redis — reintenta |
| Langfuse no arranca | Espera ~1 min (migraciones); luego `docker compose logs langfuse-web` |
| No veo trazas en Langfuse | Faltan claves o flag: `LANGFUSE_ENABLED=true` + claves en `.env`, y reinicia la app |
| promptfoo no llega a la app | La app debe estar levantada (`docker compose up -d`) y con el índice ingestado |
