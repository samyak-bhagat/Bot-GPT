# BOT GPT Backend

FastAPI + LangGraph conversational backend with a LangChain model factory, embedding factory, PostgreSQL/pgvector-ready persistence, and Streamlit demo UI.

Detailed architecture and API/data design documentation is available in `SYSTEM_DESIGN.md`.

## Quick start

Works on Windows, macOS, and Linux with Docker Desktop (or Docker Engine + Compose plugin).

1. Copy env file:
   - `cp .env.example .env`
2. Start services:
   - `docker compose up -d --build`
3. API docs:
   - `http://localhost:8000/docs`
4. Streamlit UI:
   - `http://localhost:8501`

## Runtime modes

### A) Fully local (default)

- LLM provider: Ollama (Docker service)
- Embeddings: HuggingFace sentence-transformers
- Default model pull is automated by compose via `ollama-pull` service.

Recommended `.env` values:

```env
DEFAULT_PROVIDER=ollama
DEFAULT_MODEL=qwen3:1.7b
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_PULL_MODEL=qwen3:1.7b
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### B) OpenAI mode

Recommended `.env` values:

```env
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

You can also keep `EMBEDDING_PROVIDER=huggingface` in OpenAI mode if you want local embeddings.

## Ollama auto-pull in Docker Compose

`docker-compose.yml` includes:

- `ollama` service (model runtime)
- `ollama-pull` one-shot service (pulls `OLLAMA_PULL_MODEL`)
- `api` waits for `ollama-pull` completion before startup

So this manual command is usually not needed anymore:

```bash
docker exec -it botgpt-ollama ollama pull <model>
```

To change the model, update both:

- `OLLAMA_PULL_MODEL`
- `DEFAULT_MODEL`

Then rerun:

```bash
docker compose up -d --build
```

## Demo commands

Use these during your interview demo (Swagger or terminal).

1. Health check
   - `GET /health`
2. Ingest a document
   - `POST /api/v1/documents` (multipart file)
3. Create conversation
   - `POST /api/v1/conversations`
4. Send message
   - `POST /api/v1/conversations/{id}/messages`
5. Fetch full conversation
   - `GET /api/v1/conversations/{id}`
6. Fetch cost breakdown
   - `GET /api/v1/conversations/{id}/costs`
7. List conversations
   - `GET /api/v1/conversations`
8. Delete conversation
   - `DELETE /api/v1/conversations/{id}`

### Curl quick run

```bash
# 0) Set base URL
BASE="http://localhost:8000"
USER_ID="demo@example.com"

# 1) Create conversation
curl -s -X POST "$BASE/api/v1/conversations" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: $USER_ID" \
  -d '{"provider":"groq","model":"llama-3.1-70b-versatile","document_ids":[]}'

# 2) Send message (replace CONVERSATION_ID)
curl -s -X POST "$BASE/api/v1/conversations/CONVERSATION_ID/messages" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: $USER_ID" \
  -d '{"content":"Give me a short summary","provider":"groq","model":"llama-3.1-70b-versatile"}'

# 3) Cost breakdown
curl -s -X GET "$BASE/api/v1/conversations/CONVERSATION_ID/costs" \
  -H "X-User-Id: $USER_ID"
```

## Implemented

- FastAPI app entrypoint and health endpoint
- Versioned API router layout (`/api/v1`)
- Chat model factory (`ChatGroq`, `ChatOpenAI`, `ChatOllama`)
- Embedding model factory (default HuggingFace sentence-transformers, optional OpenAI small)
- Conversation CRUD + message persistence
- RAG routing path (`open` vs `rag`) in LangGraph
- Retrieval adapter with pgvector SQL path (Postgres) + fallback semantic ranker (sqlite/local)
- Document ingestion with chunking + embedding generation
- Cost tracking (LLM + embeddings) and conversation-level breakdown endpoint
- Container setup for API, Postgres+pgvector, Ollama, Streamlit
- Streamlit UI with chat, history, delete flow, and cost analytics panel
- Pytest + CI workflow

## Basic smoke validation

- Open `http://localhost:8501`
- Ingest one document from the **Knowledge base** tab
- Start a new chat and send one message
- Confirm:
  - assistant response is generated
  - conversation cost breakdown is visible
  - document-linked conversation shows embedding cost rows

## Provider switching and failures

- You can switch provider/model directly in Streamlit sidebar; each new message uses the selected values.
- If a provider is selected but not configured (for example missing `OPENAI_API_KEY`), the API now returns a clear 400 error instead of mock output.
- If upstream provider calls fail (network/model/runtime issues), API returns a 502 with provider/model context.

## Dependency notes

- Redis has been removed from active runtime path and compose stack.
- Current Python dependencies match active imports/runtime usage for API + Streamlit + ingestion.

## Interview flow (10 minutes)

1. Show architecture (`DESIGN.md`) and explain factory pattern choices.
2. Show `/docs` and execute: create conversation -> send message -> fetch costs.
3. Open Streamlit and demonstrate:
   - conversation history
   - switching conversations
   - delete flow with confirmation
   - cost dashboard (LLM + embedding + per-document table/chart)
4. Close with trade-offs and roadmap (auth, queue workers, tracing, caching, migrations).
