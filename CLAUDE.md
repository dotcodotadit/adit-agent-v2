# Adit-Agent — Codebase Guide

## Project Overview

Adit-Agent is a Telegram bot backed by an agentic LLM pipeline. Users send messages (text, images, audio, video, documents) and receive streamed answers powered by a ReAct tool-using loop. The bot supports chat, agent, and deep-reasoning modes.

## Running

```bash
# 1. Copy and fill in credentials
cp .env.example .env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run database migrations (SQLite by default)
python -m app.database.migrations create

# 4. Start the bot
python -m app.main
```

## Architecture

```
app/
├── main.py                  # Entry point: startup, polling, graceful shutdown
├── config.py                # All settings via pydantic-settings (from .env)
├── dependencies.py          # AppContainer: wires DB → vector store → providers → memory → orchestrator
│
├── agent/                   # Intelligent core (no Telegram dependency)
│   ├── context_builder.py   # LLMRouter protocol, persona/system prompts, token-budgeted prompt assembly
│   ├── planner.py           # Heuristic gating + LLM-based task decomposition (Plan, PlanStep)
│   ├── executor.py          # Streaming ReAct loop (AgentEvent stream, tool calling, safety gating)
│   ├── orchestrator.py      # Top-level coordinator: memory → plan → execute → reflect → persist
│   └── memory_manager.py    # Short-term history (SQL) + long-term recall (relational + vector)
│
├── providers/
│   └── base.py              # ProviderRouter: OpenAI-compatible failover, streaming, embeddings, vision, STT
│
├── memory/
│   ├── vector_store.py      # Async ChromaDB wrapper (VectorStore)
│   └── embeddings.py        # EmbeddingService (thin wrapper over provider.embed)
│
├── multimodal/
│   ├── base.py              # ProcessedMedia result type, detect_kind()
│   ├── documents.py         # PDF/DOCX/text extraction (synchronous, run in thread)
│   ├── images.py            # Pillow metadata + vision provider description
│   ├── audio.py             # STT transcription via provider.transcribe()
│   ├── video.py             # ffmpeg keyframe sampling + vision descriptions + summary
│   └── pipeline.py          # MediaPipeline dispatcher (single entry point for bot layer)
│
├── bot/
│   ├── telegram_bot.py      # build_application(): wires Application, handlers, services
│   ├── services.py          # BotServices bundle (orchestrator, memory, media, dev, confirm)
│   ├── handlers.py          # Message handler + StreamRenderer (live-editing Telegram message)
│   ├── commands.py          # Slash commands: /start /help /new /mode /dev /thoughts /raw /whoami /forget /stats
│   ├── middlewares.py       # Access gate (group -1) + global error handler
│   ├── safety.py            # ConfirmationManager: inline buttons for dangerous tools
│   └── dev_mode.py          # DevModeStore: per-user dev/thoughts/raw flags
│
├── tools/                   # Auto-discovered capabilities advertised to the LLM
│   ├── base.py              # Tool, ToolResult, ToolContext, resolve_in_sandbox
│   ├── registry.py          # ToolRegistry singleton + @tool decorator
│   ├── filesystem/          # read_file, search_files, write_file
│   ├── web/                 # web_search (DuckDuckGo), web_scraper
│   ├── media/               # image_reader, audio_reader, video_reader
│   └── system/              # shell (dangerous), browser (dangerous), process
│
└── database/
    ├── models.py            # SQLAlchemy 2.0 ORM: User, Conversation, Message, ToolCall, Memory, Attachment
    ├── migrations.py        # Alembic (if configured) or create_all fallback
    └── session.py           # DatabaseSessionManager, get_session()
```

## Key Design Decisions

### Provider Contract
`app/agent/context_builder.LLMRouter` is a `Protocol` defined at the bottom of the dependency graph. `ProviderRouter` in `app/providers/base.py` implements it. Everything in `app/agent/` depends only on the protocol, never the concrete router, making the agent testable with fakes.

### Tool Registration
Tools self-register at import time via `@tool(...)`. `ToolRegistry.discover()` (called at startup in `AppContainer._init_orchestrator`) walks `app/tools/` and imports every module. **Footgun**: `ToolRegistry.__len__` makes an empty registry falsy, so `@tool(registry=<empty_reg>)` silently falls back to the global registry. Tests that need isolated registries must manually copy tool objects in after registration.

### Streaming
`Executor.run()` and `Orchestrator.stream()` are async generators of `AgentEvent`. The bot's `StreamRenderer` edits a single Telegram message in place as events arrive, throttled to ~1 edit/second. The final answer replaces transient "Thinking…" state.

### Safety / Dangerous Tools
Tools marked `dangerous=True` are gated by the `Confirmer` callback. The bot wires this to `ConfirmationManager`, which sends an inline keyboard and parks the agent loop on an `asyncio.Future` until the user approves or a 120-second timeout denies.

### Memory
Two layers:
- **Short-term**: last N messages from `messages` table, loaded as OpenAI chat messages per turn.
- **Long-term**: `Memory` rows with optional ChromaDB vectors. Semantic recall uses embeddings; falls back to importance-ordered relational recall when vectors are unavailable.

### Multimodal
The bot downloads attachments to `settings.upload_dir`, then calls `MediaPipeline.process()`. The result's `.as_context()` is appended to the user's message text so the agent sees it as inline content.

## Testing

```bash
pytest                   # run all 89 tests
pytest -v tests/test_executor.py   # one module
```

Tests use in-memory SQLite (`session_factory` fixture) and `FakeRouter` (scriptable `LLMResponse` list). No real LLM or Telegram connection required.

## Environment Variables (key ones)

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Required to run the bot |
| `FREEMODEL_API_KEY` / `FREEMODEL_BASE_URL` | First-priority LLM provider |
| `LLM_DEFAULT_MODEL` | Model name sent to providers (default: `gpt-4o-mini`) |
| `AGENT_MAX_STEPS` | ReAct iteration cap (default: 12) |
| `REQUIRE_TOOL_CONFIRMATION` | Gate dangerous tools behind inline buttons |
| `DATABASE_URL` | SQLAlchemy async URL (default: SQLite) |

Full list in `.env.example`.
