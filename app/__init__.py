"""Adit-Agent — a streaming ReAct Telegram bot backed by a tool-using LLM pipeline.

Package layout::

    app/
    ├── agent/       – orchestrator, executor (ReAct), planner, context builder, memory manager
    ├── bot/         – Telegram handlers, commands, safety gates, developer mode
    ├── database/    – SQLAlchemy 2.0 models, migrations, session management
    ├── memory/      – vector store (ChromaDB) and embedding service
    ├── multimodal/  – document / image / audio / video pipelines
    ├── providers/   – LLM provider router (OpenAI-compatible, priority failover)
    ├── tools/       – tool registry + concrete tools (filesystem/web/system/media)
    ├── utils/       – logging (loguru), shared helpers
    ├── config.py    – pydantic-settings singleton
    ├── dependencies.py – application container (startup/shutdown lifecycle)
    └── main.py      – entry point

Start here: :mod:`app.main` to run the bot,
:mod:`app.agent.orchestrator` to understand the agent pipeline,
:mod:`app.dependencies` to see how everything is wired.
"""
