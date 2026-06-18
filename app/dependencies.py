"""Application-wide dependency container and lifecycle for Adit-Agent.

This module implements a small, explicit dependency-injection container. Rather
than scattering global singletons across modules, all shared, long-lived
resources (database engine, vector store, provider clients, agent orchestrator)
are constructed once in :meth:`AppContainer.startup` and torn down in
:meth:`AppContainer.shutdown`.

The container favors *lazy, ordered* initialization so failures surface with a
clear message about which subsystem could not start, and *reverse-order*
teardown so resources are released safely.

>>> container = AppContainer()
>>> await container.startup()
>>> ...                       # use container.orchestrator, container.db, ...
>>> await container.shutdown()

Design notes
------------
* The container holds *interfaces by attribute*, typed as ``Any`` where the
  concrete classes live in sibling modules that may not yet exist. This keeps
  ``dependencies.py`` importable during incremental development.
* Heavy imports are performed lazily inside ``startup`` to keep process start
  fast and to avoid import cycles with the agent/provider packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.config import Settings, get_settings
from app.utils.logger import get_logger

if TYPE_CHECKING:
    # Imported only for type-checkers; avoids runtime import cycles.
    from sqlalchemy.ext.asyncio import AsyncEngine

log = get_logger(__name__)

__all__ = ["AppContainer", "StartupError", "get_container"]


class StartupError(RuntimeError):
    """Raised when a container subsystem fails to initialize during startup.

    The message names the subsystem that failed, and the originating error is
    chained so the root cause is preserved.
    """


@dataclass
class AppContainer:
    """Owns and wires the application's shared resources.

    Attributes are populated during :meth:`startup`. Accessing a resource
    before startup raises :class:`RuntimeError` via :meth:`_require`.
    """

    settings: Settings = field(default_factory=get_settings)

    # Populated during startup (typed loosely to avoid hard import coupling).
    db_engine: "AsyncEngine | None" = None
    session_factory: Any = None
    vector_store: Any = None
    provider_router: Any = None
    memory_manager: Any = None
    orchestrator: Any = None

    _started: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def startup(self) -> "AppContainer":
        """Initialize all subsystems in dependency order.

        Order matters: database -> vector store -> providers -> memory ->
        orchestrator. Each step logs success so partial-startup failures are
        easy to locate.
        """
        if self._started:
            log.debug("Container already started; skipping re-initialization.")
            return self

        log.info("Starting Adit-Agent container (env={})", self.settings.environment)

        # Ordered so dependents start after their dependencies. Each step is
        # wrapped to add context and to roll back partial startup on failure.
        steps: tuple[tuple[str, Any], ...] = (
            ("database", self._init_database),
            ("vector store", self._init_vector_store),
            ("providers", self._init_providers),
            ("memory", self._init_memory),
            ("orchestrator", self._init_orchestrator),
        )
        for label, init in steps:
            try:
                await init()
            except Exception as exc:  # noqa: BLE001 - re-raised with context below
                log.error("Failed to initialize {}: {}", label, exc)
                # Tear down anything that did come up so a partial startup does
                # not leak resources (e.g. an open database engine).
                await self.shutdown()
                raise StartupError(
                    f"Container startup failed while initializing {label}: {exc}"
                ) from exc

        self._started = True
        log.success("Container startup complete.")
        return self

    async def shutdown(self) -> None:
        """Release resources in reverse initialization order.

        Each teardown is guarded so one failure does not prevent the rest from
        running — best-effort cleanup is the goal during shutdown.
        """
        log.info("Shutting down Adit-Agent container.")

        async def _safe(label: str, coro: Any) -> None:
            if coro is None:
                return
            try:
                await coro
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                log.warning("Error during {} shutdown: {}", label, exc)

        # Orchestrator / memory may hold provider clients; close them first.
        if self.orchestrator is not None and hasattr(self.orchestrator, "aclose"):
            await _safe("orchestrator", self.orchestrator.aclose())
        if self.provider_router is not None and hasattr(self.provider_router, "aclose"):
            await _safe("provider_router", self.provider_router.aclose())
        if self.db_engine is not None:
            await _safe("database", self.db_engine.dispose())

        self._started = False
        log.success("Container shutdown complete.")

    # ------------------------------------------------------------------ #
    # Subsystem initializers (lazy imports inside each)
    # ------------------------------------------------------------------ #
    async def _init_database(self) -> None:
        """Create the async SQLAlchemy engine and session factory."""
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        self.db_engine = create_async_engine(
            self.settings.database_url,
            echo=self.settings.database_echo,
            pool_pre_ping=True,
        )
        self.session_factory = async_sessionmaker(
            self.db_engine, expire_on_commit=False
        )

        # Bring the schema up (Alembic if configured, else create_all).
        from app.database.migrations import run_migrations

        await run_migrations(self.db_engine)
        log.info("Database engine ready ({}).", self.settings.database_url)

    async def _init_vector_store(self) -> None:
        """Initialize the persistent ChromaDB vector store.

        Kept defensive: a missing/optional vector store should degrade the
        agent (no long-term recall) rather than crash startup.
        """
        try:
            import chromadb

            self.vector_store = chromadb.PersistentClient(
                path=str(self.settings.vector_store_dir)
            )
            log.info(
                "Vector store ready ({}).", self.settings.vector_store_dir
            )
        except Exception as exc:  # noqa: BLE001
            self.vector_store = None
            log.warning("Vector store unavailable, long-term memory disabled: {}", exc)

    async def _init_providers(self) -> None:
        """Build the provider router from enabled providers in priority order."""
        from app.providers import ProviderRouter

        enabled = self.settings.enabled_providers()
        if not enabled:
            log.warning(
                "No LLM providers enabled — set at least one *_API_KEY in .env. "
                "The agent will be unavailable until one is configured."
            )
            self.provider_router = None
            return

        log.info(
            "Enabled providers (priority order): {}",
            ", ".join(p.name for p in enabled),
        )
        self.provider_router = ProviderRouter(enabled, self.settings)

    async def _init_memory(self) -> None:
        """Construct the memory manager (short-term + long-term stores)."""
        from app.agent.memory_manager import MemoryManager

        self.memory_manager = MemoryManager(
            session_factory=self.session_factory,
            settings=self.settings,
            vector_store=self.vector_store,
            provider_router=self.provider_router,
        )
        log.info("Memory manager initialized.")

    async def _init_orchestrator(self) -> None:
        """Construct the top-level agent orchestrator (planner + executor)."""
        if self.provider_router is None:
            log.warning(
                "Skipping orchestrator init — no provider router available."
            )
            self.orchestrator = None
            return

        from app.agent.orchestrator import Orchestrator
        from app.tools.registry import get_registry

        registry = get_registry()
        registry.discover()  # auto-import and register all tool modules

        self.orchestrator = Orchestrator(
            provider_router=self.provider_router,
            registry=registry,
            memory_manager=self.memory_manager,
            settings=self.settings,
        )
        log.info("Orchestrator initialized with {} tool(s).", len(registry))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _require(self, attr: str) -> Any:
        """Return a started resource or raise if startup has not run."""
        value = getattr(self, attr)
        if not self._started or value is None:
            raise RuntimeError(
                f"Resource {attr!r} requested before startup or unavailable. "
                "Call `await container.startup()` first."
            )
        return value


# Process-wide container singleton. Constructed on first access so importing
# this module has no side effects (important for tests).
_container: AppContainer | None = None


def get_container() -> AppContainer:
    """Return the lazily-created, process-wide :class:`AppContainer`."""
    global _container
    if _container is None:
        _container = AppContainer()
    return _container
