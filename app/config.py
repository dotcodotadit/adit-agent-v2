"""Application configuration for Adit-Agent.

All runtime configuration is centralized here using ``pydantic-settings``.
Values are loaded, in order of precedence, from:

1. Explicit keyword arguments to :class:`Settings`.
2. Environment variables.
3. The ``.env`` file at the project root.
4. Field defaults declared below.

Access the singleton via :func:`get_settings`, which is cached so the ``.env``
file is parsed only once per process.

>>> from app.config import get_settings
>>> settings = get_settings()
>>> settings.telegram_bot_token.get_secret_value()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import (
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "ProviderConfig", "ConfigError", "get_settings"]

# Project root = two levels up from this file (app/config.py -> app -> root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised when application configuration cannot be loaded or is invalid.

    Wraps lower-level errors (pydantic validation, filesystem failures) in a
    single, operator-facing exception so callers can present a clean message
    instead of a stack trace.
    """


class ProviderConfig(BaseSettings):
    """Credentials and endpoint for a single LLM provider.

    Populated from ``{PREFIX}_API_KEY`` / ``{PREFIX}_BASE_URL`` env vars by
    :class:`Settings`. A provider is considered *enabled* only when it has a
    non-empty API key.
    """

    name: str
    api_key: SecretStr = SecretStr("")
    base_url: str = ""

    @property
    def enabled(self) -> bool:
        """True when the provider has a usable API key."""
        return bool(self.api_key.get_secret_value().strip())


def _split_csv(value: str | list[str] | None) -> list[str]:
    """Parse a comma-separated env string into a clean list of tokens."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Core ----------------------------------------------------------------
    app_name: str = "Adit-Agent"
    log_level: str = "info"
    environment: str = "development"

    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    vector_store_dir: Path = PROJECT_ROOT / "data" / "vector_store"

    # ---- Telegram ------------------------------------------------------------
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_allowed_user_ids: list[int] = Field(default_factory=list)
    telegram_admin_user_ids: list[int] = Field(default_factory=list)

    # ---- LLM core ------------------------------------------------------------
    llm_provider_priority: list[str] = Field(
        default_factory=lambda: ["nara", "freemodel", "aerolink", "zerog", "zyloo"]
    )
    llm_default_model: str = "gpt-4o-mini"
    llm_request_timeout: int = 60
    llm_max_retries: int = 3
    llm_temperature: float = 0.7

    # ---- Per-provider credentials (flat env vars) ----------------------------
    nara_api_key: SecretStr = SecretStr("")
    nara_base_url: str = ""
    freemodel_api_key: SecretStr = SecretStr("")
    freemodel_base_url: str = ""
    aerolink_api_key: SecretStr = SecretStr("")
    aerolink_base_url: str = ""
    zerog_api_key: SecretStr = SecretStr("")
    zerog_base_url: str = ""
    zyloo_api_key: SecretStr = SecretStr("")
    zyloo_base_url: str = ""

    # ---- Embeddings / vector store -------------------------------------------
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chroma_collection: str = "adit_long_term"

    # ---- Database ------------------------------------------------------------
    database_url: str = "sqlite+aiosqlite:///data/db.sqlite3"
    database_echo: bool = False

    # ---- Agent behavior ------------------------------------------------------
    agent_max_steps: int = 12
    agent_max_tokens: int = 4096
    short_term_window: int = 20
    enable_planner: bool = True

    # ---- Safety --------------------------------------------------------------
    require_tool_confirmation: bool = True
    dangerous_tools: list[str] = Field(
        default_factory=lambda: ["shell", "write_file", "browser", "process"]
    )
    sandbox_root: Path = PROJECT_ROOT / "data" / "sandbox"

    # ---- Browser tool (Playwright) -------------------------------------------
    # Which Playwright browser engine to use: chromium | firefox | webkit
    playwright_browser: str = "chromium"
    # Maximum seconds to wait for a page to load before timing out.
    browser_timeout: int = 30
    # Where to store browser screenshots (auto-created if absent).
    browser_screenshot_dir: Path = PROJECT_ROOT / "data" / "cache" / "screenshots"

    # ---- Inbound rate limiting (per user) ------------------------------------
    # Throttles how many messages a single user may send within a rolling
    # window, protecting the bot (and your LLM credits) from floods/abuse.
    # Admins always bypass the limit. This is distinct from PTB's
    # AIORateLimiter, which throttles *outgoing* calls to Telegram.
    rate_limit_enabled: bool = True
    # Max messages allowed per user within the window below.
    rate_limit_max_messages: int = 5
    # Length of the rolling window, in seconds.
    rate_limit_window_seconds: float = 10.0

    # ------------------------------------------------------------------ #
    # Validators / coercion
    # ------------------------------------------------------------------ #
    @field_validator(
        "telegram_allowed_user_ids",
        "telegram_admin_user_ids",
        mode="before",
    )
    @classmethod
    def _parse_int_csv(cls, value: object) -> list[int]:
        tokens = _split_csv(value if isinstance(value, (str, list)) else None)
        result: list[int] = []
        for token in tokens:
            try:
                result.append(int(token))
            except ValueError as exc:
                raise ValueError(
                    "expected a comma-separated list of integer user IDs, "
                    f"but got non-integer value {token!r}"
                ) from exc
        return result

    @field_validator(
        "llm_provider_priority",
        "dangerous_tools",
        mode="before",
    )
    @classmethod
    def _parse_str_csv(cls, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return _split_csv(value if isinstance(value, str) else None)

    @field_validator("log_level")
    @classmethod
    def _normalize_level(cls, value: str) -> str:
        allowed = {"debug", "info", "warning", "error", "critical"}
        normalized = value.lower().strip()
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized

    @model_validator(mode="after")
    def _ensure_directories(self) -> "Settings":
        """Create required data directories so downstream code can assume them."""
        for directory in (
            self.data_dir,
            self.upload_dir,
            self.cache_dir,
            self.vector_store_dir,
            self.sandbox_root,
            self.browser_screenshot_dir,
        ):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigError(
                    f"Could not create required directory {directory}: {exc}"
                ) from exc
        return self

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    def provider_config(self, name: str) -> ProviderConfig:
        """Build a :class:`ProviderConfig` for the named provider.

        Raises
        ------
        ValueError
            If ``name`` is not a recognized provider.
        """
        key = name.lower()
        mapping = {
            "nara": (self.nara_api_key, self.nara_base_url),
            "freemodel": (self.freemodel_api_key, self.freemodel_base_url),
            "aerolink": (self.aerolink_api_key, self.aerolink_base_url),
            "zerog": (self.zerog_api_key, self.zerog_base_url),
            "zyloo": (self.zyloo_api_key, self.zyloo_base_url),
        }
        if key not in mapping:
            raise ValueError(f"Unknown provider: {name!r}")
        api_key, base_url = mapping[key]
        return ProviderConfig(name=key, api_key=api_key, base_url=base_url)

    def enabled_providers(self) -> list[ProviderConfig]:
        """Return enabled providers in priority order.

        TODO: emit a warning (not just skip) when a prioritized provider is
        disabled due to a missing key, to aid first-run debugging.
        """
        configs = [self.provider_config(name) for name in self.llm_provider_priority]
        return [cfg for cfg in configs if cfg.enabled]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, process-wide :class:`Settings` singleton.

    Raises
    ------
    ConfigError
        If settings cannot be loaded — e.g. an environment variable or ``.env``
        entry fails validation. The underlying error is chained for debugging.
    """
    try:
        return Settings()
    except ConfigError:
        # Already an operator-facing error (e.g. from directory creation).
        raise
    except ValidationError as exc:
        raise ConfigError(
            "Invalid configuration — check your .env / environment variables:\n"
            f"{exc}"
        ) from exc
