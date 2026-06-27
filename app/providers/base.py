"""LLM provider router for Adit-Agent.

:class:`ProviderRouter` is the concrete implementation of the
:class:`~app.agent.context_builder.LLMRouter` contract the agent core depends on.
It speaks the OpenAI-compatible Chat Completions API (via the ``openai`` async
SDK) and fans requests out across the configured providers in priority order,
failing over to the next provider when one errors. This gives the bot resilience
when running on a chain of free / low-tier OpenAI-compatible endpoints.

Beyond the core ``complete`` / ``stream`` / ``embed`` methods the agent uses, the
router also implements the duck-typed contracts the media tools expect —
``vision(prompt, images)`` and ``transcribe(audio_path, language)`` — so the same
object satisfies every consumer in the system.

Responses are normalized into the agent's transport types
(:class:`~app.agent.context_builder.LLMResponse`,
:class:`~app.agent.context_builder.StreamChunk`,
:class:`~app.agent.context_builder.ToolCallRequest`) so the rest of the codebase
never touches a raw SDK object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator

from app.agent.context_builder import (
    ChatMessage,
    LLMResponse,
    StreamChunk,
    ToolCallRequest,
)
from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.config import ProviderConfig, Settings

log = get_logger(__name__)

__all__ = ["ProviderError", "ProviderRouter"]

# Default model used for audio transcription when no override is supplied.
_DEFAULT_TRANSCRIBE_MODEL = "whisper-1"


class ProviderError(RuntimeError):
    """Raised when every configured provider fails (or none are configured).

    The message names the operation and, where useful, chains the last
    underlying provider error so the root cause is preserved.
    """


@dataclass(slots=True)
class _Provider:
    """A named provider together with its instantiated async SDK client."""

    name: str
    client: Any  # openai.AsyncOpenAI


class ProviderRouter:
    """Routes LLM calls across configured providers with priority failover.

    Construct via :meth:`from_settings` (the common path) or pass explicit
    :class:`~app.config.ProviderConfig` objects. Each call tries providers in the
    order given; the first to succeed wins, and the rest are skipped.
    """

    def __init__(
        self,
        providers: list["ProviderConfig"],
        settings: "Settings",
    ) -> None:
        self._settings = settings
        self._default_model = settings.llm_default_model
        self._providers: list[_Provider] = []
        self._build_clients(providers)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_settings(cls, settings: "Settings") -> "ProviderRouter":
        """Build a router from the enabled providers in ``settings``."""
        return cls(settings.enabled_providers(), settings)

    def _build_clients(self, providers: list["ProviderConfig"]) -> None:
        """Instantiate an ``AsyncOpenAI`` client per enabled provider."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError(
                "The 'openai' package is required for the provider router."
            ) from exc

        for cfg in providers:
            if not cfg.enabled:
                continue
            client = AsyncOpenAI(
                api_key=cfg.api_key.get_secret_value(),
                base_url=cfg.base_url or None,
                timeout=float(self._settings.llm_request_timeout),
                max_retries=self._settings.llm_max_retries,
            )
            self._providers.append(_Provider(name=cfg.name, client=client))

        if self._providers:
            log.info(
                "Provider router ready with {} provider(s): {}.",
                len(self._providers),
                ", ".join(p.name for p in self._providers),
            )
        else:
            log.warning("Provider router has no enabled providers; calls will fail.")

    @property
    def has_providers(self) -> bool:
        """True when at least one provider is configured."""
        return bool(self._providers)

    # ------------------------------------------------------------------ #
    # Chat completion (non-streaming)
    # ------------------------------------------------------------------ #
    async def complete(
        self,
        messages: list[ChatMessage] | str,
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a single completion, failing over across providers."""
        kwargs = self._chat_kwargs(
            messages, model, tools, tool_choice, temperature, max_tokens
        )
        last_err: Exception | None = None
        for provider in self._providers:
            try:
                resp = await provider.client.chat.completions.create(**kwargs)
                return self._parse_response(resp)
            except Exception as exc:  # noqa: BLE001 - try the next provider
                last_err = exc
                log.warning("Provider {!r} failed on complete(): {}", provider.name, exc)
        raise ProviderError(f"All providers failed on complete(): {last_err}") from last_err

    # ------------------------------------------------------------------ #
    # Chat completion (streaming)
    # ------------------------------------------------------------------ #
    async def stream(
        self,
        messages: list[ChatMessage] | str,
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion, yielding text deltas then a terminal ``done``.

        Failover only applies to *opening* the stream: if a provider errors
        before the first chunk, the next is tried. Once tokens have started
        flowing, an error propagates (the executor handles it) since a half-
        emitted answer cannot be transparently retried elsewhere.
        """
        kwargs = self._chat_kwargs(
            messages, model, tools, tool_choice, temperature, max_tokens
        )
        kwargs["stream"] = True

        last_err: Exception | None = None
        for provider in self._providers:
            try:
                sdk_stream = await provider.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - opening failed; try next
                last_err = exc
                log.warning("Provider {!r} failed to open stream: {}", provider.name, exc)
                continue

            async for chunk in self._consume_stream(sdk_stream, model=kwargs["model"]):
                yield chunk
            return

        raise ProviderError(f"All providers failed on stream(): {last_err}") from last_err

    async def _consume_stream(
        self, sdk_stream: Any, *, model: str
    ) -> AsyncIterator[StreamChunk]:
        """Translate an SDK chat stream into :class:`StreamChunk` events."""
        content_parts: list[str] = []
        # index -> accumulated tool call fields
        tool_acc: dict[int, dict[str, str]] = {}
        finish_reason = "stop"

        async for event in sdk_stream:
            if not event.choices:
                continue
            choice = event.choices[0]
            delta = choice.delta

            text = getattr(delta, "content", None)
            if text:
                content_parts.append(text)
                yield StreamChunk(type="text", text=text)

            for tcd in getattr(delta, "tool_calls", None) or []:
                slot = tool_acc.setdefault(tcd.index, {"id": "", "name": "", "args": ""})
                if tcd.id:
                    slot["id"] = tcd.id
                fn = getattr(tcd, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        tool_calls = [
            ToolCallRequest(
                id=slot["id"] or f"call_{idx}",
                name=slot["name"],
                arguments=_safe_json(slot["args"]),
            )
            for idx, slot in sorted(tool_acc.items())
            if slot["name"]
        ]
        yield StreamChunk(
            type="done",
            response=LLMResponse(
                content="".join(content_parts),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                model=model,
            ),
        )

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    async def embed(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]]:
        """Return one embedding vector per input string."""
        if not texts:
            return []
        model = model or self._settings.embedding_model
        last_err: Exception | None = None
        for provider in self._providers:
            try:
                resp = await provider.client.embeddings.create(model=model, input=texts)
                return [item.embedding for item in resp.data]
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.debug("Provider {!r} does not support embed(): {}", provider.name, exc)
        raise ProviderError(f"All providers failed on embed(): {last_err}") from last_err

    # ------------------------------------------------------------------ #
    # Vision (used by image_reader / video_reader tools)
    # ------------------------------------------------------------------ #
    async def vision(
        self,
        *,
        prompt: str,
        images: list[str],
        model: str | None = None,
    ) -> str:
        """Describe/answer about ``images`` (data URLs) given ``prompt``."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}} for url in images
        )
        response = await self.complete(
            [{"role": "user", "content": content}],
            model=model or self._default_model,
        )
        return response.content

    # ------------------------------------------------------------------ #
    # Transcription (used by audio_reader tool)
    # ------------------------------------------------------------------ #
    async def transcribe(
        self,
        *,
        audio_path: str,
        language: str | None = None,
        model: str | None = None,
    ) -> str:
        """Transcribe a local audio file to text."""
        model = model or _DEFAULT_TRANSCRIBE_MODEL
        last_err: Exception | None = None
        for provider in self._providers:
            try:
                with open(audio_path, "rb") as handle:
                    resp = await provider.client.audio.transcriptions.create(
                        model=model,
                        file=handle,
                        language=language or None,
                    )
                return getattr(resp, "text", "") or ""
            except FileNotFoundError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("Provider {!r} failed on transcribe(): {}", provider.name, exc)
        raise ProviderError(
            f"All providers failed on transcribe(): {last_err}"
        ) from last_err

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def aclose(self) -> None:
        """Close every underlying SDK client (best-effort)."""
        for provider in self._providers:
            try:
                await provider.client.close()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                log.warning("Error closing provider {!r}: {}", provider.name, exc)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _chat_kwargs(
        self,
        messages: list[ChatMessage] | str,
        model: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Assemble the keyword arguments for a chat completion call."""
        kwargs: dict[str, Any] = {
            "model": model or self._default_model,
            "messages": _coerce_messages(messages),
            "temperature": (
                temperature if temperature is not None else self._settings.llm_temperature
            ),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        return kwargs

    @staticmethod
    def _parse_response(resp: Any) -> LLMResponse:
        """Normalize an SDK chat completion into an :class:`LLMResponse`."""
        if not resp.choices:
            return LLMResponse(content="", finish_reason="stop", model=getattr(resp, "model", ""))
        choice = resp.choices[0]
        message = choice.message

        tool_calls: list[ToolCallRequest] = []
        for tc in getattr(message, "tool_calls", None) or []:
            tool_calls.append(
                ToolCallRequest(
                    id=tc.id or f"call_{len(tool_calls)}",
                    name=tc.function.name,
                    arguments=_safe_json(tc.function.arguments),
                )
            )

        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
            }

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            model=getattr(resp, "model", ""),
            usage=usage,
            raw=resp,
        )


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #
def _coerce_messages(messages: list[ChatMessage] | str) -> list[ChatMessage]:
    """Accept either a ready message list or a bare user string."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return messages


def _safe_json(raw: str | None) -> dict[str, Any]:
    """Parse a tool-call arguments JSON string, tolerating malformed input."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
