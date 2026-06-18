"""Tests for the provider router (JSON helpers + ProviderError path)."""

from __future__ import annotations

import pytest

from app.providers.base import ProviderError, ProviderRouter, _safe_json, _coerce_messages


# --------------------------------------------------------------------------- #
# Module helpers
# --------------------------------------------------------------------------- #
class TestSafeJson:
    def test_valid_object(self):
        assert _safe_json('{"a": 1}') == {"a": 1}

    def test_empty_string(self):
        assert _safe_json("") == {}

    def test_none(self):
        assert _safe_json(None) == {}

    def test_malformed(self):
        assert _safe_json("{bad json}") == {}

    def test_array_returns_empty(self):
        # We only accept objects, not bare arrays.
        assert _safe_json("[1,2,3]") == {}


class TestCoerceMessages:
    def test_list_passthrough(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _coerce_messages(msgs) is msgs

    def test_string_wraps_as_user(self):
        result = _coerce_messages("hello")
        assert result == [{"role": "user", "content": "hello"}]


# --------------------------------------------------------------------------- #
# ProviderRouter — no providers configured
# --------------------------------------------------------------------------- #
class TestProviderRouterNoProviders:
    def _router_no_providers(self, settings) -> ProviderRouter:
        return ProviderRouter([], settings)

    @pytest.mark.asyncio
    async def test_complete_raises_provider_error(self, settings):
        router = self._router_no_providers(settings)
        with pytest.raises(ProviderError):
            await router.complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_embed_raises_provider_error(self, settings):
        router = self._router_no_providers(settings)
        with pytest.raises(ProviderError):
            await router.embed(["test"])

    @pytest.mark.asyncio
    async def test_stream_raises_provider_error(self, settings):
        router = self._router_no_providers(settings)
        with pytest.raises(ProviderError):
            chunks = []
            async for chunk in router.stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

    def test_has_providers_false(self, settings):
        router = self._router_no_providers(settings)
        assert not router.has_providers


# --------------------------------------------------------------------------- #
# ProviderRouter — embed with empty list
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_embed_empty_list_returns_empty(settings):
    router = ProviderRouter([], settings)
    # Edge case: empty list bypasses the provider entirely.
    result = await router.embed([])
    assert result == []
