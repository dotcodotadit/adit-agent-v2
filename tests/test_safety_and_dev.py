"""Tests for the safety confirmation manager and developer-mode store."""

from __future__ import annotations

import asyncio

import pytest

from app.bot.dev_mode import DevFlags, DevModeStore
from app.bot.safety import ConfirmationManager


# --------------------------------------------------------------------------- #
# DevModeStore
# --------------------------------------------------------------------------- #
class TestDevModeStore:
    def test_flags_default_all_off(self):
        store = DevModeStore()
        flags = store.flags(42)
        assert not flags.dev
        assert not flags.thoughts
        assert not flags.raw
        assert not flags.any_on()

    def test_toggle_dev_on_and_off(self):
        store = DevModeStore()
        new_val = store.toggle(1, "dev")
        assert new_val is True
        assert store.flags(1).dev
        store.toggle(1, "dev")
        assert not store.flags(1).dev

    def test_enabling_thoughts_enables_dev(self):
        store = DevModeStore()
        store.toggle(1, "thoughts")
        flags = store.flags(1)
        assert flags.thoughts
        assert flags.dev  # implicitly on

    def test_enabling_raw_enables_dev(self):
        store = DevModeStore()
        store.toggle(1, "raw")
        flags = store.flags(1)
        assert flags.raw
        assert flags.dev

    def test_disabling_dev_clears_dependents(self):
        store = DevModeStore()
        store.toggle(1, "thoughts")
        store.toggle(1, "raw")
        # Now turn dev off.
        store.toggle(1, "dev")  # was True, becomes False
        flags = store.flags(1)
        assert not flags.dev
        assert not flags.thoughts
        assert not flags.raw

    def test_flags_are_per_user(self):
        store = DevModeStore()
        store.toggle(1, "dev")
        assert not store.flags(2).dev

    def test_summary_string(self):
        flags = DevFlags(dev=True, thoughts=False, raw=True)
        s = flags.summary()
        assert "dev=on" in s
        assert "raw=on" in s
        assert "thoughts=off" in s


# --------------------------------------------------------------------------- #
# ConfirmationManager — timeout path (no real Telegram needed)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_confirmation_times_out_and_denies():
    manager = ConfirmationManager(timeout=0.05)

    class FakeBot:
        async def send_message(self, chat_id, text, reply_markup=None):
            return type("M", (), {"message_id": 1})()

    class FakeTool:
        name = "shell"
        description = "Run shell"
        dangerous = True

    confirmer = manager.make_confirmer(FakeBot(), chat_id=999)
    result = await confirmer(FakeTool(), {"command": "ls"})
    assert result is False  # denied on timeout


@pytest.mark.asyncio
async def test_confirmation_allow_via_callback():
    manager = ConfirmationManager(timeout=2.0)
    received_token = None

    class CapturingBot:
        async def send_message(self, chat_id, text, reply_markup=None):
            nonlocal received_token
            # Extract the token from the keyboard callback data.
            buttons = reply_markup.inline_keyboard[0]
            allow_data = buttons[0].callback_data  # "cf:<token>:allow"
            received_token = allow_data.split(":")[1]
            return type("M", (), {"message_id": 1})()

    class FakeTool:
        name = "write_file"
        description = "Write a file"
        dangerous = True

    class FakeUpdate:
        class FakeQuery:
            data = None
            async def answer(self):
                pass
            async def edit_message_text(self, text):
                pass
        callback_query = FakeQuery()

    confirmer = manager.make_confirmer(CapturingBot(), chat_id=1)
    future_result: list[bool] = []

    async def run():
        result = await confirmer(FakeTool(), {"path": "x.txt"})
        future_result.append(result)

    task = asyncio.create_task(run())
    # Give the coroutine a moment to send the message and park on the future.
    await asyncio.sleep(0.05)

    # Simulate the user tapping "Allow".
    assert received_token is not None
    update = FakeUpdate()
    update.callback_query.data = f"cf:{received_token}:allow"
    await manager.handle_callback(update, None)

    await asyncio.wait_for(task, timeout=1.0)
    assert future_result == [True]


@pytest.mark.asyncio
async def test_confirmation_deny_via_callback():
    manager = ConfirmationManager(timeout=2.0)
    received_token = None

    class CapturingBot2:
        async def send_message(self, chat_id, text, reply_markup=None):
            nonlocal received_token
            deny_data = reply_markup.inline_keyboard[0][1].callback_data
            received_token = deny_data.split(":")[1]
            return type("M", (), {"message_id": 1})()

    class FakeTool2:
        name = "browser"
        description = "Open browser"
        dangerous = True

    class FakeUpdate2:
        class FakeQuery2:
            data = None
            async def answer(self): pass
            async def edit_message_text(self, text): pass
        callback_query = FakeQuery2()

    confirmer = manager.make_confirmer(CapturingBot2(), chat_id=2)
    future_result: list[bool] = []

    async def run():
        result = await confirmer(FakeTool2(), {})
        future_result.append(result)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.05)

    update = FakeUpdate2()
    update.callback_query.data = f"cf:{received_token}:deny"
    await manager.handle_callback(update, None)

    await asyncio.wait_for(task, timeout=1.0)
    assert future_result == [False]
