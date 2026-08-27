from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "hermes_discord_realtime",
    ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("hermes_discord_realtime", PACKAGE)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(PACKAGE)

from hermes_discord_realtime.pcm_stream import FRAME_SIZE, RealtimePCMSource
from hermes_discord_realtime.adapter import DiscordRealtimeAdapter, register
from hermes_discord_realtime.realtime_voice import (
    RealtimeVoiceSession,
    discord_pcm_to_realtime,
    realtime_pcm_to_discord,
)
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource


class FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.events = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        event = await self.events.get()
        if event is None:
            raise StopAsyncIteration
        return json.dumps(event)

    async def send(self, payload):
        event = json.loads(payload)
        self.sent.append(event)
        if event["type"] == "session.update":
            await self.events.put({"type": "session.updated", "session": {}})

    async def close(self):
        await self.events.put(None)


class FakePluginContext:
    def __init__(self):
        self.platform = None

    def register_platform(self, **kwargs):
        self.platform = kwargs


class FakeChannel:
    def __init__(self):
        self.messages = []

    async def send(self, message):
        self.messages.append(message)


class FakeDiscordClient:
    def __init__(self, channel):
        self.channel = channel

    def get_channel(self, _channel_id):
        return self.channel


def test_plugin_replaces_discord_factory():
    ctx = FakePluginContext()
    register(ctx)
    assert ctx.platform["name"] == "discord"
    adapter = ctx.platform["adapter_factory"](PlatformConfig(enabled=True))
    assert isinstance(adapter, DiscordRealtimeAdapter)


@pytest.mark.asyncio
async def test_spoken_turn_uses_existing_hermes_message_handler():
    adapter = DiscordRealtimeAdapter(PlatformConfig(enabled=True))
    channel = FakeChannel()
    adapter._client = FakeDiscordClient(channel)
    adapter._voice_text_channels[123] = 999
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="999",
        user_id="456",
        user_name="Jamie",
        chat_type="channel",
    )
    adapter._voice_sources[123] = source.to_dict()
    seen = []

    async def handler(event):
        seen.append(event)
        return "The Hermes answer"

    adapter.set_message_handler(handler)
    result = await adapter._consult_hermes(123, 456, "check my calendar")

    assert result == "The Hermes answer"
    assert seen[0].text == "check my calendar"
    assert seen[0].source.chat_id == "999"
    assert seen[0].source.user_id == "456"
    assert seen[0].message_type.value == "text"
    assert "Voice · Realtime" in channel.messages[0]


def test_pcm_conversions():
    discord_pcm = np.array(
        [[1000, 3000], [3000, 5000], [-1000, 1000], [1000, 3000]],
        dtype=np.int16,
    ).tobytes()
    assert np.frombuffer(discord_pcm_to_realtime(discord_pcm), dtype=np.int16).tolist() == [
        3000,
        1000,
    ]

    realtime_pcm = np.array([100, -200], dtype=np.int16).tobytes()
    converted = np.frombuffer(realtime_pcm_to_discord(realtime_pcm), dtype=np.int16)
    assert converted.reshape(-1, 2).tolist() == [
        [100, 100],
        [100, 100],
        [-200, -200],
        [-200, -200],
    ]


def test_streaming_source_append_and_cancel():
    source = RealtimePCMSource()
    source.start_speech_stream("one")
    source.append_speech_stream("one", b"\x01\x02" * 100)
    frame = source.read()
    assert len(frame) == FRAME_SIZE
    assert frame.startswith(b"\x01\x02" * 100)
    assert source.cancel_speech_stream("one") > 0
    assert source.read() == b"\x00" * FRAME_SIZE


@pytest.mark.asyncio
async def test_realtime_requires_hermes_tool_and_speaks_its_result():
    websocket = FakeWebSocket()
    connect = AsyncMock(return_value=websocket)
    consult = AsyncMock(return_value="Hermes result")
    sink = RealtimePCMSource()
    session = RealtimeVoiceSession(
        guild_id=123,
        api_key="test-key",
        audio_sink=sink,
        consult_hermes=consult,
        websocket_connect=connect,
    )
    await session.start()
    try:
        update = websocket.sent[0]
        assert update["session"]["tool_choice"] == "required"
        assert update["session"]["tools"][0]["name"] == "consult_hermes"

        session._current_turn_user_id = 456
        await session._handle_event(
            {
                "type": "response.done",
                "response": {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "consult_hermes",
                            "call_id": "call-1",
                            "arguments": json.dumps({"request": "check my calendar"}),
                        }
                    ]
                },
            }
        )
        await asyncio.gather(*session._tool_tasks)
        consult.assert_awaited_once_with(123, 456, "check my calendar")
        assert websocket.sent[-2]["item"]["output"] == "Hermes result"
        assert websocket.sent[-1]["type"] == "response.create"

        pcm = np.array([10, -10], dtype=np.int16).tobytes()
        await session._handle_event(
            {
                "type": "response.output_audio.delta",
                "item_id": "item-1",
                "content_index": 0,
                "delta": base64.b64encode(pcm).decode("ascii"),
            }
        )
        assert sink.read() != b"\x00" * FRAME_SIZE
    finally:
        await session.close()
