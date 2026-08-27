"""OpenAI Realtime transport used by the Discord platform override."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2.1-mini"
DEFAULT_INSTRUCTIONS = """
You are the realtime voice interface for a Hermes personal agent.
For every user request, call consult_hermes exactly once. Put a faithful,
complete transcription of the user's request in the request argument. Never
answer from your own knowledge before calling the tool. After the tool returns,
speak its answer naturally and faithfully. Do not mention the tool or handoff.
""".strip()


def discord_pcm_to_realtime(pcm: bytes) -> bytes:
    """Convert Discord 48 kHz stereo s16le PCM to 24 kHz mono s16le."""
    if not pcm:
        return b""
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2")
    samples = samples[: len(samples) - (len(samples) % 4)]
    if not len(samples):
        return b""
    stereo = samples.reshape(-1, 2).astype(np.int32)
    mono_48k = (stereo[:, 0] + stereo[:, 1]) // 2
    mono_24k = (mono_48k[0::2] + mono_48k[1::2]) // 2
    return mono_24k.astype("<i2").tobytes()


def realtime_pcm_to_discord(pcm: bytes) -> bytes:
    """Convert OpenAI 24 kHz mono s16le PCM to Discord 48 kHz stereo."""
    if not pcm:
        return b""
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) - (len(pcm) % 2)], dtype="<i2")
    if not len(samples):
        return b""
    upsampled = np.repeat(samples, 2)
    stereo = np.repeat(upsampled[:, None], 2, axis=1)
    return stereo.astype("<i2", copy=False).tobytes()


class RealtimeVoiceSession:
    """One OpenAI Realtime WebSocket bound to one Discord guild."""

    def __init__(
        self,
        *,
        guild_id: int,
        api_key: str,
        audio_sink: Any,
        consult_hermes: Callable[[int, int, str], Awaitable[str]],
        model: str = DEFAULT_MODEL,
        voice: str = "marin",
        vad: str = "semantic_vad",
        instructions: str = "",
        max_tool_output_chars: int = 12_000,
        websocket_connect: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.guild_id = int(guild_id)
        self.api_key = api_key
        self.audio_sink = audio_sink
        self.consult_hermes = consult_hermes
        self.model = model or DEFAULT_MODEL
        self.voice = voice
        self.vad = vad if vad in {"semantic_vad", "server_vad"} else "semantic_vad"
        self.instructions = instructions.strip() or DEFAULT_INSTRUCTIONS
        self.max_tool_output_chars = max(1000, int(max_tool_output_chars))
        self._websocket_connect = websocket_connect

        self._ws: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._audio_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=1000)
        self._sender_task: Optional[asyncio.Task] = None
        self._receiver_task: Optional[asyncio.Task] = None
        self._tool_tasks: Set[asyncio.Task] = set()
        self._send_lock = asyncio.Lock()
        self._ready_future: Optional[asyncio.Future] = None
        self._connected = False
        self._closed = False
        self._turn_generation = 0
        self._latest_user_id = 0
        self._current_turn_user_id = 0
        self._response_active = False
        self._current_stream_id: Optional[str] = None
        self._current_item_id: Optional[str] = None
        self._current_content_index = 0
        self.last_error = ""

    @property
    def is_connected(self) -> bool:
        return self._connected and not self._closed

    async def start(self) -> None:
        if self.is_connected:
            return
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for Realtime voice")
        connect = self._websocket_connect
        if connect is None:
            from websockets.asyncio.client import connect as websocket_connect

            connect = websocket_connect

        self._loop = asyncio.get_running_loop()
        self._ready_future = self._loop.create_future()
        self._closed = False
        try:
            self._ws = await connect(
                f"{OPENAI_REALTIME_URL}?model={self.model}",
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
                max_size=16 * 1024 * 1024,
            )
            self._sender_task = asyncio.create_task(self._audio_sender())
            self._receiver_task = asyncio.create_task(self._event_receiver())
            await self._send(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "model": self.model,
                        "instructions": self.instructions,
                        "output_modalities": ["audio"],
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm", "rate": 24000},
                                "turn_detection": {"type": self.vad},
                            },
                            "output": {
                                "format": {"type": "audio/pcm"},
                                "voice": self.voice,
                            },
                        },
                        "tools": [
                            {
                                "type": "function",
                                "name": "consult_hermes",
                                "description": (
                                    "Send the user's request to their Hermes agent. "
                                    "Call exactly once for every user turn."
                                ),
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "request": {
                                            "type": "string",
                                            "description": "Faithful complete user request",
                                        }
                                    },
                                    "required": ["request"],
                                    "additionalProperties": False,
                                },
                            }
                        ],
                        "tool_choice": "required",
                    },
                }
            )
            await asyncio.wait_for(asyncio.shield(self._ready_future), timeout=10)
            self._connected = True
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        self._closed = True
        self._connected = False
        if self._current_stream_id:
            self.audio_sink.cancel_speech_stream(self._current_stream_id)
        tasks = [self._sender_task, self._receiver_task, *self._tool_tasks]
        for task in tasks:
            if task and task is not asyncio.current_task():
                task.cancel()
        if self._ws is not None:
            with suppress(Exception):
                await self._ws.close()
        for task in tasks:
            if task and task is not asyncio.current_task():
                with suppress(asyncio.CancelledError, Exception):
                    await task
        self._tool_tasks.clear()
        self._ws = None

    def feed_audio_threadsafe(self, user_id: int, discord_pcm: bytes) -> bool:
        if not self.is_connected or self._loop is None:
            return False
        pcm = discord_pcm_to_realtime(discord_pcm)
        if not pcm:
            return True
        self._loop.call_soon_threadsafe(self._queue_audio, int(user_id), pcm)
        return True

    def _queue_audio(self, user_id: int, pcm: bytes) -> None:
        if not self.is_connected:
            return
        try:
            self._audio_queue.put_nowait((user_id, pcm))
        except asyncio.QueueFull:
            self.last_error = "Realtime input audio queue overflow"

    async def _audio_sender(self) -> None:
        try:
            while not self._closed:
                user_id, pcm = await self._audio_queue.get()
                self._latest_user_id = user_id
                await self._send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode("ascii"),
                    }
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._mark_failed(exc)

    async def _event_receiver(self) -> None:
        try:
            async for raw in self._ws:
                await self._handle_event(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                await self._mark_failed(exc)

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("type", "")
        if event_type == "session.updated":
            if self._ready_future is not None and not self._ready_future.done():
                self._ready_future.set_result(True)
        elif event_type == "response.created":
            self._response_active = True
        elif event_type == "input_audio_buffer.speech_started":
            await self._interrupt_output()
        elif event_type == "response.output_audio.delta":
            self._append_output_audio(event)
        elif event_type == "response.output_audio.done":
            if self._current_stream_id:
                self.audio_sink.finish_speech_stream(self._current_stream_id)
        elif event_type == "response.done":
            self._response_active = False
            self._handle_response_done(event)
        elif event_type == "error":
            error = event.get("error") or {}
            message = error.get("message") or str(error) or "Unknown Realtime error"
            self.last_error = message
            if self._ready_future is not None and not self._ready_future.done():
                self._ready_future.set_exception(RuntimeError(message))

    def _append_output_audio(self, event: Dict[str, Any]) -> None:
        try:
            provider_pcm = base64.b64decode(event.get("delta") or "")
        except Exception:
            return
        stream_id = event.get("item_id") or event.get("response_id") or uuid.uuid4().hex
        if self._current_stream_id != stream_id:
            if self._current_stream_id:
                self.audio_sink.cancel_speech_stream(self._current_stream_id)
            self._current_stream_id = stream_id
            self._current_item_id = event.get("item_id")
            self._current_content_index = int(event.get("content_index", 0) or 0)
            self.audio_sink.start_speech_stream(stream_id)
        self.audio_sink.append_speech_stream(
            stream_id, realtime_pcm_to_discord(provider_pcm)
        )

    def _handle_response_done(self, event: Dict[str, Any]) -> None:
        generation = self._turn_generation
        user_id = self._current_turn_user_id
        for item in (event.get("response") or {}).get("output") or []:
            if item.get("type") != "function_call" or item.get("name") != "consult_hermes":
                continue
            task = asyncio.create_task(self._run_hermes_tool(item, generation, user_id))
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)

    async def _run_hermes_tool(self, item: Dict[str, Any], generation: int, user_id: int) -> None:
        call_id = item.get("call_id")
        if not call_id:
            return
        try:
            request = str(json.loads(item.get("arguments") or "{}").get("request") or "").strip()
        except (TypeError, ValueError, json.JSONDecodeError):
            request = ""
        try:
            answer = (
                await self.consult_hermes(self.guild_id, user_id, request)
                if request
                else "I couldn't understand that request. Please ask again."
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Hermes Realtime consultation failed")
            answer = "Hermes hit an error handling that. Please try again."

        if not self.is_connected:
            return
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(answer or "Hermes completed without text.")[
                        : self.max_tool_output_chars
                    ],
                },
            }
        )
        if generation != self._turn_generation:
            return
        await self._send(
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio"],
                    "tool_choice": "none",
                    "instructions": (
                        "Speak the consult_hermes result naturally and faithfully. "
                        "Do not add facts."
                    ),
                },
            }
        )

    async def _interrupt_output(self) -> None:
        self._turn_generation += 1
        self._current_turn_user_id = self._latest_user_id
        stream_id = self._current_stream_id
        item_id = self._current_item_id
        content_index = self._current_content_index
        played_ms = self.audio_sink.cancel_speech_stream(stream_id) if stream_id else 0
        self._current_stream_id = None
        self._current_item_id = None
        if self._response_active:
            with suppress(Exception):
                await self._send({"type": "response.cancel"})
        if item_id and played_ms > 0:
            with suppress(Exception):
                await self._send(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": item_id,
                        "content_index": content_index,
                        "audio_end_ms": played_ms,
                    }
                )

    async def _send(self, event: Dict[str, Any]) -> None:
        if self._ws is None or self._closed:
            raise RuntimeError("Realtime WebSocket is not connected")
        async with self._send_lock:
            await self._ws.send(json.dumps(event))

    async def _mark_failed(self, exc: Exception) -> None:
        self.last_error = str(exc)
        self._connected = False
        logger.error("OpenAI Realtime voice failed: %s", exc)
