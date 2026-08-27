"""Thread-safe streaming PCM source for Discord voice playback."""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Optional

import discord


FRAME_MS = 20
SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_WIDTH = 2
FRAME_SIZE = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * FRAME_MS // 1000


class RealtimePCMSource(discord.AudioSource):
    """Continuous Discord source fed incrementally by OpenAI audio deltas.

    Discord's player thread calls :meth:`read` every 20 ms. The asyncio thread
    appends provider audio through :meth:`append`. Keeping one source alive
    avoids file buffering and lets barge-in discard unheard audio immediately.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: Deque[bytes] = deque()
        self._head_offset = 0
        self._stream_id: Optional[str] = None
        self._stream_finished = False
        self._played_bytes = 0
        self._closed = False

    def is_opus(self) -> bool:
        return False

    def start_speech_stream(self, stream_id: str) -> None:
        with self._lock:
            self._chunks.clear()
            self._head_offset = 0
            self._stream_id = str(stream_id)
            self._stream_finished = False
            self._played_bytes = 0

    def append_speech_stream(self, stream_id: str, pcm: bytes) -> bool:
        if not pcm:
            return True
        with self._lock:
            if self._closed or self._stream_id != str(stream_id):
                return False
            self._chunks.append(bytes(pcm))
            return True

    def finish_speech_stream(self, stream_id: str) -> bool:
        with self._lock:
            if self._stream_id != str(stream_id):
                return False
            self._stream_finished = True
            return True

    def cancel_speech_stream(self, stream_id: str) -> int:
        with self._lock:
            if self._stream_id != str(stream_id):
                return 0
            played_ms = self._played_bytes * 1000 // (
                SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH
            )
            self._chunks.clear()
            self._head_offset = 0
            self._stream_id = None
            self._stream_finished = False
            self._played_bytes = 0
            return played_ms

    def _take_locked(self, size: int) -> bytes:
        out = bytearray()
        while self._chunks and len(out) < size:
            head = self._chunks[0]
            remaining = head[self._head_offset :]
            take = min(size - len(out), len(remaining))
            out.extend(remaining[:take])
            self._head_offset += take
            if self._head_offset >= len(head):
                self._chunks.popleft()
                self._head_offset = 0
        self._played_bytes += len(out)
        if not self._chunks and self._stream_finished:
            self._stream_id = None
            self._stream_finished = False
            self._played_bytes = 0
        return bytes(out)

    def read(self) -> bytes:
        with self._lock:
            if self._closed:
                return b""
            frame = self._take_locked(FRAME_SIZE)
        if len(frame) < FRAME_SIZE:
            frame += b"\x00" * (FRAME_SIZE - len(frame))
        return frame

    def cleanup(self) -> None:
        with self._lock:
            self._closed = True
            self._chunks.clear()
            self._stream_id = None

