"""Discord platform override that adds OpenAI Realtime voice."""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from plugins.platforms.discord import adapter as core

from .pcm_stream import RealtimePCMSource
from .realtime_voice import DEFAULT_MODEL, RealtimeVoiceSession

logger = logging.getLogger(__name__)


class RealtimeVoiceReceiver(core.VoiceReceiver):
    """Core receiver with a low-latency callback after Opus decoding."""

    def __init__(
        self,
        voice_client: Any,
        allowed_user_ids: Optional[set] = None,
        pcm_callback: Optional[Callable[[int, bytes], bool]] = None,
    ) -> None:
        super().__init__(voice_client, allowed_user_ids=allowed_user_ids)
        self._pcm_callback = pcm_callback

    def _on_packet(self, data: bytes) -> None:
        if not self._running or self._paused:
            return
        self._packet_debug_count += 1
        if len(data) < 16:
            return
        if (data[0] >> 6) != 2 or (data[1] & 0x7F) != 0x78:
            return

        first_byte = data[0]
        _, _, _seq, _timestamp, ssrc = struct.unpack_from(">BBHII", data, 0)
        if ssrc == self._bot_ssrc:
            return

        cc = first_byte & 0x0F
        has_extension = bool(first_byte & 0x10)
        has_padding = bool(first_byte & 0x20)
        header_size = 12 + (4 * cc) + (4 if has_extension else 0)
        if len(data) < header_size + 4:
            return

        ext_data_len = 0
        if has_extension:
            ext_offset = 12 + (4 * cc)
            ext_data_len = struct.unpack_from(">H", data, ext_offset + 2)[0] * 4

        header = bytes(data[:header_size])
        payload_with_nonce = data[header_size:]
        if len(payload_with_nonce) < 4:
            return
        nonce = bytearray(24)
        nonce[:4] = payload_with_nonce[-4:]
        encrypted = bytes(payload_with_nonce[:-4])

        try:
            import nacl.secret

            decrypted = nacl.secret.Aead(self._secret_key).decrypt(
                encrypted, header, bytes(nonce)
            )
        except Exception:
            return

        if ext_data_len and len(decrypted) > ext_data_len:
            decrypted = decrypted[ext_data_len:]
        if has_padding:
            if not decrypted:
                return
            pad_len = decrypted[-1]
            if pad_len == 0 or pad_len > len(decrypted):
                return
            decrypted = decrypted[:-pad_len]
            if not decrypted:
                return

        if self._dave_session:
            with self._lock:
                user_id = self._ssrc_to_user.get(ssrc, 0)
            if user_id:
                try:
                    import davey

                    decrypted = self._dave_session.decrypt(
                        user_id, davey.MediaType.audio, decrypted
                    )
                except Exception as exc:
                    if "Unencrypted" not in str(exc):
                        return

        try:
            if ssrc not in self._decoders:
                self._decoders[ssrc] = core.discord.opus.Decoder()
            pcm = self._decoders[ssrc].decode(decrypted)
            with self._lock:
                user_id = self._ssrc_to_user.get(ssrc, 0)
            consumed = False
            if user_id and self._pcm_callback is not None:
                consumed = bool(self._pcm_callback(user_id, pcm))
            if consumed:
                return
            with self._lock:
                self._buffers[ssrc].extend(pcm)
                self._last_packet_time[ssrc] = time.monotonic()
        except Exception:
            with self._lock:
                self._decoders.pop(ssrc, None)


class DiscordRealtimeAdapter(core.DiscordAdapter):
    """The standard Discord adapter plus a Realtime speech front end."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._realtime_sessions: Dict[int, RealtimeVoiceSession] = {}
        self._realtime_sources: Dict[int, RealtimePCMSource] = {}
        self._realtime_cfg = self._load_realtime_config()

    @staticmethod
    def _load_realtime_config() -> Dict[str, Any]:
        values: Dict[str, Any] = {
            "enabled": False,
            "model": DEFAULT_MODEL,
            "voice": "marin",
            "vad": "semantic_vad",
            "instructions": "",
            "max_tool_output_chars": 12_000,
        }
        try:
            from hermes_cli.config import read_raw_config

            raw = read_raw_config() or {}
            configured = ((raw.get("discord") or {}).get("realtime_voice") or {})
            if isinstance(configured, dict):
                values.update({k: v for k, v in configured.items() if k in values})
        except Exception:
            logger.exception("Could not read discord.realtime_voice config")
        return values

    def realtime_voice_enabled(self) -> bool:
        value = self._realtime_cfg.get("enabled", False)
        return value is True or str(value).lower() in {"1", "true", "yes", "on"}

    def realtime_voice_active(self, guild_id: int) -> bool:
        session = self._realtime_sessions.get(int(guild_id))
        return bool(session and session.is_connected)

    async def _open_realtime(self, guild_id: int, source: RealtimePCMSource) -> None:
        from agent.secret_scope import get_secret

        api_key = (get_secret("OPENAI_API_KEY", "") or "").strip()
        if not api_key:
            api_key = (get_secret("VOICE_TOOLS_OPENAI_KEY", "") or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        cfg = self._realtime_cfg
        session = RealtimeVoiceSession(
            guild_id=guild_id,
            api_key=api_key,
            audio_sink=source,
            consult_hermes=self._consult_hermes,
            model=str(cfg.get("model") or DEFAULT_MODEL),
            voice=str(cfg.get("voice") or "marin"),
            vad=str(cfg.get("vad") or "semantic_vad"),
            instructions=str(cfg.get("instructions") or ""),
            max_tool_output_chars=int(cfg.get("max_tool_output_chars") or 12_000),
        )
        await session.start()
        self._realtime_sessions[guild_id] = session

    def _feed_realtime(self, guild_id: int, user_id: int, pcm: bytes) -> bool:
        session = self._realtime_sessions.get(guild_id)
        if session is None or not session.is_connected:
            return False
        guild = self._client.get_guild(guild_id) if self._client else None
        if not self._is_allowed_user(str(user_id), guild=guild, is_dm=False):
            return True
        consumed = session.feed_audio_threadsafe(user_id, pcm)
        if consumed:
            loop = getattr(self, "_voice_event_loop", None)
            if loop is not None:
                loop.call_soon_threadsafe(self._reset_voice_timeout, guild_id)
        return consumed

    async def _consult_hermes(self, guild_id: int, user_id: int, transcript: str) -> str:
        """Run one spoken request through this gateway's normal Hermes handler."""
        handler = getattr(self, "_message_handler", None)
        if handler is None:
            return "Hermes is not ready yet."

        # `/voice join` stores this binding immediately after join returns.
        for _ in range(20):
            text_channel_id = self._voice_text_channels.get(guild_id)
            if text_channel_id:
                break
            await asyncio.sleep(0.05)
        else:
            return "This voice channel is not linked to a Hermes text session."

        source_data = getattr(self, "_voice_sources", {}).get(guild_id)
        if source_data:
            source = SessionSource.from_dict(source_data)
            source.user_id = str(user_id)
            source.user_name = str(user_id)
        else:
            source = SessionSource(
                platform=core.Platform.DISCORD,
                chat_id=str(text_channel_id),
                user_id=str(user_id),
                user_name=str(user_id),
                chat_type="channel",
            )

        try:
            channel = self._client.get_channel(text_channel_id)
            if channel:
                safe = transcript[:2000].replace("@everyone", "@\u200beveryone").replace(
                    "@here", "@\u200bhere"
                )
                await channel.send(f"**[Voice · Realtime]** <@{user_id}>: {safe}")
        except Exception:
            logger.debug("Could not echo Realtime transcript", exc_info=True)

        channel_prompt = None
        try:
            channel_prompt = self._resolve_channel_prompt(str(text_channel_id))
            if asyncio.iscoroutine(channel_prompt):
                channel_prompt = await channel_prompt
        except Exception:
            pass

        event = MessageEvent(
            source=source,
            text=transcript,
            message_type=MessageType.TEXT,
            raw_message=SimpleNamespace(guild_id=guild_id, guild=None),
            channel_prompt=channel_prompt if isinstance(channel_prompt, str) else None,
            metadata={"realtime_voice_proxy": True},
            allow_gateway_control=True,
        )
        response = await handler(event)
        text, _ttl = self._unwrap_ephemeral(response)
        if text:
            try:
                text = self.prepare_tts_text(text)
            except Exception:
                pass
        return str(text or "Hermes completed that without a text response.")

    async def join_voice_channel(
        self, channel: Any, *, text_channel_id: int = None, source: dict = None
    ) -> bool:
        if not self.realtime_voice_enabled():
            return await super().join_voice_channel(
                channel, text_channel_id=text_channel_id, source=source
            )
        if not self._client or not core.DISCORD_AVAILABLE:
            return False
        guild_id = int(channel.guild.id)

        async with self._voice_locks.setdefault(guild_id, asyncio.Lock()):
            existing = self._voice_clients.get(guild_id)
            if existing and existing.is_connected():
                if existing.channel.id != channel.id:
                    await existing.move_to(channel)
                self._reset_voice_timeout(guild_id)
                return True

            vc = await channel.connect()
            self._voice_event_loop = asyncio.get_running_loop()
            self._voice_clients[guild_id] = vc
            self._reset_voice_timeout(guild_id)
            if text_channel_id is not None:
                self._voice_text_channels[guild_id] = text_channel_id
            if source is not None:
                self._voice_sources[guild_id] = source

            pcm_source = RealtimePCMSource()
            try:
                await self._open_realtime(guild_id, pcm_source)
                vc.play(pcm_source)
                self._realtime_sources[guild_id] = pcm_source
                receiver = RealtimeVoiceReceiver(
                    vc,
                    allowed_user_ids=self._allowed_user_ids,
                    pcm_callback=lambda uid, pcm: self._feed_realtime(
                        guild_id, uid, pcm
                    ),
                )
                receiver.start()
                self._voice_receivers[guild_id] = receiver
                self._voice_listen_tasks[guild_id] = asyncio.create_task(
                    self._voice_listen_loop(guild_id)
                )
                return True
            except Exception:
                logger.exception("Realtime voice failed to start; using standard voice")
                session = self._realtime_sessions.pop(guild_id, None)
                if session:
                    await session.close()
                pcm_source.cleanup()
                try:
                    if vc.is_playing():
                        vc.stop()
                    await vc.disconnect()
                finally:
                    self._voice_clients.pop(guild_id, None)

        return await super().join_voice_channel(
            channel, text_channel_id=text_channel_id, source=source
        )

    async def leave_voice_channel(self, guild_id: int) -> None:
        session = self._realtime_sessions.pop(int(guild_id), None)
        if session:
            await session.close()
        source = self._realtime_sources.pop(int(guild_id), None)
        if source:
            source.cleanup()
        await super().leave_voice_channel(guild_id)


def _build_adapter(config: Any) -> DiscordRealtimeAdapter:
    return DiscordRealtimeAdapter(config)


def register(ctx: Any) -> None:
    from hermes_cli import __version__

    if __version__ != "0.20.5":
        raise RuntimeError(
            "hermes-discord-realtime 0.1.0 supports Hermes 0.20.5 only; "
            f"found {__version__}"
        )
    ctx.register_platform(
        name="discord",
        label="Discord Realtime Voice",
        adapter_factory=_build_adapter,
        check_fn=core.discord_deps_present,
        ensure_deps_fn=core.check_discord_requirements,
        is_connected=core._is_connected,
        required_env=["DISCORD_BOT_TOKEN", "OPENAI_API_KEY"],
        install_hint="The official Hermes image already includes Discord voice dependencies.",
        setup_fn=core.interactive_setup,
        apply_yaml_config_fn=core._apply_yaml_config,
        allowed_users_env="DISCORD_ALLOWED_USERS",
        allow_all_env="DISCORD_ALLOW_ALL_USERS",
        cron_deliver_env_var="DISCORD_HOME_CHANNEL",
        standalone_sender_fn=core._standalone_send,
        max_message_length=2000,
        emoji="🎙️",
        allow_update_command=True,
    )
