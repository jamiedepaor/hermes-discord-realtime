# Hermes Discord Realtime Voice

Native, interruptible voice conversations with an existing Hermes Agent through
a Discord voice channel.

```text
Discord mobile/desktop VC
        ↕ live PCM audio
OpenAI Realtime (speech + turn detection)
        ↕ required consult_hermes call
Your existing Hermes gateway (tools + memory + personality)
```

This is a Hermes **platform plugin**. It runs inside the existing Hermes
gateway; it does not require Railway, a second server, a custom webpage, or a
replacement Nous Cloud image.

## Compatibility

- Hermes Agent `0.20.5`
- The official Nous/Docker image, which already includes Discord voice,
  WebSockets, NumPy, FFmpeg, and Opus dependencies
- OpenAI Realtime model `gpt-realtime-2.1-mini` by default

The plugin deliberately fails closed on unsupported Hermes versions.

## Install

```bash
hermes plugins install jamiedepaor/hermes-discord-realtime --enable
```

Then add this to `/opt/data/config.yaml` (or switch the cloud editor to YAML):

```yaml
discord:
  realtime_voice:
    enabled: true
    model: gpt-realtime-2.1-mini
    voice: marin
    vad: semantic_vad
```

Restart the gateway. In Discord, join a voice channel and run `/voice join`.

## Security

- Discord user authorization is inherited from Hermes (`DISCORD_ALLOWED_USERS`
  and pairing rules).
- The OpenAI API key remains in Hermes' secret store; it is never written to
  `config.yaml`.
- OpenAI handles audio and transcription. Hermes receives the resulting text.
- Every spoken request is forced through Hermes; the speech model cannot answer
  directly from its own knowledge.

