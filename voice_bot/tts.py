"""Text-to-speech providers: OpenAI, Deepgram, ElevenLabs.

Every provider streams raw **24 kHz / 16-bit / mono PCM** bytes so the output
matches ``SpeakerStream`` directly (no client-side decoding, lowest latency).
Select one with ``TTS_PROVIDER``; ``create_tts()`` builds it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

import httpx
from openai import AsyncOpenAI

from .config import settings

# The speaker plays this rate; every provider is configured to emit it.
_PCM_RATE = 24_000


class TTSProvider(Protocol):
    def stream(self, text: str) -> AsyncIterator[bytes]: ...
    async def warmup(self) -> None: ...
    async def aclose(self) -> None: ...


class OpenAITTS:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return
        async with self._client.audio.speech.with_streaming_response.create(
            model=settings.openai_tts_model,
            voice=settings.openai_tts_voice,
            input=text,
            response_format="pcm",  # 24 kHz, 16-bit, mono
            speed=settings.tts_speed,
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=4096):
                if chunk:
                    yield chunk

    async def warmup(self) -> None:
        try:
            await self._client.models.list()
        except Exception:
            pass

    async def aclose(self) -> None:
        await self._client.close()


class DeepgramTTS:
    """Deepgram Aura streaming TTS (REST, linear16 PCM)."""

    _URL = "https://api.deepgram.com/v1/speak"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Token {settings.deepgram_api_key}"},
            timeout=30.0,
        )

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return
        params = {
            "model": settings.deepgram_tts_model,
            "encoding": "linear16",
            "sample_rate": str(_PCM_RATE),
            "container": "none",  # raw PCM, no WAV header
        }
        async with self._client.stream(
            "POST", self._URL, params=params, json={"text": text}
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk

    async def warmup(self) -> None:
        try:
            await self._client.get("https://api.deepgram.com/v1/auth/token")
        except Exception:
            pass

    async def aclose(self) -> None:
        await self._client.aclose()


class ElevenLabsTTS:
    """ElevenLabs streaming TTS (Flash model, pcm_24000)."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={"xi-api-key": settings.elevenlabs_api_key},
            timeout=30.0,
        )
        self._url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/"
            f"{settings.elevenlabs_voice_id}/stream"
        )

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return
        body = {
            "text": text,
            "model_id": settings.elevenlabs_tts_model,
        }
        params = {"output_format": "pcm_24000"}  # 24 kHz, 16-bit, mono
        async with self._client.stream(
            "POST", self._url, params=params, json=body
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk

    async def warmup(self) -> None:
        try:
            await self._client.get("https://api.elevenlabs.io/v1/models")
        except Exception:
            pass

    async def aclose(self) -> None:
        await self._client.aclose()


def create_tts() -> TTSProvider:
    provider = settings.tts_provider
    if provider == "openai":
        return OpenAITTS()
    if provider == "deepgram":
        return DeepgramTTS()
    if provider == "elevenlabs":
        return ElevenLabsTTS()
    raise ValueError(f"Unknown TTS provider: {provider}")
