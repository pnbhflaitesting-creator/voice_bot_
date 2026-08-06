"""Speech-to-text providers: OpenAI, Deepgram, ElevenLabs.

Each provider takes a float32 mono @ 16 kHz utterance and returns a transcript.
Select one with ``STT_PROVIDER`` (see config). ``create_stt()`` builds it.

Note: these are batch (whole-utterance) transcribers — the clip is sent after
the turn ends. For the lowest possible STT latency you'd want *streaming* STT
(Deepgram's WebSocket), which transcribes while the user is still talking.
"""

from __future__ import annotations

from typing import Protocol

import httpx
import numpy as np
from openai import AsyncOpenAI

from .audio_utils import to_wav_bytes
from .config import settings


class STTProvider(Protocol):
    async def transcribe(self, audio: np.ndarray) -> str: ...
    async def warmup(self) -> None: ...
    async def aclose(self) -> None: ...


class OpenAISTT:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = to_wav_bytes(audio, settings.input_sample_rate)
        kwargs: dict = {
            "model": settings.openai_stt_model,
            "file": ("speech.wav", wav_bytes, "audio/wav"),
        }
        if settings.stt_language:
            kwargs["language"] = settings.stt_language
        if settings.stt_prompt:
            kwargs["prompt"] = settings.stt_prompt
        resp = await self._client.audio.transcriptions.create(**kwargs)
        return (resp.text or "").strip()

    async def warmup(self) -> None:
        try:
            await self._client.models.list()
        except Exception:
            pass

    async def aclose(self) -> None:
        await self._client.close()


class DeepgramSTT:
    """Deepgram prerecorded transcription (REST)."""

    _URL = "https://api.deepgram.com/v1/listen"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Token {settings.deepgram_api_key}"},
            timeout=30.0,
        )

    async def transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = to_wav_bytes(audio, settings.input_sample_rate)
        params = {"model": settings.deepgram_stt_model, "smart_format": "true"}
        if settings.stt_language:
            params["language"] = settings.stt_language
        resp = await self._client.post(
            self._URL,
            params=params,
            content=wav_bytes,
            headers={"Content-Type": "audio/wav"},
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data["results"]["channels"][0]["alternatives"][0]["transcript"]
        ).strip()

    async def warmup(self) -> None:
        try:
            await self._client.get("https://api.deepgram.com/v1/auth/token")
        except Exception:
            pass

    async def aclose(self) -> None:
        await self._client.aclose()


class ElevenLabsSTT:
    """ElevenLabs Scribe speech-to-text (REST)."""

    _URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={"xi-api-key": settings.elevenlabs_api_key},
            timeout=30.0,
        )

    async def transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = to_wav_bytes(audio, settings.input_sample_rate)
        data = {"model_id": settings.elevenlabs_stt_model}
        if settings.stt_language:
            data["language_code"] = settings.stt_language
        resp = await self._client.post(
            self._URL,
            data=data,
            files={"file": ("speech.wav", wav_bytes, "audio/wav")},
        )
        resp.raise_for_status()
        return (resp.json().get("text") or "").strip()

    async def warmup(self) -> None:
        try:
            await self._client.get("https://api.elevenlabs.io/v1/models")
        except Exception:
            pass

    async def aclose(self) -> None:
        await self._client.aclose()


def create_stt() -> STTProvider:
    provider = settings.stt_provider
    if provider == "openai":
        return OpenAISTT()
    if provider == "deepgram":
        return DeepgramSTT()
    if provider == "elevenlabs":
        return ElevenLabsSTT()
    raise ValueError(f"Unknown STT provider: {provider}")
