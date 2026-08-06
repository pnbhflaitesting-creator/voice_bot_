"""Speech-to-text using OpenAI's transcription models."""

from __future__ import annotations

import io
import wave

import numpy as np
from openai import AsyncOpenAI

from .config import settings


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono audio in [-1, 1] as a 16-bit PCM WAV file."""
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()


class SpeechToText:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def transcribe(self, audio: np.ndarray) -> str:
        """Return the transcript of a float32 mono @ 16 kHz utterance."""
        wav_bytes = _to_wav_bytes(audio, settings.input_sample_rate)
        # The SDK accepts a (filename, bytes, mimetype) tuple as the file.
        resp = await self._client.audio.transcriptions.create(
            model=settings.stt_model,
            file=("speech.wav", wav_bytes, "audio/wav"),
        )
        return (resp.text or "").strip()
