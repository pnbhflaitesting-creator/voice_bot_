"""Small audio helpers shared across providers."""

from __future__ import annotations

import io
import wave

import numpy as np


def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 mono in [-1, 1] to little-endian int16 samples."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono audio as a 16-bit PCM WAV file (in memory)."""
    pcm16 = float_to_pcm16(audio)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()
