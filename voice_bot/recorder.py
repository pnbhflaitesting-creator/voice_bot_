"""Debug recorder: dump each turn's audio (raw + denoised) and transcript.

Enable with ``SAVE_TURNS=true``. For every detected turn it writes three files
into ``TURNS_DIR`` (default ``recordings/``):

    <timestamp>_<n>_raw.wav     what the microphone heard (before denoising)
    <timestamp>_<n>_clean.wav   what was sent to the transcribe model (after denoise)
    <timestamp>_<n>.txt         the transcript that came back

Play the two WAVs to hear whether denoising helped or hurt, and read the .txt
to see what the STT model actually produced from the clean audio.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from .audio_utils import to_wav_bytes
from .config import settings


class TurnRecorder:
    def __init__(self) -> None:
        self.enabled = settings.save_turns
        self._n = 0
        self._dir = Path(settings.turns_dir)
        if self.enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, raw: np.ndarray, clean: np.ndarray, transcript: str) -> str:
        """Write raw/clean WAVs + transcript. Returns the shared filename stem."""
        self._n += 1
        sr = settings.input_sample_rate
        ts = dt.datetime.now().strftime("%H%M%S")
        stem = f"{ts}_{self._n:04d}"
        (self._dir / f"{stem}_raw.wav").write_bytes(to_wav_bytes(raw, sr))
        (self._dir / f"{stem}_clean.wav").write_bytes(to_wav_bytes(clean, sr))
        (self._dir / f"{stem}.txt").write_text((transcript or ""), encoding="utf-8")
        return stem
