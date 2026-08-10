"""Optional noise suppression applied to a captured utterance before STT.

This runs on the whole buffered utterance (not per-frame), so it improves STT
accuracy in noisy rooms without adding real-time processing to the mic path.
Both stages are optional and degrade gracefully if their libraries are missing.

Ordering: high-pass (cut low rumble) -> spectral denoise (cut broadband noise).
"""

from __future__ import annotations

import numpy as np

from .config import settings


class Denoiser:
    def __init__(self) -> None:
        self._mode = settings.denoise
        self._nr = None
        self._highpass = None

        if self._mode == "spectral":
            try:
                import noisereduce as nr  # type: ignore

                self._nr = nr
            except ImportError:
                print(
                    "[denoise] DENOISE=spectral but 'noisereduce' is not installed. "
                    "Run: pip install noisereduce  — continuing without denoise.",
                    flush=True,
                )
                self._mode = "none"

        if settings.highpass_hz > 0:
            self._highpass = self._make_highpass(settings.highpass_hz)

    def _make_highpass(self, cutoff_hz: int):
        """Return a function applying a Butterworth high-pass, or None."""
        try:
            from scipy.signal import butter, sosfilt  # type: ignore
        except ImportError:
            print(
                "[denoise] HIGHPASS_HZ set but 'scipy' is not installed. "
                "Run: pip install scipy — continuing without the high-pass.",
                flush=True,
            )
            return None

        nyquist = settings.input_sample_rate / 2
        sos = butter(2, cutoff_hz / nyquist, btype="highpass", output="sos")
        return lambda x: sosfilt(sos, x).astype(np.float32)

    @property
    def enabled(self) -> bool:
        return self._mode != "none" or self._highpass is not None

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Return a denoised copy of ``audio`` (float32 mono @ 16 kHz)."""
        if not self.enabled:
            return audio
        try:
            out = audio
            if self._highpass is not None:
                out = self._highpass(out)
            if self._nr is not None:
                out = self._nr.reduce_noise(
                    y=out,
                    sr=settings.input_sample_rate,
                    stationary=settings.denoise_stationary,
                    prop_decrease=settings.denoise_strength,
                ).astype(np.float32)
            return out
        except Exception as exc:  # pragma: no cover - never break a turn on this
            print(f"[denoise] skipped ({exc})", flush=True)
            return audio
