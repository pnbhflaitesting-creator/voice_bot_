"""Acoustic echo cancellation (pure NumPy, no compiled dependency).

When the microphone hears the bot's own audio (built-in mic near the speakers,
leakage, etc.) barge-in falls apart — the bot interrupts and answers itself.
AEC removes the played audio (the *reference*) from the mic signal so only the
user's voice remains.

This is a **constrained partitioned frequency-domain adaptive filter** (a.k.a.
multi-delay block frequency-domain / MDF adaptive filter), the same family used
by WebRTC's echo canceller. It cancels the *linear* echo path (speaker → room →
mic). It has no nonlinear processing or double-talk detector, so it reduces echo
rather than annihilating it; for heavy double-talk a hardware/WebRTC AEC or a
headset is still better. Half-duplex remains the guaranteed-safe fallback.

Everything runs at the 16 kHz mic rate, one 512-sample frame at a time.
"""

from __future__ import annotations

import numpy as np


class EchoCanceller:
    def __init__(self, frame_size: int, filter_len: int, mu: float = 0.3) -> None:
        self.B = frame_size
        self.N = 2 * frame_size          # FFT size (50% overlap, overlap-save)
        self.P = max(1, -(-filter_len // frame_size))  # partitions (ceil)
        self.bins = self.N // 2 + 1
        self.mu = mu

        self._W = np.zeros((self.P, self.bins), dtype=np.complex128)  # filter (freq)
        self._X = np.zeros((self.P, self.bins), dtype=np.complex128)  # ref history (freq)
        self._x_prev = np.zeros(self.B)                              # overlap-save tail
        self._power = np.zeros(self.bins)                            # smoothed ref power

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Return the mic frame with the echo of ``ref`` removed.

        ``mic`` and ``ref`` are float32 arrays of length ``frame_size``.
        """
        mic = mic.astype(np.float64)
        ref = ref.astype(np.float64)

        # If nothing is playing, there is no echo to cancel.
        if not np.any(ref) and not np.any(self._x_prev):
            self._x_prev = ref
            return mic.astype(np.float32)

        # Overlap-save input block: [previous B ref samples, current B].
        x_block = np.concatenate([self._x_prev, ref])
        self._x_prev = ref
        Xf = np.fft.rfft(x_block)

        # Shift the newest reference block into the partition history.
        self._X = np.roll(self._X, 1, axis=0)
        self._X[0] = Xf

        # Estimated echo = sum over partitions of W·X, then the valid (last B).
        Yf = np.sum(self._W * self._X, axis=0)
        y = np.fft.irfft(Yf, n=self.N)[self.B:]

        # Error / cleaned output.
        e = mic - y

        # Adapt the filter (normalized, gradient-constrained to keep it causal).
        Ef = np.fft.rfft(np.concatenate([np.zeros(self.B), e]))
        inst_power = np.sum(np.abs(self._X) ** 2, axis=0)
        self._power = 0.9 * self._power + 0.1 * inst_power
        step = self.mu / (self._power + 1e-6)
        for p in range(self.P):
            grad = np.conj(self._X[p]) * (step * Ef)
            g = np.fft.irfft(grad, n=self.N)
            g[self.B:] = 0.0  # constrain to the first B taps of this partition
            self._W[p] += np.fft.rfft(g)

        return e.astype(np.float32)

    def reset(self) -> None:
        self._W[:] = 0
        self._X[:] = 0
        self._x_prev[:] = 0
        self._power[:] = 0
