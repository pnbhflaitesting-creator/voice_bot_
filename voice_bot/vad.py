"""Silero VAD wrapper and a turn-taking state machine.

The :class:`TurnDetector` is fed one 512-sample (16 kHz, float32) frame at a
time. It tracks whether the user is currently speaking and decides when a turn
has ended (a natural pause), emitting the collected audio for transcription.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import torch
from silero_vad import load_silero_vad

from .config import settings


class TurnEvent(Enum):
    """What the detector observed after consuming a frame."""

    NONE = auto()          # nothing notable
    SPEECH_START = auto()  # user just started talking
    TURN_END = auto()      # user finished a turn; audio is ready


@dataclass
class TurnResult:
    event: TurnEvent
    audio: np.ndarray | None = None  # float32 mono @ 16 kHz, only on TURN_END


class TurnDetector:
    """Streaming VAD + end-of-turn detection.

    Usage::

        det = TurnDetector()
        for frame in frames:              # each frame is 512 float32 samples
            result = det.process(frame)
            if result.event is TurnEvent.TURN_END:
                transcribe(result.audio)
    """

    def __init__(self) -> None:
        # Loads the Silero VAD model (torch backend). Small and CPU-friendly.
        self._model = load_silero_vad()
        torch.set_num_threads(1)  # lowest latency for tiny per-frame calls

        self._sample_rate = settings.input_sample_rate
        self._threshold = settings.vad_threshold

        frame_ms = settings.frame_ms
        self._min_speech_frames = max(1, round(settings.min_speech_ms / frame_ms))
        self._hangover_frames = max(1, round(settings.silence_hangover_ms / frame_ms))
        self._min_turn_frames = max(1, round(settings.min_turn_ms / frame_ms))
        pad_frames = max(0, round(settings.speech_pad_ms / frame_ms))

        # Rolling buffer of recent frames so we can prepend a little audio
        # before the detected speech start (prevents clipped first words).
        self._preroll: deque[np.ndarray] = deque(maxlen=pad_frames)

        self._reset()

    def _reset(self) -> None:
        self._model.reset_states()
        self._speaking = False
        self._speech_run = 0        # consecutive speech frames (for start)
        self._silence_run = 0       # consecutive silence frames (for end)
        self._buffer: list[np.ndarray] = []

    def _speech_prob(self, frame: np.ndarray) -> float:
        tensor = torch.from_numpy(frame)
        return self._model(tensor, self._sample_rate).item()

    def process(self, frame: np.ndarray) -> TurnResult:
        """Consume one 512-sample float32 frame and return what happened."""
        prob = self._speech_prob(frame)
        is_speech = prob >= self._threshold

        if not self._speaking:
            # Waiting for the user to start talking.
            self._preroll.append(frame)
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self._min_speech_frames:
                    # Confirmed start of a turn.
                    self._speaking = True
                    self._silence_run = 0
                    # Seed the buffer with the pre-roll (includes these frames).
                    self._buffer = list(self._preroll)
                    self._preroll.clear()
                    return TurnResult(TurnEvent.SPEECH_START)
            else:
                self._speech_run = 0
            return TurnResult(TurnEvent.NONE)

        # Currently in a turn: keep collecting audio.
        self._buffer.append(frame)
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._hangover_frames:
                # Trailing pause long enough -> the turn is over.
                audio = np.concatenate(self._buffer) if self._buffer else np.empty(0, np.float32)
                spoken_frames = len(self._buffer) - self._silence_run
                self._reset()
                if spoken_frames < self._min_turn_frames:
                    # Too short to be real speech; drop it and keep listening.
                    return TurnResult(TurnEvent.NONE)
                return TurnResult(TurnEvent.TURN_END, audio=audio)

        return TurnResult(TurnEvent.NONE)

    def reset(self) -> None:
        """Forget any in-progress turn (used after a barge-in)."""
        self._reset()
        self._preroll.clear()
