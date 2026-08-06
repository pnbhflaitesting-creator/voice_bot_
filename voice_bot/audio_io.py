"""Real-time microphone capture and speaker playback via sounddevice.

Two independent streams run at different sample rates:
  * Input  : 16 kHz mono float32, delivered in 512-sample frames (for VAD).
  * Output : 24 kHz mono int16 PCM (what OpenAI TTS returns).

The capture callback runs on a PortAudio thread, so frames are handed to the
asyncio world through ``loop.call_soon_threadsafe``. Playback is driven by a
dedicated worker thread pulling PCM chunks from a thread-safe queue, which makes
instant interruption (barge-in) simple: clear the queue and abort the stream.
"""

from __future__ import annotations

import asyncio
import queue
import threading

import numpy as np
import sounddevice as sd

from .config import settings


class MicrophoneStream:
    """Captures the mic and pushes 512-sample float32 frames to an asyncio queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop, frame_queue: asyncio.Queue[np.ndarray]):
        self._loop = loop
        self._queue = frame_queue
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            # Overflows are usually harmless hiccups; keep going.
            print(f"[audio-in] {status}", flush=True)
        # indata is (frames, 1) float32. Copy: the buffer is reused by PortAudio.
        frame = indata[:, 0].copy()
        self._loop.call_soon_threadsafe(self._push, frame)

    def _push(self, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # drop under back-pressure rather than block the audio thread

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=settings.input_sample_rate,
            blocksize=settings.frame_size,  # exactly one VAD frame per callback
            channels=1,
            dtype="float32",
            device=settings.input_device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


# Sentinel enqueued after a full TTS response so we know when playback drained.
_END = object()


class SpeakerStream:
    """Plays 24 kHz int16 PCM chunks; supports immediate interruption."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._stream = sd.OutputStream(
            samplerate=settings.output_sample_rate,
            channels=1,
            dtype="int16",
            device=settings.output_device,
        )
        self._interrupt = threading.Event()
        self._playing = threading.Event()  # set while audio is actually going out
        self._done_event: threading.Event | None = None
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._stream.start()
        self._thread.start()

    def close(self) -> None:
        self._running = False
        self._queue.put(_END)
        self._thread.join(timeout=2)
        self._stream.stop()
        self._stream.close()

    # -- producing audio ---------------------------------------------------
    def play(self, pcm: bytes) -> None:
        """Queue a chunk of raw 16-bit PCM for playback."""
        self._interrupt.clear()
        self._queue.put(pcm)

    def mark_end(self) -> threading.Event:
        """Mark the end of the current response. The returned event is set once
        all queued audio has finished playing (or was interrupted)."""
        done = threading.Event()
        self._done_event = done
        self._queue.put(_END)
        return done

    # -- interruption ------------------------------------------------------
    def interrupt(self) -> None:
        """Stop and discard everything currently queued/playing (barge-in)."""
        self._interrupt.set()
        # Drain anything pending.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        # Drop audio already buffered in PortAudio for an instant stop.
        self._stream.abort()
        self._stream.start()
        self._playing.clear()
        if self._done_event is not None:
            self._done_event.set()
            self._done_event = None

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    # -- worker ------------------------------------------------------------
    def _worker(self) -> None:
        while self._running:
            item = self._queue.get()
            if item is _END:
                # End of a response: mark drained and signal any waiter.
                self._playing.clear()
                if self._done_event is not None:
                    self._done_event.set()
                    self._done_event = None
                continue
            if self._interrupt.is_set():
                continue  # dropped due to barge-in
            self._playing.set()
            samples = np.frombuffer(item, dtype=np.int16)
            try:
                self._stream.write(samples)
            except Exception as exc:  # pragma: no cover - device hiccups
                print(f"[audio-out] {exc}", flush=True)
