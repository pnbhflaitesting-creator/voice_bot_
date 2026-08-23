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

    # Write audio in small sub-chunks so an interrupt takes effect within a
    # few tens of milliseconds (~20 ms at 24 kHz mono = 480 samples).
    _SUBCHUNK = 480

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._stream = sd.OutputStream(
            samplerate=settings.output_sample_rate,
            channels=1,
            dtype="int16",
            device=settings.output_device,
            latency="low",  # keep the device buffer small for snappy stops
        )
        # Guards every PortAudio call on the output stream so stream control
        # (stop/close) never races with the worker's write().
        self._stream_lock = threading.Lock()
        # Leftover odd byte carried between chunks. Some TTS providers (e.g.
        # Deepgram) stream raw PCM in chunks that don't align to the 2-byte
        # int16 sample boundary, so a chunk can have an odd byte count.
        self._carry = b""
        self._interrupt = threading.Event()
        self._playing = threading.Event()  # set while audio is actually going out
        self._done_event: threading.Event | None = None
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        # Reference tap for acoustic echo cancellation: a FIFO of the played
        # audio, resampled to the mic rate. Only built when enabled.
        self._ref_enabled = False
        self._ref = np.zeros(0, dtype=np.float32)
        self._ref_lock = threading.Lock()
        self._ref_max = settings.input_sample_rate  # cap at ~1s

    def enable_reference(self) -> None:
        """Start capturing played audio as an AEC reference (mic sample rate)."""
        self._ref_enabled = True

    def pull_reference(self, n: int) -> np.ndarray:
        """Return the next ``n`` reference samples (played audio @ mic rate),
        zero-padded if fewer are available. Consumed in lock-step with mic
        frames so the reference stays aligned with what the mic just heard."""
        with self._ref_lock:
            if self._ref.size >= n:
                out = self._ref[:n]
                self._ref = self._ref[n:]
            else:
                out = np.concatenate([self._ref, np.zeros(n - self._ref.size, np.float32)])
                self._ref = np.zeros(0, dtype=np.float32)
        return out

    def _append_reference(self, samples: np.ndarray) -> None:
        # Resample the played int16 (24 kHz) chunk to float32 at the mic rate.
        x = samples.astype(np.float32) / 32768.0
        m = int(round(x.size * settings.input_sample_rate / settings.output_sample_rate))
        if m <= 0:
            return
        ref16 = np.interp(
            np.linspace(0, x.size - 1, m), np.arange(x.size), x
        ).astype(np.float32)
        with self._ref_lock:
            self._ref = np.concatenate([self._ref, ref16])
            if self._ref.size > self._ref_max:  # keep the most recent ~1s
                self._ref = self._ref[-self._ref_max:]

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._stream.start()
        self._thread.start()

    def close(self) -> None:
        self._running = False
        self._interrupt.set()  # stop the worker writing any more audio
        self._queue.put(_END)
        self._thread.join(timeout=2)
        with self._stream_lock:
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
        """Stop and discard everything currently queued/playing (barge-in).

        We set a flag rather than abort()/start() the PortAudio stream: aborting
        from this thread while the worker is mid-write races inside ALSA and
        throws ``PaErrorCode -9999``. The worker checks the flag between small
        sub-chunks, so playback halts within ~20 ms without touching the stream
        from two threads at once.
        """
        self._interrupt.set()
        self._carry = b""
        with self._ref_lock:  # nothing more will be played -> drop the reference
            self._ref = np.zeros(0, dtype=np.float32)
        # Drain anything pending.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
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
                self._carry = b""  # drop any dangling odd byte
                self._playing.clear()
                if self._done_event is not None:
                    self._done_event.set()
                    self._done_event = None
                continue
            if self._interrupt.is_set():
                self._carry = b""
                continue  # dropped due to barge-in
            self._playing.set()
            # Align to 2-byte int16 samples, carrying any leftover odd byte to
            # the next chunk (raw PCM chunks may split mid-sample).
            data = self._carry + item
            usable = len(data) - (len(data) % 2)
            self._carry = data[usable:]
            samples = np.frombuffer(data[:usable], dtype=np.int16)
            if self._ref_enabled and samples.size:
                self._append_reference(samples)
            # Write in small sub-chunks, checking for interruption between each
            # so a barge-in stops playback almost immediately.
            for start in range(0, len(samples), self._SUBCHUNK):
                if self._interrupt.is_set() or not self._running:
                    break
                sub = samples[start:start + self._SUBCHUNK]
                try:
                    with self._stream_lock:
                        if not self._running:
                            break
                        self._stream.write(sub)
                except Exception as exc:  # pragma: no cover - device hiccups
                    print(f"[audio-out] {exc}", flush=True)
                    break
