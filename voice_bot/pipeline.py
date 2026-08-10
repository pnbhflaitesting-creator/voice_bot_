"""The real-time voice pipeline.

Flow per user turn:

    mic -> VAD/turn-detection -> STT (OpenAI) -> LLM (Gemini, streamed)
        -> sentence chunks -> TTS (OpenAI, streamed PCM) -> speaker

While the bot is speaking, the VAD keeps running. If the user starts talking
(barge-in) the current response is cancelled and playback is stopped instantly.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator

import numpy as np

from .audio_io import MicrophoneStream, SpeakerStream
from .config import settings
from .denoise import Denoiser
from .llm import create_llm
from .recorder import TurnRecorder
from .stt import create_stt
from .tts import create_tts
from .vad import TurnDetector, TurnEvent

# Boundaries: punctuation followed by whitespace/closing bracket. Requiring the
# trailing whitespace means we only split on a *confirmed* boundary (so "3.14"
# or "e.g" mid-token is never cut) and never on a period we haven't seen past.
_SENTENCE_BOUNDARY = re.compile(r"[.!?…]+[\s\"')\]]+")
# The first chunk may also break at a clause boundary (comma/semicolon/colon).
_CLAUSE_BOUNDARY = re.compile(r"[.!?…,;:]+[\s\"')\]]+")
# Emit the first chunk at the earliest boundary at or past this length, so the
# bot starts talking quickly without sounding choppy.
_FIRST_CHUNK_MIN_CHARS = 24
# Speak an in-progress buffer once it gets this long even without punctuation.
_MAX_CHARS_BEFORE_FLUSH = 200


def _next_break(buffer: str, first: bool) -> int | None:
    """Index just past the first usable boundary in ``buffer``, or None.

    For the first chunk we break at the earliest clause boundary at or past
    ``_FIRST_CHUNK_MIN_CHARS`` (low latency); afterwards only at sentence ends
    (natural prosody). A boundary counts only if some text follows it.
    """
    pattern = _CLAUSE_BOUNDARY if first else _SENTENCE_BOUNDARY
    min_chars = _FIRST_CHUNK_MIN_CHARS if first else 1
    for match in pattern.finditer(buffer):
        end = match.end()
        if end >= min_chars and end < len(buffer):
            return end
    return None


async def _sentence_chunks(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Regroup a stream of LLM tokens into speakable chunks.

    The first chunk is emitted at the earliest clause boundary (to minimise
    time-to-first-audio); later chunks are full sentences (for natural prosody).
    """
    buffer = ""
    first = True
    async for token in tokens:
        buffer += token
        while True:
            idx = _next_break(buffer, first)
            if idx is None:
                break
            chunk = buffer[:idx].strip()
            buffer = buffer[idx:].lstrip()
            if chunk:
                first = False
                yield chunk
        if len(buffer) >= _MAX_CHARS_BEFORE_FLUSH and " " in buffer:
            head, buffer = buffer.rsplit(" ", 1)
            if head.strip():
                first = False
                yield head.strip()
    if buffer.strip():
        yield buffer.strip()


class VoiceBot:
    def __init__(self) -> None:
        # Providers are chosen by env (STT_PROVIDER / TTS_PROVIDER / LLM_PROVIDER).
        self._stt = create_stt()
        self._tts = create_tts()
        self._llm = create_llm()
        self._vad = TurnDetector()
        self._denoiser = Denoiser()
        self._recorder = TurnRecorder()

        stt_lang = settings.stt_language or "auto"
        reply_lang = settings.response_language or "match input"
        print(
            f"🔌 STT={settings.stt_provider} (lang={stt_lang}) · "
            f"TTS={settings.tts_provider} · "
            f"LLM={settings.llm_provider} (reply={reply_lang})",
            flush=True,
        )
        if self._denoiser.enabled:
            bits = []
            if settings.highpass_hz > 0:
                bits.append(f"high-pass {settings.highpass_hz}Hz")
            if settings.denoise == "spectral":
                bits.append(f"spectral {settings.denoise_strength}")
            print(f"🧹 Noise suppression: {', '.join(bits)}", flush=True)
        if self._recorder.enabled:
            print(f"💾 Saving each turn (raw+clean audio & transcript) to {settings.turns_dir}/", flush=True)

        self._speaker = SpeakerStream()
        self._frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        self._mic: MicrophoneStream | None = None

        # The task handling the current turn (STT -> LLM -> TTS). Cancelled on
        # barge-in so a new turn can take over immediately.
        self._response_task: asyncio.Task | None = None
        self._bot_speaking = False

    async def _warmup(self) -> None:
        """Open connections to all providers up front so the FIRST turn doesn't
        pay TLS/DNS/model cold-start (often several seconds). Runs concurrently."""
        await asyncio.gather(
            self._stt.warmup(), self._tts.warmup(), self._llm.warmup()
        )

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._mic = MicrophoneStream(loop, self._frame_queue)
        self._speaker.start()

        print("⏳ Warming up connections…", flush=True)
        await self._warmup()

        self._mic.start()
        print("🎙️  Listening… (speak into the mic, Ctrl+C to quit)\n", flush=True)
        try:
            await self._listen_loop()
        finally:
            await self._shutdown()

    async def _listen_loop(self) -> None:
        while True:
            frame = await self._frame_queue.get()

            # Half-duplex mode (no barge-in): while the bot is speaking, ignore
            # the microphone entirely. Without echo cancellation the mic hears
            # the bot's own voice from the speakers; feeding that to the VAD/STT
            # makes the bot transcribe and answer itself. Dropping frames here
            # (and resetting the VAD when the bot finishes) prevents that loop.
            if self._bot_speaking and not settings.allow_interruptions:
                continue

            result = self._vad.process(frame)

            if result.event is TurnEvent.SPEECH_START and self._bot_speaking:
                if settings.allow_interruptions:
                    # Barge-in: user started talking over the bot.
                    await self._interrupt()

            elif result.event is TurnEvent.TURN_END and result.audio is not None:
                if self._bot_speaking:
                    # Turn finished while bot audio was still playing -> barge-in.
                    await self._interrupt()
                # Mark busy immediately so a follow-up turn can't spawn a second
                # overlapping response during STT (before audio starts playing).
                self._bot_speaking = True
                # Handle this turn concurrently so we keep reading the mic.
                self._response_task = asyncio.create_task(self._handle_turn(result.audio))

    async def _handle_turn(self, audio: np.ndarray) -> None:
        t0 = time.perf_counter()
        try:
            raw_audio = audio  # what the mic heard, before denoising
            if self._denoiser.enabled:
                audio = await asyncio.to_thread(self._denoiser.process, audio)
            transcript = await self._stt.transcribe(audio)

            # Save raw + denoised audio and the transcript for inspection. Done
            # even when the transcript is empty — that's when you most want to
            # hear what the model was given.
            if self._recorder.enabled:
                stem = await asyncio.to_thread(
                    self._recorder.save, raw_audio, audio, transcript
                )
                print(f"💾 saved {settings.turns_dir}/{stem}_[raw|clean].wav + .txt", flush=True)

            if not transcript:
                return
            t_stt = time.perf_counter()
            print(f"🧑 You:  {transcript}", flush=True)
            print("🤖 Bot:  ", end="", flush=True)

            t_first_token: float | None = None
            t_first_audio: float | None = None

            async def _timed_tokens() -> AsyncIterator[str]:
                nonlocal t_first_token
                async for tok in self._llm.stream_reply(transcript):
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    yield tok

            async for sentence in _sentence_chunks(_timed_tokens()):
                print(sentence + " ", end="", flush=True)
                async for pcm in self._tts.stream(sentence):
                    if t_first_audio is None:
                        t_first_audio = time.perf_counter()
                    self._speaker.play(pcm)
            print(flush=True)

            if settings.show_timings and t_first_audio is not None:
                stt_ms = (t_stt - t0) * 1000
                llm_ms = ((t_first_token or t_stt) - t_stt) * 1000
                tts_ms = (t_first_audio - (t_first_token or t_stt)) * 1000
                clip_s = len(audio) / settings.input_sample_rate
                print(
                    f"⏱  clip {clip_s:.1f}s · stt {stt_ms:.0f}ms · llm {llm_ms:.0f}ms · "
                    f"tts {tts_ms:.0f}ms · to-first-audio {(t_first_audio - t0) * 1000:.0f}ms",
                    flush=True,
                )

            # Wait for all queued audio to finish playing before listening again.
            done = self._speaker.mark_end()
            await asyncio.to_thread(done.wait)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - surface API/network errors
            print(f"\n[error] {exc}", flush=True)
        finally:
            if not settings.allow_interruptions:
                # Discard mic frames captured during playback (they contain the
                # bot's own audio) and clear VAD state before we start listening.
                while not self._frame_queue.empty():
                    self._frame_queue.get_nowait()
                self._vad.reset()
            self._bot_speaking = False

    async def _interrupt(self) -> None:
        """Cancel the in-flight response and stop playback."""
        self._speaker.interrupt()
        task = self._response_task
        self._response_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._bot_speaking = False

    async def _shutdown(self) -> None:
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
        if self._mic:
            self._mic.stop()
        self._speaker.close()
        for provider in (self._stt, self._tts, self._llm):
            try:
                await provider.aclose()
            except Exception:
                pass
