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
from collections.abc import AsyncIterator

import numpy as np
from openai import AsyncOpenAI

from .audio_io import MicrophoneStream, SpeakerStream
from .config import settings
from .llm import GeminiLLM
from .stt import SpeechToText
from .tts import TextToSpeech
from .vad import TurnDetector, TurnEvent

# Split streamed text on sentence-ending punctuation so we can start speaking
# before the LLM has finished the whole reply.
_SENTENCE_END = re.compile(r"(.+?[.!?…]+[\s\"')\]]*)", re.DOTALL)
# Speak an in-progress buffer once it gets this long even without punctuation.
_MAX_CHARS_BEFORE_FLUSH = 200


async def _sentence_chunks(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Regroup a stream of LLM tokens into speakable sentence-sized chunks."""
    buffer = ""
    async for token in tokens:
        buffer += token
        # Emit every complete sentence currently in the buffer.
        while True:
            match = _SENTENCE_END.match(buffer)
            if match and (match.end() < len(buffer) or buffer.endswith((" ", "\n"))):
                chunk = match.group(1).strip()
                buffer = buffer[match.end():]
                if chunk:
                    yield chunk
                continue
            break
        if len(buffer) >= _MAX_CHARS_BEFORE_FLUSH and " " in buffer:
            head, buffer = buffer.rsplit(" ", 1)
            if head.strip():
                yield head.strip()
    if buffer.strip():
        yield buffer.strip()


class VoiceBot:
    def __init__(self) -> None:
        # One OpenAI client for STT + TTS.
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self._stt = SpeechToText(self._openai)
        self._tts = TextToSpeech(self._openai)
        self._llm = GeminiLLM()
        self._vad = TurnDetector()

        self._speaker = SpeakerStream()
        self._frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        self._mic: MicrophoneStream | None = None

        # The task handling the current turn (STT -> LLM -> TTS). Cancelled on
        # barge-in so a new turn can take over immediately.
        self._response_task: asyncio.Task | None = None
        self._bot_speaking = False

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._mic = MicrophoneStream(loop, self._frame_queue)
        self._speaker.start()
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
        try:
            transcript = await self._stt.transcribe(audio)
            if not transcript:
                return
            print(f"🧑 You:  {transcript}", flush=True)
            print("🤖 Bot:  ", end="", flush=True)

            tokens = self._llm.stream_reply(transcript)
            async for sentence in _sentence_chunks(tokens):
                print(sentence + " ", end="", flush=True)
                async for pcm in self._tts.stream(sentence):
                    self._speaker.play(pcm)
            print(flush=True)

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
        await self._openai.close()
