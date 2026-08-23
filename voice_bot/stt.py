"""Speech-to-text providers: OpenAI, Deepgram, ElevenLabs.

Each provider takes a float32 mono @ 16 kHz utterance and returns a transcript.
Select one with ``STT_PROVIDER`` (see config). ``create_stt()`` builds it.

Note: these are batch (whole-utterance) transcribers — the clip is sent after
the turn ends. For the lowest possible STT latency you'd want *streaming* STT
(Deepgram's WebSocket), which transcribes while the user is still talking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Protocol
from urllib.parse import urlencode

import httpx
import numpy as np
from openai import AsyncOpenAI

from .audio_utils import float_to_pcm16, to_wav_bytes
from .config import settings

log = logging.getLogger(__name__)


class STTProvider(Protocol):
    # True for providers that consume live frames via feed() during the turn.
    streaming: bool

    async def transcribe(self, audio: np.ndarray) -> str: ...
    async def warmup(self) -> None: ...
    async def aclose(self) -> None: ...
    async def feed(self, frame: np.ndarray) -> None: ...


class OpenAISTT:
    streaming = False

    async def feed(self, frame: np.ndarray) -> None:  # not used by batch STT
        return

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = to_wav_bytes(audio, settings.input_sample_rate)
        kwargs: dict = {
            "model": settings.openai_stt_model,
            "file": ("speech.wav", wav_bytes, "audio/wav"),
        }
        if settings.stt_language:
            kwargs["language"] = settings.stt_language
        if settings.stt_prompt:
            kwargs["prompt"] = settings.stt_prompt
        resp = await self._client.audio.transcriptions.create(**kwargs)
        return (resp.text or "").strip()

    async def warmup(self) -> None:
        try:
            await self._client.models.list()
        except Exception:
            pass

    async def aclose(self) -> None:
        await self._client.close()


class DeepgramSTT:
    """Deepgram prerecorded transcription (REST)."""

    _URL = "https://api.deepgram.com/v1/listen"
    streaming = False

    async def feed(self, frame: np.ndarray) -> None:  # not used by batch STT
        return

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Token {settings.deepgram_api_key}"},
            timeout=30.0,
        )

    async def transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = to_wav_bytes(audio, settings.input_sample_rate)
        params = {"model": settings.deepgram_stt_model, "smart_format": "true"}
        if settings.stt_language:
            params["language"] = settings.stt_language
        resp = await self._client.post(
            self._URL,
            params=params,
            content=wav_bytes,
            headers={"Content-Type": "audio/wav"},
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data["results"]["channels"][0]["alternatives"][0]["transcript"]
        ).strip()

    async def warmup(self) -> None:
        try:
            await self._client.get("https://api.deepgram.com/v1/auth/token")
        except Exception:
            pass

    async def aclose(self) -> None:
        await self._client.aclose()


class ElevenLabsSTT:
    """ElevenLabs Scribe speech-to-text (REST)."""

    _URL = "https://api.elevenlabs.io/v1/speech-to-text"
    streaming = False

    async def feed(self, frame: np.ndarray) -> None:  # not used by batch STT
        return

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            headers={"xi-api-key": settings.elevenlabs_api_key},
            timeout=30.0,
        )

    async def transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = to_wav_bytes(audio, settings.input_sample_rate)
        data = {"model_id": settings.elevenlabs_stt_model}
        if settings.stt_language:
            data["language_code"] = settings.stt_language
        resp = await self._client.post(
            self._URL,
            data=data,
            files={"file": ("speech.wav", wav_bytes, "audio/wav")},
        )
        resp.raise_for_status()
        return (resp.json().get("text") or "").strip()

    async def warmup(self) -> None:
        try:
            await self._client.get("https://api.elevenlabs.io/v1/models")
        except Exception:
            pass

    async def aclose(self) -> None:
        await self._client.aclose()


class DeepgramStreamingSTT:
    """Deepgram real-time transcription over a WebSocket.

    Frames are streamed to Deepgram *while the user is still speaking* (via
    ``feed``), so by the time the turn ends there's almost nothing left to do:
    ``transcribe`` sends a Finalize, waits briefly for the last words, and
    returns. If the socket is unavailable it transparently falls back to the
    batch REST API on the buffered audio, so a demo never hard-fails.
    """

    streaming = True

    def __init__(self) -> None:
        params = {
            "model": settings.deepgram_stt_model,
            "encoding": "linear16",
            "sample_rate": str(settings.input_sample_rate),
            "channels": "1",
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
        }
        if settings.stt_language:
            params["language"] = settings.stt_language
        self._url = "wss://api.deepgram.com/v1/listen?" + urlencode(params)
        self._batch_fallback = DeepgramSTT()  # safety net

        self._ws = None
        self._recv_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._finals: list[str] = []
        self._final_event = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._connecting = False
        self._last_send = 0.0
        # How long transcribe() waits after Finalize for the last words.
        self._finalize_timeout = 0.6

    # -- connection -------------------------------------------------------
    def _ws_ok(self) -> bool:
        return self._ws is not None and not getattr(self._ws, "closed", False)

    async def _connect(self) -> None:
        if self._connecting or self._ws_ok():
            return
        self._connecting = True
        try:
            import websockets  # lazy: only needed for streaming
        except ImportError:
            log.warning("STT_PROVIDER=deepgram-stream needs 'websockets' "
                        "(pip install websockets); using batch fallback.")
            self._connecting = False
            return
        headers = {"Authorization": f"Token {settings.deepgram_api_key}"}
        try:
            try:
                self._ws = await websockets.connect(self._url, additional_headers=headers)
            except TypeError:  # older websockets uses extra_headers
                self._ws = await websockets.connect(self._url, extra_headers=headers)
            self._finals.clear()
            self._recv_task = asyncio.create_task(self._receiver())
            self._keepalive_task = asyncio.create_task(self._keepalive())
            log.info("deepgram streaming: connected")
        except Exception as exc:
            log.warning("deepgram streaming: connect failed (%s); batch fallback", exc)
            self._ws = None
        finally:
            self._connecting = False

    async def _receiver(self) -> None:
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                except (ValueError, TypeError):
                    continue
                if data.get("type") != "Results":
                    continue
                alts = data.get("channel", {}).get("alternatives", [])
                text = alts[0].get("transcript", "").strip() if alts else ""
                if text and data.get("is_final"):
                    self._finals.append(text)
                    self._final_event.set()
        except Exception:
            pass
        finally:
            self._ws = None  # mark broken so we reconnect / fall back

    async def _keepalive(self) -> None:
        # Keep the socket alive during long silences (e.g. while the bot speaks).
        try:
            while self._ws_ok():
                await asyncio.sleep(5)
                if time.monotonic() - self._last_send > 4 and self._ws_ok():
                    async with self._send_lock:
                        try:
                            await self._ws.send(json.dumps({"type": "KeepAlive"}))
                        except Exception:
                            break
        except asyncio.CancelledError:
            pass

    # -- interface --------------------------------------------------------
    async def warmup(self) -> None:
        await self._connect()

    def reset(self) -> None:
        """Drop any accumulated transcript (e.g. after a barge-in or when the
        bot starts speaking) so stale words never leak into the next turn."""
        self._finals.clear()
        self._final_event.clear()

    async def feed(self, frame: np.ndarray) -> None:
        if not self._ws_ok():
            return
        pcm = float_to_pcm16(frame).tobytes()
        async with self._send_lock:
            try:
                await self._ws.send(pcm)
                self._last_send = time.monotonic()
            except Exception:
                self._ws = None  # broken; reconnect in the background
                asyncio.create_task(self._connect())

    async def transcribe(self, audio: np.ndarray) -> str:
        if not self._ws_ok():
            asyncio.create_task(self._connect())  # try to restore for next turn
            return await self._batch_fallback.transcribe(audio)

        self._final_event.clear()
        async with self._send_lock:
            try:
                await self._ws.send(json.dumps({"type": "Finalize"}))
            except Exception:
                self._ws = None
                return await self._batch_fallback.transcribe(audio)

        # Wait (bounded) for the finalize to flush the last words.
        try:
            await asyncio.wait_for(self._final_event.wait(), timeout=self._finalize_timeout)
        except asyncio.TimeoutError:
            pass

        text = " ".join(self._finals).strip()
        self._finals.clear()
        if not text:
            # Nothing came through the stream — fall back to batch on the clip.
            return await self._batch_fallback.transcribe(audio)
        return text

    async def aclose(self) -> None:
        for task in (self._recv_task, self._keepalive_task):
            if task:
                task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        await self._batch_fallback.aclose()


def create_stt() -> STTProvider:
    provider = settings.stt_provider
    if provider == "openai":
        return OpenAISTT()
    if provider == "deepgram":
        return DeepgramSTT()
    if provider == "deepgram-stream":
        return DeepgramStreamingSTT()
    if provider == "elevenlabs":
        return ElevenLabsSTT()
    raise ValueError(f"Unknown STT provider: {provider}")
