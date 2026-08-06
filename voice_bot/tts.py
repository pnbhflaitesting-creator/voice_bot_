"""Text-to-speech using OpenAI, streamed as raw PCM for low latency."""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .config import settings


class TextToSpeech:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield chunks of 24 kHz / 16-bit / mono PCM for the given text.

        ``response_format="pcm"`` avoids any client-side audio decoding, so the
        first bytes can reach the speaker as soon as the model emits them.
        """
        text = text.strip()
        if not text:
            return
        async with self._client.audio.speech.with_streaming_response.create(
            model=settings.tts_model,
            voice=settings.tts_voice,
            input=text,
            response_format="pcm",
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=4096):
                if chunk:
                    yield chunk
