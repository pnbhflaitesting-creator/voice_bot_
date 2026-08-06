"""Gemini LLM client.

We reach the Gemini endpoint through the OpenAI-compatible chat-completions
wire format, so a single, well-supported client library streams tokens. Point
``GEMINI_BASE_URL`` at your endpoint (Google's compatibility URL is the default).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .config import settings


class GeminiLLM:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
        )
        # Conversation memory. Kept in-process for the session.
        self._history: list[dict[str, str]] = [
            {"role": "system", "content": settings.system_prompt}
        ]

    def reset(self) -> None:
        self._history = [{"role": "system", "content": settings.system_prompt}]

    async def stream_reply(self, user_text: str) -> AsyncIterator[str]:
        """Append the user's turn and yield the assistant reply token by token.

        The full assistant message is committed to history once streaming ends.
        """
        self._history.append({"role": "user", "content": user_text})

        stream = await self._client.chat.completions.create(
            model=settings.gemini_model,
            messages=self._history,
            stream=True,
        )

        pieces: list[str] = []
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                pieces.append(delta)
                yield delta

        self._history.append({"role": "assistant", "content": "".join(pieces)})
