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

    def _create_kwargs(self, messages: list[dict[str, str]], **overrides) -> dict:
        kwargs: dict = {
            "model": settings.gemini_model,
            "messages": messages,
            "max_tokens": settings.max_reply_tokens,
        }
        # Disable/limit Gemini 2.5 "thinking" for a faster first token. Sent via
        # extra_body so the OpenAI SDK forwards it raw (it otherwise validates
        # reasoning_effort against a fixed set and may reject "none").
        if settings.gemini_reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": settings.gemini_reasoning_effort}
        kwargs.update(overrides)
        return kwargs

    async def warmup(self) -> None:
        """Establish the TLS connection and warm the model path so the first
        real turn doesn't pay cold-start latency. Errors are ignored."""
        try:
            await self._client.chat.completions.create(
                **self._create_kwargs(
                    [{"role": "user", "content": "hi"}], max_tokens=1, stream=False
                )
            )
        except Exception:
            pass

    async def stream_reply(self, user_text: str) -> AsyncIterator[str]:
        """Append the user's turn and yield the assistant reply token by token.

        The full assistant message is committed to history once streaming ends.
        """
        self._history.append({"role": "user", "content": user_text})

        stream = await self._client.chat.completions.create(
            **self._create_kwargs(self._history, stream=True)
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
