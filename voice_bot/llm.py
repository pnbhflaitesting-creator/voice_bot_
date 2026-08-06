"""LLM providers: Gemini and OpenAI.

Both speak the OpenAI chat-completions wire format, so one streaming client
serves both — they differ only in base URL, API key, model, and a couple of
provider-specific knobs. Select one with ``LLM_PROVIDER``; ``create_llm()``
builds it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from openai import AsyncOpenAI

from .config import settings


class LLMProvider(Protocol):
    def stream_reply(self, user_text: str) -> AsyncIterator[str]: ...
    def reset(self) -> None: ...
    async def warmup(self) -> None: ...
    async def aclose(self) -> None: ...


class OpenAICompatLLM:
    """Streaming chat over any OpenAI-compatible endpoint (OpenAI or Gemini)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        extra_body: dict | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._extra_body = extra_body or None
        self._history: list[dict[str, str]] = [
            {"role": "system", "content": settings.system_prompt}
        ]

    def reset(self) -> None:
        self._history = [{"role": "system", "content": settings.system_prompt}]

    def _kwargs(self, messages: list[dict[str, str]], **overrides) -> dict:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": settings.max_reply_tokens,
        }
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        kwargs.update(overrides)
        return kwargs

    async def warmup(self) -> None:
        try:
            await self._client.chat.completions.create(
                **self._kwargs(
                    [{"role": "user", "content": "hi"}], max_tokens=1, stream=False
                )
            )
        except Exception:
            pass

    async def stream_reply(self, user_text: str) -> AsyncIterator[str]:
        self._history.append({"role": "user", "content": user_text})
        stream = await self._client.chat.completions.create(
            **self._kwargs(self._history, stream=True)
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

    async def aclose(self) -> None:
        await self._client.close()


def create_llm() -> LLMProvider:
    provider = settings.llm_provider
    if provider == "gemini":
        extra_body = None
        if settings.gemini_reasoning_effort:
            # Sent via extra_body so the SDK forwards it raw (it otherwise
            # validates reasoning_effort and may reject "none").
            extra_body = {"reasoning_effort": settings.gemini_reasoning_effort}
        return OpenAICompatLLM(
            api_key=settings.gemini_api_key,
            base_url=settings.gemini_base_url,
            model=settings.gemini_model,
            extra_body=extra_body,
        )
    if provider == "openai":
        return OpenAICompatLLM(
            api_key=settings.openai_api_key,
            base_url=None,  # default OpenAI endpoint
            model=settings.openai_llm_model,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
