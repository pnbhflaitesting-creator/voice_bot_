"""LLM providers: Gemini and OpenAI, with optional agentic tool-calling.

Both speak the OpenAI chat-completions wire format, so one streaming client
serves both — they differ only in base URL, API key, model, and a couple of
provider-specific knobs. Select one with ``LLM_PROVIDER``; ``create_llm()``
builds it.

When an :class:`~voice_bot.agent.Agent` is supplied, the model is offered the
agent's tools. ``stream_reply`` runs a tool-calling loop: it streams a turn,
and if the model requested tool calls it executes them, appends the results,
and streams again — repeating until the model produces a spoken answer. Normal
(no-tool) turns still stream token-by-token for low latency.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol

from openai import AsyncOpenAI

from .agent import _parse_arguments
from .config import settings

if TYPE_CHECKING:
    from .agent import Agent

# Safety cap on how many tool rounds a single turn may take.
_MAX_TOOL_ROUNDS = 5


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
        agent: "Agent | None" = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._extra_body = extra_body or None
        self._agent = agent
        self._history: list[dict] = [
            {"role": "system", "content": settings.effective_system_prompt}
        ]

    def reset(self) -> None:
        self._history = [{"role": "system", "content": settings.effective_system_prompt}]

    def _kwargs(self, messages: list[dict], **overrides) -> dict:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": settings.max_reply_tokens,
            "temperature": settings.temperature,
        }
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        if self._agent is not None:
            kwargs["tools"] = self._agent.schemas()
            kwargs["tool_choice"] = "auto"
        kwargs.update(overrides)
        return kwargs

    async def warmup(self) -> None:
        try:
            await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False,
                **({"extra_body": self._extra_body} if self._extra_body else {}),
            )
        except Exception:
            pass

    async def stream_reply(self, user_text: str) -> AsyncIterator[str]:
        self._history.append({"role": "user", "content": user_text})

        for _ in range(_MAX_TOOL_ROUNDS):
            stream = await self._client.chat.completions.create(
                **self._kwargs(self._history, stream=True)
            )

            content_parts: list[str] = []
            # index -> {"id", "name", "args"} accumulated across stream deltas
            tool_calls: dict[int, dict] = {}

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                    yield delta.content
                for tcd in delta.tool_calls or []:
                    acc = tool_calls.setdefault(tcd.index, {"id": None, "name": "", "args": ""})
                    if tcd.id:
                        acc["id"] = tcd.id
                    if tcd.function and tcd.function.name:
                        acc["name"] = tcd.function.name
                    if tcd.function and tcd.function.arguments:
                        acc["args"] += tcd.function.arguments

            if not tool_calls or self._agent is None:
                # Plain answer — commit it and we're done.
                self._history.append({"role": "assistant", "content": "".join(content_parts)})
                return

            # The model asked to call tools. Record the request, run them, and
            # loop again so it can answer using the results.
            ordered = [tool_calls[i] for i in sorted(tool_calls)]
            for i, tc in enumerate(ordered):
                if not tc["id"]:
                    tc["id"] = f"call_{i}"
            self._history.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["args"] or "{}"},
                        }
                        for tc in ordered
                    ],
                }
            )
            # Run all requested tools concurrently (e.g. several web searches at
            # once), preserving order when appending their results.
            results = await asyncio.gather(
                *(
                    self._agent.execute(tc["name"], _parse_arguments(tc["args"]))
                    for tc in ordered
                )
            )
            for tc, result in zip(ordered, results):
                self._history.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result}
                )

        # Ran out of tool rounds without a final answer.
        yield "Sorry, I got stuck trying to do that."

    async def aclose(self) -> None:
        await self._client.close()


def create_llm(agent: "Agent | None" = None) -> LLMProvider:
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
            agent=agent,
        )
    if provider == "openai":
        extra_body = None
        if settings.openai_reasoning_effort:
            # e.g. "minimal" to turn reasoning off on gpt-5 / o-series. Sent via
            # extra_body so the SDK forwards it raw. Omit for non-reasoning models.
            extra_body = {"reasoning_effort": settings.openai_reasoning_effort}
        return OpenAICompatLLM(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,  # custom endpoint or default
            model=settings.openai_llm_model,
            extra_body=extra_body,
            agent=agent,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")
