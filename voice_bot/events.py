"""A tiny async pub/sub event bus.

The pipeline emits events (status changes, messages, tool calls, timings). The
web UI subscribes and forwards them to browsers over a WebSocket. In terminal-
only mode there are simply no subscribers, so ``emit`` is a cheap no-op.
"""

from __future__ import annotations

import asyncio
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        # Keep the recent history so a newly connected browser can render the
        # conversation so far.
        self._history: list[dict[str, Any]] = []
        self._history_limit = 200

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def emit(self, event: dict[str, Any]) -> None:
        # Remember conversation-relevant events for replay to new clients.
        if event.get("type") in {"message", "tool", "timings"}:
            self._history.append(event)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit :]
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
