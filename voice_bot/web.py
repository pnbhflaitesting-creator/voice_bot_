"""Optional web UI for the voice bot.

Runs the normal audio pipeline (mic + speaker stay on the machine running this
process) and, alongside it, a small FastAPI server that:

  * serves a single-page UI,
  * streams live events (transcripts, replies, tool calls, timings, status) to
    the browser over a WebSocket, and
  * accepts commands back: typed messages, and mic on/off / mode switches.

Launch with ``python -m voice_bot --web``. Open http://127.0.0.1:8000 on the
same machine (audio plays through this machine's speakers).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

# FastAPI is imported at module level (this module is only imported for the web
# feature). Importing it here — not inside create_app — is required so FastAPI
# can resolve the `sock: WebSocket` annotation, which is a string under
# `from __future__ import annotations`.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .config import settings

if TYPE_CHECKING:
    from .pipeline import VoiceBot

log = logging.getLogger(__name__)

_UI_FILE = Path(__file__).parent / "web_ui.html"


def create_app(bot: "VoiceBot") -> FastAPI:
    app = FastAPI()

    @app.get("/")
    async def index() -> "HTMLResponse":
        return HTMLResponse(_UI_FILE.read_text(encoding="utf-8"))

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        queue = bot.events.subscribe()

        # Replay the conversation so far to this fresh client.
        for event in bot.events.history():
            await sock.send_json(event)
        await sock.send_json({"type": "status", "status": "listening"})

        async def pump() -> None:
            while True:
                event = await queue.get()
                await sock.send_json(event)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                msg = await sock.receive_json()
                await _handle_client_message(bot, msg)
        except WebSocketDisconnect:
            pass
        except Exception:  # pragma: no cover - client hiccups
            log.exception("websocket error")
        finally:
            pump_task.cancel()
            bot.events.unsubscribe(queue)

    return app


async def _handle_client_message(bot: "VoiceBot", msg: dict) -> None:
    kind = msg.get("type")
    if kind == "user_text":
        asyncio.create_task(bot.respond_to_text(msg.get("text", "")))
    elif kind == "mic":
        bot.set_mic_enabled(bool(msg.get("on")))
    elif kind == "mode":
        bot.set_mode(str(msg.get("mode", "voice")))


async def run_web() -> None:
    """Run the pipeline and the web server together in one event loop."""
    import uvicorn

    from .pipeline import VoiceBot
    bot = VoiceBot()
    app = create_app(bot)
    config = uvicorn.Config(
        app, host=settings.web_host, port=settings.web_port, log_level="warning"
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # let __main__ own Ctrl+C

    url = f"http://{settings.web_host}:{settings.web_port}"
    print(f"🌐 Web UI at {url}  (audio plays on this machine)", flush=True)

    await asyncio.gather(bot.run(), server.serve())
