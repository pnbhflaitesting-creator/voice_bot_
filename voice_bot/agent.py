"""Agentic tool layer: session memory + a set of demo tools.

The LLM is given these tools via native function-calling (works with both
Gemini and OpenAI over the OpenAI wire format — no MCP required). The model
decides when to call a tool; if a tool needs information the user didn't give
(e.g. a city for the weather), the model asks for it instead of guessing.

All tools here are keyless so the demo works out of the box:
  - get_weather(location)      real data via Open-Meteo (geocode + forecast)
  - get_current_time(timezone) local or named-timezone clock
  - calculate(expression)      safe arithmetic evaluator
  - web_search(query)          DuckDuckGo Instant Answer
  - remember / recall / list_memories   session memory (until Ctrl+C)

Session memory lives in-process for the whole session and is cleared when the
program exits.
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import math
import operator
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

import httpx

log = logging.getLogger(__name__)


def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """Real DuckDuckGo web search (synchronous). Raises if the library is absent.

    Supports both the current ``ddgs`` package and the older
    ``duckduckgo_search`` name.
    """
    try:
        from ddgs import DDGS  # current package name
    except ImportError:  # pragma: no cover - fallback to the old name
        from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))

# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------


class SessionMemory:
    """A simple key/value store that persists for the session (until Ctrl+C)."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def all(self) -> dict[str, str]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (for the calculate tool)
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau}
_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "abs": abs, "round": round, "min": min, "max": max,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


# WMO weather codes -> human descriptions (Open-Meteo).
_WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain", 71: "light snow",
    73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers",
    81: "showers", 82: "violent showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # JSON schema for the arguments
    run: Callable[..., Awaitable[str]]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Agent:
    """Owns the tools, the shared HTTP client, and the session memory."""

    def __init__(self) -> None:
        self.memory = SessionMemory()
        self._http = httpx.AsyncClient(timeout=15.0)
        self._tools: dict[str, Tool] = {t.name: t for t in self._build_tools()}

    # -- interface used by the LLM ----------------------------------------
    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    async def execute(self, name: str, arguments: dict) -> str:
        tool = self._tools.get(name)
        arg_str = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        print(f"\n🛠️  {name}({arg_str})", flush=True)
        log.info("tool call: %s(%s)", name, arg_str)
        t = time.perf_counter()
        if tool is None:
            log.warning("tool call: unknown tool %s", name)
            return f"Error: no tool named {name}."
        try:
            result = await tool.run(**arguments)
        except TypeError as exc:  # bad/missing arguments from the model
            log.warning("tool %s bad arguments: %s", name, exc)
            return f"Error: bad arguments for {name}: {exc}"
        except Exception as exc:  # pragma: no cover - network/tool errors
            log.exception("tool %s failed", name)
            return f"Error running {name}: {exc}"
        log.info("tool %s -> %r (%.0fms)", name, result, (time.perf_counter() - t) * 1000)
        return result

    async def aclose(self) -> None:
        await self._http.aclose()

    # -- tool registry ----------------------------------------------------
    def _build_tools(self) -> list[Tool]:
        return [
            Tool(
                "get_weather",
                "Get the current weather for a location (city, town, or place). "
                "If the user did not specify a location, ask them for one instead "
                "of calling this tool.",
                {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or place name, e.g. 'Istanbul' or 'Paris, France'.",
                        }
                    },
                    "required": ["location"],
                },
                self._get_weather,
            ),
            Tool(
                "get_current_time",
                "Get the current date and time, optionally for a named IANA timezone.",
                {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone like 'Europe/Istanbul'. Omit for local time.",
                        }
                    },
                    "required": [],
                },
                self._get_current_time,
            ),
            Tool(
                "calculate",
                "Evaluate an arithmetic expression, e.g. '2*(3+4)' or 'sqrt(144)'.",
                {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "The math expression."}
                    },
                    "required": ["expression"],
                },
                self._calculate,
            ),
            Tool(
                "web_search",
                "Look up a quick factual answer from the web (DuckDuckGo).",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for."}
                    },
                    "required": ["query"],
                },
                self._web_search,
            ),
            Tool(
                "remember",
                "Store a fact for the rest of this session, e.g. the user's name or a preference.",
                {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "What the fact is about, e.g. 'name'."},
                        "value": {"type": "string", "description": "The value to remember."},
                    },
                    "required": ["key", "value"],
                },
                self._remember,
            ),
            Tool(
                "recall",
                "Retrieve a fact stored earlier this session.",
                {
                    "type": "object",
                    "properties": {"key": {"type": "string", "description": "What to recall."}},
                    "required": ["key"],
                },
                self._recall,
            ),
            Tool(
                "list_memories",
                "List everything remembered so far this session.",
                {"type": "object", "properties": {}, "required": []},
                self._list_memories,
            ),
        ]

    # -- tool implementations --------------------------------------------
    async def _get_weather(self, location: str = "") -> str:
        location = (location or "").strip()
        if not location:
            return "No location was given. Ask the user which city they want the weather for."
        geo = await self._http.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en"},
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return f"I couldn't find a place called {location}."
        place = results[0]
        wx = await self._http.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
            },
        )
        wx.raise_for_status()
        cur = wx.json()["current"]
        desc = _WMO.get(cur["weather_code"], "unknown conditions")
        name = place["name"]
        country = place.get("country", "")
        return (
            f"Weather in {name}{', ' + country if country else ''}: {desc}, "
            f"{cur['temperature_2m']}°C (feels like {cur['apparent_temperature']}°C), "
            f"humidity {cur['relative_humidity_2m']}%, wind {cur['wind_speed_10m']} km/h."
        )

    async def _get_current_time(self, timezone: str = "") -> str:
        tz = None
        if timezone:
            try:
                from zoneinfo import ZoneInfo

                tz = ZoneInfo(timezone)
            except Exception:
                return f"I don't recognise the timezone '{timezone}'."
        now = datetime.now(tz)
        where = timezone or "local time"
        return now.strftime(f"It's %A, %B %-d, %Y at %-I:%M %p ({where}).")

    async def _calculate(self, expression: str) -> str:
        try:
            value = _safe_eval(ast.parse(expression, mode="eval"))
        except Exception:
            return f"I couldn't evaluate '{expression}'."
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return f"{expression} = {value}"

    async def _web_search(self, query: str) -> str:
        # Real web search via DuckDuckGo SERP (the `ddgs` package), which returns
        # actual result pages — unlike the Instant Answer API, which only covers
        # entities/definitions. Runs in a thread since the library is sync.
        try:
            results = await asyncio.to_thread(_ddg_search, query, 5)
        except Exception as exc:
            log.warning("web_search via ddgs failed: %s", exc)
            results = None

        if results:
            lines = [f"Top web results for '{query}':"]
            for i, r in enumerate(results, 1):
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                lines.append(f"{i}. {title} — {body}")
            return "\n".join(lines)

        # Fallback: DuckDuckGo Instant Answer (works for definitions/entities).
        resp = await self._http.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("AbstractText"):
            return data["AbstractText"]
        if data.get("Answer"):
            return str(data["Answer"])
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text"):
                return topic["Text"]
        return (
            f"I couldn't find results for '{query}'. "
            "(Install the 'ddgs' package for full web search.)"
        )

    async def _remember(self, key: str, value: str) -> str:
        self.memory.set(key, value)
        return f"Got it — I'll remember that {key} is {value}."

    async def _recall(self, key: str) -> str:
        value = self.memory.get(key)
        if value is None:
            return f"I don't have anything stored for '{key}' yet."
        return f"{key} is {value}."

    async def _list_memories(self) -> str:
        data = self.memory.all()
        if not data:
            return "I'm not remembering anything yet this session."
        return "Here's what I remember: " + "; ".join(f"{k}: {v}" for k, v in data.items()) + "."


def _parse_arguments(raw: str) -> dict:
    """Parse tool-call arguments JSON, tolerating empty/malformed strings."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
