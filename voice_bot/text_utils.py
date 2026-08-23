"""Turn LLM markdown into clean text for TTS.

LLMs love to emit markdown (**bold**, `code`, [links](url), bullet lists, …).
Spoken verbatim, a TTS voice reads "star star" and "backtick". This strips the
formatting markers while keeping the words, so only the words are spoken. The UI
still receives the original markdown and renders it (bold, etc.).
"""

from __future__ import annotations

import re

_LINK = re.compile(r"!?\[([^\]]+)\]\([^)]*\)")          # [text](url) / ![alt](url)
_BOLD = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)       # **x** / __x__
_ITALIC = re.compile(r"(?<!\w)([*_])(.+?)\1(?!\w)", re.DOTALL)  # *x* / _x_
_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
_CODE = re.compile(r"`+([^`]*)`+")
_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_QUOTE = re.compile(r"(?m)^\s{0,3}>\s?")
_BULLET = re.compile(r"(?m)^\s*[-*+]\s+")
_NUMBERED = re.compile(r"(?m)^\s*\d+[.)]\s+")


def speakable(text: str) -> str:
    """Strip markdown formatting, returning plain text safe to read aloud."""
    text = _LINK.sub(r"\1", text)
    text = _BOLD.sub(r"\2", text)
    text = _ITALIC.sub(r"\2", text)
    text = _STRIKE.sub(r"\1", text)
    text = _CODE.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _BULLET.sub("", text)
    text = _NUMBERED.sub("", text)
    # Remove any leftover asterisks/backticks (e.g. a marker split across chunks).
    text = text.replace("`", "").replace("*", "")
    return text.strip()
