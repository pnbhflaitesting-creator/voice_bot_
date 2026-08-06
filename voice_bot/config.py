"""Central configuration for the voice bot.

All values are read from environment variables (see ``.env.example``) so the
same code runs in different setups without edits. Import ``settings`` anywhere
you need a value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load a local .env file if present. Real environment variables always win.
load_dotenv()


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _get_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # ---- OpenAI (STT + TTS) -------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY", ""))
    # STT model. "gpt-4o-mini-transcribe" is fast + cheap; "whisper-1" is the
    # most widely compatible fallback.
    stt_model: str = field(default_factory=lambda: _get("STT_MODEL", "gpt-4o-mini-transcribe"))
    # ISO-639-1 language hint (e.g. "en", "sk", "de"). Pinning the language
    # greatly improves accuracy AND speed on short clips vs. auto-detection.
    # Leave empty for auto-detect.
    stt_language: str = field(default_factory=lambda: _get("STT_LANGUAGE", ""))
    # Optional biasing prompt: names/jargon the model should spell correctly.
    stt_prompt: str = field(default_factory=lambda: _get("STT_PROMPT", ""))
    # TTS model + voice. "tts-1" has the lowest time-to-first-audio;
    # "gpt-4o-mini-tts" sounds better but starts a bit slower.
    tts_model: str = field(default_factory=lambda: _get("TTS_MODEL", "gpt-4o-mini-tts"))
    tts_voice: str = field(default_factory=lambda: _get("TTS_VOICE", "alloy"))
    tts_speed: float = field(default_factory=lambda: _get_float("TTS_SPEED", 1.0))
    # Print a per-turn latency breakdown (stt / llm / tts).
    show_timings: bool = field(default_factory=lambda: _get_bool("SHOW_TIMINGS", True))

    # ---- Gemini (LLM) -------------------------------------------------------
    # The user already has a Gemini endpoint. We talk to it through the OpenAI
    # wire format, so any OpenAI-compatible Gemini endpoint works. Google's own
    # compatibility endpoint is the default below.
    gemini_api_key: str = field(default_factory=lambda: _get("GEMINI_API_KEY", ""))
    gemini_base_url: str = field(
        default_factory=lambda: _get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    )
    gemini_model: str = field(default_factory=lambda: _get("GEMINI_MODEL", "gemini-2.0-flash"))
    system_prompt: str = field(
        default_factory=lambda: _get(
            "SYSTEM_PROMPT",
            "You are a helpful, friendly voice assistant. Keep replies short, "
            "conversational, and easy to speak aloud. Avoid markdown, lists, and "
            "long numbers unless asked.",
        )
    )

    # ---- Audio --------------------------------------------------------------
    # Silero VAD requires 16 kHz mono. Do not change the input rate.
    input_sample_rate: int = 16_000
    # Silero VAD processes exactly 512 samples per call at 16 kHz (~32 ms).
    frame_size: int = 512
    # OpenAI TTS "pcm" output is 24 kHz, 16-bit, mono, little-endian.
    output_sample_rate: int = 24_000

    input_device: int | None = field(
        default_factory=lambda: (
            int(os.environ["INPUT_DEVICE"]) if os.environ.get("INPUT_DEVICE") else None
        )
    )
    output_device: int | None = field(
        default_factory=lambda: (
            int(os.environ["OUTPUT_DEVICE"]) if os.environ.get("OUTPUT_DEVICE") else None
        )
    )

    # ---- VAD / turn-taking --------------------------------------------------
    # Speech probability above this counts as speech (0..1).
    vad_threshold: float = field(default_factory=lambda: _get_float("VAD_THRESHOLD", 0.5))
    # A turn only starts after this much continuous speech (debounces noise).
    min_speech_ms: int = field(default_factory=lambda: _get_int("MIN_SPEECH_MS", 200))
    # A turn ends after this much trailing silence (the "end of turn" pause).
    silence_hangover_ms: int = field(default_factory=lambda: _get_int("SILENCE_HANGOVER_MS", 500))
    # Ignore turns shorter than this after trimming (filters coughs/clicks).
    min_turn_ms: int = field(default_factory=lambda: _get_int("MIN_TURN_MS", 350))
    # Keep this much audio before detected speech start (avoids clipped words).
    speech_pad_ms: int = field(default_factory=lambda: _get_int("SPEECH_PAD_MS", 250))

    # If True, the user can interrupt the bot while it is speaking (barge-in).
    # Recommended: use headphones, otherwise the mic hears the bot and may
    # interrupt itself (there is no acoustic echo cancellation here).
    allow_interruptions: bool = field(
        default_factory=lambda: _get_bool("ALLOW_INTERRUPTIONS", True)
    )

    def validate(self) -> None:
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_size / self.input_sample_rate


settings = Settings()
