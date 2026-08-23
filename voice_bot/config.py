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
    # ---- Provider selection -------------------------------------------------
    # Which service handles each stage. Mix and match freely.
    #   stt_provider: openai | deepgram | elevenlabs
    #   tts_provider: openai | deepgram | elevenlabs
    #   llm_provider: gemini | openai
    stt_provider: str = field(default_factory=lambda: _get("STT_PROVIDER", "openai").lower())
    tts_provider: str = field(default_factory=lambda: _get("TTS_PROVIDER", "openai").lower())
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "gemini").lower())

    # ---- API keys -----------------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: _get("GEMINI_API_KEY", ""))
    deepgram_api_key: str = field(default_factory=lambda: _get("DEEPGRAM_API_KEY", ""))
    elevenlabs_api_key: str = field(default_factory=lambda: _get("ELEVENLABS_API_KEY", ""))

    # ---- STT (shared) -------------------------------------------------------
    # ISO-639-1 language hint (e.g. "en", "sk", "de"). Pinning the language
    # improves accuracy AND speed vs. auto-detection. Empty = auto-detect.
    stt_language: str = field(default_factory=lambda: _get("STT_LANGUAGE", ""))
    # Optional biasing prompt (OpenAI): names/jargon to spell correctly.
    stt_prompt: str = field(default_factory=lambda: _get("STT_PROMPT", ""))
    # Per-provider STT model (only the selected provider's value is used).
    openai_stt_model: str = field(
        default_factory=lambda: _get("STT_MODEL", "gpt-4o-mini-transcribe")
    )
    deepgram_stt_model: str = field(
        default_factory=lambda: _get("DEEPGRAM_STT_MODEL", "nova-3")
    )
    elevenlabs_stt_model: str = field(
        default_factory=lambda: _get("ELEVENLABS_STT_MODEL", "scribe_v1")
    )

    # ---- TTS (shared + per-provider) ----------------------------------------
    tts_speed: float = field(default_factory=lambda: _get_float("TTS_SPEED", 1.0))
    # OpenAI: "tts-1" is lowest latency; "gpt-4o-mini-tts" sounds nicer.
    openai_tts_model: str = field(default_factory=lambda: _get("TTS_MODEL", "gpt-4o-mini-tts"))
    openai_tts_voice: str = field(default_factory=lambda: _get("TTS_VOICE", "alloy"))
    # Deepgram Aura voice model (voice is baked into the model name).
    deepgram_tts_model: str = field(
        default_factory=lambda: _get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
    )
    # ElevenLabs: Flash v2.5 is the low-latency model. Default voice = Rachel.
    elevenlabs_tts_model: str = field(
        default_factory=lambda: _get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
    )
    elevenlabs_voice_id: str = field(
        default_factory=lambda: _get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    )

    # Print a per-turn latency breakdown (stt / llm / tts).
    show_timings: bool = field(default_factory=lambda: _get_bool("SHOW_TIMINGS", True))

    # ---- LLM: Gemini --------------------------------------------------------
    # Talked to through the OpenAI-compatible wire format, so any compatible
    # Gemini endpoint works. Google's compatibility endpoint is the default.
    gemini_base_url: str = field(
        default_factory=lambda: _get(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    )
    gemini_model: str = field(default_factory=lambda: _get("GEMINI_MODEL", "gemini-2.0-flash"))
    # Gemini 2.5 models "think" before answering by default, adding latency and
    # tokens. For a voice bot we want the fastest first token: "none" disables
    # thinking. Values: none/low/medium/high, or empty to not send the param.
    gemini_reasoning_effort: str = field(
        default_factory=lambda: _get("GEMINI_REASONING_EFFORT", "none")
    )

    # ---- LLM: OpenAI --------------------------------------------------------
    openai_llm_model: str = field(default_factory=lambda: _get("OPENAI_LLM_MODEL", "gpt-4o-mini"))
    # Optional custom endpoint (Azure deployment, local server, gateway, …).
    # Empty = the standard OpenAI API.
    openai_base_url: str = field(default_factory=lambda: _get("OPENAI_BASE_URL", ""))
    # Reasoning effort for reasoning-capable OpenAI models (gpt-5, o-series):
    # "minimal" is effectively reasoning-off; also low/medium/high. Leave EMPTY
    # for non-reasoning models (e.g. gpt-4o-mini), which reject this parameter.
    openai_reasoning_effort: str = field(
        default_factory=lambda: _get("OPENAI_REASONING_EFFORT", "")
    )

    # ---- LLM (shared) -------------------------------------------------------
    # Cap the reply length so the LLM doesn't ramble (faster, more responsive).
    max_reply_tokens: int = field(default_factory=lambda: _get_int("MAX_REPLY_TOKENS", 400))
    # Sampling temperature (0 = deterministic, higher = more varied).
    temperature: float = field(default_factory=lambda: _get_float("TEMPERATURE", 0.2))
    # ---- TTS chunking (how soon the bot starts speaking) --------------------
    # The reply is streamed to TTS in chunks so audio starts before the LLM has
    # finished. The FIRST chunk is spoken as soon as the LLM emits a clause
    # boundary at/after this many characters (lower = starts sooner, choppier).
    tts_first_chunk_chars: int = field(
        default_factory=lambda: _get_int("TTS_FIRST_CHUNK_CHARS", 24)
    )
    # If no sentence boundary appears, flush a chunk once it reaches this length.
    tts_max_chunk_chars: int = field(
        default_factory=lambda: _get_int("TTS_MAX_CHUNK_CHARS", 200)
    )

    # Spoken on startup before listening. Empty = no greeting.
    greeting: str = field(
        default_factory=lambda: _get("GREETING", "How can I help you today?")
    )
    system_prompt: str = field(
        default_factory=lambda: _get(
            "SYSTEM_PROMPT",
            "You are a helpful, friendly voice assistant. Keep replies short, "
            "conversational, and easy to speak aloud. Avoid markdown, lists, and "
            "long numbers unless asked.",
        )
    )
    # Force the assistant to always reply in this language regardless of what
    # language the user speaks (e.g. "English", "Turkish"). Empty = match input.
    response_language: str = field(default_factory=lambda: _get("RESPONSE_LANGUAGE", ""))

    # ---- Agent (tool calling) ----------------------------------------------
    # Give the LLM access to tools (weather, time, math, web search, memory).
    agent_enabled: bool = field(default_factory=lambda: _get_bool("AGENT_ENABLED", True))

    @property
    def effective_system_prompt(self) -> str:
        prompt = self.system_prompt
        if self.response_language:
            prompt += f" Always respond in {self.response_language}, no matter what language the user speaks."
        if self.agent_enabled:
            prompt += (
                " You have tools available (weather, current time, a calculator, "
                "web search, and a memory to remember and recall facts). Use them "
                "when they help answer accurately. If a tool needs information the "
                "user hasn't given — like a city for the weather — ask them for it "
                "first instead of guessing. Keep spoken answers short and natural."
            )
        return prompt

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
    # After the bot starts speaking, ignore barge-in for this long. Prevents the
    # bot's own audio (if the mic hears it) from instantly interrupting itself.
    echo_guard_ms: int = field(default_factory=lambda: _get_int("ECHO_GUARD_MS", 400))
    # Acoustic echo cancellation: subtract the bot's played audio from the mic so
    # barge-in works even when the mic hears the speaker. Enables true full-duplex.
    echo_cancellation: bool = field(
        default_factory=lambda: _get_bool("ECHO_CANCELLATION", False)
    )
    # AEC adaptive-filter length (echo tail it can model) and step size.
    aec_filter_ms: int = field(default_factory=lambda: _get_int("AEC_FILTER_MS", 250))
    aec_mu: float = field(default_factory=lambda: _get_float("AEC_MU", 0.3))

    # ---- Noise suppression (applied to the utterance before STT) ------------
    # "none"     : off (default).
    # "spectral" : spectral-gating denoise via the `noisereduce` package
    #              (pip install noisereduce). Good for steady background noise.
    denoise: str = field(default_factory=lambda: _get("DENOISE", "none").lower())
    # How aggressively to reduce noise (0..1). Higher removes more noise but can
    # muffle speech.
    denoise_strength: float = field(default_factory=lambda: _get_float("DENOISE_STRENGTH", 0.8))
    # Stationary = assume a constant noise profile (fan/hum); non-stationary
    # adapts over time (recommended for varied noise).
    denoise_stationary: bool = field(
        default_factory=lambda: _get_bool("DENOISE_STATIONARY", False)
    )
    # High-pass filter cutoff in Hz to cut low-frequency rumble/hum (0 = off).
    # ~80-100 Hz is safe for speech. Uses scipy if available.
    highpass_hz: int = field(default_factory=lambda: _get_int("HIGHPASS_HZ", 0))

    # ---- Web UI -------------------------------------------------------------
    # Where the optional web UI (python -m voice_bot --web) listens.
    web_host: str = field(default_factory=lambda: _get("WEB_HOST", "127.0.0.1"))
    web_port: int = field(default_factory=lambda: _get_int("WEB_PORT", 8000))

    # ---- Logging ------------------------------------------------------------
    # A fresh log file is created per session in log_dir, recording every step
    # (calls, transcripts, replies, tool calls, timings, errors).
    log_dir: str = field(default_factory=lambda: _get("LOG_DIR", "logs"))
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO").upper())
    # Also echo logs to the console (the app already prints a friendly view, so
    # this is off by default).
    log_to_console: bool = field(default_factory=lambda: _get_bool("LOG_TO_CONSOLE", False))

    # ---- Debug recording ----------------------------------------------------
    # Save, for each detected turn, the raw audio (what the mic heard), the
    # denoised audio (what is sent to STT), and the transcript. Lets you listen
    # to both and compare against what the model returned.
    save_turns: bool = field(default_factory=lambda: _get_bool("SAVE_TURNS", False))
    turns_dir: str = field(default_factory=lambda: _get("TURNS_DIR", "recordings"))

    # Which API key each provider needs.
    _KEY_FOR_PROVIDER = {
        "openai": ("OPENAI_API_KEY", "openai_api_key"),
        "gemini": ("GEMINI_API_KEY", "gemini_api_key"),
        "deepgram": ("DEEPGRAM_API_KEY", "deepgram_api_key"),
        "deepgram-stream": ("DEEPGRAM_API_KEY", "deepgram_api_key"),
        "elevenlabs": ("ELEVENLABS_API_KEY", "elevenlabs_api_key"),
    }

    def validate(self) -> None:
        valid = {
            "stt": {"openai", "deepgram", "deepgram-stream", "elevenlabs"},
            "tts": {"openai", "deepgram", "elevenlabs"},
            "llm": {"gemini", "openai"},
        }
        for stage, chosen in (
            ("stt", self.stt_provider),
            ("tts", self.tts_provider),
            ("llm", self.llm_provider),
        ):
            if chosen not in valid[stage]:
                raise RuntimeError(
                    f"{stage.upper()}_PROVIDER='{chosen}' is invalid; "
                    f"choose one of: {', '.join(sorted(valid[stage]))}."
                )

        # Only require the keys for the providers actually selected.
        needed = {self.stt_provider, self.tts_provider, self.llm_provider}
        missing = []
        for provider in needed:
            env_name, attr = self._KEY_FOR_PROVIDER[provider]
            if not getattr(self, attr):
                missing.append(env_name)
        if missing:
            raise RuntimeError(
                "Missing required environment variables for the selected "
                "providers: " + ", ".join(sorted(set(missing)))
                + ". Copy .env.example to .env and fill them in."
            )

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_size / self.input_sample_rate


settings = Settings()
