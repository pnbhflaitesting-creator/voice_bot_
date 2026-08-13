# Real-time Python Voice Bot

A real-time voice assistant in **pure Python** with **swappable providers** for
each stage:

| Stage | Providers (pick one via `.env`) |
| --- | --- |
| 🎧 **VAD / turn-taking** | Silero VAD |
| 📝 **STT** | `openai` · `deepgram` · `deepgram-stream` (real-time) · `elevenlabs` |
| 🧠 **LLM** | `gemini` · `openai` |
| 🔊 **TTS** | `openai` · `deepgram` · `elevenlabs` |

Token-streaming + sentence-chunked TTS for low latency, plus **barge-in**
(interrupt the bot by speaking) and an **agent layer** — the LLM can call tools
(weather, time, math, web search) and keeps a **session memory**.

```
🎙️ mic ─▶ Silero VAD (turn detection) ─▶ OpenAI STT ─▶ Gemini LLM (streamed)
                                                              │
                          speaker ◀── OpenAI TTS ◀── sentence chunks
```

## How it works

1. The microphone is captured continuously at 16 kHz in 512-sample frames.
2. Each frame goes to **Silero VAD**. A small state machine (`voice_bot/vad.py`) decides when a **turn starts** (enough continuous speech) and when it **ends** (a trailing pause of `SILENCE_HANGOVER_MS`).
3. The collected utterance is transcribed by **OpenAI** (`voice_bot/stt.py`).
4. The transcript is sent to **Gemini** and the reply is **streamed** token by token (`voice_bot/llm.py`).
5. Tokens are regrouped into sentences and each sentence is sent to **OpenAI TTS** as it completes, so the bot starts speaking before the full reply is generated (`voice_bot/tts.py`, `voice_bot/pipeline.py`).
6. If you start talking while the bot is speaking, VAD detects it and the current response is cancelled and playback stops instantly (**barge-in**).

## Setup

Requires Python 3.10+ and PortAudio (for `sounddevice`).

```bash
# System audio library
#   macOS:          brew install portaudio
#   Debian/Ubuntu:  sudo apt-get install portaudio19-dev

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then fill in your keys
```

In `.env`, pick your providers and set only the keys they need:

- `STT_PROVIDER` / `TTS_PROVIDER` / `LLM_PROVIDER`
- `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY` — whichever the selected providers require (the app tells you if one is missing)

> The Gemini client speaks the OpenAI-compatible chat format. The default
> `GEMINI_BASE_URL` is Google's compatibility endpoint. If you proxy Gemini
> yourself, point `GEMINI_BASE_URL` at your own URL — no code changes needed.

## Run

```bash
python -m voice_bot            # start talking
python -m voice_bot --devices  # list audio devices (to set INPUT_DEVICE/OUTPUT_DEVICE)
```

Then just talk. Pause when you're done and the bot will reply. Press **Ctrl+C** to quit.

> 💡 **Use headphones.** Barge-in listens while the bot speaks; without
> headphones the mic hears the bot's own voice and may interrupt itself (there's
> no acoustic echo cancellation). Set `ALLOW_INTERRUPTIONS=false` for strict
> half-duplex if you're on speakers.

## Configuration

Everything is environment-driven (see `.env.example`). Highlights:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VAD_THRESHOLD` | `0.5` | Speech probability cutoff (higher = less sensitive) |
| `MIN_SPEECH_MS` | `200` | Speech must last this long to begin a turn |
| `SILENCE_HANGOVER_MS` | `700` | Trailing silence that ends a turn |
| `MIN_TURN_MS` | `350` | Ignore utterances shorter than this |
| `SPEECH_PAD_MS` | `250` | Audio kept before speech start (avoids clipped words) |
| `ALLOW_INTERRUPTIONS` | `true` | Enable barge-in |
| `STT_MODEL` | `gpt-4o-mini-transcribe` | OpenAI transcription model |
| `TTS_MODEL` / `TTS_VOICE` | `gpt-4o-mini-tts` / `alloy` | OpenAI TTS |
| `GEMINI_MODEL` | `gemini-2.0-flash` | LLM model name |

## Agent tools & memory

The bot is an **agent**: the LLM can call tools (native function-calling — no
MCP needed) and keeps a **session memory** that lasts until you Ctrl+C. Enabled
by default (`AGENT_ENABLED=true`). All demo tools are keyless:

| Tool | What it does |
| --- | --- |
| `get_weather(location)` | Real weather via Open-Meteo. **Asks for the city if you don't say one.** |
| `get_current_time(timezone)` | Local or named-timezone clock |
| `calculate(expression)` | Safe arithmetic (`2*(3+4)`, `sqrt(144)`) |
| `web_search(query)` | Real web search (DuckDuckGo SERP via the `ddgs` package) |
| `remember` / `recall` / `list_memories` | Store & retrieve facts for the session |

Try saying:

- *"What's the weather?"* → it asks **which city**, then answers with real data.
- *"Remember that my favourite colour is teal."* … later … *"What's my favourite colour?"*
- *"What's the square root of 2025?"* · *"What time is it in Tokyo?"*

When a tool runs you'll see it in the console, e.g. `🛠️  get_weather(location='Tokyo')`.
The model streams a normal spoken answer when no tool is needed, so latency for
plain chat is unchanged. Add your own tool in `voice_bot/agent.py` by appending
a `Tool(...)` to `_build_tools` — its JSON schema is sent to the model
automatically. Set `AGENT_ENABLED=false` to disable tools entirely.

## Choosing providers

Set the three provider switches in `.env` and supply only the matching keys:

```bash
STT_PROVIDER=deepgram      # openai | deepgram | elevenlabs
TTS_PROVIDER=elevenlabs    # openai | deepgram | elevenlabs
LLM_PROVIDER=gemini        # gemini | openai

DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
GEMINI_API_KEY=...
```

All three TTS providers stream **24 kHz / 16-bit mono PCM**, so they drop into
the same speaker path with no re-encoding. Per-provider model/voice knobs
(`DEEPGRAM_TTS_MODEL`, `ELEVENLABS_VOICE_ID`, …) are in `.env.example`.

**Recommended for lowest latency:** `STT_PROVIDER=deepgram-stream` (real-time),
`TTS_PROVIDER=elevenlabs` (Flash v2.5) or `deepgram` (Aura). These are the
industry-standard fast options and will crush an OpenAI-`tts-1` bottleneck.

### Streaming STT (`deepgram-stream`)

The batch STT providers can't start until you *stop* talking, then upload the
whole clip — so a 3-second sentence costs ~1–2s of pure post-speech STT.
`deepgram-stream` instead streams your audio to Deepgram over a WebSocket
**while you speak**, so at end-of-turn it just sends a Finalize and returns the
last words (~100–300ms). Turn-taking is still driven by Silero VAD; Deepgram
only transcribes. If the socket can't connect it transparently falls back to the
batch REST API, so it never hard-fails. Needs `pip install websockets`.

## What "fast" means (industry reference)

The benchmark for natural conversation is **~800 ms voice-to-voice** (you stop
talking → bot audio starts); under 500 ms is excellent. Getting there requires
*streaming at every stage*. This project streams the **LLM → TTS** path already;
the remaining serial cost is **batch STT** (the clip is sent after you stop). The
lowest-latency setups replace that with **streaming STT** (transcribe while you
talk) or a **speech-to-speech "live" API** (Gemini Live / OpenAI Realtime) — see
the note at the end.

## Logs

Every run writes a fresh log file to `logs/session_<timestamp>.log` recording
each step — provider config, per-turn STT/LLM/TTS calls with timings, the
transcript and the reply, every tool call and its result, barge-ins, and full
tracebacks for any error. The console keeps its friendly emoji view; the log
file is the detailed trace for debugging.

```bash
LOG_DIR=logs
LOG_LEVEL=INFO        # DEBUG for more detail
LOG_TO_CONSOLE=false  # true to also stream logs to the terminal
```

A typical turn in the log:

```
10:15:02.184 INFO  voice_bot.pipeline | turn 3: start, clip=1.80s (28800 samples)
10:15:02.185 INFO  voice_bot.pipeline | turn 3: STT (deepgram) …
10:15:03.010 INFO  voice_bot.pipeline | turn 3: STT done in 826ms -> 'what is the weather in tokyo'
10:15:03.400 INFO  voice_bot.agent    | tool call: get_weather(location='Tokyo')
10:15:03.951 INFO  voice_bot.agent    | tool get_weather -> 'Weather in Tokyo, Japan: ...' (551ms)
10:15:04.220 INFO  voice_bot.pipeline | turn 3: reply -> 'It's clear and 24 degrees in Tokyo.'
10:15:04.221 INFO  voice_bot.pipeline | turn 3: timings clip=1.8s stt=826ms llm=390ms tts=270ms to-first-audio=1486ms
```

## Suppressing background noise

There are **two different noise problems** — pick the fix for yours:

**A. Noise *triggers* the bot** (it replies to a fan, keyboard, or other people
talking). This is turn-detection, not audio quality. Tighten the VAD — no extra
dependencies:

```bash
VAD_THRESHOLD=0.7      # 0..1, higher = only confident speech counts
MIN_SPEECH_MS=300      # require longer continuous speech to start a turn
```

**B. Noise *garbles* what you say** (STT mishears you in a noisy room). Enable
denoising of the utterance before it's transcribed:

```bash
pip install noisereduce scipy

DENOISE=spectral       # spectral-gating denoise
DENOISE_STRENGTH=0.8   # 0..1 (higher = more aggressive)
HIGHPASS_HZ=90         # also cut low rumble/hum (optional)
```

Denoising runs on the buffered utterance (off the real-time mic path, in a
worker thread) so it adds a little processing but not streaming latency. If the
libraries aren't installed it prints a note and simply runs without denoising.

**Hear what the model hears.** To debug transcription, dump every turn to disk:

```bash
SAVE_TURNS=true
TURNS_DIR=recordings
```

Each turn writes `<time>_<n>_raw.wav` (mic input), `<time>_<n>_clean.wav` (what
STT actually received, after denoising), and `<time>_<n>.txt` (the transcript).
Play the two WAVs back-to-back to check whether denoising is helping or muffling
your speech — and compare against the transcript.

> For heavier, real-time per-frame suppression the industry uses **RNNoise**,
> **DeepFilterNet**, or **Krisp**; and hardware/OS echo-and-noise cancellation
> (or a headset mic) beats any of this. Ask if you want DeepFilterNet wired in.

## Speeding it up & fixing accuracy

Every turn prints a latency breakdown (`SHOW_TIMINGS=true`):

```
⏱  stt 620ms · llm 410ms · tts 300ms · to-first-audio 1330ms
```

Use it to see *where* the time goes, then tune:

**Accuracy — do this first.** If the bot mishears you, it's almost always
language auto-detection guessing wrong on a short clip. **Pin your language:**

```bash
STT_LANGUAGE=en          # the language you SPEAK — or tr, de, es, fr, ...
# STT_PROMPT=names, jargon, product terms it keeps misspelling
```

> ⚠️ **`STT_LANGUAGE` is not translation** — it tells STT which language you're
> *speaking*. And OpenAI's `gpt-4o(-mini)-transcribe` is
> [known to ignore it](https://community.openai.com/t/gpt-4o-transcribe-language-enforcement/1357014)
> and transcribe the wrong language anyway. If you need the language *enforced*,
> use a provider that respects it:
> ```bash
> STT_PROVIDER=deepgram   # strict language + fast (recommended)
> STT_LANGUAGE=en
> DEEPGRAM_API_KEY=...
> # or, staying on OpenAI:  STT_MODEL=whisper-1   (strict but slower)
> ```
> To always **reply** in one language no matter what's spoken, set
> `RESPONSE_LANGUAGE=English` (this steers the LLM, independent of STT).

**Latency levers, biggest first:**

| Lever | Change | Effect |
| --- | --- | --- |
| **STT model** | `STT_MODEL=gpt-4o-mini-transcribe` | **`whisper-1` is ~3–5× slower — never use it for real-time** |
| Gemini thinking | `GEMINI_REASONING_EFFORT=none` | 2.5 models "think" before replying by default; this is a big first-token win |
| End-of-turn wait | `SILENCE_HANGOVER_MS=400` | −100–300 ms dead time before every reply (too low = it cuts you off mid-pause) |
| LLM model | `GEMINI_MODEL=gemini-2.5-flash-lite` | Faster first token than `flash` |
| TTS model | `TTS_MODEL=tts-1` | Lower time-to-first-audio than `gpt-4o-mini-tts` |
| Reply length | `MAX_REPLY_TOKENS=200` | Shorter replies finish sooner |
| STT language | `STT_LANGUAGE=en` | Faster + more accurate STT |

Connections are warmed at startup so the *first* turn isn't penalised by
cold-start TLS/DNS.

A good **low-latency profile**:

```bash
STT_MODEL=gpt-4o-mini-transcribe      # NOT whisper-1
STT_LANGUAGE=en
GEMINI_MODEL=gemini-2.5-flash-lite    # or gemini-2.0-flash
GEMINI_REASONING_EFFORT=none          # disable 2.5 "thinking"
TTS_MODEL=tts-1
SILENCE_HANGOVER_MS=400
```

> ⚠️ Your `whisper-1 → 8.7 s` result is exactly the trap: `whisper-1` is the
> slowest option. `gpt-4o-mini-transcribe` + warmed connections typically brings
> STT under ~1.5 s. If it's *still* slow after switching, your network latency to
> the API is the floor — see the live-API note below.

### Want it dramatically faster? Consider a realtime/live API

This project uses three separate network hops (STT → LLM → TTS), so ~1–1.5 s
to first audio is about the floor. If you want sub-500 ms conversational
latency, the architecture to switch to is a **speech-to-speech "live" API**,
which streams audio in and out of a single session:

- **Gemini Live API** — keeps your Gemini LLM, does STT+TTS natively in one
  streaming connection. Best fit since you're already on Gemini. (This replaces
  OpenAI STT/TTS.)
- **OpenAI Realtime API** — same idea, but the model is OpenAI's, not Gemini.

Those trade the "mix and match providers" flexibility of this pipeline for much
lower latency. If you'd like, this repo can be adapted to Gemini Live.

## Project layout

```
voice_bot/
├── __main__.py    # CLI entry point (python -m voice_bot)
├── config.py      # env-driven settings + provider selection
├── audio_io.py    # mic capture + speaker playback (sounddevice), barge-in
├── audio_utils.py # WAV/PCM helpers shared by providers
├── denoise.py     # optional noise suppression (high-pass + spectral) before STT
├── recorder.py    # optional per-turn debug dump (raw/clean audio + transcript)
├── agent.py       # agent tools (weather/time/calc/search/memory) + session memory
├── logging_setup.py # per-session log file configuration
├── vad.py         # Silero VAD + turn-taking state machine
├── stt.py         # STT providers (OpenAI/Deepgram/ElevenLabs) + create_stt()
├── llm.py         # LLM providers (Gemini/OpenAI) + create_llm()
├── tts.py         # TTS providers (OpenAI/Deepgram/ElevenLabs) + create_tts()
└── pipeline.py    # orchestration: VAD → STT → LLM → sentence-chunked TTS
```

## Notes & limitations

- No acoustic echo cancellation — headphones strongly recommended for barge-in.
- Conversation history is kept in memory for the session (`llm.py`); restart clears it.
- Latency depends on your network and the chosen models; `gpt-4o-mini-*` and
  `gemini-2.0-flash` are picked for speed.
