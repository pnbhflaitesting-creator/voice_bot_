# Real-time Python Voice Bot

A fully local-driving, real-time voice assistant in **pure Python**:

- 🎧 **Silero VAD** for voice-activity detection **and turn-taking** (knows when you started and finished talking)
- 📝 **OpenAI** for **Speech-to-Text** (transcription models)
- 🧠 **Gemini** for the **LLM** (through your existing endpoint)
- 🔊 **OpenAI** for **Text-to-Speech** (streamed)
- ⛓️ Token-streaming + sentence-chunked TTS for low latency, plus **barge-in** (interrupt the bot by speaking)

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

Set your keys in `.env`:

- `OPENAI_API_KEY` — for STT + TTS
- `GEMINI_API_KEY` and (optionally) `GEMINI_BASE_URL` / `GEMINI_MODEL` — your Gemini endpoint

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

## Speeding it up & fixing accuracy

Every turn prints a latency breakdown (`SHOW_TIMINGS=true`):

```
⏱  stt 620ms · llm 410ms · tts 300ms · to-first-audio 1330ms
```

Use it to see *where* the time goes, then tune:

**Accuracy — do this first.** If the bot mishears you, it's almost always
language auto-detection guessing wrong on a short clip. **Pin your language:**

```bash
STT_LANGUAGE=en          # or sk, de, es, fr, ...
# STT_PROMPT=names, jargon, product terms it keeps misspelling
```

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
├── config.py      # env-driven settings
├── audio_io.py    # mic capture + speaker playback (sounddevice), barge-in
├── vad.py         # Silero VAD + turn-taking state machine
├── stt.py         # OpenAI speech-to-text
├── llm.py         # Gemini LLM (streamed, OpenAI-compatible)
├── tts.py         # OpenAI text-to-speech (streamed PCM)
└── pipeline.py    # orchestration: VAD → STT → LLM → sentence-chunked TTS
```

## Notes & limitations

- No acoustic echo cancellation — headphones strongly recommended for barge-in.
- Conversation history is kept in memory for the session (`llm.py`); restart clears it.
- Latency depends on your network and the chosen models; `gpt-4o-mini-*` and
  `gemini-2.0-flash` are picked for speed.
