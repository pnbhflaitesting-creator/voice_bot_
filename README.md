# Real-time Python Voice Bot

A real-time voice assistant in **pure Python** with **swappable providers** for
each stage:

| Stage | Providers (pick one via `.env`) |
| --- | --- |
| 🎧 **VAD / turn-taking** | Silero VAD |
| 📝 **STT** | `openai` · `deepgram` · `elevenlabs` |
| 🧠 **LLM** | `gemini` · `openai` |
| 🔊 **TTS** | `openai` · `deepgram` · `elevenlabs` |

Token-streaming + sentence-chunked TTS for low latency, plus **barge-in**
(interrupt the bot by speaking).

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

**Recommended for lowest latency:** `STT_PROVIDER=deepgram` (Nova),
`TTS_PROVIDER=elevenlabs` (Flash v2.5) or `deepgram` (Aura). These are the
industry-standard fast options and will crush an OpenAI-`tts-1` bottleneck.

## What "fast" means (industry reference)

The benchmark for natural conversation is **~800 ms voice-to-voice** (you stop
talking → bot audio starts); under 500 ms is excellent. Getting there requires
*streaming at every stage*. This project streams the **LLM → TTS** path already;
the remaining serial cost is **batch STT** (the clip is sent after you stop). The
lowest-latency setups replace that with **streaming STT** (transcribe while you
talk) or a **speech-to-speech "live" API** (Gemini Live / OpenAI Realtime) — see
the note at the end.

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
├── config.py      # env-driven settings + provider selection
├── audio_io.py    # mic capture + speaker playback (sounddevice), barge-in
├── audio_utils.py # WAV/PCM helpers shared by providers
├── denoise.py     # optional noise suppression (high-pass + spectral) before STT
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
