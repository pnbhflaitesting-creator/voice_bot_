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
