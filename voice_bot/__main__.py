"""Entry point: ``python -m voice_bot`` (or ``python -m voice_bot --devices``)."""

from __future__ import annotations

import asyncio
import sys

from .config import settings


def _list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())
    print(
        "\nSet INPUT_DEVICE / OUTPUT_DEVICE to the numeric id of the device you "
        "want (leave unset to use the system default)."
    )


def main() -> None:
    if "--devices" in sys.argv:
        _list_devices()
        return

    settings.validate()

    # Fresh log file for this session; everything is recorded there.
    from .logging_setup import setup_session_logging

    log_path = setup_session_logging()
    print(f"📝 Logging this session to {log_path}", flush=True)

    # Imported here so ``--devices`` works without the heavier deps loaded.
    from .pipeline import VoiceBot

    try:
        asyncio.run(VoiceBot().run())
    except KeyboardInterrupt:
        print("\n👋 Bye.", flush=True)
        import logging

        logging.getLogger("voice_bot").info("keyboard interrupt — exiting")


if __name__ == "__main__":
    main()
