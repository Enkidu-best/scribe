#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Задержка диктофона на этой машине: сколько распознаётся одна фраза (~4 с речи) тем же
вызовом, что в dictaphone_worker.py, для моделей turbo / medium / small. Медиана 3 раз.

    python3 tests/dict_latency.py [--models turbo medium small]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import engine  # noqa: E402
import tts  # noqa: E402

PHRASE = "We are checking how fast the dictation turns speech into text on this computer."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["turbo", "medium", "small"])
    a = ap.parse_args()
    eng = engine.get_engine()
    audio = tts.speech(PHRASE, "en")
    print(f"фраза {len(audio) / tts.SR:.1f} с · {eng.label}")
    for m in a.models:
        repo = engine.MODELS[m][0]
        t0 = time.monotonic()
        eng.warmup(repo)
        load = time.monotonic() - t0
        times, text = [], ""
        for _ in range(3):
            t0 = time.monotonic()
            text = eng.transcribe_text(audio, repo, "en", None)
            times.append(time.monotonic() - t0)
        print(f"ДИКТОФОН {m}: {statistics.median(times):.1f} с на фразу (загрузка {load:.0f} с) · «{text.strip()[:70]}»",
              flush=True)
        eng.unload()


if __name__ == "__main__":
    main()
