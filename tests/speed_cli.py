#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скорость расшифровки на этой машине (GitHub Actions: Windows, Linux): синтезированная запись
~2 мин, «Авто», два прохода в одном процессе; скорость - по второму (модель уже загружена).

    python3 tests/speed_cli.py [--exe scribe-cli.exe] [--sec 120]
"""
import argparse
import os
import platform
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tts  # noqa: E402
from smoke_cli import EN, silence, write_wav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe")
    ap.add_argument("--sec", type=float, default=120)
    a = ap.parse_args()
    cli = [a.exe] if a.exe else [sys.executable, str(HERE.parent / "scribe_cli.py")]
    phrases = [tts.speech(t, "en") for t in EN]
    parts, i = [silence(0.5)], 0
    while sum(len(p) for p in parts) < a.sec * tts.SR:
        parts += [phrases[i % len(phrases)], silence(0.7)]
        i += 1
    with tempfile.TemporaryDirectory() as td:
        f1 = Path(td) / "speed1.wav"
        write_wav(f1, np.concatenate(parts))
        f2 = Path(td) / "speed2.wav"
        shutil.copy(f1, f2)
        rep = Path(td) / "r.json"
        subprocess.run(cli + ["--json", str(rep), "--out", td, str(f1), str(f2)], check=True)
        r = json.loads(rep.read_text(encoding="utf-8"))
    first, second = r
    print(f"СКОРОСТЬ: {second['engine']} · запись {second['duration']:.0f} с · первый проход {first['seconds']} с "
          f"(с загрузкой модели) · второй {second['seconds']} с = {second['min_per_hour']} мин на час записи · "
          f"{platform.system()} {platform.machine()}, ядер: {os.cpu_count()}")


if __name__ == "__main__":
    main()
