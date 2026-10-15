#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дымовая проверка консольного Scribe на любой системе (GitHub Actions: Windows, Linux).

    python3 tests/smoke_cli.py                    # через scribe_cli.py (исходники)
    python3 tests/smoke_cli.py --exe путь/scribe-cli.exe   # через собранную программу

Синтезирует короткую английскую запись (и смешанную ru/en, если есть русский голос),
расшифровывает в «Авто» и проверяет: текст есть, слова узнаются, «Авто» не путает
чистую запись со смешанной и замечает смешанную.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tts  # noqa: E402

EN = ["Good morning, this is a short test of the transcription program.",
      "We are checking that speech becomes text on this computer.",
      "The weather is fine and the meeting starts at ten.",
      "Thank you for listening, the test is over."]
RU = ["Добрый день, это проверка программы расшифровки.",
      "Сейчас мы проверяем русскую речь.",
      "Встреча начнётся в десять часов.",
      "Спасибо, проверка окончена."]


def silence(sec):
    return np.zeros(int(sec * tts.SR), np.float32)


def write_wav(path, audio):
    raw = path.with_suffix(".f32")
    raw.write_bytes(audio.astype(np.float32).tobytes())
    subprocess.run(["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-f", "f32le", "-ar", str(tts.SR),
                    "-ac", "1", "-i", str(raw), str(path)], check=True)
    raw.unlink()


def words(t):
    return set(re.findall(r"[a-zа-яё]+", t.lower()))


def run(cli, files, td):
    rep = Path(td) / "report.json"
    cmd = (cli if isinstance(cli, list) else [cli]) + ["--json", str(rep), "--out", str(Path(td) / "out")] + files
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
    assert rep.exists(), f"нет отчёта (код {r.returncode})"
    return json.loads(rep.read_text(encoding="utf-8")), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", help="собранный scribe-cli(.exe); по умолчанию - scribe_cli.py из исходников")
    a = ap.parse_args()
    cli = [a.exe] if a.exe else [sys.executable, str(HERE.parent / "scribe_cli.py")]
    langs = tts.available()
    print("голоса:", langs)
    assert "en" in langs, "нет английского голоса для синтеза"
    ok = True
    with tempfile.TemporaryDirectory() as td:
        en = Path(td) / "english.wav"
        parts = [silence(0.6)]
        for t in EN:
            parts += [tts.speech(t, "en"), silence(0.8)]
        write_wav(en, np.concatenate(parts))
        files = [str(en)]
        if "ru" in langs:
            mixed = Path(td) / "mixed.wav"
            parts = [silence(0.6)]
            for r_, e_ in zip(RU, EN):
                parts += [tts.speech(r_, "ru"), silence(0.8), tts.speech(e_, "en"), silence(0.8)]
            write_wav(mixed, np.concatenate(parts))
            files.append(str(mixed))
        report, sec = run(cli, files, td)
        for r in report:
            print(json.dumps({k: v for k, v in r.items() if k != "engine_stats"}, ensure_ascii=False))
            if "error" in r:
                print("ОШИБКА:", r["error"]); ok = False; continue
            text = Path(r["out"]).read_text(encoding="utf-8")
            if r["file"].endswith("english.wav"):
                ref = words(" ".join(EN))
                share = len(ref & words(text)) / len(ref)
                print(f"english: узнано {share:.0%} слов; «Авто»: смешанная={r['probe']['mixed']}, язык={r.get('lang')}")
                ok &= share >= 0.6 and not r["probe"]["mixed"]
            else:
                cyr = len(re.findall("[а-яё]", text, re.I)); lat = len(re.findall("[a-z]", text, re.I))
                print(f"mixed: кириллица {cyr}, латиница {lat}; «Авто»: смешанная={r['probe']['mixed']}")
                ok &= r["probe"]["mixed"] and cyr > 20 and lat > 20
        print(f"всего {sec:.0f} с · движок: {report[0].get('engine')}")
    print("ИТОГ:", "OK" if ok else "ОШИБКА")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
