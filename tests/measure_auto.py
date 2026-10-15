#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Режим «Авто» (Scribe 2.3): замечает ли он второй язык и сколько это стоит.

    python3 tests/measure_auto.py                          # весь набор
    python3 tests/measure_auto.py --file "~/Downloads/запись.m4a"
    python3 tests/measure_auto.py --timing english quiet   # + время «Авто» до/после на чистых записях

Ожидание: lesson, long_insert, своя смешанная запись - «смешанная»; quiet, dictation,
english, english_accent, mix, mix_same (русские фразы с английскими словами) - нет.
--file ... --expect pure - для своей чистой записи (по умолчанию своя запись считается смешанной).
Результаты - tests/results/auto.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import multilang  # noqa: E402
import core as W  # noqa: E402 - ядро без окна (идёт и на сервере)

MODEL = "mlx-community/whisper-large-v3-turbo"   # на Windows/Linux движок сам берёт CTranslate2-версию


def asr_backend(mod=None):
    """Чем распознаёт проверяемый код: Scribe 2.4+ - движок engine.py (MLX или faster-whisper),
    код 2.1-2.3 - модуль mlx_whisper. Плюс прогрев модели."""
    if mod is None or hasattr(mod, "pu"):
        import engine
        eng = engine.get_engine()
        eng.warmup(MODEL)
        return eng
    import mlx_whisper
    mlx_whisper.transcribe(np.zeros(16000, np.float32), path_or_hf_repo=MODEL, language="ru")
    return mlx_whisper
EXPECT = {"lesson": True, "long_insert": True, "quiet": False, "dictation": False, "english": False,
          "english_accent": False, "mix": False, "mix_same": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--expect", choices=("mixed", "pure"), default="mixed")
    ap.add_argument("--timing", nargs="*", default=[])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    asr = asr_backend()
    targets = [(Path(a.file).expanduser().name, Path(a.file).expanduser(), a.expect == "mixed")] if a.file else \
        [(n, HERE / "audio" / f"{n}.m4a", EXPECT.get(n)) for n in (a.names or list(EXPECT))]
    out = {}
    for name, path, expect in targets:
        audio = W._decode_to_array(path, True)
        r = multilang.probe_languages(audio, MODEL)
        ok = "" if expect is None else ("верно" if r["mixed"] == expect else "ОШИБКА")
        share = r["confirmed_time"].get(r["other"], 0.0) / max(1e-9, r["speech_time"])
        print(f"{name:16s} {len(audio) / 16000 / 60:5.1f} мин  смешанная={r['mixed']!s:5s} {ok:7s} "
              f"основной={r['dominant']} ({r['dominant_share']:.0%}); второй язык: {r['confirmed'].get(r['other'], 0)} фраз, "
              f"{share:.1%} времени речи; фраз {r['phrases']}, сито отметило {r['flagged']}, turbo проверил {r['verified']} · {r['seconds']} с · {r['method']}")
        if r["hits"]:
            print("      фразы второго языка:", r["hits"][:8])
        out[name] = dict(r, minutes=round(len(audio) / 16000 / 60, 2), expect=expect)
    for name in a.timing:
        # «Авто» 2.2 (Whisper сам решает язык) и 2.3 (проверка + распознавание) - по очереди
        # несколько раз, медиана: так шум от других задач на видеокарте меньше влияет
        path = HERE / "audio" / f"{name}.m4a"
        audio = W._decode_to_array(path, True)
        olds, news, probes = [], [], []
        for _ in range(a.repeat):
            t0 = time.monotonic()
            W._transcribe_array(asr, audio, MODEL, None, lambda f: None, lambda: False)
            olds.append(time.monotonic() - t0)
            t0 = time.monotonic()
            segs, info = W.transcribe_auto(audio, MODEL, asr, lambda f: None, lambda: False)
            news.append(time.monotonic() - t0)
            probes.append(info["probe"]["seconds"])
        old, new, pr = (float(np.median(x)) for x in (olds, news, probes))
        print(f"время «Авто» {name} ({len(audio) / 16000 / 60:.1f} мин): 2.2 {old:.1f} с → 2.3 {new:.1f} с "
              f"(разница {new - old:+.1f} с; сама проверка {pr:.1f} с; язык {info.get('lang')}) "
              f"· замеры 2.2 {[round(x, 1) for x in olds]} 2.3 {[round(x, 1) for x in news]}")
        out.setdefault(name, {})["timing"] = dict(old=round(old, 1), new=round(new, 1), probe=round(pr, 1),
                                                  lang=info.get("lang"))
    (HERE / "results").mkdir(exist_ok=True)
    dest = HERE / "results" / "auto.json"
    prev = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    prev.update(out)
    dest.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
