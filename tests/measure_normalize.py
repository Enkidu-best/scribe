#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Нормализация громкости: тихая запись (~-35 dB, tests/audio/quiet.m4a) с ней и без неё.
Считаем долю ошибочных слов (WER) против известного текста и время.

    python3 tests/measure_normalize.py [quiet long_insert ...]
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core as W  # noqa: E402 - ядро без окна

MODEL = "mlx-community/whisper-large-v3-turbo"   # на Windows/Linux движок сам берёт CTranslate2-версию


def asr_backend(mod=None):
    """Движок распознавания (engine.py: MLX или faster-whisper) с прогревом."""
    import engine
    eng = engine.get_engine()
    eng.warmup(MODEL)
    return eng


def words(t):
    return re.findall(r"[a-zа-я0-9]+", t.lower().replace("ё", "е"))


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def main():
    names = sys.argv[1:] or ["quiet"]
    asr = asr_backend()
    out = {}
    for name in names:
        gt = json.loads((HERE / "audio" / f"{name}.json").read_text(encoding="utf-8"))
        ref = " ".join(p["text"] for p in gt["phrases"])
        for norm in (False, True):
            t0 = time.monotonic()
            a = W._decode_to_array(HERE / "audio" / f"{name}.m4a", norm)
            t1 = time.monotonic()
            segs = W._transcribe_array(asr, a, MODEL, "ru", lambda f: None, lambda: False)
            t2 = time.monotonic()
            hyp = " ".join(s["text"] for s in segs)
            rms = 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-12)
            res = dict(rms_db=round(float(rms), 1), wer=round(wer(ref, hyp), 3), decode_s=round(t1 - t0, 1),
                       asr_s=round(t2 - t1, 1), words=len(words(hyp)), ref_words=len(words(ref)))
            out[f"{name} norm={norm}"] = res
            print(name, "нормализация" if norm else "без нормализации", json.dumps(res, ensure_ascii=False))
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "normalize.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
