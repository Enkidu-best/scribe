#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Замер режима «Несколько» на тестовом наборе (tests/make_test_audio.py).

    python3 tests/measure_multilang.py --code whisper_gui.py --tag 2.2 lesson long_insert mix
    python3 tests/measure_multilang.py --code /путь/к/старому/whisper_gui.py --tag 2.1 ...
    python3 tests/measure_multilang.py --tag 2.2 --file "~/Downloads/запись.m4a"   # без эталона

Движок берётся из указанного whisper_gui.py (функция transcribe_multilang_fast с одной и
той же сигнатурой в 2.1 и 2.2) - так «было» и «стало» меряются одним скриптом.

Метрики:
  • доля фраз с правильным языком: фраза засчитана, если не меньше половины её слов
    нашлись в расшифровке (выравнивание текстов). Английская фраза, переведённая на
    русский, со словами эталона не совпадает - она не засчитывается;
  • перевод вместо расшифровки: на месте незасчитанной фразы стоит текст в другом
    алфавите (кириллица вместо латиницы или наоборот);
  • пропуск: на месте фразы в расшифровке нет ничего (Whisper молча выбросил речь);
  • вставка найдена: засчитано не меньше 80 % фраз английской вставки;
  • время на час записи.
Результаты - в tests/results/<tag>.json (в git не кладём).
"""
import argparse
import difflib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUDIO = HERE / "audio"
RESULTS = HERE / "results"
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


def norm_words(text):
    t = text.lower().replace("ё", "е")
    return re.findall(r"[a-zа-я0-9]+", t)


def script_of(words):
    cyr = sum(1 for w in words for c in w if "а" <= c <= "я")
    lat = sum(1 for w in words for c in w if "a" <= c <= "z")
    if cyr + lat == 0:
        return None
    return "ru" if cyr >= lat else "en"


def load_engine(code):
    code = Path(code).resolve()
    sys.path.insert(0, str(code.parent))
    spec = importlib.util.spec_from_file_location("scribe_under_test", code)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_engine(mod, path, asr):
    audio = mod._decode_to_array(Path(path), True)
    stats = {}
    t0 = time.monotonic()
    try:
        segs = mod.transcribe_multilang_fast(audio, MODEL, asr, lambda f: None, lambda m: None,
                                             lambda: False, stats=stats)
    except TypeError:                      # 2.1: без stats
        segs = mod.transcribe_multilang_fast(audio, MODEL, asr, lambda f: None, lambda m: None,
                                             lambda: False)
    dt = time.monotonic() - t0
    return segs, dt, len(audio) / 16000, stats


def evaluate(gt, segs):
    """Сверка с эталоном по выравниванию слов."""
    gt_words, gt_pid = [], []
    for i, p in enumerate(gt["phrases"]):
        for w in norm_words(p["text"]):
            gt_words.append(w); gt_pid.append(i)
    out_words = []
    for s in sorted(segs, key=lambda s: s["start"]):
        out_words += norm_words(s.get("text", ""))
    sm = difflib.SequenceMatcher(None, gt_words, out_words, autojunk=False)
    matched = [False] * len(gt_words)
    replaced = {}                              # фраза -> слова расшифровки на её месте
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                matched[i] = True
        elif tag == "replace":
            n = max(1, i2 - i1)
            for k, i in enumerate(range(i1, i2)):
                a = j1 + (j2 - j1) * k // n
                b = j1 + (j2 - j1) * (k + 1) // n
                replaced.setdefault(gt_pid[i], []).extend(out_words[a:b])
    rows = []
    for i, p in enumerate(gt["phrases"]):
        idx = [k for k, pid in enumerate(gt_pid) if pid == i]
        frac = sum(matched[k] for k in idx) / max(1, len(idx))
        ok = frac >= 0.5
        alt = script_of(replaced.get(i, []))
        translated = (not ok) and p["lang"] in ("ru", "en") and alt is not None and alt != p["lang"]
        dropped = (not ok) and frac == 0 and not replaced.get(i)
        rows.append(dict(i=i, lang=p["lang"], start=p["start"], ok=ok, frac=round(frac, 2),
                         translated=translated, dropped=dropped, insert=p.get("insert", False),
                         got=" ".join(replaced.get(i, []))[:120] if not ok else ""))
    return rows


def mix_report(gt, segs):
    out = []
    for p in gt["phrases"]:
        text = " ".join(s["text"].strip() for s in segs
                        if min(s["end"], p["end"] + 0.3) - max(s["start"], p["start"] - 0.3) > 0.2)
        w = p["word"]
        how = "латиницей" if re.search(rf"\b{re.escape(w)}\b", text, re.I) else (
            "другим словом латиницей" if re.search(r"[A-Za-z]{3,}", text) else "кириллицей / пропало")
        out.append(dict(word=w, got=text, how=how))
    return out


def summarize(name, rows, dt, dur):
    def share(rs):
        return f"{sum(r['ok'] for r in rs)}/{len(rs)}" if rs else "-"
    res = dict(name=name, seconds=round(dt, 1), duration=round(dur, 1),
               min_per_hour=round(dt / dur * 60, 1) if dur else None,
               phrases=share(rows), ru=share([r for r in rows if r["lang"] == "ru"]),
               en=share([r for r in rows if r["lang"] == "en"]),
               accuracy=round(sum(r["ok"] for r in rows) / max(1, len(rows)), 3),
               translated=sum(r["translated"] for r in rows), dropped=sum(r["dropped"] for r in rows))
    ins = [r for r in rows if r["insert"]]
    if ins:
        res["insert"] = share(ins)
        res["insert_found"] = sum(r["ok"] for r in ins) >= 0.8 * len(ins)
        res["insert_translated"] = sum(r["translated"] for r in ins)
        res["insert_dropped"] = sum(r["dropped"] for r in ins)
    res["bad"] = [r for r in rows if not r["ok"]][:40]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default=str(HERE.parent / "whisper_gui.py"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--file", help="своя запись без эталона: сохраняем текст и время")
    ap.add_argument("names", nargs="*", default=["lesson", "long_insert", "mix"])
    a = ap.parse_args()
    os.chdir(Path(a.code).resolve().parent)
    mod = load_engine(a.code)
    asr = asr_backend(mod)
    RESULTS.mkdir(exist_ok=True)
    out = {}
    targets = [("file", os.path.expanduser(a.file))] if a.file else [(n, AUDIO / f"{n}.m4a") for n in a.names]
    for name, path in targets:
        segs, dt, dur, stats = run_engine(mod, path, asr)
        text = [dict(start=round(s["start"], 2), end=round(s["end"], 2), text=s["text"].strip()) for s in segs]
        if name == "file":
            words = [w for s in segs for w in norm_words(s["text"])]
            res = dict(name=Path(path).name, seconds=round(dt, 1), duration=round(dur, 1),
                       min_per_hour=round(dt / dur * 60, 1), words=len(words),
                       latin_share=round(sum(script_of([w]) == "en" for w in words) / max(1, len(words)), 3))
        else:
            gt = json.loads((AUDIO / f"{name}.json").read_text(encoding="utf-8"))
            if name.startswith("mix"):
                res = dict(name=name, seconds=round(dt, 1), duration=round(dur, 1), mix=mix_report(gt, segs))
            else:
                res = summarize(name, evaluate(gt, segs), dt, dur)
        if stats:
            res["engine"] = stats
        res["segments"] = text
        out[name] = res
        brief = {k: v for k, v in res.items() if k not in ("segments", "bad", "mix")}
        print(json.dumps(brief, ensure_ascii=False))
        if name.startswith("mix"):
            for m in res["mix"]:
                print(f"   {m['word']:12s} {m['how']:24s} | {m['got']}")
    dest = RESULTS / f"{a.tag}{'-file' if a.file else ''}.json"
    prev = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    prev.update(out)
    dest.write_text(json.dumps(prev, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", dest)


if __name__ == "__main__":
    main()
