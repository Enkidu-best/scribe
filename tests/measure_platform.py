#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнение платформ (Scribe 2.4): один и тот же тестовый набор в режимах «Авто», «Русский»
и «Несколько» - на Mac (MLX), в Linux-контейнере (faster-whisper, процессор), на Windows.

    python3 tests/measure_platform.py --tag mac
    python3 tests/measure_platform.py --tag linux-docker
    python3 tests/measure_platform.py --compare mac linux-docker

Метрики на каждую запись и режим:
  • фразы - доля фраз эталона, распознанных на своём языке (как в measure_multilang.py);
  • WER - доля ошибок по словам относительно эталона (замены + пропуски + вставки);
  • «Авто» - заметил ли второй язык (ожидание: lesson, long_insert - да, остальные - нет);
  • время и минут на час записи.
Результаты - tests/results/platform-<tag>.json (в git не кладём).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import core  # noqa: E402
import engine  # noqa: E402
from measure_multilang import AUDIO, RESULTS, MODEL, evaluate, norm_words  # noqa: E402

PLAN = [("lesson", "auto"), ("lesson", "multi"),
        ("long_insert", "auto"), ("long_insert", "multi"),
        ("dictation", "auto"), ("dictation", "ru"),
        ("quiet", "auto"), ("quiet", "ru"),
        ("english", "auto"), ("english_accent", "auto"),
        ("mix", "auto"), ("mix_same", "auto")]
MIXED = {"lesson", "long_insert"}


def wer(ref, hyp):
    """Расстояние Левенштейна по словам / число слов эталона."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / max(1, len(ref))


def run(name, mode, eng):
    audio = core._decode_to_array(AUDIO / f"{name}.m4a", True)
    t0 = time.monotonic()
    info = {}
    if mode == "multi":
        segs = core.transcribe_multilang_fast(audio, MODEL, eng, lambda f: None, lambda m: None, lambda: False)
    else:
        segs, info = core.transcribe_auto(audio, MODEL, eng, lambda f: None, lambda: False,
                                          forced=None if mode == "auto" else mode)
    dt = time.monotonic() - t0
    gt = json.loads((AUDIO / f"{name}.json").read_text(encoding="utf-8"))
    rows = evaluate(gt, segs)
    ref = [w for p in gt["phrases"] for w in norm_words(p["text"])]
    hyp = [w for s in sorted(segs, key=lambda s: s["start"]) for w in norm_words(s.get("text", ""))]
    dur = len(audio) / 16000
    res = dict(name=name, mode=mode, seconds=round(dt, 1), duration=round(dur, 1),
               min_per_hour=round(dt / dur * 60, 1),
               phrases=f"{sum(r['ok'] for r in rows)}/{len(rows)}",
               accuracy=round(sum(r["ok"] for r in rows) / max(1, len(rows)), 3),
               wer=round(wer(ref, hyp), 3))
    if mode == "auto":
        probe = info.get("probe", {})
        res.update(auto_mixed=bool(probe.get("mixed")), auto_ok=bool(probe.get("mixed")) == (name in MIXED),
                   probe_seconds=probe.get("seconds"))
    res["text"] = " ".join(s["text"].strip() for s in segs)[:4000]
    return res


def compare(tags):
    data = {t: json.loads((RESULTS / f"platform-{t}.json").read_text(encoding="utf-8")) for t in tags}
    print("| запись · режим | " + " | ".join(f"{t}: фразы · WER · мин/ч" for t in tags) + " |")
    print("|---|" + "---|" * len(tags))
    for name, mode in PLAN:
        k = f"{name}:{mode}"
        cells = []
        for t in tags:
            r = data[t]["runs"].get(k)
            if not r:
                cells.append("-"); continue
            auto = "" if mode != "auto" else (" · «Авто» " + ("смеш." if r["auto_mixed"] else "один яз.")
                                                + ("" if r["auto_ok"] else " ✗"))
            cells.append(f"{r['phrases']} · {r['wer']:.1%} · {r['min_per_hour']}{auto}")
        print(f"| {name} · {mode} | " + " | ".join(cells) + " |")
    for t in tags:
        m = data[t]["meta"]
        runs = data[t]["runs"].values()
        tot = sum(r["seconds"] for r in runs); dur = sum(r["duration"] for r in runs)
        print(f"{t}: {m['engine']} · {m['machine']} · всего {tot:.0f} с на {dur:.0f} с звука "
              f"({tot / dur * 60:.1f} мин на час)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--only", nargs="*", help="записи из набора (по умолчанию - все)")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    eng = engine.get_engine()
    eng.warmup(MODEL)
    RESULTS.mkdir(exist_ok=True)
    dest = RESULTS / f"platform-{a.tag}.json"
    out = dict(meta=dict(engine=eng.label, device=getattr(eng, "device", ""),
                         machine=f"{platform.system()} {platform.machine()}, {platform.processor() or ''}".strip(", "),
                         python=platform.python_version()), runs={})
    t_all = time.monotonic()
    for name, mode in PLAN:
        if a.only and name not in a.only:
            continue
        r = run(name, mode, eng)
        out["runs"][f"{name}:{mode}"] = r
        print(json.dumps({k: v for k, v in r.items() if k != "text"}, ensure_ascii=False), flush=True)
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"всего {time.monotonic() - t_all:.0f} с -> {dest}")


if __name__ == "__main__":
    main()
