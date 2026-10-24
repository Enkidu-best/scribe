"""Задание 06, Б6 - говорящие в проверке A/B (`tests/ab_long.py`): одна диаризация по всему файлу (как в Scribe),
затем те же правила подписи (`core.diarize_and_label`, наибольшее пересечение) к фразам каждого способа.

    HEAVY_LOCK_WHO=background python3 tests/ab_speakers.py <файл> --out <папка с текстами ab_long>

Меряет для каждого способа: чистоту фраз (доля времени фразы за подписанным говорящим), фраз со сменой говорящего
внутри (другой говорит дольше 1 с), и сколько слов подписаны тем же говорящим, что в А. Разметка говорящих кэшируется
в --out (повторный запуск - без модели).
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from ab_long import load  # noqa: E402


def turns_of(file, out):
    cache = out / f"{Path(file).stem} - говорящие.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import core
    from common.lock import heavy_lock as hl
    lock = hl.HeavyLock("Scribe (проверка 06 Б)", "говорящие", who=os.environ.get("HEAVY_LOCK_WHO") or "background",
                        path=hl.default_path(), idle_after=None)
    if not lock.acquire(poll=10):
        sys.exit(75)
    try:
        holder = []
        segs = [{"start": 0.0, "end": 0.1}]
        orig = core._extract_turns
        core._extract_turns = lambda d: holder.append(orig(d)) or holder[-1]
        core.diarize_and_label(file, segs, "", 0, lambda *a: None, lambda *a: None)
        core._extract_turns = orig
    finally:
        lock.release()
    turns = [[float(s), float(e), str(k)] for s, e, k in holder[-1]]
    cache.write_text(json.dumps(turns), encoding="utf-8")
    return turns


def label(segs, turns):
    for s in segs:
        best, ov_best, total = None, 0.0, {}
        for a, b, k in turns:
            ov = max(0.0, min(s["end"], b) - max(s["start"], a))
            if ov:
                total[k] = total.get(k, 0.0) + ov
        if total:
            best = max(total, key=total.get)
            ov_best = total[best]
        s["speaker"], s["_own"], s["_all"], s["_other"] = best, ov_best, sum(total.values()), \
            max([v for k, v in total.items() if k != best] or [0.0])
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    turns = turns_of(a.file, out)
    stem = Path(a.file).stem
    ways = {k: label(load(p), turns) for k, p in (("А", out / f"{stem} - A.txt"), ("Б", out / f"{stem} - Б.txt"),
                                                    ("В", out / f"{stem} - В.txt")) if p.exists()}
    res = {"говорящих": len({k for *_, k in turns})}

    def wl(segs):
        return [(w, s["speaker"]) for s in segs for w in re.findall(r"\w+", (s.get("text") or "").lower())]
    base = wl(ways["А"])
    for k, segs in ways.items():
        own = sum(s["_own"] for s in segs)
        allv = sum(s["_all"] for s in segs) or 1
        res[f"{k}: чистота фраз, %"] = round(100 * own / allv, 1)
        res[f"{k}: фраз со сменой говорящего"] = sum(1 for s in segs if s["_other"] > 1.0)
        if k != "А":
            cur = wl(segs)
            sm = difflib.SequenceMatcher(None, [w for w, _ in base], [w for w, _ in cur], autojunk=False)
            same = tot = 0
            for m in sm.get_matching_blocks():
                for i in range(m.size):
                    tot += 1
                    same += base[m.a + i][1] == cur[m.b + i][1]
            res[f"{k}: тот же говорящий, что в А (по совпавшим словам), %"] = round(100 * same / max(1, tot), 1)
    (out / f"{stem} - говорящие итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
