"""Задание 06, Б3 - продолжение после сбоя вживую: путь Scribe (`core.transcribe_auto` → `transcribe_file` → пачки),
процесс падает на доле --die-at (os._exit - как сбой движка или закрытое окно), второй запуск продолжает.

    HEAVY_LOCK_WHO=background python3 tests/live_resume_06b.py <файл> --out <папка вне проекта> [--lang ru]

Папка готовых кусков - своя (SCRIBE_RESUME_DIR в --out), не папка Scribe. Печатает время до сбоя, время
продолжения, долю готового после сбоя и совпадение текста с распознаванием без сбоя (если оно уже есть в --out).
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_once(a, die_at):
    os.environ["SCRIBE_RESUME_DIR"] = str(Path(a.out) / "resume")
    sys.path.insert(0, str(ROOT))
    import core
    import engine
    from common.lock import heavy_lock as hl
    lock = hl.HeavyLock("Scribe (проверка 06 Б)", "продолжение после сбоя", who=os.environ.get("HEAVY_LOCK_WHO") or "background",
                        path=hl.default_path(), idle_after=None)
    if not lock.acquire(poll=10):
        return 75
    try:
        audio = core._decode_to_array(a.file)
        eng = engine.get_engine()
        store = core.resume_store(a.file, a.model, a.lang, True)

        def prog(f):
            if die_at and f >= die_at:
                print(json.dumps({"упал на": round(f, 3)}), flush=True)
                os._exit(3)
        t0 = time.time()
        segs, _ = core.transcribe_auto(audio, a.model, eng, prog, lambda: False, forced=a.lang, store=store)
        dt = round(time.time() - t0, 1)
        (Path(a.out) / f"{Path(a.file).stem} - продолжение.txt").write_text(
            "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text'].strip()}" for s in segs), encoding="utf-8")
        store.clear()
        try:
            import mlx.core as mx
            peak = round(mx.get_peak_memory() / 2**30, 2)
        except Exception:  # noqa: BLE001
            peak = None
        print(json.dumps({"распознавание, с": dt, "сегментов": len(segs), "пик памяти MLX, ГБ": peak}), flush=True)
    finally:
        lock.release()
    return 0


def words(path):
    return re.findall(r"\w+", re.sub(r"^\[[^]]*\]", "", Path(path).read_text(encoding="utf-8"), flags=re.M).lower())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--die-at", type=float, default=None)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        return run_once(a, a.die_at)
    me = [sys.executable, __file__, a.file, "--out", a.out, "--lang", a.lang, "--model", a.model, "--child"]
    t0 = time.time()
    p1 = subprocess.run(me + ["--die-at", str(a.die_at or 0.4)], capture_output=True, text=True)
    t1 = round(time.time() - t0, 1)
    sys.path.insert(0, str(ROOT))
    os.environ["SCRIBE_RESUME_DIR"] = str(Path(a.out) / "resume")
    import core
    share = core.resume_share(a.file, a.model, a.lang, True)
    t0 = time.time()
    p2 = subprocess.run(me, capture_output=True, text=True)
    res = {"первый запуск: код": p1.returncode, "первый запуск, с": t1, "готово после сбоя": share,
           "второй запуск: код": p2.returncode, "второй запуск, с": round(time.time() - t0, 1),
           "второй": [json.loads(x) for x in p2.stdout.splitlines() if x.startswith("{")],
           "папка готовых кусков после результата": any((Path(a.out) / "resume").glob("*"))}
    ref = Path(a.out) / f"{Path(a.file).stem} - В.txt"
    got = Path(a.out) / f"{Path(a.file).stem} - продолжение.txt"
    if ref.exists() and got.exists():
        x, y = words(ref), words(got)
        sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
        res["совпадение с распознаванием без сбоя, %"] = round(200 * sum(m.size for m in sm.get_matching_blocks()) / max(1, len(x) + len(y)), 1)
    if p2.returncode:
        res["ошибка"] = p2.stderr[-1500:]
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if p1.returncode == 3 and p2.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
