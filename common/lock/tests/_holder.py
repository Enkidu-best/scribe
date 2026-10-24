# -*- coding: utf-8 -*-
"""Второй процесс для тестов: держит замок во временной папке.

    python _holder.py <путь .heavy-lock> <программа> <human|background> <hold|wait|bg-checkpoint> [что делает]

hold - взять (ждать, если занято), напечатать «taken», отдать по строке «release» на stdin (или по EOF);
wait - то же, но сначала напечатать «queued», когда встал в очередь;
bg-checkpoint - фон: взять, затем каждые 0,1 с контрольная точка; печатает «taken», «yielded», «retaken»;
                выходит по строке «release».
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lock import heavy_lock as hl  # noqa: E402

path, program, who, mode = sys.argv[1:5]
what = sys.argv[5] if len(sys.argv) > 5 else ""
import os  # noqa: E402
need = float(os.environ["HL_NEED_GB"]) if os.environ.get("HL_NEED_GB") else None   # бюджет памяти (1.2.0)
lk = hl.HeavyLock(program, what, who=who, path=Path(path), idle_after=None, need_gb=need)
stop = threading.Event()


def read_stdin():
    for line in sys.stdin:
        if line.strip() == "release":
            break
    stop.set()


threading.Thread(target=read_stdin, daemon=True).start()
said = {"q": False}


def on_wait(_h, _ahead):
    if mode == "wait" and not said["q"]:
        said["q"] = True
        print("queued", flush=True)


lk.acquire(on_wait=on_wait, poll=0.05)
print("joined" if lk.joined else "taken", flush=True)
if mode == "bg-checkpoint":
    while not stop.is_set():
        if lk.checkpoint(poll=0.05):
            print("yielded", flush=True)
            if lk.held:
                print("retaken", flush=True)
        time.sleep(0.1)
else:
    stop.wait()
lk.release()
print("released", flush=True)
