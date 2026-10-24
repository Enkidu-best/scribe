#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scribe 2.7: одна нить для MLX - нет «There is no Stream(gpu, N) in current thread».

    python3 tests/test_mlx_thread.py          # проверка движка (нужна видеокарта Mac - под замком)
    python3 tests/test_mlx_thread.py --old    # + как было: та же последовательность без исполнителя

Каждая проверка запускает распознавание из НОВОЙ нити, как это делает окно (новая нить на каждый
запуск):
  1. три запуска подряд;
  2. запуск после искусственного сбоя посреди распознавания;
  3. детектор языка фраз («Авто», «Несколько») из разных нитей;
  4. файл кусками по паузам пачками (06 Б): два запуска и запуск после сбоя посреди пачек.
"""
import sys
import threading
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core  # noqa: E402
import engine  # noqa: E402
import multilang  # noqa: E402

MODEL = core.MODEL_REPO
fails = []


def in_thread(fn):
    box = {}

    def go():
        try:
            box["r"] = fn()
        except BaseException as e:          # noqa: BLE001
            box["e"] = e
    t = threading.Thread(target=go)
    t.start(); t.join()
    if "e" in box:
        raise box["e"]
    return box.get("r")


def check(cond, what):
    print(("OK   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        fails.append(what)


def old_way(audio):
    """Как было до 2.7: mlx_whisper прямо в нити запуска. Первый запуск падает, не успев
    вычислить кэш фильтров спектрограммы (так выглядит сбой нехватки памяти), второй - в новой нити."""
    import mlx_whisper
    from mlx_whisper import audio as A

    def first():
        A.mel_filters.cache_clear(); A.hanning.cache_clear()
        A.mel_filters(128)                  # ленивый массив в кэше этой нити...
        raise MemoryError("искусственный сбой (как нехватка памяти)")   # ...и нить упала
    try:
        in_thread(first)
    except MemoryError:
        pass
    try:
        in_thread(lambda: mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, language="en", verbose=False))
        return "без ошибки"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def main():
    audio = core._decode_to_array(HERE / "audio" / "english.m4a", True)[:30 * 16000]
    if "--old" in sys.argv:
        r = old_way(audio)
        print("было (без исполнителя), второй запуск после сбоя:", r, flush=True)
        from mlx_whisper import audio as A
        A.mel_filters.cache_clear(); A.hanning.cache_clear()
    eng = engine.get_engine()
    print("движок:", eng.label, flush=True)

    # 1. три запуска подряд, каждый в новой нити
    for k in range(3):
        t0 = time.monotonic()
        try:
            segs = in_thread(lambda: eng.transcribe(audio, MODEL, "en"))
            text = " ".join(s["text"] for s in segs)
            check(len(text.split()) > 30, f"запуск {k + 1}: {len(text.split())} слов, {time.monotonic() - t0:.1f} с")
        except Exception as e:
            check(False, f"запуск {k + 1}: {type(e).__name__}: {e}")

    # 2. сбой посреди распознавания - и следующий запуск
    def boom(f):
        if f > 0.2:
            raise RuntimeError("искусственный сбой посреди распознавания")
    try:
        in_thread(lambda: eng.transcribe(audio, MODEL, "en", prog=boom))
        check(False, "искусственный сбой не случился")
    except RuntimeError as e:
        check("искусственный" in str(e), "искусственный сбой дошёл до вызывающей нити")
    try:
        segs = in_thread(lambda: eng.transcribe(audio, MODEL, "en"))
        check(len(" ".join(s["text"] for s in segs).split()) > 30, "запуск после сбоя - без ошибки")
    except Exception as e:
        check(False, f"запуск после сбоя: {type(e).__name__}: {e}")

    # 3. детектор языка фраз из разных нитей (сито «Авто» и детектор «Несколько»)
    lesson = core._decode_to_array(HERE / "audio" / "lesson.m4a", True)[:60 * 16000]
    for k in range(2):
        try:
            pr = in_thread(lambda: multilang.probe_languages(lesson, MODEL))
            blocks = in_thread(lambda: multilang.plan(lesson, MODEL))
            check(pr["mixed"] and len(blocks) > 2, f"языки фраз, проход {k + 1}: смешанная={pr['mixed']}, блоков {len(blocks)}")
        except Exception as e:
            check(False, f"языки фраз, проход {k + 1}: {type(e).__name__}: {e}")

    # 4. файл кусками по паузам пачками (06 Б) из разных нитей, сбой посреди пачек - и снова
    import vad
    spans = vad.split_phrases(lesson)
    for k in range(2):
        try:
            segs = in_thread(lambda: eng.transcribe_long(lesson, spans, MODEL, "ru"))
            check(len(segs) > 3, f"файл пачками, запуск {k + 1}: {len(segs)} фраз")
        except Exception as e:
            check(False, f"файл пачками, запуск {k + 1}: {type(e).__name__}: {e}")

    def boom2(f):
        if f > 0.3:
            raise RuntimeError("искусственный сбой посреди пачек")
    try:
        in_thread(lambda: eng.transcribe_long(lesson, spans, MODEL, "ru", prog=boom2))
        check(False, "искусственный сбой посреди пачек не случился")
    except RuntimeError as e:
        check("пачек" in str(e), "сбой посреди пачек дошёл до вызывающей нити")
    try:
        segs = in_thread(lambda: eng.transcribe_long(lesson, spans, MODEL, "ru"))
        check(len(segs) > 3, "файл пачками после сбоя - без ошибки")
    except Exception as e:
        check(False, f"файл пачками после сбоя: {type(e).__name__}: {e}")

    # выгрузка из другой нити и снова распознавание
    in_thread(eng.unload)
    try:
        segs = in_thread(lambda: eng.transcribe(audio, MODEL, "en"))
        check(len(segs) > 0, "после выгрузки из другой нити - без ошибки")
    except Exception as e:
        check(False, f"после выгрузки: {type(e).__name__}: {e}")
    print("ИТОГ:", "OK" if not fails else f"ОШИБКИ: {fails}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    code = main()
    import os
    sys.stdout.flush()
    os._exit(code)
