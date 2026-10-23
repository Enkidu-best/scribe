#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scribe без окна: расшифровка из командной строки (Mac, Windows, Linux, сервер).

    python3 scribe_cli.py запись.m4a                        # «Авто», .txt рядом с записью
    python3 scribe_cli.py --lang multi --fmt docx a.mp3 b.wav
    python3 scribe_cli.py --json отчёт.json --out папка *.m4a
    python3 scribe_cli.py --engine                          # какой движок выбран на этой машине
    python3 scribe_cli.py --download-models                 # скачать нужные модели заранее

В сборке для Windows - scribe-cli.exe рядом со Scribe.exe.
Режимы: auto («Авто» - сам замечает второй язык), ru, en, multi («Несколько»).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import platform_util as pu

pu.fix_streams()
pu.setup_path()
pu.setup_cuda()

import core  # noqa: E402


def _say(*a):
    print(*a, flush=True)


def transcribe_file(path, lang, fmt, out_dir, normalize, model, eng):
    t0 = time.monotonic()
    warn = []
    try:
        audio = core._decode_to_array(path, normalize)
    except RuntimeError:
        if not normalize:
            raise
        audio = core._decode_to_array(path, False)
        warn.append("нормализация не удалась - обработано как есть")
    dur = len(audio) / core.SR_FAST
    t_dec = time.monotonic() - t0
    info = {}
    last = [0.0]

    def prog(f):
        if f >= 0 and f - last[0] >= 0.25:
            last[0] = f
            _say(f"   {path.name}: {f:.0%}")

    if lang == "multi":
        stats = {}
        segs = core.transcribe_multilang_fast(audio, model, eng, prog, lambda m: None, lambda: False, stats=stats)
        info = dict(mode="multi", engine_stats=stats)
    else:
        segs, info = core.transcribe_auto(audio, model, eng, prog, lambda: False,
                                          forced=None if lang == "auto" else lang)
        probe = info.pop("probe", {})
        info["probe"] = {k: probe.get(k) for k in ("mixed", "dominant", "confirmed", "confirmed_time",
                                                     "speech_time", "seconds", "method")}
    out = (Path(out_dir) if out_dir else path.parent) / (path.stem + "." + fmt)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "txt":
        core.write_txt(segs, out, False, False)
    elif fmt == "docx":
        core.write_docx(segs, out, path.stem, False, False)
    else:
        core.write_srt(segs, out, False)
    text = " ".join(s["text"].strip() for s in segs)
    sec = time.monotonic() - t0
    return dict(file=str(path), out=str(out), duration=round(dur, 1), seconds=round(sec, 1),
                decode_seconds=round(t_dec, 1), min_per_hour=round(sec / dur * 60, 1) if dur else None,
                words=len(text.split()), engine=eng.label, warnings=warn, **info)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scribe-cli", description="Scribe: расшифровка аудио и видео без окна")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--lang", choices=("auto", "ru", "en", "multi"), default="auto")
    ap.add_argument("--fmt", choices=("txt", "docx", "srt"), default="txt")
    ap.add_argument("--out", help="папка для результатов (по умолчанию - рядом с записью)")
    ap.add_argument("--json", help="записать отчёт в JSON")
    ap.add_argument("--model", default="turbo", help="turbo (по умолчанию), large-v3, medium, small, base")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--engine", action="store_true", help="показать выбранный движок и выйти")
    ap.add_argument("--download-models", action="store_true", help="скачать нужные модели и выйти")
    a = ap.parse_args(argv)
    cfg = core.load_config()
    core._apply_models_dir(cfg.get("models_dir"))
    import engine
    eng = engine.get_engine()
    if a.engine:
        _say(json.dumps(dict(engine=eng.name, device=getattr(eng, "device", ""), label=eng.label,
                             platform=sys.platform, settings=str(core.CONFIG_PATH), logs=str(pu.log_dir()),
                             models=str(__import__("models").hub_dir())), ensure_ascii=False))
        return 0
    if a.download_models:
        import models
        for it in models.needed(dict(cfg, model_repo=engine.MODELS[engine.model_key(a.model)][0]), eng):
            if it["local"]:
                _say(f"есть: {it['repo']}")
            elif it["optional"] and not cfg.get("hf_token"):
                _say(f"пропущено (нужен токен Hugging Face): {it['repo']}")
            else:
                _say(f"скачиваю: {it['repo']} (≈ {it['mb']} МБ)")
                models.download(it["repo"], cfg.get("hf_token") if it["optional"] else None)
        return 0
    if not a.files:
        ap.print_help()
        return 2
    model = engine.MODELS[engine.model_key(a.model)][0] if engine.model_key(a.model) in engine.MODELS else a.model
    _say(f"Scribe {core.APP_VERSION} · {engine.engine_info()} · режим {a.lang}")
    # очередь видеокарты (стандарт 2.4): командная строка - фоновый прогон, ждёт молча и уступает человеку
    # между файлами (контрольная точка); HEAVY_LOCK_WHO=human - если запускает сам Андрей
    from common.lock import heavy_lock as hl
    who = os.environ.get("HEAVY_LOCK_WHO") if os.environ.get("HEAVY_LOCK_WHO") in (hl.HUMAN, hl.BACKGROUND) \
        else hl.BACKGROUND
    lock = hl.HeavyLock("Scribe (командная строка)", "расшифровка", who=who, held_env="SCRIBE_HEAVY_LOCK_HELD",
                        path_env="SCRIBE_HEAVY_LOCK", idle_after=None)
    shown = {"t": 0.0}

    def waiting(h, ahead):
        if time.time() - shown["t"] >= 60:
            shown["t"] = time.time()
            _say(f"жду видеокарту: «{h.program}», с {h.since_text}" if h else "жду видеокарту: перед нами очередь")

    lock.acquire(on_wait=waiting, poll=2)
    report, code = [], 0
    try:
        for f in a.files:
            path = Path(f).expanduser()
            lock.set_what(f"расшифровка «{path.name}»")
            try:
                r = transcribe_file(path, a.lang, a.fmt, a.out, not a.no_normalize, model, eng)
                _say(f"готово: {path.name} → {Path(r['out']).name} · {r['seconds']} с на {r['duration']} с звука"
                     + (f" · {', '.join(r['warnings'])}" if r["warnings"] else ""))
            except Exception as e:
                r, code = dict(file=str(path), error=str(e)), 1
                _say(f"ошибка: {path.name}: {e}")
            report.append(r)
            if lock.checkpoint(on_wait=waiting, poll=2):
                _say("видеокарту ждал человек - уступил и снова получил")
    finally:
        lock.release()
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return code


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    import os
    os._exit(code)      # без разбора C++-объектов (см. whisper_gui.main)
