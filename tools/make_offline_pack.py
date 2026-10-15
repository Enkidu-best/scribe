#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Офлайн-пакет моделей Scribe для Windows и Linux (faster-whisper): один zip, который
распаковывают и выбирают в Настройках - «Загрузить модели из папки…». Интернет не нужен.

    python3 tools/make_offline_pack.py [--out папка] [--no-pyannote]

Внутри - обычная структура кэша Hugging Face (hub/models--…/refs, snapshots), но без
символических ссылок: файлы лежат прямо в snapshots/<ревизия>/, как их хранит
huggingface_hub на Windows. Поэтому архив распаковывается где угодно.

Состав:
  mobiuslabsgmbh/faster-whisper-large-v3-turbo   распознавание (MIT)
  Systran/faster-whisper-base                    проверка языка для «Авто» (MIT)
  pyannote/speaker-diarization-community-1       определение говорящих (CC-BY-4.0:
                                                 распространять можно, с указанием авторства - см. NOTICE)
Silero VAD уже входит в программу (assets/models).

pyannote на Hugging Face закрыт формой согласия; скачать его для пакета можно токеном
(настройка hf_token Scribe или переменная HF_TOKEN) или взять из локального кэша.
"""
import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPOS = [("mobiuslabsgmbh/faster-whisper-large-v3-turbo", False),
         ("Systran/faster-whisper-base", False),
         ("pyannote/speaker-diarization-community-1", True)]

NOTICE = """Scribe - офлайн-пакет моделей
=============================

Распакуйте архив в любую папку и выберите её в Scribe: Настройки - «Загрузить модели из папки…».

Модели и лицензии
-----------------
1. Whisper large-v3-turbo (OpenAI), формат CTranslate2:
   https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo - лицензия MIT.
2. Whisper base (OpenAI), формат CTranslate2:
   https://huggingface.co/Systran/faster-whisper-base - лицензия MIT.
3. pyannote speaker-diarization-community-1 (pyannoteAI):
   https://huggingface.co/pyannote/speaker-diarization-community-1
   Лицензия CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/).
   Авторство: pyannote.audio, H. Bredin и соавторы; модель распространяется без изменений.
   Если вы пользуетесь ею в работе, авторы просят ссылаться на pyannote.audio.

Whisper: Copyright (c) 2022 OpenAI (MIT License).
"""


def fetch(repo, cache, token):
    from huggingface_hub import snapshot_download
    try:                                    # сперва - из обычного кэша этой машины
        src = snapshot_download(repo, local_files_only=True)
    except Exception:
        src = snapshot_download(repo, token=token or None, cache_dir=cache)
    return Path(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--name", default="Scribe-models-2.4")
    ap.add_argument("--no-pyannote", action="store_true")
    ap.add_argument("--cache", help="папка для скачивания (по умолчанию - временная); с ней прерванная загрузка докачивается")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            import core
            token = core.load_config().get("hf_token")
        except Exception:
            token = None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    zpath = out / (a.name + ".zip")
    total = 0
    tmp = None if a.cache else tempfile.TemporaryDirectory()
    cache = a.cache or tmp.name
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        top = a.name
        z.writestr(f"{top}/NOTICE.txt", NOTICE)
        for repo, gated in REPOS:
            if gated and a.no_pyannote:
                continue
            snap = fetch(repo, cache, token if gated else None)
            rev = snap.name
            base = f"{top}/hub/models--{repo.replace('/', '--')}"
            z.writestr(f"{base}/refs/main", rev)
            n = 0
            for f in sorted(snap.rglob("*")):
                if f.is_file():               # символическая ссылка -> настоящее содержимое
                    rel = f.relative_to(snap).as_posix()
                    big = f.stat().st_size > 50 * 2**20
                    z.write(f, f"{base}/snapshots/{rev}/{rel}",
                            compress_type=zipfile.ZIP_STORED if big else zipfile.ZIP_DEFLATED)
                    n += f.stat().st_size
            total += n
            print(f"{repo}: {n / 2**20:.0f} МБ")
    if tmp:
        tmp.cleanup()
    print(f"готово: {zpath} - {zpath.stat().st_size / 2**20:.0f} МБ (модели {total / 2**20:.0f} МБ)")


if __name__ == "__main__":
    main()
