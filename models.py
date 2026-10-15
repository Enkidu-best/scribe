#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модели Scribe: какие нужны, есть ли они на диске, скачивание с прогрессом,
офлайн-пакет («Загрузить модели из папки…») и докачка библиотек CUDA для NVIDIA.

Модели лежат в обычном кэше Hugging Face (HF_HOME/hub). Офлайн-пакет - та же
структура: папка, внутри «hub/models--…». Выбрать её в Настройках - и Scribe берёт
модели оттуда, не обращаясь к интернету.
"""
import io
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import platform_util as pu

PYANNOTE = "pyannote/speaker-diarization-community-1"   # CC-BY-4.0, скачивание - с токеном HF
PYANNOTE_MB = 34
SIEVE = "base"


def hub_dir() -> Path:
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except Exception:
        base = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        return Path(base) / "hub"


def repo_dir(repo) -> Path:
    return hub_dir() / ("models--" + repo.replace("/", "--"))


def dir_size(path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def is_local(repo) -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def needed(cfg, eng):
    """Список моделей для этой машины: [dict(repo, title, mb, local, optional, why)]."""
    import engine as E
    main = cfg.get("model_repo") or E.MODELS[E.DEFAULT][0]
    items = [dict(repo=eng.repo(main), title="Распознавание речи (Whisper " + E.model_key(main) + ")",
                  mb=eng.size_mb(main), optional=False),
             dict(repo=eng.repo(SIEVE), title="Проверка языка для «Авто» (whisper-base)",
                  mb=eng.size_mb(SIEVE), optional=False)]
    dm = cfg.get("dict_model")
    if dm and E.model_key(dm) != E.model_key(main):
        items.append(dict(repo=eng.repo(dm), title=f"Диктофон (Whisper {E.model_key(dm)})",
                          mb=eng.size_mb(dm), optional=False))
    items.append(dict(repo=PYANNOTE, title="Определение говорящих (pyannote)", mb=PYANNOTE_MB, optional=True))
    for it in items:
        it["local"] = is_local(it["repo"])
    return items


def online(timeout=3):
    """Есть ли связь с Hugging Face (для докачки модели, которой нет на диске)."""
    try:
        with _open_url("https://huggingface.co", timeout):
            return True
    except Exception:
        return False


def download(repo, token=None):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, token=token or None)


class Downloader:
    """Скачивание нескольких моделей в фоне; прогресс - по размеру папки в кэше."""

    def __init__(self, items, token=None):
        self.items = [it for it in items if not it["local"]]
        self.token = token
        self.total = sum(it["mb"] for it in self.items) * 1024 * 1024
        self.base = {it["repo"]: dir_size(repo_dir(it["repo"])) for it in self.items}
        self.current = None
        self.error = None
        self.done = False

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            for it in self.items:
                self.current = it
                download(it["repo"], self.token if it["repo"] == PYANNOTE else None)
                it["local"] = True
        except Exception as e:
            self.error = str(e)
        self.done = True

    def progress(self):
        got = sum(max(0, dir_size(repo_dir(it["repo"])) - self.base[it["repo"]]) for it in self.items)
        return got, self.total


def use_folder(folder):
    """«Загрузить модели из папки…»: папка офлайн-пакета (внутри hub/models--…) или сама hub.
    -> (hf_home или None, сообщение)"""
    folder = Path(folder)
    inner = [p for p in folder.iterdir() if p.is_dir() and (p / "hub").is_dir()] if folder.is_dir() else []
    if not (folder / "hub").is_dir() and len(inner) == 1:
        folder = inner[0]                   # «Извлечь всё» в Windows кладёт архив в папку с тем же именем
    if (folder / "hub").is_dir():
        home = folder
    elif folder.name == "hub" or any(p.name.startswith("models--") for p in folder.iterdir()):
        home = folder.parent if folder.name == "hub" else None
        if home is None:
            return None, "В папке есть модели, но нет подпапки «hub». Выберите папку офлайн-пакета целиком."
    else:
        return None, "В этой папке нет моделей (ожидается папка офлайн-пакета с подпапкой «hub»)."
    apply_home(str(home))
    found = [p.name.replace("models--", "").replace("--", "/") for p in (home / "hub").iterdir()
             if p.name.startswith("models--")]
    return str(home), "Найдены модели: " + ", ".join(sorted(found))


def apply_home(home):
    """Перенаправить кэш Hugging Face в другую папку - и сразу, и для следующих запусков."""
    if not home:
        return
    os.environ["HF_HOME"] = home
    os.environ.pop("HF_HUB_CACHE", None)
    if "huggingface_hub.constants" in sys.modules:
        c = sys.modules["huggingface_hub.constants"]
        c.HF_HOME = home
        c.HF_HUB_CACHE = os.path.join(home, "hub")
        c.HUGGINGFACE_HUB_CACHE = c.HF_HUB_CACHE


# ----------------------------------------------- ускорение NVIDIA (Windows)
NVIDIA_PKGS = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12")   # CTranslate2 4.x: CUDA 12 + cuDNN 9


def _open_url(url, timeout):
    """HTTPS с сертификатами certifi (в сборке и в Python с python.org системных может не быть)."""
    import ssl
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    return urllib.request.urlopen(url, timeout=timeout, context=ctx)


def _wheel_url(pkg, plat):
    """Самая свежая сборка пакета NVIDIA для этой платформы (у cuDNN - только 9.x: с ним собран
    CTranslate2 4.x; у новых выпусков сборки для Windows бывают не сразу)."""
    with _open_url(f"https://pypi.org/pypi/{pkg}/json", 20) as r:
        data = json.loads(r.read().decode("utf-8"))

    def key(v):
        return [int(x) for x in v.split(".") if x.isdigit()]
    versions = sorted(data["releases"], key=key, reverse=True)
    if pkg == "nvidia-cudnn-cu12":
        versions = [v for v in versions if v.startswith("9.")]
    for v in versions:
        for f in data["releases"][v]:
            if f["filename"].endswith(".whl") and plat in f["filename"] and not f.get("yanked"):
                return f["url"], f["size"]
    raise RuntimeError(f"нет сборки {pkg} для {plat}")


def nvidia_size_mb():
    try:
        return sum(_wheel_url(p, "win_amd64")[1] for p in NVIDIA_PKGS) / 1024 / 1024
    except Exception:
        return 1200.0


def nvidia_download(progress=lambda got, total: None, cancel=lambda: False):
    """Скачать cuBLAS и cuDNN (официальные пакеты NVIDIA с PyPI) и положить их DLL
    в папку данных Scribe. После перезапуска faster-whisper работает на видеокарте."""
    dest = pu.cuda_dir()
    dest.mkdir(parents=True, exist_ok=True)
    urls = [_wheel_url(p, "win_amd64") for p in NVIDIA_PKGS]
    total, got = sum(s for _, s in urls), 0
    for url, _ in urls:
        buf = io.BytesIO()
        with _open_url(url, 60) as r:
            while True:
                if cancel():
                    raise RuntimeError("отменено")
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                buf.write(chunk)
                got += len(chunk)
                progress(got, total)
        with zipfile.ZipFile(buf) as z:
            for n in z.namelist():
                if n.lower().endswith(".dll"):
                    with z.open(n) as src, open(dest / Path(n).name, "wb") as out:
                        shutil.copyfileobj(src, out)
    (dest / "ready.txt").write_text(time.strftime("%Y-%m-%d"), encoding="utf-8")
    return dest
