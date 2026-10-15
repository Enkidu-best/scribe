#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Всё, что зависит от системы (Mac / Windows / Linux), - в одном месте:
пути к настройкам и логам, открытие файла и папки, адрес канала для процесса
диктофона, ffmpeg из сборки, библиотеки CUDA для видеокарты NVIDIA.

Модуль лёгкий (только стандартная библиотека; platformdirs - если есть): его
импортирует и процесс диктофона, который запускает Legal Knowledge Desk.
"""
import os
import platform
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
IS_APPLE_SILICON = IS_MAC and platform.machine() == "arm64"
FROZEN = bool(getattr(sys, "frozen", False))
APP = "Scribe"

# слова для окна: на Windows и Linux не «Mac» и не «Терминал»
HERE = "на этом Mac" if IS_MAC else "на этом компьютере"
SHELL = "в Терминале" if IS_MAC else "в командной строке"
HAND = "pointinghand" if IS_MAC else "hand2"     # курсор-ссылка: «pointinghand» есть только в Tk для Mac
if IS_MAC:
    FFMPEG_HINT = "Не найден ffmpeg. В Терминале:\n  brew install ffmpeg"
    MIC_HINT = "Установите один раз в Терминале:\n\nbrew install portaudio\n"
elif IS_WIN:
    FFMPEG_HINT = ("Не найден ffmpeg. Он входит в установщик Scribe - переустановите программу. "
                   "При запуске из исходников: winget install Gyan.FFmpeg")
    MIC_HINT = "Установите один раз в командной строке:\n\n"
else:
    FFMPEG_HINT = "Не найден ffmpeg. В командной строке:\n  sudo apt install ffmpeg"
    MIC_HINT = "Установите один раз в командной строке:\n\nsudo apt install libportaudio2\n"


def fix_streams(name="scribe.log"):
    """Оконная сборка (Scribe.exe) запускается без консоли: sys.stdout и sys.stderr - None, и любой
    print/tqdm (полоска скачивания моделей) уронил бы программу. Пишем такой вывод в журнал.
    В консоли Windows (scribe-cli.exe) - UTF-8, иначе кириллица в перенаправленном выводе падает."""
    if sys.stdout is None or sys.stderr is None:
        try:
            d = log_dir()
            d.mkdir(parents=True, exist_ok=True)
            f = open(d / name, "a", encoding="utf-8", buffering=1)
        except Exception:
            f = open(os.devnull, "w", encoding="utf-8")
        if sys.stdout is None:
            sys.stdout = f
        if sys.stderr is None:
            sys.stderr = f
    for st in (sys.stdout, sys.stderr):
        try:
            if (getattr(st, "encoding", "") or "").lower().replace("-", "") != "utf8":
                st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def source_dir() -> Path:
    """Папка с ресурсами программы: рядом с кодом или внутри сборки PyInstaller."""
    if FROZEN and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def resource(*parts) -> Path:
    return source_dir().joinpath(*parts)


def _pd(kind):
    try:
        import platformdirs
        return Path(getattr(platformdirs, kind)(APP, appauthor=False))
    except Exception:
        base = Path(os.environ.get("APPDATA") or Path.home() / ".config") if IS_WIN else Path.home() / ".config"
        return base / APP


def config_path() -> Path:
    """Файл настроек. На Mac - прежний ~/.audio2text.json (настройки не теряются)."""
    if IS_MAC:
        return Path.home() / ".audio2text.json"
    return _pd("user_config_dir") / "settings.json"


def log_dir() -> Path:
    """Папка логов. На Mac - ~/Library/Logs/Scribe, как в 2.2."""
    if IS_MAC:
        return Path.home() / "Library" / "Logs" / APP
    return _pd("user_log_dir")


def data_dir() -> Path:
    """Данные программы (например, библиотеки CUDA, скачанные по кнопке)."""
    return _pd("user_data_dir")


def open_path(path):
    """Открыть файл программой по умолчанию."""
    path = str(path)
    if IS_WIN:
        os.startfile(path)                    # noqa: S606 - так открывают файлы в Windows
    elif IS_MAC:
        subprocess.run(["open", path], check=False)
    else:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def reveal(path):
    """Показать файл в папке (Finder / Проводник / файловый менеджер)."""
    path = Path(path)
    if IS_WIN:
        subprocess.run(["explorer", f"/select,{path}"], check=False)
    elif IS_MAC:
        subprocess.run(["open", "-R", str(path)], check=False)
    else:
        subprocess.Popen(["xdg-open", str(path.parent)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_folder(path):
    path = str(path)
    if IS_WIN:
        os.startfile(path)                    # noqa: S606
    elif IS_MAC:
        subprocess.run(["open", path], check=False)
    else:
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def worker_address():
    """Адрес канала к процессу диктофона: на Mac и Linux - сокет AF_UNIX,
    на Windows - именованный канал AF_PIPE. Протокол сообщений тот же."""
    tag = secrets.token_hex(6)
    if IS_WIN:
        return rf"\\.\pipe\scribe_{tag}", "AF_PIPE"
    return os.path.join(tempfile.gettempdir(), f"scribe_{tag}.sock"), "AF_UNIX"


def setup_path():
    """ffmpeg/ffprobe и ollama должны находиться и при запуске не из терминала."""
    extra = []
    if FROZEN:
        extra.append(str(source_dir() / "ffmpeg"))
        extra.append(str(Path(sys.executable).parent / "ffmpeg"))
    if IS_MAC:
        extra += ["/opt/homebrew/bin", "/usr/local/bin", os.path.expanduser("~/.local/bin")]
    if IS_WIN:
        extra.append(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama"))
    cur = os.environ.get("PATH", "").split(os.pathsep)
    for p in extra:
        if os.path.isdir(p) and p not in cur:
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
    if IS_WIN and FROZEN:
        _hide_console_windows()


def _hide_console_windows():
    """Scribe.exe - оконная программа без консоли. Без этого каждый вызов ffmpeg/ffprobe
    открывал бы на миг чёрное окно консоли."""
    if getattr(subprocess.Popen, "_scribe_nowin", False):
        return
    base = subprocess.Popen

    class _Popen(base):
        _scribe_nowin = True

        def __init__(self, *a, **k):
            k["creationflags"] = k.get("creationflags", 0) | 0x08000000   # CREATE_NO_WINDOW
            super().__init__(*a, **k)

    subprocess.Popen = _Popen


def cuda_dir() -> Path:
    return data_dir() / "cuda"


def setup_cuda():
    """Библиотеки CUDA (cuBLAS, cuDNN), скачанные кнопкой «Ускорение NVIDIA», - в путь
    поиска DLL. Вызывать до первого импорта ctranslate2."""
    d = cuda_dir()
    if IS_WIN and d.is_dir():
        try:
            os.add_dll_directory(str(d))
        except Exception:
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def has_nvidia() -> bool:
    """Есть ли видеокарта NVIDIA с драйвером (по nvidia-smi)."""
    import shutil
    exe = shutil.which("nvidia-smi")
    if not exe and IS_WIN:
        cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(cand) if cand.exists() else None
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False


def ollama_installed() -> bool:
    import shutil
    if shutil.which("ollama"):
        return True
    if IS_MAC:
        return os.path.isdir("/Applications/Ollama.app")
    if IS_WIN:
        return Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")).exists()
    return False


def start_ollama():
    """Тихо запустить Ollama, если она установлена. -> True, если что-то запустили."""
    import shutil
    if IS_MAC and os.path.isdir("/Applications/Ollama.app"):
        subprocess.run(["open", "-g", "-a", "Ollama"], check=False)
        return True
    if IS_WIN:
        app = Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"))
        if app.exists():
            subprocess.Popen([str(app)], creationflags=0x08000000)      # CREATE_NO_WINDOW
            return True
    exe = shutil.which("ollama")
    if exe:
        kw = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if IS_WIN:
            kw["creationflags"] = 0x08000000
        else:
            kw["start_new_session"] = True
        subprocess.Popen([exe, "serve"], **kw)
        return True
    return False
