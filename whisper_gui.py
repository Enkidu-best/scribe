#!/usr/bin/env python3
"""
Scribe - локальная транскрипция аудио и видео в текст на MLX Whisper (Apple Silicon).

Одно окно с боковым меню (оформление - как у Legal Knowledge Desk, см. theme.py):
  • «Расшифровка» - файлы перетаскиванием или кнопкой (можно целую папку);
    язык: авто / русский / английский / несколько; формат: .txt / .docx / .srt;
    говорящие (pyannote), таймкоды, по желанию - краткое содержание отдельным
    документом (локальная модель через Ollama, см. summary.py);
    у каждого файла свой статус и кнопки «Открыть»; расшифровку можно остановить.
  • «Диктофон» - живая диктовка с микрофона (dictaphone.py).
  • «Настройки» - модель, нормализация громкости, токен Hugging Face,
    модель для краткого содержания, папка моделей.

Результат сохраняется рядом с исходным файлом.
"""

import os
import io
import re
import json
import time
import queue
import threading
import subprocess
import contextlib
from pathlib import Path
import sys
import platform

# На Apple Silicon MLX работает только в arm64. Если приложение запустили в
# x86_64 (Rosetta или .app с чужой архитектурой) - перезапускаем себя нативно,
# иначе mlx.core не загрузится (ошибка "incompatible architecture").
if (sys.platform == "darwin" and platform.machine() == "x86_64"
        and not os.environ.get("SCRIBE_ARM64")):
    os.environ["SCRIBE_ARM64"] = "1"
    try:
        os.execvp("arch", ["arch", "-arm64", sys.executable] + sys.argv)
    except Exception:
        pass

# В собранном .app (py2app) главный исполняемый файл - сам Scribe, а не python.
# Поэтому процесс распознавания диктофона запускается перезапуском приложения
# с этим флагом. Ловим его до поднятия интерфейса и уходим в воркер.
if len(sys.argv) >= 2 and sys.argv[1] == "--dictaphone-worker":
    import dictaphone_worker
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    dictaphone_worker.main()
    sys.exit(0)

# --- PATH: чтобы ffmpeg/ffprobe/ollama нашлись даже при запуске из Finder ---
for _p in ("/opt/homebrew/bin", "/usr/local/bin", os.path.expanduser("~/.local/bin")):
    if os.path.isdir(_p) and _p not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
except Exception:
    import tkinter as tk
    from tkinter import messagebox
    r = tk.Tk(); r.withdraw()
    messagebox.showerror(
        "Нужен модуль customtkinter",
        f"Установите в Терминале:\n\n\"{sys.executable}\" -m pip install customtkinter")
    raise SystemExit(1)

import theme as T
import summary
from summary import Cancelled

HAS_DND = False
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES

    class RootTk(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.TkdndVersion = TkinterDnD._require(self)

    HAS_DND = True
except Exception:
    RootTk = ctk.CTk


# ---------------------------------------------------------------------------
# Turbo - модель по умолчанию (быстрая). Large v3 доступна в Настройках как
# опция «на всякий», но в разы медленнее при том же результате.
MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
MODELS = {
    "Turbo - быстро (рекомендуется)": "mlx-community/whisper-large-v3-turbo",
    "Large v3 - точнее, но в разы медленнее": "mlx-community/whisper-large-v3-mlx",
}
# Язык - сегменты; «Несколько» включает блочный мультиязычный движок.
LANG_SEG = ["Авто", "Русский", "English", "Несколько"]
LANG_CODE = {"Русский": "ru", "English": "en"}   # «Авто»/«Несколько» → None
LANG_HINT = {
    "Авто": "Язык определится сам; если в записи два языка, это тоже будет учтено.",
    "Несколько": "Для записей, где точно перемешаны русский и английский: язык определяется "
                 "для каждой фразы, без предварительной проверки.",
}
LANG_NAME = {"ru": "русская", "en": "английская"}
FMT_SEG = {"Word": "docx", "Текст": "txt", "Субтитры": "srt"}
TEXT_EXTS = {".txt", ".docx", ".srt"}      # готовые расшифровки: из них можно подвести итоги
NSPK = {"Авто": 0, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}


def _model_title(repo):
    for k, v in MODELS.items():
        if v == repo:
            return k
    return list(MODELS)[0]


def _models_base(cfg):
    return cfg.get("models_dir") or os.environ.get("HF_HOME") \
        or os.path.expanduser("~/.cache/huggingface")


def _models_dir_path(cfg):
    return os.path.join(_models_base(cfg), "hub")


def _models_dir_display(cfg):
    return _models_dir_path(cfg).replace(os.path.expanduser("~"), "~")


def _apply_models_dir(d):
    if d:
        os.environ["HF_HOME"] = d

AUDIO_EXTS = {
    # аудио
    ".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".oga", ".opus",
    ".aiff", ".aif", ".wma", ".m4b", ".amr", ".mka", ".ape", ".wv", ".caf", ".ac3",
    # видео
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".wmv", ".avi", ".flv",
    ".3gp", ".3g2", ".ts", ".mts", ".m2ts", ".mpg", ".mpeg", ".asf", ".vob", ".ogv",
}
AUDIO_TYPES = [("Аудио и видео", " ".join("*" + e for e in sorted(AUDIO_EXTS))),
               ("Готовые расшифровки", "*.txt *.docx *.srt"),
               ("Все файлы", "*.*")]

# Этапы обработки файла: строка под шкалой называет этап прямо.
STEP_NAMES = {"prep": "Подготовка звука", "lang": "Разбивка на фразы и определение языка",
              "asr": "Распознавание речи", "diar": "Определение говорящих", "sum": "Итоги"}
STEP_CHIP = {"prep": "Подготовка", "lang": "Языки", "asr": "Распознавание", "diar": "Говорящие",
             "sum": "Итоги", "save": "Сохранение"}
DIAR_STEPS = {"segmentation": "разбивка на фрагменты", "speaker_counting": "подсчёт говорящих",
              "embeddings": "голосовые отпечатки", "discrete_diarization": "группировка"}

# Список поддерживаемых форматов - показываем в «Настройках».
FORMATS_AUDIO = "MP3, WAV, M4A, AAC, FLAC, OGG, OPUS, AIFF, WMA, M4B, AMR"
FORMATS_VIDEO = "MP4, MOV, MKV, WEBM, WMV, AVI, FLV, 3GP, TS, MPG"

APP_VERSION = "2.3"
CONFIG_PATH = Path(os.path.expanduser("~/.audio2text.json"))
# Пробуем модели по порядку: сначала новая community-1 (её тянут свежие версии
# pyannote и она точнее), затем классическая 3.1 как запасная.
DIAR_MODELS = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
]
# Что проверять на доступ: community-1 самодостаточна; 3.1 требует ещё segmentation-3.0.
DIAR_CHECK = [
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
]


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
def parse_ts(text: str) -> float:
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return 0.0
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def ffprobe_duration(path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _clock(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def group_paragraphs(segments, gap=1.2, max_chars=380):
    paras, buf, start, prev_end = [], [], None, None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if start is None:
            start = seg["start"]
        big_pause = prev_end is not None and (seg["start"] - prev_end) > gap
        cur = " ".join(buf)
        long_enough = len(cur) > max_chars and cur.rstrip().endswith((".", "!", "?", "…"))
        if buf and (big_pause or long_enough):
            paras.append((start, " ".join(buf).strip()))
            buf, start = [], seg["start"]
        buf.append(text)
        prev_end = seg["end"]
    if buf:
        paras.append((start, " ".join(buf).strip()))
    return paras


def group_by_speaker(segments, gap=1.2, max_chars=380):
    """Блоки (start, speaker, text): новый блок при смене говорящего,
    большой паузе или когда абзац стал длинным."""
    blocks, buf, start, cur, prev_end = [], [], None, None, None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        spk = seg.get("speaker", "Спикер ?")
        if cur is None:
            cur, start = spk, seg["start"]
        speaker_changed = spk != cur
        big_pause = prev_end is not None and (seg["start"] - prev_end) > gap
        cur_text = " ".join(buf)
        long_enough = len(cur_text) > max_chars and cur_text.rstrip().endswith((".", "!", "?", "…"))
        if buf and (speaker_changed or big_pause or long_enough):
            blocks.append((start, cur, " ".join(buf).strip()))
            buf, start, cur = [], seg["start"], spk
        buf.append(text)
        prev_end = seg["end"]
    if buf:
        blocks.append((start, cur, " ".join(buf).strip()))
    return blocks


# Запись - во временный файл рядом и затем os.replace: результат перезаписывается
# даже если старый файл открыт, а при сбое старый файл не теряется.
def _tmp_for(path):
    return path.with_name("." + path.name + ".tmp")


def write_txt(segments, path, with_ts, diarized):
    lines = []
    if diarized:
        prev_spk = None
        for start, spk, text in group_by_speaker(segments):
            ts = f"[{_clock(start)}] " if with_ts else ""
            if spk != prev_spk:
                lines.append(f"{ts}{spk}: {text}")
            else:
                lines.append(f"{ts}{text}")
            lines.append("")
            prev_spk = spk
    else:
        for start, text in group_paragraphs(segments):
            p = f"[{_clock(start)}] " if with_ts else ""
            lines.append(f"{p}{text}"); lines.append("")
    tmp = _tmp_for(path)
    tmp.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_srt(segments, path, diarized):
    blocks = []
    for i, seg in enumerate(segments, 1):
        text = seg["text"].strip()
        if diarized and seg.get("speaker"):
            text = f"{seg['speaker']}: {text}"
        blocks.append(str(i))
        blocks.append(f"{_clock(seg['start'], True)} --> {_clock(seg['end'], True)}")
        blocks.append(text); blocks.append("")
    tmp = _tmp_for(path)
    tmp.write_text("\n".join(blocks), encoding="utf-8")
    os.replace(tmp, path)


def write_docx(segments, path, title, with_ts, diarized):
    from docx import Document
    from docx.shared import RGBColor
    doc = Document()
    doc.add_heading(title, level=1)
    if diarized:
        prev_spk = None
        for start, spk, text in group_by_speaker(segments):
            p = doc.add_paragraph()
            if with_ts:
                ts = p.add_run(f"[{_clock(start)}] "); ts.italic = True
                ts.font.color.rgb = RGBColor(0x5f, 0x6b, 0x7a)
            if spk != prev_spk:
                r = p.add_run(f"{spk}: "); r.bold = True
                r.font.color.rgb = RGBColor(0x1f, 0x38, 0x64)
            p.add_run(text)
            prev_spk = spk
    else:
        for start, text in group_paragraphs(segments):
            p = doc.add_paragraph()
            if with_ts:
                t = p.add_run(f"[{_clock(start)}] "); t.italic = True
                t.font.color.rgb = RGBColor(0x5f, 0x6b, 0x7a)
            p.add_run(text)
    tmp = _tmp_for(path)
    doc.save(str(tmp))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Звук: один проход ffmpeg в память (и нормализация громкости в том же проходе).
SR_FAST = 16000


def _decode_to_array(path, normalize=True, cancel=None):
    """ОДИН проход ffmpeg -> float32 моно 16 кГц в память. loudnorm вшит сюда же,
    так что отдельного прохода нормализации не нужно. Читаем по кускам, чтобы
    «Остановить» прерывал и долгую подготовку звука."""
    import numpy as np
    import tempfile
    cmd = ["ffmpeg", "-nostdin", "-i", str(path)]
    if normalize:
        cmd += ["-af", "loudnorm"]
    cmd += ["-ar", str(SR_FAST), "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le",
            "-hide_banner", "-loglevel", "error", "-"]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        chunks = []
        while True:
            if cancel is not None and cancel():
                proc.kill()
                proc.wait()
                raise Cancelled()
            buf = proc.stdout.read(1 << 20)
            if not buf:
                break
            chunks.append(buf)
        proc.wait()
        if proc.returncode != 0:
            err.seek(0)
            raise RuntimeError("ffmpeg: " + err.read().decode("utf-8", "ignore")[-300:])
    return np.frombuffer(b"".join(chunks), dtype=np.float32).copy()


class _ProgressCatcher(io.TextIOBase):
    """Ловит строки «[00:12.000 --> 00:15.000] текст», которые mlx_whisper печатает
    при verbose=True, и превращает их в прогресс. Заодно - точка отмены: если
    нажали «Остановить», исключение из print прерывает распознавание сразу."""

    def __init__(self, duration, cb, cancel=lambda: False):
        self.duration = duration or 0
        self.cb = cb
        self.cancel = cancel
        self.buf = ""

    def write(self, s):
        if self.cancel():
            raise Cancelled()
        self.buf += s
        if "\n" in self.buf:
            *lines, self.buf = self.buf.split("\n")
            for line in lines:
                m = re.search(r'-->\s*([\d:.]+)\s*\]', line)
                if m and self.duration > 0:
                    self.cb(min(1.0, parse_ts(m.group(1)) / self.duration))
        return len(s)


def _transcribe_array(mlx_whisper, audio, model_repo, lang, prog, cancel, initial_prompt=None, temperature=0.0):
    """Один вызов mlx_whisper по массиву в памяти, с прогрессом и отменой."""
    catcher = _ProgressCatcher(len(audio) / SR_FAST, prog, cancel)
    with contextlib.redirect_stdout(catcher):
        res = mlx_whisper.transcribe(
            audio, path_or_hf_repo=model_repo, language=lang, initial_prompt=initial_prompt,
            temperature=temperature, condition_on_previous_text=False, verbose=True)
    return res.get("segments", [])


def transcribe_auto(audio, model_repo, mlx_whisper, prog, cancel, stage=None, forced=None, on_mixed=None):
    """«Авто» (2.3): сначала дешёвая проверка языков по всем фразам (multilang.probe_languages:
    сито whisper-base + перепроверка turbo).
    В записи есть и русский, и английский - обрабатываем как «Несколько»; иначе - один язык,
    найденный проверкой (а не только по первым 30 с, как решал бы сам Whisper).
    forced - язык, выбранный вручную: режим не меняем, только сообщаем о другом языке.
    -> (segments, info)"""
    import multilang
    stage = stage or (lambda k, d="": None)
    probe = multilang.probe_languages(audio, model_repo, cancel)
    info = dict(probe=probe)
    if forced:
        other = {"ru": "en", "en": "ru"}[forced]
        if probe["dominant"] == other or multilang.has_language(probe, other):
            info["forced_other"] = other
        lang = forced
    elif probe["mixed"]:
        if on_mixed:
            on_mixed(probe)
        info["mode"] = "mixed"
        segs = transcribe_multilang_fast(audio, model_repo, mlx_whisper, prog, lambda m: None, cancel, stage=stage)
        return segs, info
    else:
        # язык, найденный проверкой по всей записи (а не только по первым 30 с, как решил бы
        # сам Whisper) - если проверка в нём уверена
        lang = probe["dominant"] if probe["dominant_share"] >= 0.8 and probe["phrases"] >= 3 else None
    info.update(mode="single", lang=lang)
    stage("asr")
    prog(0)
    return _transcribe_array(mlx_whisper, audio, model_repo, lang, prog, cancel), info


def transcribe_multilang_fast(audio, model_repo, mlx_whisper, prog, stat, cancel, stage=None, stats=None):
    """Режим «Несколько»: язык - для каждой фразы (см. multilang.py)."""
    import multilang

    def block(clip, lang, prompt, bprog, temperature=0.0):
        return _transcribe_array(mlx_whisper, clip, model_repo, lang, bprog, cancel, initial_prompt=prompt,
                                 temperature=temperature)

    return multilang.transcribe(audio, model_repo, block, prog, cancel,
                                stage=stage or (lambda k, d="": None), stats=stats)


# ---------------------------------------------------------------------------
_PIPELINES = {}   # модель диаризации грузится один раз на запуск приложения


def _load_pipeline(token):
    """Загружает pyannote pipeline независимо от версии: логинимся заранее,
    токен подхватывается из кэша/окружения, а не через параметр (у разных
    версий pyannote параметр называется по-разному: token / use_auth_token)."""
    if token in _PIPELINES:
        return _PIPELINES[token]
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
    except Exception:
        pass
    from pyannote.audio import Pipeline
    errors = []
    for model in DIAR_MODELS:
        for kw in ({}, {"token": token}):
            try:
                pl = Pipeline.from_pretrained(model, **kw)
                if pl is not None:
                    _PIPELINES[token] = pl
                    return pl
            except Exception as e:
                errors.append(f"{model}: {e}")
    raise RuntimeError(
        "Ни одна модель диаризации не загрузилась. Обычно причина - не приняты "
        "условия модели на Hugging Face. Откройте и нажмите Agree / Accept на:\n"
        "huggingface.co/pyannote/speaker-diarization-community-1\n\n"
        + "\n".join(errors[:4]))


def _mk_hook(prog, stat, cancel):
    """Колбэк прогресса pyannote: шаги диаризации - в шкалу и строку этапа (по-русски),
    заодно точка отмены."""
    last = [None]

    def hook(step_name, step_artifact=None, file=None, total=None, completed=None):
        if cancel():
            raise Cancelled()
        try:
            if step_name and step_name != last[0]:
                last[0] = step_name
                stat(DIAR_STEPS.get(step_name, "обработка"))
            if total and completed is not None and float(total) > 0:
                prog(min(1.0, float(completed) / float(total)))
        except Exception:
            pass
    return hook


def _extract_turns(diar):
    """Достаёт (start, end, speaker) из результата любой версии pyannote."""
    ann = getattr(diar, "speaker_diarization", diar)
    turns = []
    if hasattr(ann, "itertracks"):
        for item in ann.itertracks(yield_label=True):
            seg, spk = item[0], item[-1]
            turns.append((seg.start, seg.end, spk))
    else:
        for turn, spk in ann:
            turns.append((turn.start, turn.end, spk))
    return turns


def diarize_and_label(audio_path, segments, token, num_speakers, log, prog, cancel=lambda: False):
    import torch
    log("загрузка модели")
    pipeline = _load_pipeline(token)
    device = "cpu"
    try:
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps")); device = "mps"
    except Exception:
        device = "cpu"
    # Готовим аудио сами через ffmpeg и подаём waveform напрямую - минуя torchcodec,
    # который ломается на свежих связках PyTorch/ffmpeg. Без нормализации - как раньше.
    data = _decode_to_array(audio_path, normalize=False)
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, N)
    payload = {"waveform": waveform, "sample_rate": SR_FAST}
    kwargs = {"num_speakers": num_speakers} if num_speakers and num_speakers > 0 else {}
    hook = _mk_hook(prog, log, cancel)

    def _run(pl):
        try:
            return pl(payload, hook=hook, **kwargs)
        except TypeError:
            return pl(payload, **kwargs)  # старые версии без hook

    try:
        diar = _run(pipeline)
    except Cancelled:
        raise
    except Exception:
        if device != "mps":
            raise
        log("повтор на процессоре")
        pipeline.to(torch.device("cpu"))
        diar = _run(pipeline)
    turns = _extract_turns(diar)
    mapping, n = {}, 0
    for _, _, spk in turns:
        if spk not in mapping:
            n += 1; mapping[spk] = f"Спикер {n}"

    def speaker_at(a, b):
        best, best_ov = None, 0.0
        for s, e, spk in turns:
            ov = max(0.0, min(b, e) - max(a, s))
            if ov > best_ov:
                best_ov, best = ov, spk
        return mapping.get(best, "Спикер ?")

    for seg in segments:
        seg["speaker"] = speaker_at(seg["start"], seg["end"])
    return segments


def free_asr_models():
    """Перед итогами выгружаем Whisper и pyannote: qwen3.5:9b занимает ~6 ГБ, и вместе
    с ними (и с работающим LKD) Mac на 16 ГБ начинает задыхаться."""
    import gc
    try:
        from mlx_whisper.transcribe import ModelHolder
        ModelHolder.model = None
        ModelHolder.model_path = None
    except Exception:
        pass
    _PIPELINES.clear()
    try:
        import multilang
        multilang._LID.clear()
    except Exception:
        pass
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def read_transcript(path):
    """Текст готовой расшифровки (.txt / .docx / .srt) - для итогов по файлам из прошлых запусков."""
    path = Path(path)
    if path.suffix.lower() == ".docx":
        from docx import Document
        paras = [p.text for p in Document(str(path)).paragraphs]
        if paras and paras[0].strip() == path.stem:      # заголовок, который пишет сам Scribe
            paras = paras[1:]
        return "\n".join(t for t in paras if t.strip())
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".srt":
        keep = [ln for ln in text.splitlines()
                if ln.strip() and not ln.strip().isdigit() and "-->" not in ln]
        return " ".join(keep)
    return text


def ollama_installed():
    import shutil
    return os.path.isdir("/Applications/Ollama.app") or bool(shutil.which("ollama"))


def _fmt_dur(sec):
    if not sec:
        return ""
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_size(n):
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.0f} {unit}" if unit in ("Б", "КБ") else f"{n:.1f} {unit}".replace(".", ",")
        n /= 1024


def _short(name, n=58):
    return name if len(name) <= n else name[:n - 12] + "…" + name[-11:]


# ---------------------------------------------------------------------------
# Интерфейс
class Job:
    """Один файл в списке: путь, статус и результаты."""
    def __init__(self, path):
        self.path = path
        try:
            self.size = path.stat().st_size
        except OSError:
            self.size = 0
        self.duration = 0.0
        self.state = "queue"       # queue / run / done / error / cancel
        self.stage = ""
        self.out = None
        self.summary = None
        self.error = ""
        self.warn = ""
        self.row = None
        self.info = ""             # пометка о том, как обработано (например, «смешанная запись»)
        self.kind = "text" if path.suffix.lower() in TEXT_EXTS else "audio"
        self.segments = None       # сегменты держим до закрытия окна - для «Подвести итоги»
        self.diarized = False
        if self.kind == "text":     # готовая расшифровка: сразу «готово», можно подвести итоги
            self.state, self.out = "done", path
            sp = summary.summary_path(path)
            self.summary = sp if sp.exists() else None


class FileRow(ctk.CTkFrame):
    """Строка списка файлов - как строка таблицы «Документы» в LKD."""

    def __init__(self, master, job, app):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.job, self.app = job, app
        self.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self, text="", image=T.icon("file-text" if job.kind == "text" else "audio-lines", T.MUTED, 18)
                     ).grid(row=0, column=0, rowspan=2, padx=(14, 10), pady=(12, 0), sticky="n")
        T.label(self, _short(job.path.name, 64), 14, bold=True, anchor="w"
                ).grid(row=0, column=1, sticky="ew", pady=(10, 0))
        self.meta = T.label(self, "", 12, color=T.MUTED, anchor="w")
        self.meta.grid(row=1, column=1, sticky="ew", pady=(0, 8))
        self.chip = T.chip(self)
        self.chip.grid(row=0, column=2, rowspan=2, padx=(8, 4), sticky="e")
        self.rm = T.button(self, "", "x", kind="ghost", height=28, command=lambda: app.remove_job(job))
        self.rm.grid(row=0, column=3, rowspan=2, padx=(2, 10))
        self.acts = ctk.CTkFrame(self, fg_color="transparent", width=1, height=1)
        self.note = T.label(self, "", 12, color=T.BAD, anchor="w", justify="left", wraplength=460)
        ctk.CTkFrame(self, height=1, fg_color=T.BORDER, corner_radius=0
                     ).grid(row=4, column=0, columnspan=4, sticky="ew")
        self.refresh()

    def refresh(self):
        j = self.job
        meta = [x for x in ("готовая расшифровка" if j.kind == "text" else "", _fmt_dur(j.duration),
                            _fmt_size(j.size) if j.size else "", j.path.parent.name) if x]
        self.meta.configure(text="  ·  ".join(meta))
        kind, text = {"queue": ("neutral", "В очереди"), "run": ("acc", j.stage or "Идёт…"),
                      "done": ("ok", "Готово"), "error": ("bad", "Ошибка"),
                      "cancel": ("warn", "Остановлено")}[j.state]
        T.set_chip(self.chip, text, kind)
        for w in self.acts.winfo_children():
            w.destroy()
        if j.state == "done" and j.out:
            self.acts.grid(row=2, column=1, columnspan=3, sticky="w", pady=(0, 10))
            T.button(self.acts, "Открыть", "file-text", height=28, size=13, width=100,
                     command=lambda: _open(j.out)).pack(side="left", padx=(0, 4))
            if j.summary:
                T.button(self.acts, "Итоги", "sparkles", height=28, size=13, width=90,
                         command=lambda: _open(j.summary)).pack(side="left", padx=2)
            else:
                # галочку «Итоги» забыли - итоги из готовой расшифровки, без повторного распознавания
                b = T.button(self.acts, "Подвести итоги", "sparkles", kind="ghost", height=28, size=13, width=140,
                             command=lambda: self.app.summarize_job(j))
                b.configure(border_width=1, border_color=T.BORDER_STRONG)
                if self.app.running or not ollama_installed():
                    b.configure(state="disabled")
                b.pack(side="left", padx=2)
            T.button(self.acts, "", "folder-open", kind="ghost", height=28,
                     command=lambda: subprocess.run(["open", "-R", str(j.out)], check=False)
                     ).pack(side="left", padx=2)
        else:
            self.acts.grid_remove()
        if j.state == "error":
            msg, color = j.error, T.BAD
        else:
            msg = "\n".join(x for x in (j.info if j.state == "done" else "", j.warn) if x)
            color = T.WARN if j.warn else T.ACCENT
        if msg:
            self.note.configure(text=msg, text_color=color)
            self.note.grid(row=3, column=1, columnspan=3, sticky="ew", pady=(0, 10))
        else:
            self.note.grid_remove()
        self.rm.configure(state="disabled" if self.app.running else "normal")


def _open(path):
    subprocess.run(["open", str(path)], check=False)


def _first_line(e, n=220):
    s = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return s if len(s) <= n else s[:n] + "…"


class App:
    def __init__(self, root, cfg):
        self.root = root
        self.cfg = cfg
        self.jobs = []
        self.q = queue.Queue()
        self.running = False
        self.cancel_evt = threading.Event()
        self.start_time = None
        self.cur_job = None
        self.stage_text = ""
        self.dict_page = None
        self.page = None

        root.title("Scribe")
        root.geometry("1180x760")
        root.minsize(1000, 660)
        try:
            _icon = Path(__file__).with_name("scribe.png")
            if _icon.exists():
                import tkinter as _tk
                root.iconphoto(True, _tk.PhotoImage(file=str(_icon)))
        except Exception:
            pass

        root.grid_columnconfigure(2, weight=1)
        root.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        ctk.CTkFrame(root, width=1, fg_color=T.BORDER, corner_radius=0).grid(row=0, column=1, sticky="ns")
        self.content = ctk.CTkFrame(root, fg_color=T.BG, corner_radius=0)
        self.content.grid(row=0, column=2, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)
        self.pages = {}
        self.pages["files"] = self._build_files_page()
        self.pages["settings"] = self._build_settings_page()
        self.show("files")

        if HAS_DND:
            root.drop_target_register(DND_FILES)
            root.dnd_bind("<<Drop>>", self._on_drop)
            root.dnd_bind("<<DropEnter>>", lambda e: self._drag(True))
            root.dnd_bind("<<DropLeave>>", lambda e: self._drag(False))
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(120, self._drain)
        self.root.after(500, self._tick)
        self.root.after(600, self._prebuild_dict)
        self._check_ollama()

    # ================= боковое меню =================
    def _build_sidebar(self):
        side = ctk.CTkFrame(self.root, width=220, fg_color=T.SURFACE, corner_radius=0)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        side.pack_propagate(False)
        T.label(side, "Scribe", 19, bold=True, color=T.ACCENT, anchor="w").pack(fill="x", padx=22, pady=(22, 2))
        T.label(side, "Аудио и видео в текст", 12, color=T.MUTED, anchor="w").pack(fill="x", padx=22, pady=(0, 18))
        self.nav = {}
        for key, text, ic in (("files", "Расшифровка", "files"), ("dict", "Диктофон", "mic")):
            self.nav[key] = self._nav_btn(side, text, ic, key)
            self.nav[key].pack(fill="x", padx=12, pady=1)

        bottom = ctk.CTkFrame(side, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=12, pady=(0, 14))
        ctk.CTkFrame(bottom, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x", pady=(0, 8))
        self.nav["settings"] = self._nav_btn(bottom, "Настройки", "settings", "settings")
        self.nav["settings"].pack(fill="x", pady=1)
        self.theme_btn = self._nav_btn(bottom, "", "sun-moon", None)
        self.theme_btn.configure(command=self._cycle_theme)
        self.theme_btn.pack(fill="x", pady=1)
        self._theme_label()
        T.label(bottom, f"Scribe {APP_VERSION}\nлокально на Apple MLX", 11, color=T.FAINT,
                anchor="w", justify="left", wraplength=180).pack(fill="x", padx=10, pady=(8, 0))

    def _nav_btn(self, parent, text, ic, key):
        b = ctk.CTkButton(parent, text=text, anchor="w", height=36, corner_radius=8, border_width=0,
                          border_color=T.SURFACE, fg_color="transparent", hover_color=T.SURFACE2,
                          text_color=T.TEXT, text_color_disabled=T.FAINT,
                          font=T.font(14), image=T.icon(ic, T.MUTED, 18), compound="left",
                          command=(lambda: self.show(key)) if key else None)
        b._ic = ic
        return b

    def show(self, key):
        # Сначала меню (все цвета - парами светлая/тёмная), отрисовать его, и только потом
        # строить/показывать страницу: иначе тяжёлое построение «Диктофона» задерживает
        # перерисовку и пункт меню на миг мигает чужим цветом.
        for k, b in self.nav.items():
            on = k == key
            b.configure(fg_color=T.ACCENT_SOFT if on else "transparent",
                        hover_color=T.ACCENT_SOFT if on else T.SURFACE2,
                        border_color=T.ACCENT_SOFT if on else T.SURFACE,
                        text_color=T.ACCENT if on else T.TEXT, font=T.font(14, bold=on),
                        image=T.icon(b._ic, T.ACCENT if on else T.MUTED, 18))
        self.root.update_idletasks()
        if key == "dict":
            self._prebuild_dict()
        for k, p in self.pages.items():
            if k == key:
                p.grid(row=0, column=0, sticky="nsew")
            else:
                p.grid_remove()
        self.page = key
        if key == "dict":
            self.dict_page.activate()

    def _prebuild_dict(self):
        """Страницу «Диктофон» строим заранее (без запуска распознавания): тяжёлое
        построение в момент щелчка по меню задерживало перерисовку."""
        if self.dict_page is None:
            from dictaphone import DictaphonePage
            self.dict_page = DictaphonePage(self.content, self.cfg.get("model_repo", MODEL_REPO), self.cfg)
            self.pages["dict"] = self.dict_page

    def _theme_label(self):
        names = {"system": "Системная", "light": "Светлая", "dark": "Тёмная"}
        self.theme_btn.configure(text=f"Тема: {names.get(self.cfg.get('theme', 'system'), 'Системная')}")

    def _cycle_theme(self):
        order = ["system", "light", "dark"]
        cur = self.cfg.get("theme", "system")
        nxt = order[(order.index(cur) + 1) % 3] if cur in order else "light"
        self.cfg["theme"] = nxt
        ctk.set_appearance_mode(nxt)
        self._theme_label()
        save_config(self.cfg)

    # ================= «Расшифровка» =================
    def _build_files_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=28, pady=(22, 14))
        T.label(head, "Расшифровка", 22, bold=True, anchor="w").pack(fill="x")
        T.label(head, "Аудио и видео → текст. Всё распознаётся на этом Mac; результат сохраняется "
                      "рядом с исходным файлом.", 13, color=T.MUTED, anchor="w").pack(fill="x", pady=(2, 0))

        # --- слева: файлы ---
        self.files_card = T.card(page)
        self.files_card.grid(row=1, column=0, sticky="nsew", padx=(28, 8))
        self._build_empty(self.files_card)
        self._build_list(self.files_card)

        # --- справа: параметры ---
        opts = T.card(page)
        opts.grid(row=1, column=1, sticky="nsew", padx=(8, 28))
        self._build_options(opts)

        # --- низ: прогресс и запуск ---
        bar = ctk.CTkFrame(page, fg_color="transparent")
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=28, pady=(14, 20))
        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True, padx=(0, 18))
        prow = ctk.CTkFrame(left, fg_color="transparent")
        prow.pack(fill="x")
        self.progress = ctk.CTkProgressBar(prow, height=6)
        self.progress.set(0)
        self.progress.pack(side="left", fill="x", expand=True, pady=(4, 0))
        self.timer = T.label(prow, "00:00", 13, color=T.MUTED, width=54, anchor="e")
        self.timer.pack(side="right", padx=(10, 0))
        self.status = T.label(left, "Добавьте файлы - и нажмите «Расшифровать».", 13, color=T.MUTED,
                              anchor="w")
        self.status.pack(fill="x", pady=(4, 0))
        self.run_btn = T.button(bar, "Расшифровать", "play", kind="primary", width=200, height=42,
                                size=15, command=self.start)
        self.run_btn.pack(side="right")
        self._refresh_files()
        return page

    def _build_empty(self, parent):
        self.empty = ctk.CTkFrame(parent, fg_color=T.SURFACE, border_width=2, border_color=T.BORDER_STRONG,
                                  corner_radius=T.RADIUS)
        inner = ctk.CTkFrame(self.empty, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        ic = ctk.CTkLabel(inner, text="", image=T.icon("upload", T.ACCENT, 36))
        ic.pack(pady=(0, 10))
        title = T.label(inner, "Перетащите сюда аудио или видео" if HAS_DND else "Выберите аудио или видео",
                        17, bold=True)
        title.pack()
        sub = T.label(inner, "Можно несколько файлов или целую папку.\n"
                             "MP3, M4A, WAV, MP4, MOV и всё, что открывает ffmpeg.",
                      13, color=T.MUTED, justify="center")
        sub.pack(pady=(4, 14))
        T.button(inner, "Выбрать файлы", "plus", kind="primary", width=170, command=self.add_files).pack()
        for w in (self.empty, inner, ic, title, sub):
            w.bind("<Button-1>", lambda e: self.add_files())

    def _build_list(self, parent):
        self.list_wrap = ctk.CTkFrame(parent, fg_color="transparent")
        hdr = ctk.CTkFrame(self.list_wrap, fg_color="transparent")
        hdr.pack(fill="x", padx=14, pady=(10, 8))
        self.list_title = T.label(hdr, "Файлы", 15, bold=True, anchor="w")
        self.list_title.pack(side="left")
        self.clear_btn = T.button(hdr, "Очистить", "trash-2", kind="ghost", height=30, size=13, width=110,
                                  command=self.clear_files)
        self.clear_btn.pack(side="right")
        self.add_btn = T.button(hdr, "Добавить", "plus", height=30, size=13, width=110, command=self.add_files)
        self.add_btn.pack(side="right", padx=6)
        ctk.CTkFrame(self.list_wrap, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x")
        self.rows = ctk.CTkScrollableFrame(self.list_wrap, fg_color="transparent", corner_radius=0)
        self.rows.pack(fill="both", expand=True, padx=(1, 1), pady=(0, 6))

    def _build_options(self, card):
        o = ctk.CTkFrame(card, fg_color="transparent")
        o.pack(fill="both", expand=True, padx=18, pady=16)
        ctk.CTkFrame(o, width=280, height=0, fg_color="transparent").pack()   # держит ширину колонки
        cfg = self.cfg

        T.section(o, "Язык записи").pack(fill="x")
        self.lang_seg = T.Segmented(o, LANG_SEG, command=lambda v: self._lang_hint(), size=12)
        self.lang_seg.set(cfg.get("lang", "Авто") if cfg.get("lang") in LANG_SEG else "Авто")
        self.lang_seg.pack(fill="x", pady=(6, 4))
        self.lang_note = T.label(o, "", 12, color=T.MUTED, anchor="w", justify="left", wraplength=280)
        self.lang_note.pack(fill="x")
        self._lang_hint()

        T.section(o, "Формат результата").pack(fill="x", pady=(16, 0))
        self.fmt_seg = T.Segmented(o, list(FMT_SEG), command=lambda v: self._sync_opts(), size=12)
        self.fmt_seg.set(cfg.get("fmt", "Word") if cfg.get("fmt") in FMT_SEG else "Word")
        self.fmt_seg.pack(fill="x", pady=(6, 4))
        T.label(o, "Каким файлом сохранить расшифровку:\nWord - .docx  ·  Текст - .txt  ·  Субтитры - .srt",
                12, color=T.MUTED, anchor="w", justify="left", wraplength=280).pack(fill="x")

        ctk.CTkFrame(o, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x", pady=16)
        T.section(o, "Дополнительно").pack(fill="x", pady=(0, 8))

        self.diar_var = ctk.BooleanVar(value=cfg.get("diar", bool(cfg.get("hf_token"))))
        self.diar_cb = ctk.CTkCheckBox(o, text="Определять говорящих", variable=self.diar_var,
                                       font=T.font(14), command=self._sync_opts, checkbox_width=20,
                                       checkbox_height=20)
        self.diar_cb.pack(anchor="w")
        self.nspk_row = ctk.CTkFrame(o, fg_color="transparent")
        T.label(self.nspk_row, "Сколько человек", 13, color=T.MUTED).pack(side="left")
        self.nspk_var = ctk.StringVar(value=cfg.get("nspk", "Авто") if cfg.get("nspk") in NSPK else "Авто")
        ctk.CTkOptionMenu(self.nspk_row, variable=self.nspk_var, values=list(NSPK), width=90, height=28,
                          font=T.font(13), dropdown_font=T.font(13)).pack(side="left", padx=(10, 0))
        self.token_link = T.label(o, "Нужен токен Hugging Face - задать в Настройках", 12, color=T.ACCENT,
                                  anchor="w", justify="left", wraplength=250, cursor="pointinghand")
        self.token_link.bind("<Button-1>", lambda e: self.show("settings"))
        self.diar_hint = T.label(o, "Метки «Спикер 1, 2…». Заметно дольше.", 12, color=T.MUTED, anchor="w",
                                 justify="left", wraplength=250)

        self.ts_var = ctk.BooleanVar(value=cfg.get("ts", False))
        self.ts_cb = ctk.CTkCheckBox(o, text="Таймкоды в тексте", variable=self.ts_var, font=T.font(14),
                                     checkbox_width=20, checkbox_height=20)
        self.ts_cb.pack(anchor="w", pady=(12, 0))

        self.sum_var = ctk.BooleanVar(value=cfg.get("summary", False))
        self.sum_cb = ctk.CTkCheckBox(o, text="Итоги (отдельный файл)", variable=self.sum_var, font=T.font(14),
                                      checkbox_width=20, checkbox_height=20)
        self.sum_cb.pack(anchor="w", pady=(12, 0))
        self.sum_hint = T.label(o, "", 12, color=T.MUTED, anchor="w", justify="left", wraplength=250)
        self.sum_hint.pack(fill="x", padx=(30, 0), pady=(2, 0))
        self._sync_opts()

    def _lang_hint(self):
        v = self.lang_seg.get()
        self.lang_note.configure(text=LANG_HINT.get(
            v, f"Вся запись распознаётся как «{v}». Если в ней окажется и другой язык - предупредим."))

    def _sync_opts(self):
        """Согласовать опции между собой: без токена говорящих не включить,
        для субтитров таймкоды и так есть, без Ollama нет итогов."""
        has_token = bool(self.cfg.get("hf_token"))
        if not has_token:
            self.diar_var.set(False)
        self.diar_cb.configure(state="normal" if has_token else "disabled")
        for w in (self.nspk_row, self.token_link, self.diar_hint):
            w.pack_forget()
        if not has_token:
            self.token_link.pack(fill="x", padx=(30, 0), pady=(2, 0), after=self.diar_cb)
        elif self.diar_var.get():
            self.nspk_row.pack(fill="x", padx=(30, 0), pady=(6, 0), after=self.diar_cb)
            self.diar_hint.pack(fill="x", padx=(30, 0), pady=(4, 0), after=self.nspk_row)
        srt = self.fmt_seg.get() == "Субтитры"
        self.ts_cb.configure(state="disabled" if srt else "normal")
        ext = "Word" if self.fmt_seg.get() == "Word" else ".txt"
        if ollama_installed():
            self.sum_cb.configure(state="normal")
            self.sum_hint.configure(text=f"Суть, решения и задачи - отдельным документом ({ext}) рядом с "
                                         "полной расшифровкой. Готовит локальная модель через Ollama.",
                                    text_color=T.MUTED)
        else:
            self.sum_var.set(False)
            self.sum_cb.configure(state="disabled")
            self.sum_hint.configure(text="Нужна Ollama (ollama.com) - локальная модель для текста.",
                                    text_color=T.WARN)

    # ---------- файлы ----------
    def _refresh_files(self):
        if self.jobs:
            self.empty.pack_forget()
            self.list_wrap.pack(fill="both", expand=True)
            self.list_title.configure(text=f"Файлы · {len(self.jobs)}")
        else:
            self.list_wrap.pack_forget()
            self.empty.pack(fill="both", expand=True, padx=14, pady=14)
        st = "disabled" if self.running else "normal"
        self.clear_btn.configure(state=st)

    def _add_paths(self, paths):
        found = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                found += sorted(x for x in path.rglob("*")
                                if x.suffix.lower() in AUDIO_EXTS and not x.name.startswith("."))
            elif path.suffix.lower() in AUDIO_EXTS or path.suffix.lower() in TEXT_EXTS:
                found.append(path)
        have = {j.path for j in self.jobs}
        new = []
        for path in found:
            if path not in have:
                have.add(path)
                job = Job(path)
                job.row = FileRow(self.rows, job, self)
                job.row.pack(fill="x")
                self.jobs.append(job)
                new.append(job)
        skipped = len(paths) - len(found) if not any(Path(p).is_dir() for p in paths) else 0
        if skipped > 0 and not new:
            self._set_status("Эти файлы не похожи на аудио, видео или готовую расшифровку.", T.WARN)
        self._refresh_files()
        audio = [j for j in new if j.kind == "audio"]
        if audio:
            threading.Thread(target=self._probe, args=(audio,), daemon=True).start()

    def _probe(self, jobs):
        for j in jobs:
            self.q.put(("job", j, {"duration": ffprobe_duration(j.path)}))

    def add_files(self):
        if self.running:
            return
        self._add_paths(filedialog.askopenfilenames(title="Выберите файлы", filetypes=AUDIO_TYPES))

    def _on_drop(self, event):
        self._drag(False)
        if self.page != "files":
            self.show("files")
        if not self.running:
            self._add_paths(self.root.tk.splitlist(event.data))
        return "copy"

    def _drag(self, on):
        c = T.ACCENT if on else T.BORDER_STRONG
        self.empty.configure(border_color=c)
        self.files_card.configure(border_color=T.ACCENT if on else T.BORDER)
        return "copy"

    def remove_job(self, job):
        if self.running:
            return
        job.row.destroy()
        self.jobs.remove(job)
        self._refresh_files()

    def clear_files(self):
        if self.running:
            return
        for j in self.jobs:
            j.row.destroy()
        self.jobs.clear()
        self._refresh_files()
        self.progress.set(0)
        self._set_status("Добавьте файлы - и нажмите «Расшифровать».")

    # ---------- статус/таймер ----------
    def _stat(self, m): self.q.put(("status", m))
    def _prog(self, v): self.q.put(("prog", v))
    def _job(self, job, **upd): self.q.put(("job", job, upd))

    def _stages(self, job, keys, prefix="", normalize=True):
        """Строка этапа: «Файл 2 из 5 · Шаг 3 из 5 · Распознавание речи - 43 %».
        Номера шагов считаются по включённым опциям."""
        def go(key, detail=""):
            if key == "save":
                text = f"{prefix}Сохранение"
            else:
                name = STEP_NAMES[key]
                if key == "prep" and normalize:
                    name = "Подготовка звука и нормализация громкости"
                text = f"{prefix}Шаг {keys.index(key) + 1} из {len(keys)} · {name}" + (f": {detail}" if detail else "")
            self.q.put(("stage", job, STEP_CHIP[key], text))
        go.keys = keys
        return go

    def _set_status(self, text, color=T.MUTED):
        self.status.configure(text=text, text_color=color)

    def _tick(self):
        if self.running and self.start_time:
            el = int(time.monotonic() - self.start_time)
            self.timer.configure(text=f"{el // 60:02d}:{el % 60:02d}")
        self.root.after(500, self._tick)

    def _drain(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "status":
                    if not self.cancel_evt.is_set():
                        self._set_status(msg[1])
                elif kind == "stage":
                    _, job, chip, text = msg
                    job.stage = chip
                    self.cur_job = job
                    self.stage_text = text
                    if not self.cancel_evt.is_set():
                        self._set_status(text)
                    if job.row is not None and job.row.winfo_exists() and job.state == "run":
                        T.set_chip(job.row.chip, chip, "acc")
                elif kind == "prog":
                    v = msg[1]
                    if v < 0:
                        self.progress.configure(mode="indeterminate"); self.progress.start()
                    else:
                        try: self.progress.stop()
                        except Exception: pass
                        self.progress.configure(mode="determinate"); self.progress.set(v)
                    j = self.cur_job
                    if j is not None and j.state == "run":
                        T.set_chip(j.row.chip, f"{j.stage} · {int(v * 100)} %" if v >= 0 else j.stage, "acc")
                        if self.stage_text and not self.cancel_evt.is_set():
                            self._set_status(f"{self.stage_text} - {int(v * 100)} %" if v >= 0 else self.stage_text)
                elif kind == "job":
                    _, job, upd = msg
                    for k, v in upd.items():
                        setattr(job, k, v)
                    if upd.get("state") == "run":
                        self.cur_job = job
                    if job.row is not None and job.row.winfo_exists():
                        job.row.refresh()
                elif kind == "token":
                    self._token_result(*msg[1])
                elif kind == "ollama":
                    self._ollama_result(*msg[1])
                elif kind == "done":
                    self._finish(msg[1])
        except queue.Empty:
            pass
        self.root.after(120, self._drain)

    # ---------- запуск ----------
    def _save_prefs(self):
        self.cfg.update(lang=self.lang_seg.get(), fmt=self.fmt_seg.get(), diar=self.diar_var.get(),
                        nspk=self.nspk_var.get(), ts=self.ts_var.get(), summary=self.sum_var.get())
        save_config(self.cfg)

    def start(self):
        if self.running:
            self.cancel_evt.set()
            self.run_btn.configure(state="disabled", text="Останавливаю…")
            self._set_status("Останавливаю - дождусь ближайшей точки остановки…", T.WARN)
            return
        if not self.jobs:
            self.add_files()
            return
        audio_jobs = [j for j in self.jobs if j.kind == "audio"]
        if not audio_jobs:
            self._set_status("В списке только готовые расшифровки - для них есть «Подвести итоги».", T.WARN)
            return
        pending = [j for j in audio_jobs if j.state != "done"] or audio_jobs
        sel = self.lang_seg.get()
        fmt = FMT_SEG[self.fmt_seg.get()]
        opts = dict(
            lang=LANG_CODE.get(sel), multilang=(sel == "Несколько"),
            model=self.cfg.get("model_repo", MODEL_REPO), fmt=fmt,
            with_ts=self.ts_var.get() and fmt != "srt",
            diar=self.diar_var.get() and bool(self.cfg.get("hf_token")),
            nspk=NSPK[self.nspk_var.get()], token=self.cfg.get("hf_token", ""),
            normalize=self.cfg.get("normalize", True),
            summary=self.sum_var.get(), sum_model=self.cfg.get("summary_model", summary.DEFAULT_MODEL),
            outs={id(j): self._out_path(j, fmt, audio_jobs) for j in pending})
        self._save_prefs()
        for j in pending:
            j.state, j.error, j.warn, j.stage, j.out, j.summary, j.info = "queue", "", "", "", None, None, ""
            j.segments, j.diarized = None, False
        self._begin_run()
        threading.Thread(target=self._worker, args=(pending, opts), daemon=True).start()

    @staticmethod
    def _out_path(job, fmt, jobs):
        """a.mp4 и a.m4a в одной папке не должны перезаписывать общий a.txt:
        если имя занято другим файлом этого пакета - «a (m4a).txt»."""
        p = job.path
        clash = any(o is not job and o.path.parent == p.parent and o.path.stem == p.stem for o in jobs)
        if clash:
            return p.with_name(f"{p.stem} ({p.suffix[1:].lower()}).{fmt}")
        return p.with_suffix("." + fmt)

    def _begin_run(self):
        self.running = True
        for j in self.jobs:
            j.row.refresh()
        self.cancel_evt.clear()
        self.start_time = time.monotonic()
        self.stage_text = ""
        self.run_btn.configure(text="Остановить")
        T.restyle(self.run_btn, "danger", "square")
        self.add_btn.configure(state="disabled")
        self._refresh_files()
        self.nav["files"].configure(text="Расшифровка  ·  идёт")
        self.progress.configure(mode="determinate"); self.progress.set(0)

    def summarize_job(self, job):
        """«Подвести итоги» у готового файла: итоги из готовой расшифровки, без повторного распознавания."""
        if self.running:
            return
        opts = dict(sum_model=self.cfg.get("summary_model", summary.DEFAULT_MODEL))
        job.state, job.warn, job.error = "run", "", ""
        self._begin_run()
        threading.Thread(target=self._summary_worker, args=(job, opts), daemon=True).start()

    def _summary_worker(self, job, o):
        cancel = self.cancel_evt.is_set
        stage = self._stages(job, ["sum"])
        self.q.put(("stage", job, STEP_CHIP["sum"], "Шаг 1 из 1 · Итоги"))
        try:
            if job.segments:
                text = summary.transcript_text(job.segments, job.diarized)
            else:
                text = read_transcript(job.out)
            summ = self._make_summary(job, text, job.out, job.duration, o, stage, cancel)
            self._job(job, state="done", summary=summ)
            _open(summ)
            self.q.put(("done", ("ok", f"Итоги готовы: {summ.name}.")))
        except Cancelled:
            self._job(job, state="done")
            self.q.put(("done", ("warn", "Итоги остановлены.")))
        except Exception as e:
            self._job(job, state="done", warn=f"Итоги не сделаны: {_first_line(e)}")
            self.q.put(("done", ("warn", "Итоги не сделаны - причина под файлом в списке.")))

    def _worker(self, jobs, o):
        cancel = self.cancel_evt.is_set
        try:
            self._stat("Загружаю модель…")
            import mlx_whisper
        except Exception as e:
            self.q.put(("done", ("error", "Не найден mlx_whisper:\n  pip install mlx-whisper\n\n" + str(e)))); return
        if o["fmt"] == "docx":
            try:
                import docx  # noqa
            except Exception:
                self.q.put(("done", ("error", "Для Word нужен python-docx:\n  pip install python-docx"))); return
        if o["diar"]:
            try:
                import pyannote.audio  # noqa
                import torch  # noqa
            except Exception:
                self.q.put(("done", ("error", "Для говорящих нужен pyannote:\n  pip install pyannote.audio"))); return

        done, failed, cancelled = [], 0, False
        n = len(jobs)
        for i, job in enumerate(jobs, 1):
            if cancel():
                cancelled = True
                self._job(job, state="cancel")
                continue
            pre = f"Файл {i} из {n} · " if n > 1 else ""
            keys = ["prep"] + (["lang"] if o["multilang"] else []) + ["asr"] + (["diar"] if o["diar"] else []) \
                + (["sum"] if o["summary"] else [])
            stage = self._stages(job, keys, pre, o["normalize"])
            try:
                done.append(self._run_one(job, o, mlx_whisper, stage, cancel))
            except Cancelled:
                cancelled = True
                self._job(job, state="cancel")
            except FileNotFoundError as e:
                if "ffmpeg" in str(e) or "ffprobe" in str(e):
                    self._job(job, state="error", error="Не найден ffmpeg")
                    self.q.put(("done", ("error", "Не найден ffmpeg. В Терминале:\n  brew install ffmpeg"))); return
                failed += 1
                self._job(job, state="error", error=_first_line(e))
            except Exception as e:
                failed += 1
                self._job(job, state="error", error=_first_line(e))

        if done and not cancelled:
            out, summ, _ = done[0]
            _open(out)
            if summ:
                _open(summ)
        warned = sum(1 for _, _, w in done if w)
        if cancelled:
            msg = ("warn", f"Остановлено. Готово файлов: {len(done)} из {n}.")
        elif failed:
            msg = ("warn", f"Готово {len(done)} из {n}. Ошибки - в списке файлов.")
        elif warned:
            msg = ("warn", f"Готово: {len(done)}. Есть предупреждения - см. список файлов.")
        else:
            msg = ("ok", f"Готово. Файлов: {len(done)}.")
        self.q.put(("done", msg))

    def _run_one(self, job, o, mlx_whisper, stage, cancel):
        path = job.path
        if not path.exists():
            raise RuntimeError("Файл не найден - его переместили или удалили.")
        self._job(job, state="run")
        stage("prep")
        self._prog(-1)
        warn = []
        try:
            audio = _decode_to_array(path, o["normalize"], cancel)
        except RuntimeError:
            if not o["normalize"]:
                raise
            audio = _decode_to_array(path, False, cancel)
            warn.append("Нормализация не удалась - обработано как есть.")
        if not len(audio):
            raise RuntimeError("В файле не найден звук.")
        if cancel():
            raise Cancelled()
        duration = len(audio) / SR_FAST
        self._job(job, duration=duration)
        if o["multilang"]:
            segments = transcribe_multilang_fast(audio, o["model"], mlx_whisper, self._prog, lambda m: None, cancel,
                                                 stage=stage)
        else:
            stage("prep", "проверка языков")

            def on_mixed(probe):
                stage.keys.insert(stage.keys.index("asr"), "lang")    # «Авто» стал «Несколько»: +1 шаг
                self._job(job, info="Найдены русский и английский - обработано как смешанная запись.")

            segments, info = transcribe_auto(audio, o["model"], mlx_whisper, self._prog, cancel, stage=stage,
                                             forced=o["lang"], on_mixed=on_mixed)
            other = info.get("forced_other")
            if other:
                warn.append(f"В записи есть {LANG_NAME[other]} речь - для смешанной записи выберите "
                            "«Авто» или «Несколько».")
        del audio

        diar_used = False
        if o["diar"]:
            stage("diar")
            self._prog(-1)
            try:
                segments = diarize_and_label(path, segments, o["token"], o["nspk"],
                                             lambda d: stage("diar", d), self._prog, cancel)
                diar_used = True
            except Cancelled:
                raise
            except Exception as e:
                warn.append(f"Говорящие не определены: {_first_line(e)}")
        self._job(job, segments=segments, diarized=diar_used)

        out = o["outs"][id(job)]
        stage("save")
        if o["fmt"] == "txt":
            write_txt(segments, out, o["with_ts"], diar_used)
        elif o["fmt"] == "docx":
            write_docx(segments, out, path.stem, o["with_ts"], diar_used)
        else:
            write_srt(segments, out, diar_used)
        self._job(job, out=out)

        summ = None
        if o["summary"]:
            try:
                summ = self._make_summary(job, summary.transcript_text(segments, diar_used), out, duration, o,
                                          stage, cancel)
            except Cancelled:
                raise
            except Exception as e:
                warn.append(f"Итоги не сделаны: {_first_line(e)}")
        self._prog(1.0)
        warn = "\n".join(warn)
        self._job(job, state="done", summary=summ, warn=warn)
        return out, summ, warn

    def _make_summary(self, job, text, out, duration, o, stage, cancel):
        if len(text) < 300:
            raise RuntimeError("запись слишком короткая, подводить нечего.")
        stage("sum", "подключение к Ollama")
        self._prog(-1)
        free_asr_models()
        if not summary.ensure_ollama(lambda m: stage("sum", m)):
            raise RuntimeError("Ollama не запущена. Откройте приложение Ollama (ollama.com) и повторите.")
        model = summary.pick_model(o["sum_model"], summary.list_models())
        if not model:
            raise RuntimeError("в Ollama нет моделей. В Терминале: ollama pull " + summary.DEFAULT_MODEL)
        stage("sum")
        data = summary.summarize(text, model, self._prog, lambda m: stage("sum", m), cancel)
        src = job.path.name if job.kind == "audio" else None
        return summary.write_summary(data, summary.summary_path(out), src, out.name, duration, model)

    def _finish(self, payload):
        kind, msg = payload
        try: self.progress.stop()
        except Exception: pass
        self.progress.configure(mode="determinate")
        self.progress.set(1.0 if kind == "ok" else self.progress.get())
        self.running = False
        self.cur_job = None
        self.run_btn.configure(state="normal", text="Расшифровать")
        T.restyle(self.run_btn, "primary", "play")
        self.add_btn.configure(state="normal")
        self.nav["files"].configure(text="Расшифровка")
        for j in self.jobs:
            if j.state in ("queue", "run"):
                j.state = "done" if j.kind == "text" else "cancel"
            j.row.refresh()
        self._refresh_files()
        el = int(time.monotonic() - self.start_time) if self.start_time else 0
        if kind == "error":
            self._set_status("Ошибка.", T.BAD)
            messagebox.showerror("Ошибка", msg)
        else:
            self._set_status(f"{msg}  За {el // 60:02d}:{el % 60:02d}.", T.OK if kind == "ok" else T.WARN)

    # ================= «Настройки» =================
    def _build_settings_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        scroll = ctk.CTkScrollableFrame(page, fg_color="transparent", corner_radius=0)
        scroll.pack(fill="both", expand=True)
        wrap = ctk.CTkFrame(scroll, fg_color="transparent")
        wrap.pack(fill="x", padx=16, pady=(10, 20))
        T.label(wrap, "Настройки", 22, bold=True, anchor="w").pack(fill="x")
        T.label(wrap, "Меняются редко. Всё сохраняется само.", 13, color=T.MUTED, anchor="w").pack(fill="x", pady=(2, 6))

        # --- Распознавание ---
        c = self._card(wrap, "Распознавание")
        self.model_var = ctk.StringVar(value=_model_title(self.cfg.get("model_repo", MODEL_REPO)))
        ctk.CTkOptionMenu(c, variable=self.model_var, values=list(MODELS), width=380, height=32,
                          font=T.font(13), dropdown_font=T.font(13), command=self._save_model).pack(anchor="w")
        self._hint(c, "Turbo - быстрая и рекомендуемая. Large v3 в разы медленнее при том же результате.")
        self.norm_var = ctk.BooleanVar(value=self.cfg.get("normalize", True))
        ctk.CTkCheckBox(c, text="Нормализовать громкость", variable=self.norm_var, font=T.font(14),
                        checkbox_width=20, checkbox_height=20, command=self._save_norm).pack(anchor="w", pady=(14, 0))
        self._hint(c, "Выравнивает тихие и громкие места: меньше ошибок и выдуманного текста на тишине. "
                      "Добавляет немного времени. Лучше не выключать.")

        # --- Итоги ---
        c = self._card(wrap, "Итоги")
        r = ctk.CTkFrame(c, fg_color="transparent"); r.pack(fill="x")
        self.ollama_chip = T.chip(r, "Проверяю Ollama…")
        self.ollama_chip.pack(side="left")
        T.button(r, "Проверить", height=28, size=13, width=100, command=self._check_ollama).pack(side="left", padx=8)
        mr = ctk.CTkFrame(c, fg_color="transparent"); mr.pack(fill="x", pady=(10, 0))
        T.label(mr, "Модель", 13, color=T.MUTED).pack(side="left")
        self.sum_model_var = ctk.StringVar(value=self.cfg.get("summary_model", summary.DEFAULT_MODEL))
        self.sum_model_menu = ctk.CTkOptionMenu(mr, variable=self.sum_model_var, width=220, height=30,
                                                values=[self.sum_model_var.get()], font=T.font(13),
                                                dropdown_font=T.font(13), command=self._save_sum_model)
        self.sum_model_menu.pack(side="left", padx=(10, 0))
        self._hint(c, "Документ «<имя> - итоги» кладётся рядом с полной расшифровкой: суть, главное, "
                      "решения, задачи, открытые вопросы. Делается, когда отмечено «Итоги», или кнопкой "
                      "«Подвести итоги» у готового файла. Модель работает на этом Mac через Ollama - "
                      "текст никуда не уходит.")
        self._hint(c, "Память: на время итогов Scribe выгружает модели распознавания, а Ollama освобождает "
                      "память сразу после итогов. На Mac с 16 ГБ при работающем LKD лучше qwen3.5:4b.")

        # --- Говорящие ---
        c = self._card(wrap, "Определение говорящих")
        self._hint(c, "Нужен бесплатный токен Hugging Face - разовая настройка:", top=0)
        self._link(c, "1. Примите условия модели: huggingface.co/pyannote/speaker-diarization-community-1",
                   "https://huggingface.co/pyannote/speaker-diarization-community-1")
        self._link(c, "2. Создайте токен (тип Read): huggingface.co/settings/tokens",
                   "https://huggingface.co/settings/tokens")
        self._saved_token = self.cfg.get("hf_token", "")
        er = ctk.CTkFrame(c, fg_color="transparent"); er.pack(fill="x", pady=(10, 0))
        self.tok_entry = ctk.CTkEntry(er, show="•", placeholder_text="hf_…", height=32, font=T.font(13))
        self.tok_entry.pack(side="left", fill="x", expand=True)
        self.tok_entry.insert(0, self._saved_token)
        self.tok_entry.bind("<KeyRelease>", lambda e: self._sync_tok_btn())
        self.tok_btn = T.button(er, "Проверить и сохранить", kind="primary", height=32, size=13, width=200,
                                command=self._check_token)
        self.tok_btn.pack(side="left", padx=(8, 0))
        self.tok_status = T.label(c, "", 13, anchor="w", justify="left", wraplength=640)
        self.tok_status.pack(fill="x", pady=(8, 0))
        self._sync_tok_btn()

        # --- Папка моделей ---
        c = self._card(wrap, "Модели на диске")
        mr = ctk.CTkFrame(c, fg_color="transparent"); mr.pack(fill="x")
        self.models_dir_lbl = T.label(mr, _models_dir_display(self.cfg), 13, color=T.MUTED, anchor="w")
        self.models_dir_lbl.pack(side="left", fill="x", expand=True)
        T.button(mr, "Сменить…", height=30, size=13, width=100, command=self._change_models).pack(side="right")
        T.button(mr, "Открыть", "folder-open", height=30, size=13, width=100,
                 command=self._open_models).pack(side="right", padx=6)
        self._hint(c, "Сюда скачиваются модели распознавания (~1,5 ГБ). Можно открыть папку и удалить ненужное.")

        # --- Форматы ---
        c = self._card(wrap, "Поддерживаемые форматы")
        T.label(c, f"Аудио: {FORMATS_AUDIO}\nВидео: {FORMATS_VIDEO}\nи всё, что умеет открывать ffmpeg.",
                13, color=T.MUTED, anchor="w", justify="left").pack(fill="x")
        return page

    def _card(self, parent, title):
        c = T.card(parent)
        c.pack(fill="x", pady=(14, 0))
        inner = ctk.CTkFrame(c, fg_color="transparent")
        inner.pack(fill="x", padx=18, pady=16)
        T.label(inner, title, 15, bold=True, anchor="w").pack(fill="x", pady=(0, 10))
        return inner

    def _hint(self, parent, text, top=4):
        T.label(parent, text, 12, color=T.MUTED, anchor="w", justify="left", wraplength=640
                ).pack(fill="x", pady=(top, 0))

    def _link(self, parent, text, url):
        import webbrowser
        lb = T.label(parent, text, 12, color=T.ACCENT, anchor="w", cursor="pointinghand")
        lb.pack(fill="x", pady=(2, 0))
        lb.bind("<Button-1>", lambda e: webbrowser.open(url))

    def _save_model(self, _v=None):
        self.cfg["model_repo"] = MODELS[self.model_var.get()]
        save_config(self.cfg)

    def _save_norm(self):
        self.cfg["normalize"] = bool(self.norm_var.get())
        save_config(self.cfg)

    def _save_sum_model(self, v):
        self.cfg["summary_model"] = v
        save_config(self.cfg)

    def _open_models(self):
        p = _models_dir_path(self.cfg)
        try: os.makedirs(p, exist_ok=True)
        except Exception: pass
        subprocess.run(["open", p], check=False)

    def _change_models(self):
        d = filedialog.askdirectory(title="Папка для моделей")
        if not d:
            return
        self.cfg["models_dir"] = d; save_config(self.cfg)
        _apply_models_dir(d)
        self.models_dir_lbl.configure(text=_models_dir_display(self.cfg) + "  - модель докачается при первом запуске")

    # ---------- Ollama ----------
    def _check_ollama(self):
        if hasattr(self, "ollama_chip"):
            T.set_chip(self.ollama_chip, "Проверяю Ollama…", "neutral")

        def bg():
            try:
                self.q.put(("ollama", (True, summary.list_models(), "")))
            except Exception as e:
                self.q.put(("ollama", (False, [], str(e))))
        threading.Thread(target=bg, daemon=True).start()

    def _ollama_result(self, ok, models, err):
        if ok and models:
            T.set_chip(self.ollama_chip, f"Ollama работает · моделей: {len(models)}", "ok")
            cur = summary.pick_model(self.sum_model_var.get(), models)
            self.sum_model_menu.configure(values=models)
            self.sum_model_var.set(cur)
        elif ok:
            T.set_chip(self.ollama_chip, f"В Ollama нет моделей - ollama pull {summary.DEFAULT_MODEL}", "warn")
        elif ollama_installed():
            T.set_chip(self.ollama_chip, "Ollama не запущена - запустится сама при первом кратком содержании", "warn")
        else:
            T.set_chip(self.ollama_chip, "Ollama не установлена - ollama.com", "bad")

    # ---------- токен ----------
    def _sync_tok_btn(self):
        changed = self.tok_entry.get().strip() != self._saved_token
        if self._saved_token and not changed:
            self.tok_status.configure(text="✓ Токен сохранён и проверен", text_color=T.OK)
            self.tok_btn.configure(state="disabled")
        else:
            self.tok_btn.configure(state="normal")
            if not self.tok_entry.get().strip():
                self.tok_status.configure(text="", text_color=T.MUTED)

    def _check_token(self):
        token = self.tok_entry.get().strip()
        if not token:
            self.tok_status.configure(text="Введите токен.", text_color=T.WARN); return
        self.tok_status.configure(text="Проверяю…", text_color=T.MUTED)
        self.tok_btn.configure(state="disabled")
        threading.Thread(target=self._check_token_bg, args=(token,), daemon=True).start()

    def _check_token_bg(self, token):
        """Сеть - в фоне, чтобы окно не подвисало."""
        try:
            from huggingface_hub import HfApi
            api = HfApi(); who = api.whoami(token=token); name = who.get("name", "пользователь")
        except Exception as e:
            self.q.put(("token", (False, token, f"✗ Токен не принят: {_first_line(e)}", T.BAD))); return
        access = {}
        for m in DIAR_CHECK:
            try:
                api.model_info(m, token=token); access[m] = True
            except Exception:
                access[m] = False
        ok = access.get("pyannote/speaker-diarization-community-1") or (
            access.get("pyannote/speaker-diarization-3.1") and access.get("pyannote/segmentation-3.0"))
        if ok:
            self.q.put(("token", (True, token, f"✓ Токен работает ({name}). Сохранён.", T.OK)))
        else:
            self.q.put(("token", (False, token,
                                  f"Токен верный ({name}), но нет доступа к модели диаризации. "
                                  "Примите условия (ссылка 1 выше) и проверьте снова.", T.WARN)))

    def _token_result(self, ok, token, text, color):
        if ok:
            self.cfg["hf_token"] = token; save_config(self.cfg)
            self._saved_token = token
            self.diar_var.set(True)
            self._sync_opts()
        self.tok_status.configure(text=text, text_color=color)
        self.tok_btn.configure(state="disabled" if ok else "normal")

    # ---------- закрытие ----------
    def _on_close(self):
        if self.running and not messagebox.askyesno("Идёт расшифровка", "Прервать расшифровку и выйти?"):
            return
        if self.dict_page is not None and self.dict_page.has_unsaved() and not messagebox.askyesno(
                "Диктовка не сохранена", "Текст диктовки не сохранён. Выйти без сохранения?"):
            return
        self.cancel_evt.set()
        self._save_prefs()
        if self.dict_page is not None:
            self.dict_page.shutdown()
        self.root.destroy()


def main():
    cfg = load_config()
    _apply_models_dir(cfg.get("models_dir"))
    T.apply()                                    # тему - до создания окна
    ctk.set_appearance_mode(cfg.get("theme", "system"))
    root = RootTk()
    App(root, cfg)
    root.mainloop()
    # Окно закрыто, настройки сохранены, диктофон остановлен. Выходим без разбора C++-объектов
    # Metal (MLX) и onnxruntime: изредка он падал с «recursive_mutex lock failed», и macOS
    # показывала бы «Python неожиданно завершился».
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
