#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диктофон для Scribe — живая диктовка с микрофона в реальном времени.
Открывается кнопкой «🎤 Диктофон» из главного окна.

Как работает:
  захват микрофона (sounddevice) → нарезка речи на фразы по паузам
  (энергетический VAD с адаптацией под уровень микрофона) → каждая
  законченная фраза уходит в ОТДЕЛЬНЫЙ процесс распознавания
  (dictaphone_worker.py) и возвращается текстом, который дописывается в окно.

Распознавание вынесено в отдельный процесс намеренно: оно тяжёлое и держит
Python-поток (GIL). Если бы оно шло здесь же, захват микрофона «замерзал» бы на
время распознавания и речь терялась. В отдельном процессе захват идёт
непрерывно — ничего не пропадает, фразы просто распознаются с небольшим
отставанием.

Язык: «Авто» — определяется для каждой фразы отдельно (лучший вариант для
живой диктовки), либо принудительно RU/EN.
"""
import os
import sys
import secrets
import tempfile
import threading
import queue
import subprocess
from multiprocessing.connection import Listener
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path

SR = 16000
FRAME = 0.03                 # длина кадра, сек
SILENCE_HANG = 0.6           # сколько тишины закрывает фразу, сек
MIN_SPEECH = 0.4             # мин. речи во фразе, чтобы её распознавать, сек
MAX_UTTER = 18.0             # принудительно закрыть слишком длинную фразу, сек

LANG_MAP = {"Авто": None, "Русский": "ru", "English": "en"}
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# Затравка для самой первой фразы (когда контекста ещё нет) — примеры с
# пунктуацией и заглавными подсказывают Whisper ставить знаки и регистр.
PUNCT_PROMPT = "Привет! Как дела? Хорошо, спасибо. Let's continue, okay?"
_SENT_END = (".", "!", "?", "…")

# Частые «галлюцинации» Whisper на тишине/шуме — не выводим их
_HALLUCINATION = {
    "", ".", "..", "...", "продолжение следует...", "продолжение следует",
    "спасибо за просмотр", "спасибо за просмотр!", "спасибо.",
    "субтитры создавал dimatorzok", "субтитры сделал dimatorzok",
    "редактор субтитров а.синецкая корректор а.егорова",
    "thank you.", "thank you", "thanks for watching.", "thanks for watching",
    "you", "bye.",
}


class PhraseSegmenter:
    """Энергетический VAD с адаптацией под уровень микрофона.

    Первые ~0.5 c измеряет уровень фона, затем порог считает ОТНОСИТЕЛЬНО
    фона (с гистерезисом), а не по фиксированному числу. Это работает и на
    тихом встроенном микрофоне, и на громком внешнем. Логика без микрофона —
    тестируется отдельно."""

    CALIB_FRAMES = max(1, int(0.5 / FRAME))

    def __init__(self):
        self.reset()

    def reset(self):
        self._buf = []
        self._speech = 0.0
        self._silence = 0.0
        self._noise = 0.0008           # низкий стартовый фон
        self._in_speech = False
        self._calib = []

    def feed(self, frame):
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12))

        # калибровка уровня фона по первым кадрам
        if len(self._calib) < self.CALIB_FRAMES:
            self._calib.append(rms)
            if len(self._calib) == self.CALIB_FRAMES:
                self._noise = max(1e-4, float(np.percentile(self._calib, 30)))
            return None

        on = max(self._noise * 3.0, 0.0014)    # порог начала речи
        off = max(self._noise * 1.8, 0.0008)   # порог удержания (гистерезис)
        thr = off if self._in_speech else on
        out = None

        if rms > thr:                          # речь
            self._buf.append(frame)
            self._speech += FRAME
            self._silence = 0.0
            self._in_speech = True
        else:                                  # тишина
            if not self._in_speech:            # фон обновляем только вне речи
                self._noise = min(0.02, 0.97 * self._noise + 0.03 * rms)
            if self._buf:
                self._buf.append(frame)        # держим короткий хвост тишины
                self._silence += FRAME
                if self._silence >= SILENCE_HANG:
                    out = self._finish()       # вернёт None, если речи было мало

        if out is None and self._speech >= MAX_UTTER:
            out = self._finish()
        return out

    def _finish(self):
        keep = bool(self._buf) and self._speech >= MIN_SPEECH
        audio = np.concatenate(self._buf).astype(np.float32) if keep else None
        self._buf = []                 # сбрасываем фразу, фон (_noise) сохраняем
        self._speech = 0.0
        self._silence = 0.0
        self._in_speech = False
        return audio

    def flush(self):
        return self._finish()


class MicRecorder:
    """Захват микрофона в отдельном потоке + нарезка на фразы через PhraseSegmenter.
    Готовые фразы кладёт в phrase_q."""

    def __init__(self, phrase_q, level_cb=None):
        self.phrase_q = phrase_q
        self.level_cb = level_cb
        self.running = False
        self.paused = False
        self.flush_pending = False     # закрыть текущую фразу сейчас (по «Паузе»)
        self.seg = PhraseSegmenter()
        self._raw_q = queue.Queue()
        self._stream = None

    def flush_now(self):
        """Пометить, чтобы поток захвата закрыл текущую фразу немедленно.
        Сам flush делает поток _loop — без гонки за буфером сегментатора."""
        self.flush_pending = True

    def start(self):
        import sounddevice as sd
        self.running = True
        self.paused = False
        self.seg.reset()
        bs = int(FRAME * SR)

        def cb(indata, frames, t, status):
            if self.running and not self.paused:
                self._raw_q.put(indata[:, 0].copy())

        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                      blocksize=bs, callback=cb)
        self._stream.start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                frame = self._raw_q.get(timeout=0.2)
            except queue.Empty:
                if self.flush_pending:     # «Пауза» — закрыть накопленную фразу
                    self.flush_pending = False
                    tail = self.seg.flush()
                    if tail is not None:
                        self.phrase_q.put(tail)
                continue
            if self.level_cb:
                self.level_cb(float(np.sqrt(np.mean(frame ** 2) + 1e-12)))
            if self.paused:
                continue
            phrase = self.seg.feed(frame)
            if phrase is not None:
                self.phrase_q.put(phrase)

    def stop(self):
        self.running = False
        try:
            self._stream.stop(); self._stream.close()
        except Exception:
            pass
        tail = self.seg.flush()
        if tail is not None:
            self.phrase_q.put(tail)


class DictaphoneWindow(ctk.CTkToplevel):
    def __init__(self, master, model_repo=DEFAULT_MODEL):
        super().__init__(master)
        self.model_repo = model_repo
        self.title("Scribe — диктофон")
        self.geometry("640x580")
        self.minsize(540, 460)

        self.phrase_q = queue.Queue()      # фразы от захвата → воркеру
        self.ui_q = queue.Queue()          # события → главный поток (Tk)
        self.rec = None
        self._mode = "idle"                # idle / recording / paused
        self._lang = "Авто"
        self._context = ""                 # уже набранный текст (для склейки и подсказки Whisper)

        # процесс распознавания и связь с ним
        self._worker_running = True
        self._proc = None
        self._conn = None
        self._listener = None
        self._addr = None
        self._model_ready = False

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_ui)
        # процесс распознавания стартуем сразу и заранее прогреваем модель,
        # чтобы к первому «Старту» всё было готово
        threading.Thread(target=self._boot_worker, daemon=True).start()

    # ---------- интерфейс ----------
    def _build(self):
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(14, 6))
        ctk.CTkLabel(top, text="🎤  Диктофон",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        self.lang_seg = ctk.CTkSegmentedButton(top, values=list(LANG_MAP), command=self._set_lang)
        self.lang_seg.set("Авто"); self.lang_seg.pack(side="right")

        ctl = ctk.CTkFrame(self, fg_color="transparent"); ctl.pack(fill="x", padx=16, pady=(2, 6))
        self.start_btn = ctk.CTkButton(ctl, text="●  Старт", width=130, command=self._toggle_start)
        self.start_btn.pack(side="left")
        self._btn_fg = self.start_btn.cget("fg_color")
        self._btn_hover = self.start_btn.cget("hover_color")
        self.pause_btn = ctk.CTkButton(ctl, text="⏸  Пауза", width=130, fg_color="gray",
                                       command=self._toggle_pause, state="disabled")
        self.pause_btn.pack(side="left", padx=8)
        self.level = ctk.CTkProgressBar(ctl); self.level.set(0)
        self.level.pack(side="left", fill="x", expand=True, padx=(12, 0))

        self.text = ctk.CTkTextbox(self, font=ctk.CTkFont(size=15), wrap="word")
        self.text.pack(fill="both", expand=True, padx=16, pady=6)

        bottom = ctk.CTkFrame(self, fg_color="transparent"); bottom.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(bottom, text="Копировать", width=110, command=self._copy).pack(side="left")
        ctk.CTkButton(bottom, text="Сохранить…", width=110, command=self._save).pack(side="left", padx=8)
        ctk.CTkButton(bottom, text="Очистить", width=100, fg_color="gray", command=self._clear).pack(side="left")
        self.status = ctk.CTkLabel(bottom, text="Загружаю модель…", text_color="gray"); self.status.pack(side="right")

    def _set_lang(self, v):
        self._lang = v

    # ---------- процесс распознавания ----------
    def _boot_worker(self):
        """Поднимаем процесс-воркер и слушающий сокет, ждём подключения,
        затем запускаем потоки отправки фраз и приёма результатов."""
        try:
            self._addr = os.path.join(tempfile.gettempdir(),
                                      f"scribe_{secrets.token_hex(6)}.sock")
            authkey = secrets.token_bytes(16)
            self._listener = Listener(self._addr, family="AF_UNIX", authkey=authkey)
            src_dir = os.path.dirname(os.path.abspath(__file__))
            worker = os.path.join(src_dir, "dictaphone_worker.py")
            # В обычном запуске и в сборке py2app (alias) sys.executable — это
            # настоящий python: ему передаём скрипт воркера. Если это стаб
            # приложения (полная сборка) — уходим через флаг --dictaphone-worker.
            if os.path.basename(sys.executable).lower().startswith("python"):
                cmd = [sys.executable, worker,
                       self._addr, authkey.hex(), self.model_repo]
            else:
                cmd = [sys.executable, "--dictaphone-worker",
                       self._addr, authkey.hex(), self.model_repo]
            # вывод воркера пишем в лог рядом с кодом — для диагностики
            try:
                self._logf = open(os.path.join(src_dir, "_worker.log"),
                                  "w", encoding="utf-8")
            except Exception:
                self._logf = subprocess.DEVNULL
            self._proc = subprocess.Popen(cmd, stdout=self._logf, stderr=self._logf)
            self._conn = self._listener.accept()      # ждём подключения воркера
            self._conn.recv()                         # ("ready", None)
        except Exception as e:
            self.ui_q.put(("error", f"Не удалось запустить распознавание: {e}"))
            return
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._receiver, daemon=True).start()

    def _sender(self):
        """Берём готовые фразы и шлём воркеру. Тихую запись усиливаем."""
        while self._worker_running:
            try:
                audio = self.phrase_q.get(timeout=0.3)
            except queue.Empty:
                continue
            if audio is None or getattr(audio, "size", 0) == 0:
                continue
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-9))
            if rms > 0:
                gain = min(0.06 / rms, 45.0)
                if gain > 1.0:
                    audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
            # «хвост» уже набранного текста — чтобы Whisper продолжал мысль,
            # а не начинал каждую фразу с заглавной и точки
            prompt = self._context[-200:] if self._context.strip() else PUNCT_PROMPT
            try:
                self._conn.send((audio, LANG_MAP.get(self._lang), prompt))
                self.ui_q.put(("status", ("Распознаю…", "#f08c00")))
            except Exception:
                break

    def _receiver(self):
        """Принимаем результаты распознавания от воркера."""
        while self._worker_running:
            try:
                kind, val = self._conn.recv()
            except (EOFError, OSError):
                break
            if kind == "text":
                piece = self._append_piece(val)
                if piece:
                    self.ui_q.put(("text", piece))
                if self._mode == "recording":
                    self.ui_q.put(("status", ("Слушаю…", "#2f9e44")))
            elif kind == "error":
                self.ui_q.put(("error", f"Ошибка распознавания: {val}"))
            elif kind == "warmed":
                self._model_ready = True
                if self._mode == "idle":
                    self.ui_q.put(("status", ("Готов к диктовке", "gray")))

    def _append_piece(self, txt):
        """Готовим кусок к вставке: склеиваем с уже набранным как продолжение
        мысли. Заглавную ставим только в начале и после конца предложения —
        внутри мысли регистр оставляем как дал Whisper (с учётом контекста)."""
        raw = " ".join((txt or "").split()).strip()
        if not raw or raw.lower() in _HALLUCINATION:
            return ""
        ctx = self._context
        if not ctx:
            piece = raw[0].upper() + raw[1:]
        else:
            if ctx.rstrip().endswith(_SENT_END):
                raw = raw[0].upper() + raw[1:]
            sep = "" if ctx.endswith((" ", "\n")) else " "
            piece = sep + raw
        self._context = (self._context + piece)[-2000:]
        return piece

    def _shutdown_worker(self):
        self._worker_running = False
        try:
            if self._conn:
                self._conn.send(None)
        except Exception:
            pass
        for closer in (lambda: self._conn.close(),
                       lambda: self._listener.close()):
            try:
                closer()
            except Exception:
                pass
        try:
            if self._proc:
                self._proc.terminate()
        except Exception:
            pass
        try:
            if self._addr and os.path.exists(self._addr):
                os.remove(self._addr)
        except Exception:
            pass

    # ---------- управление записью ----------
    def _toggle_start(self):
        if self._mode == "idle":
            self._start()
        else:
            self._stop()

    def _start(self):
        try:
            import sounddevice  # noqa: F401
        except Exception:
            messagebox.showinfo(
                "Нужен модуль микрофона",
                "Установите один раз:\n\nbrew install portaudio\n"
                "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 "
                "-m pip install sounddevice")
            return
        self._mode = "recording"
        self.start_btn.configure(text="⏹  Стоп", fg_color="#c0392b", hover_color="#a5342a")
        self.pause_btn.configure(state="normal", text="⏸  Пауза")
        self.status.configure(
            text="Слушаю…" if self._model_ready else "Слушаю… (модель загружается)",
            text_color="#2f9e44")
        self.rec = MicRecorder(self.phrase_q, level_cb=lambda v: self.ui_q.put(("level", v)))
        try:
            self.rec.start()
        except Exception as e:
            self._mode = "idle"
            self.start_btn.configure(text="●  Старт", fg_color=self._btn_fg, hover_color=self._btn_hover)
            self.pause_btn.configure(state="disabled")
            messagebox.showerror("Микрофон", f"Не удалось открыть микрофон:\n{e}\n\n"
                                 "Проверьте разрешение на доступ к микрофону в «Системных настройках».")
            return

    def _stop(self):
        self._mode = "idle"
        if self.rec:
            self.rec.stop()            # флашит последнюю фразу в очередь
        self.start_btn.configure(text="●  Старт", fg_color=self._btn_fg, hover_color=self._btn_hover)
        self.pause_btn.configure(state="disabled", text="⏸  Пауза")
        self.level.set(0)
        self.status.configure(text="Остановлено", text_color="gray")

    def _toggle_pause(self):
        if self._mode == "recording":
            self._mode = "paused"; self.rec.paused = True
            self.rec.flush_now()           # закрыть текущую фразу сразу
            self.pause_btn.configure(text="▶  Продолжить")
            self.status.configure(text="Пауза", text_color="orange")
        elif self._mode == "paused":
            self._mode = "recording"; self.rec.paused = False
            self.pause_btn.configure(text="⏸  Пауза")
            self.status.configure(text="Слушаю…", text_color="#2f9e44")

    # ---------- обновление UI ----------
    def _drain_ui(self):
        try:
            while True:
                kind, val = self.ui_q.get_nowait()
                if kind == "text":
                    self.text.insert("end", val)   # кусок уже со своим разделителем
                    self.text.see("end")
                elif kind == "level":
                    self.level.set(min(1.0, val * 20))
                elif kind == "status":
                    txt, color = val
                    self.status.configure(text=txt, text_color=color)
                elif kind == "error":
                    self.status.configure(text=val, text_color="red")
        except queue.Empty:
            pass
        self.after(100, self._drain_ui)

    # ---------- действия ----------
    def _copy(self):
        t = self.text.get("1.0", "end").strip()
        if t:
            self.clipboard_clear(); self.clipboard_append(t)
            self.status.configure(text="Скопировано в буфер", text_color="gray")

    def _save(self):
        t = self.text.get("1.0", "end").strip()
        if not t:
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".txt", initialfile="Диктовка",
            filetypes=[("Текст", "*.txt"), ("Word", "*.docx")])
        if not path:
            return
        try:
            if path.lower().endswith(".docx"):
                from docx import Document
                doc = Document()
                for para in t.split("\n"):
                    doc.add_paragraph(para)
                doc.save(path)
            else:
                Path(path).write_text(t + "\n", encoding="utf-8")
            self.status.configure(text="Сохранено", text_color="green")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def _clear(self):
        self.text.delete("1.0", "end")
        self._context = ""

    def _on_close(self):
        try:
            self._mode = "idle"
            if self.rec:
                self.rec.stop()
        except Exception:
            pass
        self._shutdown_worker()
        self.destroy()
