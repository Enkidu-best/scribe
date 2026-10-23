#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диктофон для Scribe - живая диктовка с микрофона в реальном времени.
Страница «Диктофон» главного окна.

Как работает:
  захват микрофона (sounddevice) → нарезка речи на фразы детектором речи
  Silero VAD (vad.py; фраза закрывается паузой от 0,5 с, длинная речь без пауз
  режется в самой тихой точке последних 3 с, не позже 25 с) → каждая
  законченная фраза уходит в ОТДЕЛЬНЫЙ процесс распознавания
  (dictaphone_worker.py) и возвращается текстом, который дописывается в окно.

Распознавание вынесено в отдельный процесс намеренно: оно тяжёлое и держит
Python-поток (GIL). Если бы оно шло здесь же, захват микрофона «замерзал» бы на
время распознавания и речь терялась. Процесс-воркер запускает и Legal Knowledge
Desk (голосовой ввод), поэтому его протокол не меняем.

Надёжность:
  • воркер упал или завис - перезапускаем, неотвеченная фраза отправляется заново;
  • микрофон пропал (отключили AirPods) - переподключаемся к устройству по умолчанию.

Язык: «Авто» - определяется для каждой фразы отдельно (лучший вариант для
живой диктовки), либо принудительно RU/EN.
"""
import os
import re
import sys
import time
import secrets
import threading
import queue
import subprocess
from multiprocessing.connection import Listener
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path

import platform_util as pu
import theme as T
import vad
from common.lock import heavy_lock as hl                   # очередь видеокарты - общий модуль (стандарт 2.4)
from common.lock.lock_ui import texts as lt, tk_busy

SR = 16000
FRAME = vad.CHUNK            # 512 отсчётов = 32 мс - кадр детектора речи
NO_AUDIO_SEC = 1.5           # столько без кадров с микрофона - считаем, что он пропал
ANSWER_TIMEOUT = 90.0        # столько ждём ответа воркера, потом перезапускаем
WEAK_SPEECH = 0.4            # меньше речи по детектору - фраза под подозрением на «галлюцинацию»

LANG_MAP = {"Авто": None, "Русский": "ru", "English": "en"}
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MIC = "Системный (по умолчанию)"

# Затравка для самой первой фразы (когда контекста ещё нет) - примеры с
# пунктуацией и заглавными подсказывают Whisper ставить знаки и регистр.
PUNCT_PROMPT = "Привет! Как дела? Хорошо, спасибо. Let's continue, okay?"
_SENT_END = (".", "!", "?", "…")
_SENT = re.compile(r"[^.!?…]+[.!?…]+")

# Частые «галлюцинации» Whisper на тишине/шуме. Отбрасываем их, только если
# детектор речи почти не слышал речи во фразе: настоящее «спасибо» остаётся.
_HALLUCINATION = {
    "", ".", "..", "...", "продолжение следует...", "продолжение следует",
    "спасибо за просмотр", "спасибо за просмотр!", "спасибо.",
    "субтитры создавал dimatorzok", "субтитры сделал dimatorzok",
    "редактор субтитров а.синецкая корректор а.егорова",
    "thank you.", "thank you", "thanks for watching.", "thanks for watching",
    "you", "bye.",
}


def last_sentence(text, limit=100):
    """Подсказка Whisper: последнее законченное предложение, не длиннее limit знаков."""
    text = " ".join(text.split())
    sents = _SENT.findall(text)
    s = sents[-1].strip() if sents else text
    return s[-limit:].strip()


def input_devices():
    """Имена микрофонов (устройств с входом)."""
    try:
        import sounddevice as sd
        return [d["name"] for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
    except Exception:
        return []


def _refresh_portaudio():
    """PortAudio запоминает список устройств при старте - после подключения/отключения
    наушников его нужно перечитать."""
    try:
        import sounddevice as sd
        sd._terminate()
        sd._initialize()
    except Exception:
        pass


class MicRecorder:
    """Захват микрофона в отдельном потоке + нарезка на фразы (vad.StreamSegmenter).
    Готовые фразы кладёт в phrase_q как dict(audio, speech). Если звук перестал
    приходить (устройство отключили) - переподключается к устройству по умолчанию."""

    def __init__(self, phrase_q, level_cb=None, event_cb=None, device=None):
        self.phrase_q = phrase_q
        self.level_cb = level_cb
        self.event_cb = event_cb or (lambda kind, val: None)
        self.device = device           # имя устройства или None - системный
        self.running = False
        self.paused = False
        self.flush_pending = False     # закрыть текущую фразу сейчас (по «Паузе»)
        self.seg = vad.StreamSegmenter()
        self._raw_q = queue.Queue()
        self._stream = None
        self._last_frame = 0.0
        self.device_name = ""

    def flush_now(self):
        """Пометить, чтобы поток захвата закрыл текущую фразу немедленно.
        Сам flush делает поток _loop - без гонки за буфером сегментатора."""
        self.flush_pending = True

    def _open(self, device):
        import sounddevice as sd

        def cb(indata, frames, t, status):
            if self.running and not self.paused:
                self._raw_q.put(indata[:, 0].copy())
            self._last_frame = time.monotonic()

        names = input_devices()
        dev = device if device in names else None
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME,
                                      callback=cb, device=dev)
        self._stream.start()
        self._last_frame = time.monotonic()
        try:
            self.device_name = sd.query_devices(dev, "input")["name"]
        except Exception:
            self.device_name = dev or ""

    def start(self):
        self.running = True
        self.paused = False
        self.seg.reset()
        self._open(self.device)
        threading.Thread(target=self._loop, daemon=True).start()

    def _close_stream(self):
        try:
            self._stream.stop(); self._stream.close()
        except Exception:
            pass

    def reconnect(self, device=None):
        """Закрыть поток и открыть заново: выбранный микрофон, а если его нет - системный."""
        self._close_stream()
        _refresh_portaudio()
        for attempt in range(10):
            if not self.running:
                return False
            try:
                self._open(device if device is not None else self.device)
                self.event_cb("mic", self.device_name)
                return True
            except Exception:
                time.sleep(0.5)
        self.event_cb("mic_error", "Микрофон не найден")
        return False

    def _loop(self):
        while self.running:
            try:
                frame = self._raw_q.get(timeout=0.2)
            except queue.Empty:
                if self.flush_pending:     # «Пауза» - закрыть накопленную фразу
                    self.flush_pending = False
                    tail = self.seg.flush()
                    if tail is not None:
                        self.phrase_q.put(tail)
                stream_dead = self._stream is None or not getattr(self._stream, "active", False)
                if self.running and (stream_dead or time.monotonic() - self._last_frame > NO_AUDIO_SEC):
                    self.event_cb("mic_lost", "")
                    tail = self.seg.flush()             # недосказанное не теряем
                    if tail is not None:
                        self.phrase_q.put(tail)
                    self.reconnect()
                continue
            if self.level_cb:
                self.level_cb(float(np.sqrt(np.mean(frame ** 2) + 1e-12)))
            if self.paused:
                continue
            for k in range(0, len(frame) - FRAME + 1, FRAME):
                phrase = self.seg.feed(frame[k:k + FRAME])
                if phrase is not None:
                    phrase["t"] = time.monotonic()
                    self.phrase_q.put(phrase)

    def stop(self):
        self.running = False
        self._close_stream()
        tail = self.seg.flush()
        if tail is not None:
            self.phrase_q.put(tail)


LOG_DIR = pu.log_dir()        # Mac - ~/Library/Logs/Scribe; Windows/Linux - папка логов системы
LAG_HINT = 5.0               # текст приходит позже этого (с) - на процессоре подсказать модель поменьше


def engine_key(model):
    """«small», «turbo»… по ключу или имени репозитория."""
    m = str(model).lower()
    for k in ("small", "medium", "base", "turbo", "large-v3"):
        if m == k or k in m.rsplit("/", 1)[-1]:
            return k
    return m

# Состояние → (текст, вид чипа)
_ST = {
    "loading": ("Модель загружается…", "neutral"),
    "ready": ("Готов к диктовке", "ok"),
    "listen": ("Слушаю…", "acc"),
    "listen_wait": ("Слушаю… модель ещё загружается", "acc"),
    "busy": ("Распознаю…", "warn"),
    "paused": ("Пауза", "warn"),
    "stopped": ("Остановлено", "neutral"),
    "restart": ("Перезапускаю распознавание…", "warn"),
    "mic_lost": ("Микрофон пропал - переподключаюсь…", "warn"),
}


class DictaphonePage(ctk.CTkFrame):
    """Страница «Диктофон» главного окна. Процесс распознавания поднимается при
    первом открытии страницы (activate) и живёт до закрытия приложения
    (shutdown) - повторные заходы на страницу мгновенные."""

    def __init__(self, master, model_repo=DEFAULT_MODEL, cfg=None):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.model_repo = model_repo
        self.cfg = cfg if cfg is not None else {}

        self.phrase_q = queue.Queue()      # фразы от захвата → воркеру
        self.ui_q = queue.Queue()          # события → главный поток (Tk)
        self.rec = None
        self._mode = "idle"                # idle / recording / paused
        self._lang = "Авто"
        self._context = ""                 # уже набранный текст (для склейки и подсказки Whisper)
        self._last_piece = ""              # чтобы одинаковую фразу подряд не вставлять дважды
        self._dirty = False                # есть несохранённый текст

        # процесс распознавания и связь с ним
        self._booted = False
        self._worker_running = True
        self._proc = None
        self._conn = None
        self._listener = None
        self._addr = None
        self._logf = None
        self._model_ready = False
        self._inflight = None              # фраза отправлена, ответа ещё нет - при падении шлём заново
        self._mic_lost = False             # микрофон пропал - держим этот статус, пока не вернётся
        self._restarts = []                # моменты перезапусков (защита от бесконечного цикла)
        self._lags = []                    # отставание текста от речи по последним фразам, с
        self._lag_shown = 0.0
        # на процессоре, если модель диктофона не выбрана вручную: текст отстаёт больше LAG_HINT -
        # сами переходим на Small (на 4 ядрах turbo - 9-12 с на фразу, small - 2-4 с)
        self.auto_small = False
        self.on_model_switch = None        # окно: обновить выбор в Настройках и сохранить
        self.small_ok = lambda: True       # окно: Small есть на диске или его можно скачать
        # замок видеокарты на запись (README модуля, 8.1): «Начать запись» берёт, «Остановить» отдаёт,
        # «Пауза» дольше 15 мин - отпускает сам, «Продолжить» - берёт снова
        self.hlock = hl.HeavyLock("Scribe", "диктофон", held_env="SCRIBE_HEAVY_LOCK_HELD",
                                  path_env="SCRIBE_HEAVY_LOCK")
        self.cpu_ok = lambda: False        # окно: можно ли на процессоре (четвёртый вариант окна «занята»)
        self._cpu = False                  # процесс распознавания - на процессоре, модель Small
        self._gpu_model = model_repo       # модель для видеокарты - вернуться к ней после процессора
        self._respawn = False              # перезапустить процесс распознавания между фразами
        self._wait_cancel = None           # ждём очереди видеокарты - отмена
        self._idle_tick = 0

        self._build()
        self.after(100, self._drain_ui)

    def activate(self):
        """Страницу открыли: заранее поднимаем процесс и прогреваем модель,
        чтобы к первому «Начать запись» всё было готово."""
        if not self._booted:
            self._booted = True
            threading.Thread(target=self._dispatch, daemon=True).start()
        self._fill_mics()

    # ---------- интерфейс ----------
    def _build(self):
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=28, pady=(22, 0))
        T.label(head, "Диктофон", size=22, bold=True).pack(side="left")
        self.lang_seg = T.Segmented(head, list(LANG_MAP), command=self._set_lang)
        self.lang_seg.pack(side="right")
        T.label(head, "Язык", size=13, color=T.MUTED).pack(side="right", padx=(0, 10))
        T.label(self, f"Говорите - текст появляется по мере речи. Всё распознаётся {pu.HERE}, "
                      "в сеть ничего не уходит.", size=13, color=T.MUTED, anchor="w"
                ).pack(fill="x", padx=28, pady=(2, 14))

        ctl = T.card(self)
        ctl.pack(fill="x", padx=28)
        row = ctk.CTkFrame(ctl, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(12, 6))
        self.start_btn = T.button(row, "Начать запись", "mic", kind="primary", width=170,
                                  command=self._toggle_start)
        self.start_btn.pack(side="left")
        self.pause_btn = T.button(row, "Пауза", "pause", width=130, command=self._toggle_pause,
                                  state="disabled")
        self.pause_btn.pack(side="left", padx=8)
        self.level = ctk.CTkProgressBar(row, height=6)
        self.level.set(0)
        self.level.pack(side="left", fill="x", expand=True, padx=(10, 14))
        self.chip = T.chip(row)
        self.chip.pack(side="right")
        self._status("loading")
        mrow = ctk.CTkFrame(ctl, fg_color="transparent")
        mrow.pack(fill="x", padx=14, pady=(0, 12))
        T.label(mrow, "Микрофон", 13, color=T.MUTED).pack(side="left")
        self.mic_var = ctk.StringVar(value=self.cfg.get("mic") or DEFAULT_MIC)
        self.mic_menu = ctk.CTkOptionMenu(mrow, variable=self.mic_var, values=[self.mic_var.get()], width=300,
                                          height=28, font=T.font(13), dropdown_font=T.font(13),
                                          command=self._pick_mic)
        self.mic_menu.pack(side="left", padx=(10, 6))
        T.button(mrow, "", "refresh-cw", kind="ghost", height=28,
                 command=self._fill_mics).pack(side="left")
        self.mic_note = T.label(mrow, "", 12, color=T.FAINT, anchor="w")
        self.mic_note.pack(side="left", padx=(8, 0))
        self.gpu_note = T.label(ctl, "", 12, color=T.WARN, anchor="w", justify="left", wraplength=760)

        box = T.card(self)
        box.pack(fill="both", expand=True, padx=28, pady=12)
        self.text = ctk.CTkTextbox(box, font=T.font(16), wrap="word", fg_color="transparent",
                                   border_width=0, spacing2=4, spacing3=8)
        self.text.pack(fill="both", expand=True, padx=12, pady=10)

        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=28, pady=(0, 18))
        T.button(bottom, "Копировать", "copy", width=130, command=self._copy).pack(side="left")
        T.button(bottom, "Сохранить…", "save", width=130, command=self._save).pack(side="left", padx=8)
        T.button(bottom, "Очистить", "trash-2", kind="ghost", width=110, command=self._clear).pack(side="left")
        self.words = T.label(bottom, "", size=12, color=T.FAINT)
        self.words.pack(side="right")

    def _status(self, key, text=None):
        t, kind = _ST.get(key, (key, "neutral"))
        T.set_chip(self.chip, text or t, kind)

    def _set_lang(self, v):
        self._lang = v

    # ---------- микрофон ----------
    def _fill_mics(self):
        if self._mode == "idle":
            _refresh_portaudio()
        names = [DEFAULT_MIC] + input_devices()
        self.mic_menu.configure(values=names)
        if self.mic_var.get() not in names:
            self.mic_var.set(DEFAULT_MIC)

    def _pick_mic(self, name):
        self.cfg["mic"] = None if name == DEFAULT_MIC else name
        try:
            from whisper_gui import save_config
            save_config(self.cfg)
        except Exception:
            pass
        if self.rec and self._mode != "idle":
            self.rec.device = self.cfg["mic"]
            threading.Thread(target=self.rec.reconnect, daemon=True).start()

    def _mic_event(self, kind, val):
        self.ui_q.put((kind, val))

    # ---------- процесс распознавания ----------
    def _spawn(self):
        """Поднять процесс-воркер и дождаться его подключения ("ready")."""
        # Mac/Linux - сокет AF_UNIX, Windows - именованный канал AF_PIPE; сообщения те же
        self._addr, family = pu.worker_address()
        authkey = secrets.token_bytes(16)
        self._listener = Listener(self._addr, family=family, authkey=authkey)
        src_dir = os.path.dirname(os.path.abspath(__file__))
        worker = os.path.join(src_dir, "dictaphone_worker.py")
        # В обычном запуске и в сборке py2app (alias) sys.executable - это
        # настоящий python: ему передаём скрипт воркера. Если это стаб
        # приложения (полная сборка) - уходим через флаг --dictaphone-worker.
        if os.path.basename(sys.executable).lower().startswith("python"):
            cmd = [sys.executable, worker, self._addr, authkey.hex(), self.model_repo]
        else:
            cmd = [sys.executable, "--dictaphone-worker", self._addr, authkey.hex(), self.model_repo]
        # вывод воркера - в ~/Library/Logs/Scribe (открывается в «Консоли»)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self._logf = open(LOG_DIR / "dictaphone.log", "a", encoding="utf-8")
        except Exception:
            self._logf = subprocess.DEVNULL
        env = dict(os.environ)
        if self._cpu:                                 # видеокарта занята - распознаём на процессоре
            env.update(SCRIBE_ENGINE="fw", SCRIBE_DEVICE="cpu")
        self._proc = subprocess.Popen(cmd, stdout=self._logf, stderr=self._logf, env=env)
        self._conn = self._listener.accept()          # ждём подключения воркера
        self._conn.recv()                             # ("ready", None)

    def _kill_worker(self):
        for closer in (lambda: self._conn.close(), lambda: self._listener.close(),
                       lambda: self._proc.kill(), lambda: os.remove(self._addr)):
            try:
                closer()
            except Exception:
                pass
        self._conn = None
        self._model_ready = False

    def _restart(self, why):
        """Воркер упал или завис: перезапуск. Неотвеченная фраза остаётся в _inflight
        и уйдёт заново. Больше 3 перезапусков за минуту - останавливаемся."""
        self._kill_worker()
        now = time.monotonic()
        self._restarts = [t for t in self._restarts if now - t < 60] + [now]
        if len(self._restarts) > 3:
            self.ui_q.put(("error", f"Распознавание падает - см. {LOG_DIR / 'dictaphone.log'}"))
            return False
        self.ui_q.put(("status", "restart"))
        try:
            with open(LOG_DIR / "dictaphone.log", "a", encoding="utf-8") as f:
                f.write(f"\n--- перезапуск воркера: {why} ---\n")
        except Exception:
            pass
        return True

    def _prepare(self, audio):
        """Тихую запись усиливаем."""
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-9))
        if rms > 0:
            gain = min(0.06 / rms, 45.0)
            if gain > 1.0:
                audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
        return audio

    def _prompt(self):
        # последнее законченное предложение (до 100 знаков) - Whisper продолжает мысль
        # в том же регистре, но не повторяет длинный хвост
        return last_sentence(self._context) if self._context.strip() else PUNCT_PROMPT

    def _dispatch(self):
        """Единственный поток, который говорит с воркером: фраза - ответ - следующая.
        Так подсказка всегда учитывает предыдущую фразу, а при падении воркера ясно,
        какую фразу отправить заново."""
        while self._worker_running:
            if self._respawn and self._inflight is None:  # видеокарта ↔ процессор - новый процесс
                self._respawn = False
                self._kill_worker()
            if self._conn is None:
                try:
                    self._spawn()
                    self.ui_q.put(("status", "listen" if self._mode == "recording" else "loading"))
                except Exception as e:
                    if not self._restart(f"запуск: {e}"):
                        return
                    time.sleep(1.0)
                    continue
            if self._inflight is None:
                try:
                    item = self.phrase_q.get(timeout=0.3)
                except queue.Empty:
                    if self._proc is not None and self._proc.poll() is not None and self._worker_running:
                        if not self._restart(f"процесс завершился с кодом {self._proc.returncode}"):
                            return
                    elif self._conn is not None and self._conn.poll(0):
                        self._handle(*self._safe_recv())          # «warmed» и т. п.
                    continue
                if isinstance(item, np.ndarray):
                    item = dict(audio=item, speech=len(item) / SR)
                if item is None or item.get("audio") is None or item["audio"].size == 0:
                    continue
                self._inflight = item
            item = self._inflight
            try:
                self._conn.send((self._prepare(item["audio"]), LANG_MAP.get(self._lang), self._prompt()))
                self.ui_q.put(("status", "busy"))
                t0 = time.monotonic()
                while True:
                    if not self._conn.poll(0.5):
                        if self._proc.poll() is not None:
                            raise EOFError("процесс завершился")
                        if time.monotonic() - t0 > ANSWER_TIMEOUT:
                            raise TimeoutError("нет ответа")
                        continue
                    kind, val = self._conn.recv()
                    if kind in ("text", "error"):
                        break
                    self._handle(kind, val)
            except (EOFError, OSError, TimeoutError, BrokenPipeError) as e:
                if not self._worker_running:
                    return
                if not self._restart(str(e) or type(e).__name__):
                    return
                continue                                      # _inflight уйдёт заново
            self._inflight = None
            lag = time.monotonic() - item.get("t", t0)
            self._lags = (self._lags + [lag])[-3:]
            switch = False
            if len(self._lags) == 3 and sorted(self._lags)[1] > LAG_HINT:
                if self.auto_small and engine_key(self.model_repo) != "small" and self.small_ok():
                    switch = True
                    self.auto_small = False
                    self.model_repo = "small"
                    self.ui_q.put(("switched", sorted(self._lags)[1]))
                    self._lags = []
                else:
                    self.ui_q.put(("lag", sorted(self._lags)[1]))
            if kind == "text":
                self.hlock.touch()
                piece = self._append_piece(val, item.get("speech", 9.0))
                if piece:
                    self.ui_q.put(("text", piece))
            else:
                self.ui_q.put(("error", f"Ошибка распознавания: {val}"))
            if self._mode == "recording":
                self.ui_q.put(("status", "listen"))
            if switch:                                        # следующая фраза - уже процессом с Small
                self._kill_worker()

    def _safe_recv(self):
        try:
            return self._conn.recv()
        except Exception:
            return ("lost", None)

    def _handle(self, kind, val):
        if kind == "warmed":
            self._model_ready = True
            self.ui_q.put(("status", "ready" if self._mode == "idle" else "listen"))

    def _append_piece(self, txt, speech=9.0):
        """Готовим кусок к вставке: склеиваем с уже набранным как продолжение
        мысли. Заглавную ставим только в начале и после конца предложения -
        внутри мысли регистр оставляем как дал Whisper (с учётом контекста)."""
        raw = " ".join((txt or "").split()).strip()
        raw = re.sub(r"^[\s!?.,;:…-]+", "", raw)   # «! Поэтому…» - одинокая пунктуация в начале куска
        if not re.search(r"\w", raw):
            return ""
        if raw.lower() in _HALLUCINATION and speech < WEAK_SPEECH:
            return ""                      # речи почти не было - это «галлюцинация» на шуме
        norm = re.sub(r"[^\w]+", " ", raw.lower()).strip()
        if norm and norm == self._last_piece:
            return ""                      # та же фраза подряд - повтор Whisper, не вставляем
        self._last_piece = norm
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

    def shutdown(self):
        """Закрытие приложения: остановить запись и процесс распознавания, отдать видеокарту."""
        if self._wait_cancel is not None:
            self._wait_cancel.set()
        self.hlock.release()
        try:
            self._mode = "idle"
            if self.rec:
                self.rec.stop()
        except Exception:
            pass
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

    def has_unsaved(self):
        return self._dirty and bool(self.text.get("1.0", "end").strip())

    def is_recording(self):
        return self._mode != "idle"

    # ---------- управление записью ----------
    def _toggle_start(self):
        if self._wait_cancel is not None:          # ждём очереди видеокарты - кнопка «Отменить ожидание»
            self._wait_cancel.set()
        elif self._mode == "idle":
            self._start()
        else:
            self._stop()

    # ---------- видеокарта (стандарт 2.4, 2.5)
    def _gate(self):
        """Замок видеокарты перед записью: held / wait / now / cpu, None - «Отмена»."""
        if self.hlock.held or self.hlock.try_acquire():
            self._use_gpu()
            return "held"
        ans = tk_busy.ask_busy(self.winfo_toplevel(), T, self.hlock.holder(), cpu=self.cpu_ok(),
                               queue_names=self.hlock.ahead_names())
        if ans == "now":
            self.hlock.run_now()
            self._use_gpu()
        elif ans == "cpu":
            self._use_cpu()
        return ans

    def _use_gpu(self):
        self._note("")
        if self._cpu:
            self._cpu, self.model_repo, self._respawn = False, self._gpu_model, True

    def _use_cpu(self):
        if not self._cpu:
            self._gpu_model = self.model_repo
            self._cpu, self.model_repo, self._respawn = True, "small", True
            self._model_ready = False
        self._note("Видеокарта занята - диктофон распознаёт на процессоре (модель Small), до «Остановить».")

    def _note(self, text, color=T.WARN):
        if text:
            self.gpu_note.configure(text=text, text_color=color)
            self.gpu_note.pack(fill="x", padx=14, pady=(0, 10))
        else:
            self.gpu_note.pack_forget()

    def _wait_then(self, then):
        """«Подождать очереди»: ждать замок в фоне; дождались - then(), отменили - как было."""
        cancel = self._wait_cancel = threading.Event()
        self.pause_btn.configure(state="disabled")
        self.start_btn.configure(text=lt.t("cancel_wait"), width=230)
        T.restyle(self.start_btn, "default", "x")
        T.set_chip(self.chip, lt.t("chip_waiting"), "warn")

        def run():
            ok = self.hlock.acquire(cancel=cancel,
                                    on_wait=lambda h, a: self.ui_q.put(("gpuline", lt.waiting_line(h, a))))
            self.ui_q.put(("gpuwait_done", (ok, then)))
        threading.Thread(target=run, daemon=True).start()

    def _start(self):
        mode = self._gate()
        if mode == "wait":
            self._wait_then(self._begin)
        elif mode is not None:
            self._begin()

    def _begin(self):
        try:
            import sounddevice  # noqa: F401
        except Exception:
            messagebox.showinfo(
                "Нужен модуль микрофона",
                pu.MIC_HINT + f"\"{sys.executable}\" -m pip install sounddevice")
            return
        self.activate()
        self._mode = "recording"
        self.start_btn.configure(text="Остановить")
        T.restyle(self.start_btn, "danger", "square")
        self.pause_btn.configure(state="normal", text="Пауза", image=T.icon("pause"))
        self._status("listen" if self._model_ready else "listen_wait")
        self.rec = MicRecorder(self.phrase_q, level_cb=lambda v: self.ui_q.put(("level", v)),
                               event_cb=self._mic_event, device=self.cfg.get("mic"))
        try:
            self.rec.start()
            self.mic_note.configure(text=self.rec.device_name)
        except Exception as e:
            self._mode = "idle"
            self._reset_buttons()
            self.hlock.release()
            messagebox.showerror("Микрофон", f"Не удалось открыть микрофон:\n{e}\n\n"
                                 "Проверьте доступ к микрофону: Системные настройки → "
                                 "Конфиденциальность и безопасность → Микрофон.")
            return

    def _reset_buttons(self):
        self.start_btn.configure(text="Начать запись", width=170)
        T.restyle(self.start_btn, "primary", "mic")
        self.pause_btn.configure(state="disabled", text="Пауза", image=T.icon("pause"))

    def _stop(self):
        self._mode = "idle"
        if self.rec:
            self.rec.stop()            # флашит последнюю фразу в очередь
        self._reset_buttons()
        self.level.set(0)
        self._status("stopped")
        self._note("")
        # последняя фраза ещё распознаётся - отдаём видеокарту, когда она придёт (или через 20 с)
        threading.Thread(target=self._release_after_tail, daemon=True).start()

    def _release_after_tail(self):
        end = time.monotonic() + 20
        while time.monotonic() < end and (self._inflight is not None or not self.phrase_q.empty()):
            time.sleep(0.2)
        if self._mode == "idle" and self._wait_cancel is None:
            self.hlock.release()

    def _toggle_pause(self):
        if self._mode == "recording":
            self._mode = "paused"; self.rec.paused = True
            self.rec.flush_now()           # закрыть текущую фразу сразу
            self.pause_btn.configure(text="Продолжить", image=T.icon("play"))
            self._status("paused")
            self.hlock.touch()             # с этой минуты считается простой (15 мин - отпустить видеокарту)
        elif self._mode == "paused":
            if self.hlock.idle_released:   # на паузе видеокарту отпустили - взять снова (или окно «занята»)
                mode = self._gate()
                if mode == "wait":
                    self._wait_then(self._resume)
                    return
                if mode is None:
                    return
                self.hlock.idle_released = False
            self._resume()

    def _resume(self):
        self._note("")
        self._reset_buttons()
        self.start_btn.configure(text="Остановить")
        T.restyle(self.start_btn, "danger", "square")
        self._mode = "recording"; self.rec.paused = False
        self.pause_btn.configure(state="normal", text="Пауза", image=T.icon("pause"))
        self._status("listen")
        self.hlock.touch()

    # ---------- обновление UI ----------
    def _drain_ui(self):
        try:
            while True:
                kind, val = self.ui_q.get_nowait()
                if kind == "text":
                    self.text.insert("end", val)   # кусок уже со своим разделителем
                    self.text.see("end")
                    self._dirty = True
                    self._count_words()
                elif kind == "level":
                    self.level.set(min(1.0, val * 20))
                elif kind == "status":
                    if not (self._mic_lost and val in ("listen", "busy", "listen_wait")):
                        self._status(val)
                elif kind == "mic_lost":
                    self._mic_lost = True
                    self._status("mic_lost")
                elif kind == "mic":
                    self._mic_lost = False
                    self.mic_note.configure(text=val)
                    self._fill_mics()
                    if self._mode == "recording":
                        T.set_chip(self.chip, f"Микрофон: {val}", "ok")
                elif kind == "mic_error":
                    self._mic_lost = False
                    T.set_chip(self.chip, val, "bad")
                elif kind == "error":
                    T.set_chip(self.chip, val, "bad")
                elif kind == "switched":
                    self._lag_shown = time.monotonic()
                    T.set_chip(self.chip, "Текст отставал больше 5 с - диктофон перешёл на модель Small "
                                          "(быстрее). Вернуть - в Настройках", "warn")
                    if self.on_model_switch:
                        self.on_model_switch("small")
                elif kind == "gpuline":
                    self._note(val)
                elif kind == "gpuwait_done":
                    ok, then = val
                    self._wait_cancel = None
                    self._note("")
                    if ok:
                        self.hlock.idle_released = False
                        self._use_gpu()
                        then()
                    elif self._mode == "paused":
                        self._reset_buttons()
                        self.start_btn.configure(text="Остановить")
                        T.restyle(self.start_btn, "danger", "square")
                        self.pause_btn.configure(state="normal", text="Продолжить", image=T.icon("play"))
                        self._status("paused")
                    else:
                        self._reset_buttons()
                        self._status("ready" if self._model_ready else "stopped")
                elif kind == "lag":
                    if time.monotonic() - self._lag_shown > 30:
                        self._lag_shown = time.monotonic()
                        T.set_chip(self.chip, f"Текст отстаёт на {val:.0f} с - в Настройках можно "
                                              "выбрать модель поменьше", "warn")
        except queue.Empty:
            pass
        self._idle_tick += 1
        if self._idle_tick % 50 == 0 and self._mode == "paused" and self.hlock.idle_check():
            self._note(lt.t("idle_released"), T.MUTED)      # простой на паузе 15 мин (стандарт 2.5)
        self.after(100, self._drain_ui)

    def _count_words(self):
        n = len(self.text.get("1.0", "end").split())
        self.words.configure(text=f"{n} слов" if n else "")

    # ---------- действия ----------
    def _copy(self):
        t = self.text.get("1.0", "end").strip()
        if t:
            self.clipboard_clear(); self.clipboard_append(t)
            T.set_chip(self.chip, "Скопировано", "ok")

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
            self._dirty = False
            T.set_chip(self.chip, "Сохранено", "ok")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def _clear(self):
        if self.has_unsaved() and not messagebox.askyesno(
                "Очистить", "Текст не сохранён. Очистить всё равно?", parent=self):
            return
        self.text.delete("1.0", "end")
        self._context = ""
        self._dirty = False
        self._count_words()
