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
import time
import queue
import threading
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
    import platform_util as pu
    pu.fix_streams("dictaphone.log")
    pu.setup_path()
    pu.setup_cuda()
    import dictaphone_worker
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    dictaphone_worker.main()
    sys.exit(0)

# --- PATH: ffmpeg/ffprobe (в сборке для Windows - свои) и ollama находятся при любом запуске ---
import platform_util as pu  # noqa: E402
pu.fix_streams()
pu.setup_path()

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
# Ядро без окна - core.py (его же использует консольный scribe_cli.py)
from core import (  # noqa: F401 - всё ядро доступно и как whisper_gui.<имя>
    MODEL_REPO, MODELS, LANG_SEG, LANG_CODE, LANG_HINT, LANG_NAME,
    FMT_SEG, TEXT_EXTS, NSPK, _model_title, _models_base, _models_dir_path,
    _models_dir_display, _apply_models_dir, AUDIO_EXTS, AUDIO_TYPES, STEP_NAMES, STEP_CHIP,
    DIAR_STEPS, FORMATS_AUDIO, FORMATS_VIDEO, APP_VERSION, CONFIG_PATH, DIAR_MODELS,
    DIAR_CHECK, load_config, save_config, ffprobe_duration, _clock, group_paragraphs,
    group_by_speaker, _tmp_for, write_txt, write_srt, write_docx, SR_FAST,
    _decode_to_array, _transcribe_array, transcribe_auto, transcribe_multilang_fast, _PIPELINES, _load_pipeline,
    _mk_hook, _extract_turns, diarize_and_label, free_asr_models, read_transcript, ollama_installed,
    _fmt_dur, _fmt_size, _short)

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
        est = self.app.estimate(j.duration) if j.kind == "audio" and j.state == "queue" else None
        meta = [x for x in ("готовая расшифровка" if j.kind == "text" else "", _fmt_dur(j.duration),
                            _fmt_size(j.size) if j.size else "", j.path.parent.name,
                            (f"≈ {max(10, round(est / 10) * 10)} с" if est < 90 else f"≈ {round(est / 60)} мин") + " на этом компьютере" if est else "") if x]
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
                if self.app.running:
                    b.configure(state="disabled")
                b.pack(side="left", padx=2)
            T.button(self.acts, "", "folder-open", kind="ghost", height=28,
                     command=lambda: pu.reveal(j.out)
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
    pu.open_path(path)


def _first_line(e, n=220):
    s = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    return s if len(s) <= n else s[:n] + "…"


class ModelsWindow(ctk.CTkToplevel):
    """«Загрузка моделей»: что нужно, сколько весит, скачивание с прогрессом,
    или офлайн-пакет из папки - без интернета."""

    def __init__(self, master, app, items):
        super().__init__(master)
        self.app, self.items = app, items
        self.title("Загрузка моделей")
        self.geometry("620x400")
        self.transient(master)
        wrap = ctk.CTkFrame(self, fg_color="transparent")
        wrap.pack(fill="both", expand=True, padx=22, pady=18)
        T.label(wrap, "Модели для распознавания", 18, bold=True, anchor="w").pack(fill="x")
        T.label(wrap, "Скачиваются один раз, дальше Scribe работает без интернета. Если интернета нет - "
                      "выберите папку офлайн-пакета моделей.", 13, color=T.MUTED, anchor="w", justify="left",
                wraplength=570).pack(fill="x", pady=(4, 12))
        self.rows = {}
        for it in items:
            r = ctk.CTkFrame(wrap, fg_color="transparent"); r.pack(fill="x", pady=2)
            T.label(r, it["title"], 13, anchor="w").pack(side="left")
            st = T.label(r, "", 12, color=T.MUTED, anchor="e"); st.pack(side="right")
            self.rows[it["repo"]] = st
        self.bar = ctk.CTkProgressBar(wrap, height=6); self.bar.set(0)
        self.bar.pack(fill="x", pady=(14, 4))
        self.note = T.label(wrap, "", 12, color=T.MUTED, anchor="w", justify="left", wraplength=570)
        self.note.pack(fill="x")
        btns = ctk.CTkFrame(wrap, fg_color="transparent"); btns.pack(side="bottom", fill="x", pady=(10, 0))
        self.dl_btn = T.button(btns, "Скачать", "upload", kind="primary", width=160, command=self._download)
        self.dl_btn.pack(side="left")
        T.button(btns, "Загрузить из папки…", "folder-open", width=200, command=self._from_folder).pack(side="left", padx=8)
        T.button(btns, "Позже", kind="ghost", width=90, command=self.destroy).pack(side="right")
        self._show()

    def _need(self):
        tok = self.app.cfg.get("hf_token")
        return [it for it in self.items if not it["local"] and (not it["optional"] or tok)]

    def _show(self):
        tok = self.app.cfg.get("hf_token")
        for it in self.items:
            if it["local"]:
                txt, col = "есть", T.OK
            elif it["optional"] and not tok:
                txt, col = "нужен токен HF или офлайн-пакет", T.WARN
            else:
                txt, col = f"≈ {_mb(it['mb'])}", T.MUTED
            self.rows[it["repo"]].configure(text=txt, text_color=col)
        need = self._need()
        mb = sum(it["mb"] for it in need)
        self.dl_btn.configure(text=f"Скачать ≈ {_mb(mb)}" if need else "Всё на месте",
                              state="normal" if need else "disabled")
        if not need:
            self.bar.set(1)

    def _download(self):
        import models
        self.dl = models.Downloader(self._need(), self.app.cfg.get("hf_token"))
        if not self.dl.items:
            return
        self.dl_btn.configure(state="disabled", text="Скачиваю…")
        self.dl.start()
        self.after(500, self._poll)

    def _poll(self):
        got, total = self.dl.progress()
        self.bar.set(min(1.0, got / total) if total else 0)
        cur = self.dl.current["title"] if self.dl.current else ""
        self.note.configure(text=f"{_mb(got / 1048576)} из {_mb(total / 1048576)} · {cur}", text_color=T.MUTED)
        if not self.dl.done:
            self.after(500, self._poll)
            return
        if self.dl.error:
            self.note.configure(text=f"Не удалось скачать: {_first_line(self.dl.error)}. Нет интернета? "
                                     "Можно выбрать офлайн-пакет моделей.", text_color=T.BAD)
        else:
            self.note.configure(text="Готово - модели на месте.", text_color=T.OK)
        self._show()

    def _from_folder(self):
        import models
        d = filedialog.askdirectory(title="Папка офлайн-пакета моделей", parent=self)
        if not d:
            return
        home, msg = models.use_folder(d)
        if not home:
            self.note.configure(text=msg, text_color=T.WARN)
            return
        self.app.cfg["models_dir"] = home
        save_config(self.app.cfg)
        for it in self.items:
            it["local"] = models.is_local(it["repo"])
        self.note.configure(text=msg, text_color=T.OK)
        self._show()
        self.app.after_models_changed()


def _mb(mb):
    return f"{mb / 1024:.1f} ГБ".replace(".", ",") if mb >= 1024 else f"{mb:.0f} МБ"


class App:
    def __init__(self, root, cfg):
        self.root = root
        self.cfg = cfg
        self.eng = None                 # движок распознавания - узнаём в фоне (импорт MLX/CTranslate2 небыстрый)
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
        threading.Thread(target=self._probe_engine, daemon=True).start()
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
        self.ver_lbl = T.label(bottom, f"Scribe {APP_VERSION}\nлокально", 11, color=T.FAINT,
                               anchor="w", justify="left", wraplength=180)
        self.ver_lbl.pack(fill="x", padx=10, pady=(8, 0))

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

    def _probe_engine(self):
        try:
            import engine
            eng = engine.get_engine()
            import models
            items = models.needed(self.cfg, eng)
            self.q.put(("engine", (eng, items)))
        except Exception as e:
            self.q.put(("engine", (None, str(e))))

    def _engine_ready(self, eng, items):
        self.eng = eng
        if eng is None:
            self.engine_lbl.configure(text=f"Распознавание недоступно: {_first_line(items)}", text_color=T.BAD)
            return
        self.engine_lbl.configure(text=f"Распознавание: {eng.label}", text_color=T.TEXT)
        short = {"mlx": "на Apple MLX", "cuda": "на видеокарте NVIDIA", "cpu": "на процессоре"}
        self.ver_lbl.configure(text=f"Scribe {APP_VERSION}\nлокально {short.get(getattr(eng, 'device', ''), '')}".rstrip())
        cpu = getattr(eng, "device", "") == "cpu"
        if cpu:
            self.cpu_box.pack(fill="x", pady=(14, 0), after=self.model_hint)
        if pu.IS_WIN and cpu and pu.has_nvidia():
            self.nv_box.pack(fill="x", pady=(10, 0), after=self.cpu_box)
        self._sync_opts()
        self._dict_auto()
        for j in self.jobs:
            j.row.refresh()
        if any(not it["local"] and not it["optional"] for it in items):
            self.open_models(items)

    def open_models(self, items=None):
        if self.eng is None:
            return
        import models
        items = items or models.needed(self.cfg, self.eng)
        if getattr(self, "_models_win", None) is not None and self._models_win.winfo_exists():
            self._models_win.focus(); return
        self._models_win = ModelsWindow(self.root, self, items)

    def after_models_changed(self):
        self._sync_opts()

    def _pyannote_local(self):
        if not hasattr(self, "_pyl"):
            import models
            self._pyl = models.is_local(models.PYANNOTE)
        return self._pyl

    def _diar_ready(self):
        return bool(self.cfg.get("hf_token")) or self._pyannote_local()

    # --- оценка времени по замерам на этой машине ---
    def _speed_key(self, mode, diar):
        eng = self.eng
        dev = f"{eng.name}:{getattr(eng, 'device', '')}" if eng else "?"
        return f"{dev}|{mode}|{'diar' if diar else '-'}"

    def _remember_speed(self, mode, diar, seconds, duration):
        if duration < 20:
            return
        sp = self.cfg.setdefault("speed", {})
        k = self._speed_key(mode, diar)
        r = seconds / duration
        sp[k] = round(r if k not in sp else 0.6 * sp[k] + 0.4 * r, 4)
        save_config(self.cfg)

    def estimate(self, duration):
        """Сколько займёт файл - по прошлым расшифровкам на ЭТОЙ машине (а не по цифрам с Mac)."""
        if not duration or self.eng is None:
            return None
        mode = "mixed" if self.lang_seg.get() == "Несколько" else "single"
        r = self.cfg.get("speed", {}).get(self._speed_key(mode, self.diar_var.get()))
        return r * duration if r else None

    def _prebuild_dict(self):
        """Страницу «Диктофон» строим заранее (без запуска распознавания): тяжёлое
        построение в момент щелчка по меню задерживало перерисовку."""
        if self.dict_page is None:
            from dictaphone import DictaphonePage
            model = self.cfg.get("dict_model") or self.cfg.get("model_repo", MODEL_REPO)
            self.dict_page = DictaphonePage(self.content, model, self.cfg)
            self.dict_page.on_model_switch = self._dict_switched
            self.dict_page.small_ok = self._small_ok
            self._dict_auto()
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
        T.label(head, f"Аудио и видео → текст. Всё распознаётся {pu.HERE}; результат сохраняется "
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
        self.token_link = T.label(o, "Нужен токен Hugging Face или офлайн-пакет моделей - Настройки", 12,
                                  color=T.ACCENT,
                                  anchor="w", justify="left", wraplength=250, cursor=pu.HAND)
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
        self.sum_hint.bind("<Button-1>", lambda e: None if ollama_installed() else self.explain_ollama())
        self._sync_opts()

    def _lang_hint(self):
        v = self.lang_seg.get()
        self.lang_note.configure(text=LANG_HINT.get(
            v, f"Вся запись распознаётся как «{v}». Если в ней окажется и другой язык - предупредим."))

    def _sync_opts(self):
        """Согласовать опции между собой: без токена говорящих не включить,
        для субтитров таймкоды и так есть, без Ollama нет итогов."""
        has_token = self._diar_ready()     # токен Hugging Face или модель из офлайн-пакета
        if not has_token:
            self.diar_var.set(False)
        self.diar_cb.configure(state="normal" if has_token else "disabled")
        for w in (self.nspk_row, self.token_link, self.diar_hint):
            w.pack_forget()
        cpu = self.eng is not None and getattr(self.eng, "device", "") == "cpu"
        self.diar_hint.configure(
            text="Метки «Спикер 1, 2…». Заметно дольше." + (
                " На процессоре - намного дольше: часовая запись может идти десятки минут." if cpu else ""),
            text_color=T.WARN if cpu else T.MUTED)
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
            self.sum_hint.configure(text="Нужна Ollama - бесплатная программа для локальной модели. "
                                         "Как поставить - нажмите здесь.", text_color=T.WARN)

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
                elif kind == "engine":
                    self._engine_ready(*msg[1])
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
            diar=self.diar_var.get() and self._diar_ready(),
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

    def explain_ollama(self):
        """Итоги делает Ollama - отдельная бесплатная программа. Если её нет - объясняем, как поставить."""
        import webbrowser
        if messagebox.askyesno(
                "Итоги: нужна Ollama",
                "Итоги готовит локальная языковая модель через бесплатную программу Ollama - "
                "текст никуда не уходит. Остальное в Scribe работает и без неё.\n\n"
                "Как поставить:\n1. Скачайте Ollama с ollama.com и установите.\n"
                "2. В командной строке: ollama pull qwen3.5:4b (или qwen3.5:9b, если памяти от 16 ГБ).\n"
                "3. Перезапустите Scribe.\n\nОткрыть ollama.com?"):
            webbrowser.open("https://ollama.com/download")

    def summarize_job(self, job):
        """«Подвести итоги» у готового файла: итоги из готовой расшифровки, без повторного распознавания."""
        if self.running:
            return
        if not ollama_installed():
            self.explain_ollama()
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
            import engine
            eng = engine.get_engine()
        except Exception as e:
            self.q.put(("done", ("error", "Не найден движок распознавания:\n  pip install faster-whisper\n"
                                          "(на Mac: pip install mlx-whisper)\n\n" + str(e)))); return
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
                done.append(self._run_one(job, o, eng, stage, cancel))
            except Cancelled:
                cancelled = True
                self._job(job, state="cancel")
            except FileNotFoundError as e:
                if "ffmpeg" in str(e) or "ffprobe" in str(e):
                    self._job(job, state="error", error="Не найден ffmpeg")
                    self.q.put(("done", ("error", pu.FFMPEG_HINT))); return
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

    def _run_one(self, job, o, eng, stage, cancel):
        t_start = time.monotonic()
        job_mixed = False
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
            segments = transcribe_multilang_fast(audio, o["model"], eng, self._prog, lambda m: None, cancel,
                                                 stage=stage)
        else:
            stage("prep", "проверка языков")

            def on_mixed(probe):
                nonlocal job_mixed
                job_mixed = True
                stage.keys.insert(stage.keys.index("asr"), "lang")    # «Авто» стал «Несколько»: +1 шаг
                self._job(job, info="Найдены русский и английский - обработано как смешанная запись.")

            segments, info = transcribe_auto(audio, o["model"], eng, self._prog, cancel, stage=stage,
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
        self._remember_speed("mixed" if (o["multilang"] or job_mixed) else "single", diar_used,
                             time.monotonic() - t_start, duration)
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
            raise RuntimeError(f"в Ollama нет моделей. {pu.SHELL.capitalize()}: ollama pull " + summary.DEFAULT_MODEL)
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
        self.engine_lbl = T.label(c, "Распознавание: определяю…", 13, bold=True, color=T.MUTED, anchor="w")
        self.engine_lbl.pack(fill="x", pady=(0, 10))
        self.cpu_box = ctk.CTkFrame(c, fg_color="transparent")      # показывается только на процессоре
        T.label(self.cpu_box, "Видеокарты нет - Scribe работает на процессоре: та же модель, но медленнее. "
                              "Оценка времени в карточке файла - по прошлым расшифровкам на этом компьютере.",
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(fill="x")
        dr = ctk.CTkFrame(self.cpu_box, fg_color="transparent"); dr.pack(fill="x", pady=(8, 0))
        T.label(dr, "Модель для диктофона", 13, color=T.MUTED).pack(side="left")
        dm = {"turbo": "Turbo (как для файлов)", "medium": "Medium - быстрее", "small": "Small - самая быстрая"}
        self._dict_models = dm
        import engine as _E
        cur = _E.model_key(self.cfg.get("dict_model") or "turbo")
        self.dict_model_var = ctk.StringVar(value=dm.get(cur, dm["turbo"]))
        ctk.CTkOptionMenu(dr, variable=self.dict_model_var, values=list(dm.values()), width=240, height=30,
                          font=T.font(13), dropdown_font=T.font(13), command=self._save_dict_model
                          ).pack(side="left", padx=(10, 0))
        T.label(self.cpu_box, "Если текст в диктофоне отстаёт от речи больше чем на 5 с, диктофон сам "
                              "перейдёт на Small (быстрее, чуть менее точно). Выбор здесь отключает это; "
                              "вступит в силу после перезапуска Scribe.", 12, color=T.MUTED,
                anchor="w", justify="left", wraplength=640).pack(fill="x", pady=(4, 0))
        self.nv_box = ctk.CTkFrame(c, fg_color="transparent")       # Windows + NVIDIA без библиотек CUDA
        nr = ctk.CTkFrame(self.nv_box, fg_color="transparent"); nr.pack(fill="x")
        self.nv_btn = T.button(nr, "Ускорение на видеокарте NVIDIA - скачать", "upload", height=30, size=13,
                               width=330, command=self._nvidia_download)
        self.nv_btn.pack(side="left")
        self.nv_note = T.label(self.nv_box, "Найдена видеокарта NVIDIA. Библиотеки CUDA (≈ 1,2 ГБ) скачиваются "
                                            "отдельно, чтобы не раздувать установщик.", 12, color=T.MUTED,
                               anchor="w", justify="left", wraplength=640)
        self.nv_note.pack(fill="x", pady=(4, 0))
        self.model_var = ctk.StringVar(value=_model_title(self.cfg.get("model_repo", MODEL_REPO)))
        mrow = ctk.CTkFrame(c, fg_color="transparent"); mrow.pack(fill="x")
        T.label(mrow, "Модель", 13, color=T.MUTED).pack(side="left")
        ctk.CTkOptionMenu(mrow, variable=self.model_var, values=list(MODELS), width=380, height=32,
                          font=T.font(13), dropdown_font=T.font(13), command=self._save_model
                          ).pack(side="left", padx=(10, 0))
        self.model_hint = T.label(c, "Turbo - быстрая и рекомендуемая. Large v3 в разы медленнее при том же "
                                     "результате.", 12, color=T.MUTED, anchor="w", justify="left", wraplength=640)
        self.model_hint.pack(fill="x", pady=(4, 0))
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
                      f"«Подвести итоги» у готового файла. Модель работает {pu.HERE} через Ollama - "
                      "текст никуда не уходит.")
        self._hint(c, "Память: на время итогов Scribe выгружает модели распознавания, а Ollama освобождает "
                      "память сразу после итогов. При 16 ГБ памяти и других тяжёлых программах лучше qwen3.5:4b.")

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
        mb = ctk.CTkFrame(c, fg_color="transparent"); mb.pack(fill="x", pady=(10, 0))
        T.button(mb, "Скачать модели…", "upload", height=30, size=13, width=170,
                 command=lambda: self.open_models()).pack(side="left")
        T.button(mb, "Загрузить модели из папки…", "folder-open", height=30, size=13, width=250,
                 command=self._models_from_folder).pack(side="left", padx=8)
        self._hint(c, "Без интернета: распакуйте офлайн-пакет моделей Scribe и выберите его папку - модели "
                      "возьмутся оттуда, без обращения к интернету.")

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
        lb = T.label(parent, text, 12, color=T.ACCENT, anchor="w", cursor=pu.HAND)
        lb.pack(fill="x", pady=(2, 0))
        lb.bind("<Button-1>", lambda e: webbrowser.open(url))

    def _save_dict_model(self, v):
        key = {t: k for k, t in self._dict_models.items()}[v]
        self.cfg["dict_model"] = key                 # выбран вручную - автоматика больше не трогает
        save_config(self.cfg)

    def _dict_auto(self):
        """На процессоре, пока модель диктофона не выбрана вручную, диктофон сам перейдёт на Small,
        если текст отстаёт больше 5 с."""
        page = getattr(self, "dict_page", None)
        if page is not None and self.eng is not None:
            page.auto_small = getattr(self.eng, "device", "") == "cpu" and not self.cfg.get("dict_model")

    def _small_ok(self):
        import models
        return self.eng is not None and (self.eng.is_local("small") or models.online())

    def _dict_switched(self, key):
        self.cfg["dict_model"] = key
        save_config(self.cfg)
        if hasattr(self, "dict_model_var"):
            self.dict_model_var.set(self._dict_models[key])

    def _models_from_folder(self):
        import models
        d = filedialog.askdirectory(title="Папка офлайн-пакета моделей")
        if not d:
            return
        home, msg = models.use_folder(d)
        if home:
            self.cfg["models_dir"] = home
            save_config(self.cfg)
            self.models_dir_lbl.configure(text=_models_dir_display(self.cfg))
            if hasattr(self, "_pyl"):
                del self._pyl
            self._sync_opts()
        messagebox.showinfo("Модели", msg)

    def _nvidia_download(self):
        import models
        self.nv_btn.configure(state="disabled", text="Скачиваю…")

        def run():
            try:
                models.nvidia_download(lambda g, t: self.q.put(("status", f"Ускорение NVIDIA: {g / t:.0%}")))
                self.q.put(("status", "Ускорение NVIDIA скачано - перезапустите Scribe."))
            except Exception as e:
                self.q.put(("status", f"Не удалось скачать ускорение NVIDIA: {_first_line(e)}"))
        threading.Thread(target=run, daemon=True).start()

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
        pu.open_folder(p)

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
    pu.setup_cuda()                              # библиотеки CUDA, если скачаны кнопкой «Ускорение NVIDIA»
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
