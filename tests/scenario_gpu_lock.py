#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Окно Scribe и очередь видеокарты (общий модуль common/lock, «_Общие модули 01» + Scribe 06 Г).

Своя временная папка: HOME (настройки, журналы), замок HEAVY_LOCK, аудио; движок распознавания подменён -
видеокарта не нужна; «другую программу» изображает второй процесс (common/lock/tests/_holder.py).
Окно - своё окно Scribe (не окна Андрея); снимки - только если доступен pyobjc (SCRIBE_SNAP_SITE - папка
site-packages с pyobjc, например из соседнего проекта).

    python3 tests/scenario_gpu_lock.py [папка снимков]

Сценарии (номера - README модуля, 8.1 и 9):
  S1  один добавленный файл выделяется сам: «Не начат», кнопка «Расшифровать: <файл>» (Г1, Г2)
  S2  два файла без выделения - подсказка «Щёлкните файл в списке…» (Г3)
  S3  видеокарта занята человеком: окно «Видеокарта занята: «Interview Coach», с …, репетиция» с тремя кнопками
  S4  «Подождать очереди»: «Ждёт видеокарту», строка «Жду видеокарту… · вы следующий», «Отменить ожидание»,
      шапка «занята … - в очереди»; владелец отдал - Scribe начинает сам; после - замок свободен
  S5  «Отменить ожидание»: файл снова «Не начат», в очереди пусто, замок у прежнего владельца
  S6  «Запустить сейчас»: работает поверх (now-…), шапка «работает вместе с …», после - пометка убрана
  S7  сквозной: фон держит замок → «Подождать очереди» → фон отдаёт на контрольной точке → Scribe сам → фон снова
  S8  диктофон: окно с четвёртым вариантом «На процессоре»; «Отмена» - запись не началась, замок не взят
  S9  прерванная расшифровка (06 Б): у файла и на главной кнопке «Продолжить (готово 66 %)»; другой язык - сначала
"""
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHOTS = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None

TMP = Path(tempfile.mkdtemp(prefix="scribe-gpu-lock-"))
REAL_HF = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
os.environ.update(HOME=str(TMP / "home"), HEAVY_LOCK=str(TMP / ".heavy-lock"), HF_HOME=REAL_HF,
                  HF_HUB_OFFLINE="1", SCRIBE_RESUME_DIR=str(TMP / "resume"))
os.environ.pop("HEAVY_LOCK_HELD", None)
os.environ.pop("HEAVY_LOCK_WHO", None)
(TMP / "home").mkdir()
LOCK = Path(os.environ["HEAVY_LOCK"])
sys.path.insert(0, str(ROOT))
if os.environ.get("SCRIBE_SNAP_SITE"):
    sys.path.append(os.environ["SCRIBE_SNAP_SITE"])         # только pyobjc для снимков - в конце пути

import whisper_gui as G  # noqa: E402
import engine  # noqa: E402
import models  # noqa: E402
from common.lock import heavy_lock as hl  # noqa: E402

try:
    import AppKit  # noqa: E402
    import Quartz  # noqa: E402
except Exception:
    AppKit = Quartz = None


class FakeEngine:
    """Движок без модели: «распознаёт» мгновенно - видеокарта не нужна."""
    name, label, device = "mlx", "Apple MLX (видеоядро)", "mlx"

    def is_local(self, model):
        return True

    def __getattr__(self, name):
        return lambda *a, **k: None


engine.get_engine = lambda: FakeEngine()
models.needed = lambda cfg, eng: []
G.App._probe_engine = lambda self: self.q.put(("engine", (FakeEngine(), [])))


def fake_run_one(self, job, o, eng, stage, cancel):
    """Вместо распознавания - три шага по 0,3 с (видно «идёт»), результат - .txt во временной папке."""
    for i, key in enumerate(stage.keys):
        stage(key)
        self._prog((i + 1) / len(stage.keys))
        time.sleep(0.3)
        if cancel():
            raise G.Cancelled()
    out = o["out"]
    out.write_text("проверка очереди видеокарты\n", encoding="utf-8")
    self._job(job, state="done", out=out)
    return out, None, ""


G.App._run_one = fake_run_one
import dictaphone  # noqa: E402
dictaphone.DictaphonePage.activate = lambda self: None       # процесс распознавания диктофона не поднимаем
G._open = lambda path: None                                 # не открывать результат в чужой программе


def wav(name, sec=3.0):
    p = TMP / name
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * int(16000 * sec))
    return p


HOLDER = ROOT / "common" / "lock" / "tests" / "_holder.py"
procs = []


def holder(program, what, who="human", mode="hold"):
    p = subprocess.Popen([sys.executable, str(HOLDER), str(LOCK), program, who, mode, what],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    procs.append(p)
    assert p.stdout.readline().strip() == "taken"
    return p


def release(p):
    p.stdin.write("release\n")
    p.stdin.flush()


# ------------------------------------------------------------------ окно
G.T.apply()
G.ctk.set_appearance_mode("light")
root = G.RootTk()
app = G.App(root, G.load_config())
root.geometry(f"{int(1180 * G.T.TEXT_SCALE)}x{min(int(760 * G.T.TEXT_SCALE), root.winfo_screenheight() - 90)}+60+40")
results = []


def pump(sec=0.3):
    end = time.time() + sec
    while time.time() < end:
        root.update()
        time.sleep(0.02)


def until(cond, timeout=15.0):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if cond():
            return True
        time.sleep(0.03)
    return False


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("✓ " if ok else "✗ ") + name + (f" - {detail}" if detail and not ok else ""), flush=True)


def window_id(title):
    if Quartz is None:
        return None
    best = None
    for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID):
        if w.get("kCGWindowOwnerPID") == os.getpid() and w.get("kCGWindowName") == title \
                and w.get("kCGWindowBounds", {}).get("Height", 0) > 100:
            best = w["kCGWindowNumber"]
    return best


def front():
    if AppKit is None:
        return
    for w in AppKit.NSApp.windows():
        w.setCollectionBehavior_(1 | 256)        # видно поверх полноэкранного VS Code
        w.orderFrontRegardless()


def shot(name, title="Scribe"):
    if SHOTS is None or Quartz is None:
        return
    SHOTS.mkdir(parents=True, exist_ok=True)
    pump(0.3)
    front()
    pump(0.6)
    n = window_id(title)
    if n is None:
        print("  снимок не сделан: нет окна", title)
        return
    p = SHOTS / f"{name}.png"
    subprocess.run(["screencapture", "-x", "-o", f"-l{n}", str(p)], capture_output=True)
    print("  снимок:", p if p.exists() else "не получился")


def fits(btn):
    """Текст кнопки помещается (стандарт 4.4: ничего не обрезано)."""
    root.update_idletasks()
    lbl = getattr(btn, "_text_label", None)
    return lbl is None or lbl.winfo_reqwidth() + 2 * 8 <= btn.winfo_width()


def dialog():
    for w in root.winfo_children():
        if isinstance(w, G.ctk.CTkToplevel) and hasattr(w, "buttons") and w.winfo_exists():
            return w
    return None


def labels(widget):
    out = []
    for w in widget.winfo_children():
        try:
            t = w.cget("text")
            if t:
                out.append(t)
        except Exception:
            pass
        out += labels(w)
    return out


def answer(key, snap=None, then=None):
    """Нажать кнопку окна «Видеокарта занята», когда оно откроется (окно модальное - ставим заранее)."""
    def go():
        d = dialog()
        if d is None:
            root.after(100, go)
            return
        state["dialog_text"] = " | ".join(labels(d))
        state["dialog_buttons"] = sorted(d.buttons)
        if snap:
            shot(snap, "Видеокарта занята")
        if then:
            then()
        d.buttons[key].invoke()
    root.after(300, go)


state = {}
try:
    until(lambda: app.eng is not None)
    pump(0.5)

    # S1 - один файл выделен сам, «Не начат»
    a = wav("лекция.wav")
    app._add_paths([str(a)])
    pump(0.5)
    j1 = app.jobs[0]
    check("S1 файл выделен сам", app.sel == [j1])
    check("S1 кнопка «Расшифровать: лекция.wav»", app.run_btn.cget("text") == "Расшифровать: лекция.wav",
          app.run_btn.cget("text"))
    check("S1 статус «Не начат»", j1.row.chip.cget("text") == "Не начат", j1.row.chip.cget("text"))
    check("S1 шапка «готова»", app.gpu_chip.cget("text") == "Модель: готова", app.gpu_chip.cget("text"))

    # S2 - два файла, ни один не выделен - подсказка
    app.clear_selection()
    app._add_paths([str(wav("встреча.wav")), str(wav("интервью.wav"))])
    pump(0.5)
    check("S2 два новых файла не выделены", app.sel == [])
    check("S2 подсказка", app.status.cget("text") == "Щёлкните файл в списке или нажмите «Все файлы».",
          app.status.cget("text"))
    check("S2 кнопка серая", app.run_btn.cget("state") == "disabled")
    app.select(j1)
    check("S2 выделили - подсказка ушла", app.status.cget("text") == app.HINT_START, app.status.cget("text"))
    pump(0.3)
    shot("01-files-not-started")

    # S3 + S4 - занято человеком, «Подождать очереди»
    ic = holder("Interview Coach", "репетиция")
    pump(2.2)
    check("S3 шапка «занята «Interview Coach»»", app.gpu_chip.cget("text") == "Модель: занята «Interview Coach»",
          app.gpu_chip.cget("text"))
    shot("02-header-busy")
    answer("wait", snap="03-dialog-busy")
    app.start()
    txt = state.get("dialog_text", "")
    h = hl.status(LOCK)
    check("S3 текст окна", f"Видеокарта занята: «Interview Coach», с {h.since_text}, репетиция." in txt, txt)
    check("S3 кнопки окна", state.get("dialog_buttons") == ["cancel", "now", "wait"], str(state.get("dialog_buttons")))
    check("S3 предупреждение у «Запустить сейчас»", "Обе программы будут медленнее" in txt)
    until(lambda: app.gpu_waiting)
    pump(1.5)
    check("S4 файл «Ждёт видеокарту»", j1.row.chip.cget("text") == "Ждёт видеокарту", j1.row.chip.cget("text"))
    st = app.status.cget("text")
    check("S4 строка ожидания", st.startswith("Жду видеокарту: «Interview Coach», с") and st.endswith("вы следующий"), st)
    check("S4 «Отменить ожидание»", app.all_btn.cget("text") == "Отменить ожидание", app.all_btn.cget("text"))
    check("S4 текст кнопок не обрезан", fits(app.all_btn) and fits(app.run_btn),
          f"{app.all_btn._text_label.winfo_reqwidth()}/{app.all_btn.winfo_width()} "
          f"{app.run_btn._text_label.winfo_reqwidth()}/{app.run_btn.winfo_width()}")
    pump(2.2)
    check("S4 шапка «в очереди»", app.gpu_chip.cget("text") == "Модель: занята «Interview Coach» - в очереди",
          app.gpu_chip.cget("text"))
    check("S4 Scribe в очереди", [w.program for w in hl.queue(LOCK)] == ["Scribe"])
    shot("04-waiting-in-queue")
    release(ic)
    check("S4 Scribe начал сам", until(lambda: j1.state in ("run", "done")))
    until(lambda: not app.running)
    check("S4 файл готов", j1.state == "done", j1.state)
    check("S4 замок свободен после", hl.status(LOCK) is None and hl.queue(LOCK) == [])
    jr = (hl.queue_dir(LOCK) / "journal.log").read_text(encoding="utf-8")
    check("S4 журнал: Scribe ждал, взял, отдал",
          all(f"\t{e}\tScribe\t" in jr for e in ("wait", "take", "release")))
    pump(0.3)
    shot("05-done")

    # S5 - «Отменить ожидание»
    j2 = app.jobs[1]
    app.select(j2)
    hd = holder("Herald", "озвучка книги")
    answer("wait")
    app.start()
    until(lambda: app.gpu_waiting)
    pump(0.5)
    app._all_or_stop()                                     # «Отменить ожидание»
    until(lambda: not app.running)
    check("S5 файл снова «Не начат»", j2.state == "queue" and j2.row.chip.cget("text") == "Не начат",
          j2.row.chip.cget("text"))
    check("S5 очередь пуста, замок у Herald", hl.queue(LOCK) == [] and hl.status(LOCK).program == "Herald")
    check("S5 строка", app.status.cget("text").startswith("Ожидание видеокарты отменено"), app.status.cget("text"))

    # S6 - «Запустить сейчас»
    seen = {}

    def watch():
        if app.running:
            seen["al"] = [w.program for w in hl.alongside(LOCK)]
            app._gpu_tick()
            seen["chip"] = app.gpu_chip.cget("text")
        else:
            root.after(50, watch)
    answer("now")
    root.after(400, watch)
    app.start()
    until(lambda: "chip" in seen)
    until(lambda: not app.running)
    check("S6 работал поверх (now-…)", seen.get("al") == ["Scribe"], str(seen.get("al")))
    check("S6 шапка «работает вместе с «Herald»»", seen.get("chip") == "Модель: работает вместе с «Herald»",
          str(seen.get("chip")))
    check("S6 файл готов", j2.state == "done")
    check("S6 пометка убрана, замок у Herald", hl.alongside(LOCK) == [] and hl.status(LOCK).program == "Herald")
    release(hd)
    until(lambda: hl.status(LOCK) is None)

    # S7 - сквозной: фон держит, человек ждёт, фон уступает на контрольной точке
    bg = holder("Herald (ночной прогон)", "дообучение", who="background", mode="bg-checkpoint")
    j3 = app.jobs[2]
    app.select(j3)
    answer("wait", snap="06-dialog-background")
    t0 = time.time()
    app.start()
    txt = state.get("dialog_text", "")
    check("S7 окно говорит «фоновый прогон»", "фоновый прогон" in txt, txt)
    check("S7 фон уступил, Scribe начал сам", until(lambda: j3.state in ("run", "done"), 10))
    took = time.time() - t0
    until(lambda: not app.running)
    check("S7 файл готов", j3.state == "done")
    check("S7 фон снова держит замок", until(lambda: (hl.status(LOCK) or hl.Holder("-")).program == "Herald (ночной прогон)", 10))
    check("S7 журнал: yield", "\tyield\tHerald (ночной прогон)" in (hl.queue_dir(LOCK) / "journal.log").read_text(encoding="utf-8"))
    print(f"  S7: от «Подождать» до готового файла {took:.1f} с")

    # S8 - диктофон: четвёртый вариант «На процессоре» (микрофон не открываем: начало записи - заглушка)
    release(bg)
    until(lambda: hl.status(LOCK) is None)
    hd2 = holder("Interview Coach", "репетиция")
    dictaphone.DictaphonePage._begin = lambda self: state.setdefault("begin", True)
    app._cpu_dict = True
    app.show("dict")
    pump(0.8)
    dp = app.dict_page
    answer("cancel", snap="07-dialog-dictaphone-cpu")
    dp._toggle_start()
    pump(0.3)
    check("S8 в окне есть «На процессоре»", state.get("dialog_buttons") == ["cancel", "cpu", "now", "wait"],
          str(state.get("dialog_buttons")))
    check("S8 «Отмена» - запись не началась", dp._mode == "idle" and not dp.hlock.held)
    answer("wait")
    dp._toggle_start()
    pump(0.5)
    check("S8 «Подождать очереди»: «Отменить ожидание» без обрезки",
          dp.start_btn.cget("text") == "Отменить ожидание" and fits(dp.start_btn), dp.start_btn.cget("text"))
    check("S8 диктофон в очереди", [w.program for w in hl.queue(LOCK)] == ["Scribe"],
          f"{[(w.program, w.what) for w in hl.queue(LOCK)]} держит {hl.status(LOCK)}")
    dp._toggle_start()                                     # «Отменить ожидание»
    pump(1.5)
    check("S8 отменили - «Начать запись», очередь пуста",
          dp.start_btn.cget("text") == "Начать запись" and hl.queue(LOCK) == [] and dp._wait_cancel is None)
    answer("wait")
    dp._toggle_start()
    pump(0.3)
    release(hd2)
    check("S8 освободилось - запись начинается сама", until(lambda: state.get("begin") and dp.hlock.held, 10))
    dp.hlock.release()

    # S9 - прерванная расшифровка (06 Б): готовые куски на диске - «Продолжить (готово N %)» у файла и на кнопке
    app.show("files")
    pump(0.5)
    app.clear_selection()
    sem = wav("семинар.wav", 2.0)
    app._add_paths([str(sem)])
    pump(0.5)
    j9 = next(x for x in app.jobs if x.path == sem)
    st = G.resume_store(sem, app.cfg.get("model_repo", G.MODEL_REPO), G.LANG_CODE.get(j9.opts["lang"]),
                        app.cfg.get("normalize", True))
    st.open([(0.0, 30.0), (31.0, 60.0), (61.0, 90.0)])
    st.add(0, [{"start": 0.0, "end": 5.0, "text": " готово"}])
    st.add(1, [{"start": 31.0, "end": 35.0, "text": " тоже"}])
    app._probe([j9])
    pump(0.5)
    check("S9 у файла - «Продолжить (готово 66 %)»", "Продолжить (готово 66 %)" in labels(j9.row.acts),
          str(labels(j9.row.acts)))
    check("S9 главная кнопка «Продолжить: семинар.wav (готово 66 %)» без обрезки",
          app.run_btn.cget("text") == "Продолжить: семинар.wav (готово 66 %)" and fits(app.run_btn),
          app.run_btn.cget("text"))
    shot("08-resume")
    other = "English" if j9.opts["lang"] != "English" else "Русский"
    app.lang_seg.set(other)
    app._opt_changed("lang")
    pump(0.3)
    check("S9 другой язык - готовые куски не те: «Расшифровать: семинар.wav»",
          app.run_btn.cget("text") == "Расшифровать: семинар.wav" and not j9.resume, app.run_btn.cget("text"))
finally:
    for p in procs:
        if p.poll() is None:
            p.kill()
        p.wait()
        for f in (p.stdin, p.stdout):
            f.close()
    try:
        app.cancel_evt.set()
        if app.dict_page is not None:
            app.dict_page.shutdown()
        root.destroy()
    except Exception:
        pass
    shutil.rmtree(TMP, ignore_errors=True)

bad = [r for r in results if not r[1]]
print(f"\nИтог: {len(results) - len(bad)} из {len(results)}")
sys.stdout.flush()
os._exit(1 if bad else 0)
