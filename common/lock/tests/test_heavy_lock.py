# -*- coding: utf-8 -*-
"""Проверки модуля очереди видеокарты (README, раздел 9: L1-L9). Только временная папка (стандарт 6.10),
без видеокарты и моделей.

    python3 -m unittest discover -s "_общие модули/lock/tests" -v
"""
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from lock import heavy_lock as hl  # noqa: E402
from lock.lock_ui import texts as tx  # noqa: E402

SH = HERE.parent / "heavy_lock.sh"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def wait_for(cond, timeout=10.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return False


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="heavylock-test-"))
        self.lock = self.tmp / ".heavy-lock"
        self.procs = []
        self._env = {k: os.environ.pop(k) for k in ("HEAVY_LOCK_HELD", "HEAVY_LOCK_WHO") if k in os.environ}
        os.environ["HEAVY_LOCK"] = str(self.lock)          # даже по ошибке - не настоящая очередь

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait()
            for f in (p.stdin, p.stdout):
                if f is not None:
                    f.close()
        os.environ.pop("HEAVY_LOCK", None)
        os.environ.update(self._env)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def mk(self, program="Тест", who=hl.HUMAN, **kw):
        return hl.HeavyLock(program, kw.pop("what", "проверка"), who=who, path=self.lock, **kw)

    def holder_proc(self, program, who=hl.HUMAN, mode="hold", what="держит"):
        p = subprocess.Popen([sys.executable, str(HERE / "_holder.py"), str(self.lock), program, who, mode, what],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        self.procs.append(p)
        return p

    @staticmethod
    def expect(p, word, timeout=10.0):
        out = []

        def rd():
            for line in p.stdout:
                out.append(line.strip())
                if line.strip() == word:
                    return
        th = threading.Thread(target=rd, daemon=True)
        th.start()
        th.join(timeout)
        if word not in out:
            raise AssertionError(f"не дождался «{word}», было: {out}")

    @staticmethod
    def send(p, line="release"):
        p.stdin.write(line + "\n")
        p.stdin.flush()

    def journal(self):
        f = hl.queue_dir(self.lock) / "journal.log"
        return [ln.split("\t") for ln in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


class PageLink(Base):
    """Ссылка «Открыть страницу «Видеокарта»» (README, 12): только там, где лежат «_общие модули»."""

    def test_no_page_in_temp(self):
        self.assertIsNone(hl.page_tools(self.lock))
        self.assertFalse(hl.status_json(self.lock)["page"])
        self.assertFalse(hl.open_page(self.lock))

    def test_page_opens_without_waiting(self):
        tools = self.tmp / "_общие модули" / "lock" / "tools"
        tools.mkdir(parents=True)
        (tools / "gpu_page.py").write_text("", encoding="utf-8")
        self.assertEqual(hl.page_tools(self.lock), tools)
        self.assertTrue(hl.status_json(self.lock)["page"])
        calls = []
        real = hl.subprocess.Popen
        hl.subprocess.Popen = lambda cmd, **kw: calls.append(cmd)
        try:
            self.assertTrue(hl.open_page(self.lock))
        finally:
            hl.subprocess.Popen = real
        self.assertEqual(calls[0][-1], str(tools / "gpu_page.py"))     # нет .app - сам сервер страницы


class TwoProcesses(Base):
    """L1: второй ждёт и получает замок после первого."""

    def test_second_waits(self):
        a = self.holder_proc("Interview Coach", what="репетиция")
        self.expect(a, "taken")
        b = self.mk("Scribe")
        self.assertFalse(b.try_acquire())
        h = hl.status(self.lock)
        self.assertEqual((h.program, h.pid, h.what, h.who), ("Interview Coach", a.pid, "репетиция", hl.HUMAN))
        got = {}
        th = threading.Thread(target=lambda: got.setdefault("ok", b.acquire(poll=0.05)))
        th.start()
        self.assertTrue(wait_for(lambda: len(hl.queue(self.lock)) == 1))
        self.assertEqual(hl.queue(self.lock)[0].program, "Scribe")
        self.send(a)
        th.join(10)
        self.assertTrue(got.get("ok"))
        self.assertEqual(hl.status(self.lock).program, "Scribe")
        self.assertEqual(hl.queue(self.lock), [])
        b.release()
        self.assertFalse(self.lock.exists())
        events = [r[1] for r in self.journal()]
        for e in ("take", "wait", "release"):
            self.assertIn(e, events)
        held = [r for r in self.journal() if r[1] == "release" and r[2] == "Scribe"]
        self.assertTrue(held and held[0][6] != "")            # сколько держал

    def test_same_process_two_locks(self):
        a, b = self.mk("Scribe", what="файлы"), self.mk("Scribe", what="диктофон")
        self.assertTrue(a.try_acquire())
        self.assertFalse(b.try_acquire())                     # второй объект - не владелец
        b.release()
        self.assertTrue(self.lock.exists())
        a.release()
        self.assertFalse(self.lock.exists())


class StaleOwner(Base):
    """L2: мёртвый владелец снимается сразу; живой - не снимается и через 4 ч."""

    def test_dead_owner(self):
        a = self.holder_proc("Herald")
        self.expect(a, "taken")
        a.send_signal(signal.SIGKILL)
        a.wait()
        b = self.mk("Scribe")
        self.assertIsNone(hl.status(self.lock))
        self.assertTrue(b.try_acquire())
        self.assertIn("stale", [r[1] for r in self.journal()])
        b.release()

    def test_alive_old_owner_kept(self):
        a = self.holder_proc("Herald", mode="hold", what="книга")
        self.expect(a, "taken")
        old = time.time() - 4 * 3600
        os.utime(self.lock, (old, old))
        b = self.mk("Scribe")
        self.assertFalse(b.try_acquire())
        self.assertEqual(hl.status(self.lock).program, "Herald")

    def test_legacy_dead_pid(self):
        self.lock.mkdir()
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        (self.lock / "owner").write_text(f"Понятный документ Fri Oct 23 19:05:25 MSK 2026 pid {dead.pid}\n")
        self.assertTrue(self.mk().try_acquire())

    def test_legacy_without_pid(self):
        self.lock.mkdir()
        (self.lock / "owner").write_text("Понятный документ\n")
        old = time.time() - 3600
        os.utime(self.lock, (old, old))
        self.assertFalse(self.mk().try_acquire())             # час - ещё держит
        old = time.time() - 4 * 3600
        os.utime(self.lock, (old, old))
        self.assertTrue(self.mk().try_acquire())              # больше 3 ч без номера процесса - брошен

    def test_being_written(self):
        self.lock.mkdir()                                     # owner ещё не записан
        self.assertFalse(self.mk().try_acquire())
        self.assertEqual(hl.status(self.lock).program, "?")


class CancelWait(Base):
    """L3: отмена из очереди - файл очереди убран, замок не взят."""

    def test_cancel(self):
        a = self.holder_proc("Herald")
        self.expect(a, "taken")
        b = self.mk("Scribe")
        cancel = threading.Event()
        res = {}
        th = threading.Thread(target=lambda: res.setdefault("ok", b.acquire(cancel=cancel, poll=0.05)))
        th.start()
        self.assertTrue(wait_for(lambda: len(hl.queue(self.lock)) == 1))
        self.assertTrue(b.waiting)
        cancel.set()
        th.join(5)
        self.assertFalse(res["ok"])
        self.assertEqual(hl.queue(self.lock), [])
        self.assertFalse(b.waiting)
        self.assertEqual(hl.status(self.lock).program, "Herald")
        self.assertIn("cancel", [r[1] for r in self.journal()])

    def test_dead_waiter_ignored(self):
        q = hl.queue_dir(self.lock)
        q.mkdir(parents=True)
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        (q / "20260101-000000-000000-X-1.json").write_text(json.dumps(
            {"program": "X", "pid": dead.pid, "who": "human", "kind": "wait", "ts": 1}))
        self.assertEqual(hl.queue(self.lock), [])
        self.assertTrue(self.mk(who=hl.BACKGROUND).try_acquire())


class RunNow(Base):
    """L4: «Запустить сейчас» - замок не наш, `now-…` виден, убран после."""

    def test_run_now(self):
        a = self.holder_proc("Interview Coach")
        self.expect(a, "taken")
        b = self.mk("Scribe")
        self.assertFalse(b.try_acquire())
        b.run_now()
        self.assertTrue(b.now and not b.held)
        al = hl.alongside(self.lock)
        self.assertEqual([x.program for x in al], ["Scribe"])
        self.assertEqual(hl.queue(self.lock), [])             # поверх - не очередь
        self.assertEqual(tx.header_state(b)[0], "alongside")
        b.release()
        self.assertEqual(hl.alongside(self.lock), [])
        self.assertEqual(hl.status(self.lock).program, "Interview Coach")
        self.assertIn("now", [r[1] for r in self.journal()])

    def test_now_then_free(self):
        a = self.holder_proc("Herald")
        self.expect(a, "taken")
        b = self.mk("Scribe")
        b.try_acquire()
        b.run_now()
        self.send(a)
        self.expect(a, "released")
        self.assertTrue(b.try_acquire())                      # освободилось - замок наш, «поверх» снято
        self.assertFalse(b.now)
        self.assertEqual(hl.alongside(self.lock), [])
        b.release()


class HumanFirst(Base):
    """L5: человек важнее фона."""

    def test_order_humans_first(self):
        a = self.holder_proc("Herald")
        self.expect(a, "taken")
        bg, hu = self.mk("Ночной прогон", who=hl.BACKGROUND), self.mk("Scribe", who=hl.HUMAN)
        res = {}
        t_bg = threading.Thread(target=lambda: res.setdefault("bg", bg.acquire(poll=0.05)))
        t_bg.start()
        self.assertTrue(wait_for(lambda: len(hl.queue(self.lock)) == 1))
        t_hu = threading.Thread(target=lambda: res.setdefault("hu", hu.acquire(poll=0.05)))
        t_hu.start()
        self.assertTrue(wait_for(lambda: len(hl.queue(self.lock)) == 2))
        self.assertEqual([w.program for w in hl.queue(self.lock)], ["Scribe", "Ночной прогон"])
        self.assertEqual(hu.ahead(), 0)
        self.assertEqual(bg.ahead(), 1)
        self.send(a)
        t_hu.join(10)
        self.assertTrue(res.get("hu"))
        self.assertEqual(hl.status(self.lock).program, "Scribe")
        self.assertTrue(t_bg.is_alive())                      # фон всё ещё ждёт
        hu.release()
        t_bg.join(10)
        self.assertTrue(res.get("bg"))
        bg.release()

    def test_background_does_not_take_when_human_waits(self):
        a = self.holder_proc("Herald")
        self.expect(a, "taken")
        hu = self.mk("Scribe")
        th = threading.Thread(target=lambda: hu.acquire(poll=0.05))
        th.start()
        self.assertTrue(wait_for(lambda: len(hl.queue(self.lock)) == 1))
        bg = self.mk("Claude Code", who=hl.BACKGROUND)
        self.send(a)
        self.expect(a, "released")
        self.assertFalse(bg.try_acquire())                    # свободно, но ждёт человек
        th.join(10)
        self.assertTrue(hu.held)
        hu.release()
        self.assertTrue(bg.try_acquire())
        bg.release()

    def test_background_yields_at_checkpoint(self):
        bgp = self.holder_proc("Herald (ночь)", who=hl.BACKGROUND, mode="bg-checkpoint", what="дообучение")
        self.expect(bgp, "taken")
        self.assertTrue(hl.status(self.lock).background)
        hu = self.mk("Scribe")
        self.assertFalse(hu.try_acquire())
        self.assertTrue(hu.acquire(wait=10, poll=0.05))       # фон уступил на контрольной точке
        self.assertEqual(hl.status(self.lock).program, "Scribe")
        self.assertEqual([w.program for w in hl.queue(self.lock)], ["Herald (ночь)"])
        hu.release()
        self.expect(bgp, "retaken")
        self.assertEqual(hl.status(self.lock).program, "Herald (ночь)")
        self.assertIn("yield", [r[1] for r in self.journal()])
        self.send(bgp)

    def test_checkpoint_noop_without_human(self):
        bg = self.mk("Фон", who=hl.BACKGROUND)
        self.assertTrue(bg.try_acquire())
        self.assertFalse(bg.checkpoint())
        self.assertTrue(bg.held)
        bg.release()


class Idle(Base):
    """L6: простой 15 минут (подмена часов)."""

    def test_idle_release_and_resume(self):
        c = Clock()
        lk = self.mk("Interview Coach", clock=c)
        self.assertTrue(lk.try_acquire())
        c.t += 14 * 60
        self.assertFalse(lk.idle_check())
        lk.touch()
        c.t += 14 * 60
        self.assertFalse(lk.idle_check())
        c.t += 60
        self.assertTrue(lk.idle_check())
        self.assertFalse(lk.held)
        self.assertTrue(lk.idle_released)
        self.assertFalse(self.lock.exists())
        self.assertIn("idle", [r[1] for r in self.journal()])
        self.assertTrue(lk.resume())
        self.assertTrue(lk.held and not lk.idle_released)
        lk.release()

    def test_resume_busy(self):
        c = Clock()
        lk = self.mk("Interview Coach", clock=c)
        lk.try_acquire()
        c.t += 15 * 60
        self.assertTrue(lk.idle_check())
        other = self.mk("Herald")
        self.assertTrue(other.try_acquire())
        self.assertFalse(lk.resume())                         # занято - окно «Видеокарта занята»
        other.release()
        self.assertTrue(lk.resume())
        lk.release()

    def test_background_never_idle(self):
        c = Clock()
        lk = self.mk("Фон", who=hl.BACKGROUND, clock=c)
        lk.try_acquire()
        c.t += 5 * 3600
        self.assertFalse(lk.idle_check())
        self.assertTrue(lk.held)
        lk.release()


class Compat(Base):
    """L7: совместимость со старыми копиями (IC coach/lock.py, Herald core/heavylock.py, shell)."""

    def test_reads_legacy_owner(self):
        self.lock.mkdir()
        (self.lock / "owner").write_text(f"Понятный документ Fri Oct 23 19:05:25 MSK 2026 pid {os.getpid()}\n")
        h = hl.status(self.lock)
        self.assertEqual((h.program, h.pid), ("Понятный документ", os.getpid()))
        self.assertEqual(h.since, dt.datetime(2026, 10, 23, 19, 5, 25))
        self.assertTrue(h.legacy)
        self.assertFalse(self.mk().try_acquire())

    def test_old_code_reads_new_owner(self):
        lk = self.mk("Scribe")
        lk.try_acquire()
        txt = (self.lock / "owner").read_text(encoding="utf-8").strip()
        m = re.match(r"(.+?)\s+(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b", txt)          # как IC owner() / Herald owner()
        self.assertEqual(m.group(1), "Scribe")
        self.assertEqual(int(re.search(r"pid (\d+)", txt).group(1)), os.getpid())   # как IC _stale()
        sed = subprocess.run(["sed", "-nE", r"s/.* pid ([0-9]+).*/\1/p", str(self.lock / "owner")],
                             capture_output=True, text=True).stdout.strip()   # как CD / LDB heavy_lock.sh
        self.assertEqual(sed, str(os.getpid()))
        lk.release()

    def test_held_outside(self):
        os.environ["HEAVY_LOCK_HELD"] = "1"
        try:
            lk = self.mk()
            self.assertTrue(lk.try_acquire())
            self.assertFalse(self.lock.exists())
            lk.release()
        finally:
            del os.environ["HEAVY_LOCK_HELD"]
        os.environ["SCRIBE_HEAVY_LOCK_HELD"] = "1"
        try:
            self.assertTrue(self.mk(held_env="SCRIBE_HEAVY_LOCK_HELD").try_acquire())
            self.assertFalse(self.lock.exists())
        finally:
            del os.environ["SCRIBE_HEAVY_LOCK_HELD"]

    def test_who_from_env(self):
        os.environ["HEAVY_LOCK_WHO"] = "background"
        try:
            self.assertEqual(self.mk(who=None).who, hl.BACKGROUND)
        finally:
            del os.environ["HEAVY_LOCK_WHO"]
        self.assertEqual(self.mk(who=None).who, hl.HUMAN)

    def test_unavailable_folder(self):
        lk = hl.HeavyLock("Scribe", path=self.tmp / "нет такой папки" / ".heavy-lock")
        self.assertTrue(lk.try_acquire())                     # работаем без замка (README, 5.8)
        self.assertTrue(lk.unavailable)
        lk.release()

    def test_default_path_env(self):
        self.assertEqual(hl.default_path(), self.lock)
        del os.environ["HEAVY_LOCK"]
        os.environ["SCRIBE_TEST_LOCK"] = str(self.tmp / "x")
        try:
            self.assertEqual(hl.default_path("SCRIBE_TEST_LOCK"), self.tmp / "x")
        finally:
            del os.environ["SCRIBE_TEST_LOCK"]
            os.environ["HEAVY_LOCK"] = str(self.lock)


class Shell(Base):
    """L8: heavy_lock.sh take / status / release / run / yield."""

    def sh(self, *args, **kw):
        return subprocess.run(["bash", str(SH), *args], capture_output=True, text=True, encoding="utf-8",
                              timeout=kw.pop("timeout", 30), **kw)

    def sleeper(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.procs.append(p)
        return p

    def test_take_status_release(self):
        s = self.sleeper()
        r = self.sh("take", "Herald", "дообучение F5", "--pid", str(s.pid))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        h = hl.status(self.lock)
        self.assertEqual((h.program, h.pid, h.who, h.what), ("Herald", s.pid, hl.BACKGROUND, "дообучение F5"))
        st = self.sh("status").stdout
        self.assertIn("Видеокарта занята: «Herald»", st)
        self.assertIn("фон", st)
        self.assertEqual(self.sh("release", "Herald", "--pid", "1").returncode, 0)   # чужой - не трогает
        self.assertTrue(self.lock.exists())
        self.sh("release", "Herald", "--pid", str(s.pid))
        self.assertFalse(self.lock.exists())
        self.assertIn("Видеокарта свободна.", self.sh("status").stdout)
        js = json.loads(self.sh("status", "--json").stdout)
        self.assertTrue(js["free"])

    def test_take_timeout(self):
        a = self.holder_proc("Interview Coach")
        self.expect(a, "taken")
        s = self.sleeper()
        r = self.sh("take", "Herald", "x", "--pid", str(s.pid), "--wait-min", "0.02")
        self.assertEqual(r.returncode, 75)

    def test_run(self):
        r = self.sh("run", "Понятный документ", "сценарии", "--",
                    sys.executable, "-c", "import os; print('HELD=' + os.environ['HEAVY_LOCK_HELD']); "
                    "print('DIR=' + str(os.path.isdir(os.environ['HEAVY_LOCK'])))")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("HELD=1", r.stdout)
        self.assertIn("DIR=True", r.stdout)
        self.assertFalse(self.lock.exists())

    def test_run_with_options(self):
        r = self.sh("run", "Понятный документ", "прогон", "--who", "background", "--wait-min", "5", "--",
                    sys.executable, "-c", "import json, os, sys; sys.path.insert(0, os.environ['LOCKMOD']); "
                    "from lock import heavy_lock as hl; h = hl.status(os.environ['HEAVY_LOCK']); "
                    "print('WHO=' + h.who + ' WHAT=' + h.what)",
                    env={**os.environ, "LOCKMOD": str(HERE.parents[1])})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("WHO=background WHAT=прогон", r.stdout)

    def test_yield(self):
        s = self.sleeper()
        self.sh("take", "Herald (ночь)", "глава 3", "--pid", str(s.pid))
        self.assertEqual(self.sh("yield", "Herald (ночь)", "--pid", str(s.pid)).returncode, 0)   # никто не ждёт
        self.assertEqual(hl.status(self.lock).pid, s.pid)
        hu = self.mk("Scribe")
        th = threading.Thread(target=lambda: hu.acquire(poll=0.05))
        th.start()
        self.assertTrue(wait_for(lambda: len(hl.queue(self.lock)) == 1))
        y = subprocess.Popen(["bash", str(SH), "yield", "Herald (ночь)", "--pid", str(s.pid)],
                             stdout=subprocess.PIPE, text=True)
        self.procs.append(y)
        th.join(10)
        self.assertTrue(hu.held)
        self.assertEqual(hl.status(self.lock).program, "Scribe")
        hu.release()
        self.assertEqual(y.wait(20), 0)
        h = hl.status(self.lock)
        self.assertEqual((h.program, h.pid, h.what), ("Herald (ночь)", s.pid, "глава 3"))


class OwnBackground(Base):
    """L10 (1.4.0, задание 06 Настольного помощника, И): человек не ждёт фон своей же программы."""

    def test_human_joins_own_program_background(self):
        bg = self.mk("Настольный помощник", who=hl.BACKGROUND, what="векторы правил и готовые вопросы")
        self.assertTrue(bg.try_acquire())
        human = self.mk("Настольный помощник", what="ответ на вопрос по правилам")
        t0 = time.monotonic()
        self.assertTrue(human.acquire(wait=5, poll=0.1))
        self.assertLess(time.monotonic() - t0, 1.0)                 # сразу, не после пересборки
        self.assertTrue(human.joined and human.now)
        self.assertEqual(hl.status(self.lock).what, "векторы правил и готовые вопросы")   # замок по-прежнему у фона
        human.release()
        self.assertTrue(bg.held and self.lock.is_dir())               # фон работает дальше
        bg.release()
        self.assertFalse(self.lock.exists())

    def test_other_program_still_waits_for_background(self):
        bg = self.mk("Настольный помощник", who=hl.BACKGROUND)
        self.assertTrue(bg.try_acquire())
        other = self.mk("Interview Coach")
        self.assertFalse(other.try_acquire())                        # чужой фон - очередь и уступка на контрольной точке
        own_bg = self.mk("Настольный помощник", who=hl.BACKGROUND, what="ещё фон")
        self.assertFalse(own_bg.try_acquire())                       # фон своего фона не обходит
        bg.release()

    def test_own_human_holder_is_not_joined(self):
        h1 = self.mk("Настольный помощник")
        self.assertTrue(h1.try_acquire())
        h2 = self.mk("Настольный помощник", what="второй вопрос")
        self.assertFalse(h2.try_acquire())                           # человек человека - по очереди, как раньше
        h1.release()


class Texts(unittest.TestCase):
    """L9: тексты RU = EN по ключам, алфавит."""

    def test_keys_and_alphabet(self):
        ru, en = tx.TEXTS["ru"], tx.TEXTS["en"]
        self.assertEqual(set(ru), set(en))
        strip = lambda s: re.sub(r"\{\w+\}", "", s)      # noqa: E731
        for k, v in ru.items():
            self.assertNotRegex(strip(v), r"[A-Za-z]", f"латиница в RU: {k}")
        for k, v in en.items():
            self.assertNotRegex(strip(v), r"[А-Яа-яЁё]", f"кириллица в EN: {k}")
        for k in ru:
            self.assertEqual(re.findall(r"\{\w+\}", ru[k]), re.findall(r"\{\w+\}", en[k]), k)

    def test_describe(self):
        h = hl.Holder("Interview Coach", 1, "репетиция", since=dt.datetime.now().replace(hour=16, minute=31))
        self.assertEqual(tx.describe(h), "Видеокарта занята: «Interview Coach», с 16:31, репетиция.")
        self.assertEqual(tx.describe(hl.Holder("?", 1, since=h.since)), "Видеокарта занята: «другая программа», с 16:31.")
        self.assertEqual(tx.waiting_line(h, 0), "Жду видеокарту: «Interview Coach», с 16:31, репетиция · вы следующий")
        self.assertIn("ahead of you: 2", tx.waiting_line(h, 2, "en"))
        old = dt.datetime(2026, 10, 22, 9, 5)
        self.assertEqual(hl.since_text(old, now=dt.datetime(2026, 10, 23, 12, 0)), "22.10 09:05")


class CpuOption(unittest.TestCase):
    """«На процессоре» с прогнозом (доп. 21:40): медленнее видеокарты больше чем в 3 раза - «очень медленно»."""

    def test_cpu_option(self):
        self.assertIsNone(tx.cpu_option(None))
        self.assertEqual(tx.cpu_option(True)[0], "На процессоре")
        label, hint, slow = tx.cpu_option({"seconds": 90, "factor": 1.2})
        self.assertEqual((label, slow), ("На процессоре", False))
        self.assertIn("≈ 2 мин", hint)
        label, hint, slow = tx.cpu_option({"seconds": 4 * 3600 + 600, "factor": 12}, "ru")
        self.assertEqual((label, slow), ("На процессоре (очень медленно)", True))
        self.assertIn("≈ 4 ч 10 мин (в 12 раз дольше", hint)
        self.assertIn("12 times longer", tx.cpu_option({"seconds": 600, "factor": 12}, "en")[1])
        self.assertEqual(tx.eta_text(20), "20 с")


class StatusJson(Base):
    def test_status_json(self):
        a = self.holder_proc("Herald", what="книга")
        self.expect(a, "taken")
        me = self.mk("Interview Coach")
        d = hl.status_json(me=me)
        self.assertFalse(d["free"])
        self.assertEqual(d["holder"]["program"], "Herald")
        self.assertEqual(d["holder"]["what"], "книга")
        self.assertFalse(d["me"]["held"])
        self.assertEqual(tx.header_state(me)[:2], ("busy", "занята «Herald»"))
        self.assertEqual(tx.header_state(None, model_ok=False)[0], "nomodel")


if __name__ == "__main__":
    unittest.main(verbosity=2)
