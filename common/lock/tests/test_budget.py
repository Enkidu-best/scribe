# -*- coding: utf-8 -*-
"""Диспетчер видеокарты по бюджету памяти (README, раздел 10; «_Общие модули 02»). Только временная папка, без
видеокарты и моделей: «другую программу» изображает второй процесс (_holder.py).

    python3 "_общие модули/lock/tests/test_budget.py"
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from lock import heavy_lock as hl  # noqa: E402

HOLDER = HERE / "_holder.py"
SH = HERE.parent / "heavy_lock.sh"


class Budget(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hl-budget-"))
        self.lock = self.tmp / ".heavy-lock"
        self._env = {k: os.environ.pop(k) for k in ("HEAVY_LOCK_HELD", "HEAVY_LOCK_WHO", "HEAVY_LOCK") if k in os.environ}
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                try:
                    p.stdin.write("release\n")
                    p.stdin.flush()
                    p.wait(5)
                except (OSError, subprocess.TimeoutExpired):
                    p.kill()
            for f in (p.stdin, p.stdout):
                f.close()
        os.environ.update(self._env)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def settings(self, **kw):
        hl.save_settings(self.lock, **kw)

    def other(self, program="Herald", who="human", need=None, what="озвучка книги"):
        env = dict(os.environ, **({"HL_NEED_GB": str(need)} if need else {}))
        p = subprocess.Popen([sys.executable, str(HOLDER), str(self.lock), program, who, "hold", what],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        self.procs.append(p)
        self.assertIn(p.stdout.readline().strip(), ("taken", "joined"))
        return p

    def mine(self, need=5.0, who="human", program="Scribe"):
        return hl.HeavyLock(program, "лекция", who=who, path=self.lock, idle_after=None, need_gb=need)

    # B1: по оценке рядом никто не запускается - как до 1.2.0
    def test_unmeasured_waits(self):
        self.other()                                   # Herald без заявленной памяти, в таблице - оценка
        lk = self.mine(need=3)
        self.assertFalse(lk.try_acquire())
        self.assertFalse(lk.joined)

    # B2: замерено и помещается - работают вместе
    def test_measured_fits_joins(self):
        self.settings(needs_gb={"Herald": {"gb": 6, "measured": True}})
        self.other()
        lk = self.mine(need=5)
        self.assertTrue(lk.try_acquire())
        self.assertTrue(lk.joined and lk.now and lk.held)
        self.assertEqual([h.program for h in hl.alongside(self.lock)], ["Scribe"])
        self.assertEqual(hl.used_gb(self.lock)[0], 11.0)
        self.assertEqual(hl.status_json(self.lock)["used_gb"], 11.0)
        lk.release()
        self.assertEqual(hl.alongside(self.lock), [])
        self.assertEqual(hl.status(self.lock).program, "Herald")   # замок владельца не тронут

    # B3: не помещается - ждёт, а не падает
    def test_does_not_fit_waits(self):
        self.other(need=6)
        lk = self.mine(need=8)
        self.assertFalse(lk.try_acquire())

    # B4: «Только по очереди» - вместе не работают даже если помещается
    def test_one_at_a_time(self):
        self.settings(mode="one_at_a_time")
        self.other(need=6)
        self.assertFalse(self.mine(need=2).try_acquire())

    # B5: фон рядом - только если не ждёт человек; «Пауза фону» - не встаёт рядом
    def test_background_after_humans_and_pause(self):
        self.other(need=6)
        hum = self.mine(need=9, program="Понятный документ")      # человек не помещается - встаёт в очередь
        hum._qfile = hum._entry("wait")
        bg = self.mine(need=2, who="background", program="Прогон")
        self.assertFalse(bg.try_acquire())                        # ждёт человек - фон рядом не встаёт
        hum._drop(hum._qfile)
        hum._qfile = None
        self.settings(pause_background=True)
        self.assertFalse(bg.try_acquire())
        self.settings(pause_background=False)
        self.assertTrue(bg.try_acquire() and bg.joined)
        bg.release()

    # B6: фон уступает человеку на контрольной точке; «Ночь - всё фону» - не уступает; пауза - уступает всегда
    def test_checkpoint_modes(self):
        bg = self.mine(need=None, who="background", program="Прогон")
        self.assertTrue(bg.try_acquire())
        hum = self.mine(need=None, program="Herald")
        hum._qfile = hum._entry("wait")
        self.settings(mode="night")
        self.assertFalse(bg.checkpoint(poll=0.05))
        self.settings(mode="mine_first")
        hum._drop(hum._qfile)
        hum._qfile = None
        self.settings(pause_background=True)
        stop = threading.Event()
        threading.Timer(0.5, stop.set).start()
        self.assertTrue(bg.checkpoint(cancel=stop, poll=0.05))   # уступил и ждёт кнопки, а не забирает снова
        self.assertFalse(bg.held)
        self.assertIsNone(hl.status(self.lock))

    # B7: замок свободен, но рядом работает другой и места не хватает - ждать
    def test_free_lock_but_alongside_full(self):
        p = self.other(need=6)
        joiner = self.mine(need=6, program="Interview Coach")
        self.assertTrue(joiner.try_acquire() and joiner.joined)
        p.stdin.write("release\n")
        p.stdin.flush()
        p.wait(5)
        self.assertIsNone(hl.status(self.lock))
        big = self.mine(need=9, program="Понятный документ")
        self.assertFalse(big.try_acquire())                       # 6 рядом + 9 > 13
        small = self.mine(need=5, program="Scribe")
        self.assertTrue(small.try_acquire() and not small.joined)  # 6 + 5 - помещается, замок теперь у Scribe
        small.release()
        joiner.release()

    # B8: «Видеокарта» в командной строке: режим, память, кто работает
    def test_cli_gpu_and_mode(self):
        self.other(need=6)
        out = subprocess.run(["bash", str(SH), "--path", str(self.lock), "mode", "one_at_a_time"],
                             capture_output=True, text=True).stdout
        self.assertIn("Режим: Только по очереди", out)
        self.assertIn("занято 6 из 13 ГБ", out)
        self.assertIn("«Herald»", out)
        self.assertEqual(json.loads(hl.settings_path(self.lock).read_text(encoding="utf-8"))["mode"], "one_at_a_time")
        out = subprocess.run(["bash", str(SH), "--path", str(self.lock), "pause-bg", "on"], capture_output=True, text=True).stdout
        self.assertIn("фон на паузе", out)

    # B9: фон в run - с пониженным приоритетом процесса
    def test_background_low_priority(self):
        cmd = hl._run_cmd(["echo", "x"], hl.BACKGROUND)
        if sys.platform == "darwin":
            self.assertEqual(cmd[:2], ["taskpolicy", "-b"])
        self.assertEqual(hl._run_cmd(["echo", "x"], hl.HUMAN), ["echo", "x"])

    # B10: старый владелец (строка owner, без owner.json) - память неизвестна, рядом не встаём
    def test_legacy_holder_unknown(self):
        self.settings(needs_gb={"Legal Doc Builder": {"gb": 6, "measured": True}})
        self.lock.mkdir()
        (self.lock / "owner").write_text(f"Legal Doc Builder Fri Oct 17 01:00:00 2026 pid {os.getpid()}\n", encoding="utf-8")
        lk = self.mine(need=2)
        self.assertTrue(lk.try_acquire() and lk.joined)           # строка старой копии - по таблице (замерено)
        lk.release()
        self.settings(needs_gb={"Legal Doc Builder": {"gb": 6, "measured": False}})
        self.assertFalse(self.mine(need=2).try_acquire())

    # B11: «Пауза фону» при свободной видеокарте - фон не берёт, человек берёт
    def test_pause_blocks_background_on_free_gpu(self):
        self.settings(pause_background=True)
        bg = self.mine(need=None, who="background", program="Прогон")
        self.assertFalse(bg.try_acquire())
        hum = self.mine(need=None, program="Herald")
        self.assertTrue(hum.try_acquire())
        hum.release()
        self.settings(pause_background=False)
        self.assertTrue(bg.try_acquire())
        bg.release()

    # B12: «Отпустить» - фон уступает на контрольной точке и ждёт «Вернуть»; другим фоном очередь не загораживает
    def test_let_go_and_back(self):
        bg = self.mine(need=None, who="background", program="Прогон")
        self.assertTrue(bg.try_acquire())
        hl.let_go(self.lock, bg.pid)
        self.assertTrue(hl.status_json(self.lock)["holder"]["off"])
        box = {}
        t = threading.Thread(target=lambda: box.update(y=bg.checkpoint(poll=0.05)))
        t.start()
        for _ in range(100):
            if bg.waiting:
                break
            threading.Event().wait(0.02)
        self.assertTrue(bg.waiting and hl.status(self.lock) is None)
        self.assertTrue(hl.status_json(self.lock)["queue"][0]["off"])
        other = hl.HeavyLock("Второй прогон", who="background", path=self.lock, idle_after=None, pid=os.getppid())
        self.assertTrue(other.try_acquire())                      # отпущенный не стоит впереди
        other.release()
        hl.let_go(self.lock, bg.pid, on=False)                    # «Вернуть»
        t.join(5)
        self.assertTrue(box["y"] and bg.held)
        bg.release()

    # B13: «Отпустить» не действует на работу человека; отметка умершего процесса снимается
    def test_let_go_human_and_dead(self):
        hl.let_go(self.lock, os.getpid())
        hum = self.mine(need=None, program="Herald")
        self.assertTrue(hum.try_acquire())
        hum.release()
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        hl.let_go(self.lock, p.pid)
        self.assertFalse(hl.held_off(self.lock, p.pid))
        self.assertFalse((hl.queue_dir(self.lock) / f"off-{p.pid}").exists())
        out = subprocess.run(["bash", str(SH), "--path", str(self.lock), "back", str(os.getpid())],
                             capture_output=True, text=True).stdout
        self.assertIn("Режим:", out)
        self.assertFalse(hl.held_off(self.lock, os.getpid()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
