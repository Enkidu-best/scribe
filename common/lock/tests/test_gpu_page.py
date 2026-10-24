# -*- coding: utf-8 -*-
"""Страница «Видеокарта» (tools/gpu_page.py, README, 12): свой сервер на свободном порту, свой замок во временной папке,
безголовый Chromium - окон и браузера Андрея не открывает. «Другую программу» и фон изображают процессы _holder.py.
Пропускается, если Playwright не установлен.

    <python с playwright> "_общие модули/lock/tests/test_gpu_page.py" [папка для снимков]
"""
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from lock import heavy_lock as hl  # noqa: E402

SHOTS = Path(sys.argv.pop(1)) if __name__ == "__main__" and len(sys.argv) > 1 else None
HOLDER = HERE / "_holder.py"
SERVER = HERE.parent / "tools" / "gpu_page.py"
A11Y = HERE.parents[1] / "a11y"

try:
    from playwright.sync_api import sync_playwright
except Exception:                       # noqa: BLE001 - нет Playwright - проверка пропускается
    sync_playwright = None


@unittest.skipIf(sync_playwright is None, "нет Playwright")
@unittest.skipIf(not SERVER.exists(), "копия в программе - страница только в «_общие модули»")
class GpuPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hl-gpu-page-"))
        cls.lock = cls.tmp / ".heavy-lock"
        hl.save_settings(cls.lock, budget_gb=13, needs_gb={"Herald": {"gb": 6, "measured": True}})
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0))
            cls.port = so.getsockname()[1]
        cls.url = f"http://127.0.0.1:{cls.port}"
        cls.srv = subprocess.Popen([sys.executable, str(SERVER), "--port", str(cls.port), "--lock", str(cls.lock),
                                    "--no-open"], stdout=subprocess.PIPE, text=True)
        assert cls.srv.stdout.readline().strip() == cls.url
        cls.procs = []
        cls.herald = cls.hold("Herald", "human", "озвучка книги")
        cls.bg = cls.hold("Прогон тестов", "background", "pytest", queue=True)
        cls.pw = sync_playwright().start()
        cls.br = cls.pw.chromium.launch()

    @classmethod
    def hold(cls, program, who, what, queue=False):
        p = subprocess.Popen([sys.executable, str(HOLDER), str(cls.lock), program, who, "wait" if queue else "hold", what],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        cls.procs.append(p)
        if not queue:
            assert p.stdout.readline().strip() in ("taken", "joined")
        else:
            for _ in range(100):
                if any(h.program == program for h in hl.queue(cls.lock)):
                    break
                time.sleep(0.05)
        return p

    @classmethod
    def tearDownClass(cls):
        cls.br.close()
        cls.pw.stop()
        for p in [*cls.procs, cls.srv]:
            if p.poll() is None:
                p.kill()
                p.wait(5)
            for f in (p.stdin, p.stdout):
                if f:
                    f.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def page(self, lang="ru", scheme="light"):
        pg = self.br.new_page(viewport={"width": 1100, "height": 900}, locale="ru-RU" if lang == "ru" else "en-US",
                              color_scheme=scheme)
        self.errors = []
        pg.on("pageerror", lambda e: self.errors.append(str(e)))
        pg.goto(self.url)
        pg.wait_for_selector("#working li .who")
        return pg

    def test_shows_working_waiting_memory(self):
        pg = self.page()
        work = pg.inner_text("#working")
        self.assertIn("«Herald»", work)
        self.assertIn("озвучка книги", work)
        self.assertIn("6 ГБ (замер)", work)
        self.assertIn("«Прогон тестов»", pg.inner_text("#waiting"))
        self.assertIn("Занято 6 из 13 ГБ", pg.inner_text("#memtext"))
        self.assertEqual(pg.locator("#working button").count(), 0)      # работа человека - без «Отпустить»
        self.assertEqual(pg.inner_text("#waiting button"), "Отпустить")
        self.assertEqual(pg.evaluate("getComputedStyle(document.body).fontSize"), "17px")   # 100 % = 17 px (стандарт 1.5)
        if SHOTS:
            SHOTS.mkdir(parents=True, exist_ok=True)
            pg.screenshot(path=str(SHOTS / "gpu-page-ru-light.png"), full_page=True)
        self.assertEqual(self.errors, [])
        pg.close()

    def test_mode_pause_let_go(self):
        pg = self.page()
        pg.click("label.mode:has(input[value=one_at_a_time])")
        pg.wait_for_function("document.querySelector('label.mode.on input').value === 'one_at_a_time'")
        self.assertEqual(hl.settings(self.lock)["mode"], "one_at_a_time")
        pg.click("#pause")
        pg.wait_for_function("document.getElementById('pause').getAttribute('aria-checked') === 'true'")
        self.assertTrue(hl.settings(self.lock)["pause_background"])
        pg.click("#pause")
        pg.wait_for_function("document.getElementById('pause').getAttribute('aria-checked') === 'false'")
        pid = self.bg.pid
        pg.click("#waiting button[data-pid]")
        pg.wait_for_selector("#waiting .off")
        self.assertTrue(hl.held_off(self.lock, pid))
        self.assertIn("ждёт «Вернуть»", pg.inner_text("#waiting"))
        pg.click("#waiting button[data-pid]")
        pg.wait_for_function("!document.querySelector('#waiting .off')")
        self.assertFalse(hl.held_off(self.lock, pid))
        hl.save_settings(self.lock, mode="mine_first")
        self.assertEqual(self.errors, [])
        pg.close()

    def test_foreign_site_cannot_change(self):
        req = urllib.request.Request(self.url + "/api/pause", data=b'{"on": true}', method="POST",
                                     headers={"Content-Type": "application/json", "Origin": "https://example.com"})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(e.exception.code, 403)
        self.assertFalse(hl.settings(self.lock)["pause_background"])

    def test_english_has_no_cyrillic(self):
        pg = self.page(lang="en")
        txt = pg.inner_text("body")
        # имена программ и «что делает» приходят из очереди как есть - их убираем
        for s in ("Herald", "озвучка книги", "Прогон тестов", "pytest"):
            txt = txt.replace(s, "")
        self.assertFalse([c for c in txt if "а" <= c.lower() <= "я" or c in "ёЁ"], txt[:300])
        self.assertIn("6 of 13 GB in use", txt)
        pg.close()

    def test_contrast_both_themes(self):
        if not (A11Y / "contrast.py").exists():
            self.skipTest("нет модуля a11y")
        sys.path.insert(0, str(A11Y.parent))
        from a11y import contrast
        for scheme in ("light", "dark"):
            pg = self.page(scheme=scheme)
            pg.click("#aa")                                   # всплывающее «Aa» - тоже проверить
            bad = contrast.check(pg)
            self.assertFalse(bad, f"{scheme}: " + "; ".join(contrast.summary(bad)))
            if SHOTS:
                pg.screenshot(path=str(SHOTS / f"gpu-page-ru-{scheme}-aa.png"), full_page=True)
            pg.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
