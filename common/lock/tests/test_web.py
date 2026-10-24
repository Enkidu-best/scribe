# -*- coding: utf-8 -*-
"""Веб-окно «Видеокарта занята» (lock_ui/web): безголовый Chromium, своя временная страница, без окон Андрея.
Пропускается, если Playwright не установлен.

    <python с playwright> "_общие модули/lock/tests/test_web.py" [папка для снимков]
"""
import functools
import http.server
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
UI = HERE.parent / "lock_ui"
# папка для снимков - только при запуске напрямую (под pytest argv - его параметры, «-q» - не папка)
SHOTS = Path(sys.argv.pop(1)) if __name__ == "__main__" and len(sys.argv) > 1 else None

try:
    from playwright.sync_api import sync_playwright
except Exception:                       # noqa: BLE001 - нет Playwright - проверка пропускается
    sync_playwright = None

PAGE = """<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<link rel="stylesheet" href="lock_ui/web/heavy-lock.css"><style>body{{font:16px system-ui;margin:0;height:100vh}}</style>
</head><body><script src="lock_ui/web/heavy-lock.js"></script></body></html>"""
HOLDER = {"program": "Interview Coach", "pid": 1, "what": "репетиция", "who": "human", "since_text": "16:31"}


@unittest.skipIf(sync_playwright is None, "нет Playwright")
class WebDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="heavylock-web-"))
        shutil.copytree(UI, cls.tmp / "lock_ui")
        for lang in ("ru", "en"):
            (cls.tmp / f"{lang}.html").write_text(PAGE.format(lang=lang), encoding="utf-8")
        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a):
                pass
        handler = functools.partial(Quiet, directory=str(cls.tmp))
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def open(self, lang="ru", width=1000):
        pg = self.browser.new_page(viewport={"width": width, "height": 700})
        pg.goto(f"{self.base}/{lang}.html")
        pg.evaluate("HeavyLockUI.ready")
        return pg

    def ask(self, pg, opts):
        pg.evaluate("([h, o]) => { window.ans = 'pending'; HeavyLockUI.ask(h, o).then(a => window.ans = a); }",
                    [HOLDER, opts])
        pg.wait_for_selector(".hl-dialog")

    def test_ru_buttons_and_answers(self):
        pg = self.open()
        self.ask(pg, {"lang": "ru", "cpu": True, "queue": ["Herald"]})
        txt = pg.inner_text(".hl-dialog")
        for s in ("Видеокарта занята: «Interview Coach», с 16:31, репетиция.", "Подождать очереди",
                  "Запустить сейчас", "На процессоре", "Отмена", "В очереди перед вами: «Herald».",
                  "Обе программы будут медленнее"):
            self.assertIn(s, txt)
        if SHOTS:
            SHOTS.mkdir(parents=True, exist_ok=True)
            pg.screenshot(path=str(SHOTS / "web-busy-ru.png"))
        cut = pg.evaluate("[...document.querySelectorAll('.hl-dialog *')].filter(e => e.scrollWidth > e.clientWidth + 1)"
                          ".map(e => e.textContent)")
        self.assertEqual(cut, [])                       # ничего не обрезано (стандарт 4.4)
        pg.click("button[data-answer=now]")
        self.assertEqual(pg.evaluate("window.ans"), "now")
        self.assertEqual(pg.query_selector_all(".hl-dialog"), [])
        self.ask(pg, {"lang": "ru"})
        self.assertIsNone(pg.query_selector("button[data-answer=cpu]"))
        pg.keyboard.press("Escape")
        self.assertIsNone(pg.evaluate("window.ans"))
        self.ask(pg, {"lang": "ru"})
        pg.click("button[data-answer=wait]")
        self.assertEqual(pg.evaluate("window.ans"), "wait")
        pg.close()

    def test_cpu_slow(self):
        pg = self.open()
        self.ask(pg, {"lang": "ru", "cpu": {"seconds": 7200, "factor": 12}})
        b = pg.inner_text("button[data-answer=cpu]")
        self.assertEqual(b, "На процессоре (очень медленно)")
        self.assertIn("≈ 2 ч (в 12 раз дольше", pg.inner_text(".hl-dialog"))
        self.assertNotIn("hl-primary", pg.get_attribute("button[data-answer=cpu]", "class"))
        pg.click("button[data-answer=cpu]")
        self.assertEqual(pg.evaluate("window.ans"), "cpu")
        pg.close()

    def test_en_and_narrow(self):
        pg = self.open("en", width=420)
        self.ask(pg, {"lang": "en"})
        txt = pg.inner_text(".hl-dialog")
        self.assertIn("The GPU is busy: “Interview Coach”, since 16:31, репетиция.", txt)   # «что» - слова владельца
        self.assertIn("Wait in line", txt)
        w = pg.evaluate("document.documentElement.scrollWidth <= innerWidth")
        self.assertTrue(w)
        pg.close()

    def test_over_modal_dialog(self):
        """Окно программы - модальный <dialog> (верхний слой): окно очереди поверх него, кнопки нажимаются, Esc
        закрывает только окно очереди (найдено в Legal Doc Builder, «Из документа»)."""
        pg = self.open()
        pg.evaluate("() => { const d = document.createElement('dialog'); d.id = 'app-dlg';"
                    " d.innerHTML = '<textarea style=\"width:600px;height:400px\"></textarea>';"
                    " document.body.append(d); d.showModal(); }")
        self.ask(pg, {"lang": "ru", "cpu": True})
        pg.click("button[data-answer=cpu]", timeout=3000)
        self.assertEqual(pg.evaluate("window.ans"), "cpu")
        self.ask(pg, {"lang": "ru"})
        pg.keyboard.press("Escape")
        self.assertIsNone(pg.evaluate("window.ans"))
        self.assertTrue(pg.evaluate("document.getElementById('app-dlg').open"))
        pg.close()

    def test_header_state(self):
        pg = self.open()
        st = pg.evaluate("HeavyLockUI.headerState({holder: {program: 'Herald'}, me: {waiting: true}}, 'ru')")
        self.assertEqual(st, {"state": "busy_queued", "text": "занята «Herald» - в очереди", "kind": "warn"})
        st = pg.evaluate("HeavyLockUI.headerState({holder: null, me: {}}, 'en', {modelOk: false})")
        self.assertEqual(st["text"], "no model")
        pg.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
