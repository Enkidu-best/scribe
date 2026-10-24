# -*- coding: utf-8 -*-
"""Тексты и состояния окна «Видеокарта занята» (README, 4.3 и 7). Источник - texts.json рядом:
те же строки читает веб-виджет (web/heavy-lock.js)."""
from __future__ import annotations

import json
from pathlib import Path

TEXTS: dict[str, dict[str, str]] = json.loads(Path(__file__).with_name("texts.json").read_text(encoding="utf-8"))

# цвет строки состояния (стандарт 2.1) - имена как у чипов тем Scribe / Herald (theme.set_chip)
STATE_KIND = {"ready": "ok", "working": "acc", "busy": "warn", "busy_queued": "warn", "alongside": "warn",
              "nomodel": "neutral", "error": "bad"}


def t(key: str, lang: str = "ru", **kw) -> str:
    s = TEXTS.get(lang, TEXTS["ru"]).get(key) or TEXTS["ru"][key]
    return s.format(**kw) if kw else s


def _program(holder, lang: str) -> str:
    p = getattr(holder, "program", "") if holder is not None else ""
    return t("someone", lang) if p in ("", "?") else p


def describe(holder, lang: str = "ru") -> str:
    """«Видеокарта занята: «Interview Coach», с 16:31, репетиция.»"""
    what = getattr(holder, "what", "") if holder is not None else ""
    since = holder.since_text if holder is not None else "?"
    if what:
        return t("busy", lang, program=_program(holder, lang), since=since, what=what)
    return t("busy_nowhat", lang, program=_program(holder, lang), since=since)


def waiting_line(holder, ahead: int, lang: str = "ru") -> str:
    """Строка ожидания: «Жду видеокарту: «Herald», с 16:31, озвучка книги · вы следующий»."""
    if holder is None:
        head = t("waiting_queue", lang)
    elif getattr(holder, "what", ""):
        head = t("waiting", lang, program=_program(holder, lang), since=holder.since_text, what=holder.what)
    else:
        head = t("waiting_nowhat", lang, program=_program(holder, lang), since=holder.since_text)
    tail = t("waiting_next", lang) if ahead <= 0 else t("waiting_ahead", lang, n=ahead)
    return f"{head} · {tail}"


SLOW_FACTOR = 3.0      # медленнее видеокарты больше чем во столько раз - «очень медленно», кнопка не выделена (доп. 21:40)


def eta_text(seconds: float, lang: str = "ru") -> str:
    """«40 с», «12 мин», «2 ч 10 мин» - прогноз для кнопки «На процессоре»."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return t("eta_sec", lang, n=max(5, round(s / 5) * 5))
    m = round(s / 60)
    if m < 60:
        return t("eta_min", lang, n=m)
    h, m = divmod(m, 60)
    return t("eta_h", lang, h=h) if m == 0 else t("eta_h_min", lang, h=h, n=m)


def cpu_option(cpu, lang: str = "ru") -> tuple[str, str, bool] | None:
    """Кнопка «На процессоре»: (надпись, пояснение, очень медленно). cpu: False / None - кнопки нет; True - без прогноза;
    {"seconds": прогноз на процессоре, "factor": во сколько раз медленнее видеокарты} - с прогнозом (замер программы)."""
    if not cpu:
        return None
    if cpu is True:
        return t("cpu", lang), t("cpu_hint", lang), False
    eta = eta_text(float(cpu.get("seconds") or 0), lang) if cpu.get("seconds") else ""
    x = float(cpu.get("factor") or 1.0)
    if x > SLOW_FACTOR:
        return t("cpu_slow_btn", lang), t("cpu_slow", lang, eta=eta or "?", x=f"{x:.0f}"), True
    return t("cpu", lang), (t("cpu_eta", lang, eta=eta) if eta else t("cpu_hint", lang)), False


def header_state(lock=None, lang: str = "ru", model_ok: bool = True, error: bool = False,
                 working: bool = False) -> tuple[str, str, str]:
    """Строка состояния в шапке: (состояние, текст, цвет). lock - HeavyLock этой программы (или None)."""
    from .. import heavy_lock as hl
    if error:
        st = "error"
    elif lock is not None and lock.now:
        return "alongside", t("st_alongside", lang, program=_program(lock.holder(), lang)), STATE_KIND["alongside"]
    elif working or (lock is not None and lock.held):
        st = "working"
    else:
        h = hl.status(lock.path) if lock is not None else None
        if h is not None or (lock is not None and lock.waiting):
            key = "busy_queued" if lock.waiting else "busy"
            return key, t("st_" + key, lang, program=_program(h, lang)), STATE_KIND[key]
        st = "ready" if model_ok else "nomodel"
    return st, t("st_" + st, lang), STATE_KIND[st]
