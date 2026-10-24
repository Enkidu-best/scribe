# -*- coding: utf-8 -*-
"""Окно «Видеокарта занята» для программ на customtkinter (Herald, Scribe, Anonymizer) - README, 6.3.

ask_busy(root, T, holder, lang, cpu=False, queue_names=()) → "wait" / "now" / "cpu" / None. cpu: True - кнопка
«На процессоре»; {"seconds": …, "factor": …} - с прогнозом, медленнее видеокарты больше чем в 3 раза - «очень медленно».
T - модуль темы программы (theme.py: label, button, цвета MUTED / WARN / FAINT): окно выглядит как
остальная программа. Модальное, по центру окна программы; Esc и крестик - «Отмена», Enter - «Подождать очереди».
"""
from __future__ import annotations

from .. import heavy_lock as hl
from .texts import cpu_option, describe, t


def ask_busy(root, T, holder, lang: str = "ru", cpu: bool = False, queue_names=(), on_open=None) -> str | None:
    import customtkinter as ctk

    dlg = ctk.CTkToplevel(root)
    dlg.title(t("title", lang))
    dlg.resizable(False, False)
    dlg.transient(root)
    res = {"ans": None}
    wrap = ctk.CTkFrame(dlg, fg_color="transparent")
    wrap.pack(fill="both", expand=True, padx=24, pady=20)
    # заголовок - сама строка «Видеокарта занята: «…», с 16:31, …» (в строке окна - «Видеокарта занята»)
    T.label(wrap, describe(holder, lang), 16, bold=True, anchor="w", justify="left", wraplength=500).pack(fill="x")
    if holder is not None and getattr(holder, "background", False):
        T.label(wrap, t("busy_bg", lang), 13, color=T.MUTED, anchor="w", justify="left",
                wraplength=500).pack(fill="x", pady=(4, 0))
    if queue_names:
        T.label(wrap, t("queue_ahead", lang, names=", ".join(f"«{n}»" for n in queue_names)), 13, color=T.MUTED,
                anchor="w", justify="left", wraplength=500).pack(fill="x", pady=(4, 0))

    def close(ans):
        res["ans"] = ans
        dlg.destroy()

    opts = ctk.CTkFrame(wrap, fg_color="transparent")
    opts.pack(fill="x", pady=(16, 0))
    opts.grid_columnconfigure(1, weight=1)
    rows = [("wait", "primary", "pause", t("wait", lang), t("wait_hint", lang), T.MUTED),
            ("now", "default", "play", t("now", lang), t("now_hint", lang), T.WARN)]
    co = cpu_option(cpu, lang)        # cpu: True или {"seconds", "factor"} - прогноз по замеру программы (доп. 21:40)
    if co:
        label, hint, slow = co
        rows.append(("cpu", "ghost" if slow else "default", "cpu", label, hint, T.WARN if slow else T.MUTED))
    buttons = {}
    for r, (key, kind, ic, label, hint, color) in enumerate(rows):
        b = T.button(opts, label, ic if _has_icon(T, ic) else None, kind=kind, width=250, height=38,
                     command=lambda k=key: close(k))
        if kind == "ghost":
            b.configure(border_width=1, border_color=T.BORDER_STRONG)
        b.grid(row=r, column=0, sticky="w", pady=4)
        T.label(opts, hint, 12, color=color, anchor="w", justify="left", wraplength=280
                ).grid(row=r, column=1, sticky="w", padx=(14, 0))
        buttons[key] = b
    foot = ctk.CTkFrame(wrap, fg_color="transparent")
    foot.pack(fill="x", pady=(14, 0))
    buttons["cancel"] = T.button(foot, t("cancel", lang), kind="ghost", width=110, command=lambda: close(None))
    buttons["cancel"].pack(side="right")
    if hl.page_tools() is not None:           # страница «Видеокарта» (README, 12) - окно остаётся открытым
        buttons["page"] = T.button(foot, t("open_page", lang), kind="ghost", command=hl.open_page)
        buttons["page"].pack(side="left")

    dlg.bind("<Return>", lambda e: close("wait"))
    dlg.bind("<Escape>", lambda e: close(None))
    dlg.protocol("WM_DELETE_WINDOW", lambda: close(None))
    dlg.update_idletasks()                       # по центру окна программы, чуть выше середины
    w, h = dlg.winfo_reqwidth(), dlg.winfo_reqheight()
    x = root.winfo_rootx() + (root.winfo_width() - w) // 2
    y = root.winfo_rooty() + (root.winfo_height() - h) // 3
    dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
    dlg.after(50, lambda: (dlg.lift(), dlg.focus_force(), dlg.grab_set()))
    dlg.buttons = buttons                        # для проверок: нажать кнопку без мыши
    if on_open is not None:
        dlg.after(80, lambda: on_open(dlg))
    root.wait_window(dlg)
    return res["ans"]


def _has_icon(T, name: str) -> bool:
    try:
        return (T.ICONS / f"{name}.png").exists()
    except Exception:
        return False
