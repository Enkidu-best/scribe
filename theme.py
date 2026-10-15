#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оформление Scribe - в стиле Legal Knowledge Desk (web/static/app.css):
нейтральный фон, один акцент (тёмно-синий), тонкие границы, карточки,
светлая и тёмная тема, системный шрифт без засечек, иконки Lucide.

Цвета - пары (светлая, тёмная): customtkinter сам берёт нужную по теме.
Иконки лежат в assets/icons как чёрные PNG-маски (отрисованы из Lucide один
раз) и перекрашиваются здесь под цвет и тему.
"""
from pathlib import Path

import sys
import customtkinter as ctk

# --- токены (те же значения, что в LKD) ---
BG = ("#f5f6f8", "#0f1218")
SURFACE = ("#ffffff", "#171b23")
SURFACE2 = ("#f0f2f5", "#1d222c")
BORDER = ("#dfe3e8", "#2a313d")
BORDER_STRONG = ("#c9cfd7", "#3a4250")
TEXT = ("#1b2230", "#e5e9f0")
MUTED = ("#5f6b7a", "#a3adbd")
FAINT = ("#8a94a3", "#7d8796")
ACCENT = ("#1f3864", "#8fb0e8")
ACCENT_HOVER = ("#2b4a7e", "#a7c2ef")
ACCENT_SOFT = ("#e6ecf5", "#1e2a40")
ACCENT_TEXT = ("#ffffff", "#0f1218")
OK = ("#2e7d4f", "#6cc28f")
OK_SOFT = ("#e5f3ea", "#16271d")
WARN = ("#9a6200", "#e3b25a")
WARN_SOFT = ("#fbf1dc", "#2b2416")
BAD = ("#b3261e", "#f08b84")
BAD_SOFT = ("#fbe7e5", "#2e1a19")

RADIUS = 10
RADIUS_SM = 8
# системный шрифт: macOS (= -apple-system в LKD), Windows, Linux
FAMILY = {"darwin": ".AppleSystemUIFont", "win32": "Segoe UI"}.get(sys.platform, "DejaVu Sans")

ICONS = Path(__file__).with_name("assets") / "icons"


def font(size=14, bold=False):
    return ctk.CTkFont(family=FAMILY, size=size, weight="bold" if bold else "normal")


def apply():
    """Подменяем цвета встроенной темы customtkinter на токены LKD - тогда все
    стандартные виджеты сразу выглядят как надо."""
    ctk.set_default_color_theme("blue")
    t = ctk.ThemeManager.theme
    t["CTk"]["fg_color"] = list(BG)
    t["CTkToplevel"]["fg_color"] = list(BG)
    t["CTkFrame"].update(corner_radius=RADIUS, fg_color=list(SURFACE), top_fg_color=list(SURFACE2),
                         border_color=list(BORDER))
    t["CTkButton"].update(corner_radius=RADIUS_SM, border_width=1, fg_color=list(SURFACE),
                          hover_color=list(SURFACE2), border_color=list(BORDER_STRONG),
                          text_color=list(TEXT), text_color_disabled=list(FAINT))
    t["CTkLabel"]["text_color"] = list(TEXT)
    t["CTkEntry"].update(corner_radius=RADIUS_SM, border_width=1, fg_color=list(SURFACE),
                         border_color=list(BORDER_STRONG), text_color=list(TEXT),
                         placeholder_text_color=list(FAINT))
    t["CTkCheckBox"].update(corner_radius=5, border_width=2, fg_color=list(ACCENT),
                            hover_color=list(ACCENT_HOVER), border_color=list(BORDER_STRONG),
                            checkmark_color=list(ACCENT_TEXT), text_color=list(TEXT),
                            text_color_disabled=list(FAINT))
    t["CTkSwitch"].update(fg_color=list(BORDER_STRONG), progress_color=list(ACCENT),
                          button_color=list(SURFACE), button_hover_color=list(SURFACE2),
                          text_color=list(TEXT), text_color_disabled=list(FAINT))
    t["CTkProgressBar"].update(fg_color=list(SURFACE2), progress_color=list(ACCENT))
    t["CTkOptionMenu"].update(corner_radius=RADIUS_SM, fg_color=list(SURFACE2), button_color=list(SURFACE2),
                              button_hover_color=list(BORDER), text_color=list(TEXT),
                              text_color_disabled=list(FAINT))
    t["CTkScrollbar"].update(button_color=list(BORDER_STRONG), button_hover_color=list(FAINT))
    t["CTkTextbox"].update(corner_radius=RADIUS_SM, fg_color=list(SURFACE), border_color=list(BORDER),
                           text_color=list(TEXT), scrollbar_button_color=list(BORDER_STRONG),
                           scrollbar_button_hover_color=list(FAINT))
    t["CTkScrollableFrame"]["label_fg_color"] = list(SURFACE2)
    t["DropdownMenu"].update(fg_color=list(SURFACE), hover_color=list(SURFACE2), text_color=list(TEXT))
    t["CTkFont"].update(family=FAMILY, size=14)


# --- иконки ---
_icon_cache = {}


def icon(name, color=MUTED, size=16):
    key = (name, color, size)
    if key in _icon_cache:
        return _icon_cache[key]
    from PIL import Image
    try:
        mask = Image.open(ICONS / f"{name}.png").getchannel("A")
    except Exception:
        return None

    def tint(c):
        im = Image.new("RGBA", mask.size, c)
        im.putalpha(mask)
        return im

    img = ctk.CTkImage(light_image=tint(color[0]), dark_image=tint(color[1]), size=(size, size))
    _icon_cache[key] = img
    return img


# --- элементы ---
_BTN = {
    "default": dict(fg_color=SURFACE, hover_color=SURFACE2, border_color=BORDER_STRONG, border_width=1,
                    text_color=TEXT, ic=MUTED),
    "primary": dict(fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=ACCENT, border_width=1,
                    text_color=ACCENT_TEXT, ic=ACCENT_TEXT),
    "ghost": dict(fg_color="transparent", hover_color=SURFACE2, border_color=SURFACE, border_width=0,
                  text_color=TEXT, ic=MUTED),
    "danger": dict(fg_color=SURFACE, hover_color=BAD_SOFT, border_color=BORDER_STRONG, border_width=1,
                   text_color=BAD, ic=BAD),
}


def button(master, text="", icon_name=None, kind="default", size=14, height=34, **kw):
    st = dict(_BTN[kind])
    ic = st.pop("ic")
    img = icon(icon_name, ic, 16) if icon_name else None
    opts = dict(text=text, image=img, compound="left", height=height, corner_radius=RADIUS_SM,
                font=font(size, bold=(kind == "primary")), **st)
    if not text:
        opts["width"] = height
    opts.update(kw)
    return ctk.CTkButton(master, **opts)


def restyle(btn, kind, icon_name=None):
    """Сменить вид кнопки на лету (например, «Расшифровать» ↔ «Остановить»)."""
    st = dict(_BTN[kind])
    ic = st.pop("ic")
    st.pop("border_width", None)
    if icon_name:
        st["image"] = icon(icon_name, ic, 16)
    btn.configure(**st)


def card(master, **kw):
    opts = dict(fg_color=SURFACE, border_width=1, border_color=BORDER, corner_radius=RADIUS)
    opts.update(kw)
    return ctk.CTkFrame(master, **opts)


def label(master, text="", size=14, bold=False, color=TEXT, **kw):
    return ctk.CTkLabel(master, text=text, font=font(size, bold), text_color=color, **kw)


def section(master, text):
    """Подзаголовок блока: как h4 в LKD - приглушённый, полужирный."""
    return label(master, text, size=12, bold=True, color=MUTED, anchor="w")


_CHIP = {"neutral": (SURFACE2, MUTED), "acc": (ACCENT_SOFT, ACCENT), "ok": (OK_SOFT, OK),
         "warn": (WARN_SOFT, WARN), "bad": (BAD_SOFT, BAD)}


def chip(master, text="", kind="neutral"):
    bg, fg = _CHIP[kind]
    return ctk.CTkLabel(master, text=text, fg_color=bg, text_color=fg, corner_radius=11, height=22,
                        font=font(12), padx=9)


def set_chip(ch, text, kind):
    bg, fg = _CHIP[kind]
    ch.configure(text=text, fg_color=bg, text_color=fg)


class Segmented(ctk.CTkFrame):
    """Переключатель-сегменты как .seg в LKD: серая подложка, выбранный - акцентом."""

    def __init__(self, master, values, command=None, size=13):
        super().__init__(master, fg_color=SURFACE2, border_width=1, border_color=BORDER,
                         corner_radius=RADIUS_SM)
        self._command = command
        self._value = values[0]
        self._btns = {}
        for i, v in enumerate(values):
            b = ctk.CTkButton(self, text=v, width=20, height=28, corner_radius=6, border_width=0, font=font(size),
                              command=lambda v=v: self._click(v))
            b.grid(row=0, column=i, sticky="ew", padx=(3 if i == 0 else 1, 3 if i == len(values) - 1 else 1),
                   pady=3)
            self.grid_columnconfigure(i, weight=1)
            self._btns[v] = b
        self._paint()

    def _click(self, v):
        if str(self._btns[v].cget("state")) == "disabled":
            return
        self.set(v)
        if self._command:
            self._command(v)

    def _paint(self):
        for v, b in self._btns.items():
            if v == self._value:
                b.configure(fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=ACCENT_TEXT)
            else:
                b.configure(fg_color=SURFACE2, hover_color=BORDER, text_color=MUTED)

    def get(self):
        return self._value

    def set(self, v):
        if v in self._btns:
            self._value = v
            self._paint()

    def set_state(self, state):
        for b in self._btns.values():
            b.configure(state=state)
