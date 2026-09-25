#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сборка Scribe в настоящий .app через py2app (alias-режим).

Зачем alias-режим: приложение НЕ копирует внутрь тяжёлые пакеты (mlx, torch,
numpy и т.д.), а ссылается на уже установленные в этом Python. Поэтому .app
получается лёгким и собирается быстро. Минус — работает только на этой машине
(ссылки на локальные пути). Для личного приложения это ровно то, что нужно.

Главное, ради чего всё затевалось: Python здесь стартует ВНУТРИ процесса .app
(нет «второго бинарника»), поэтому macOS корректно спрашивает доступ к микрофону.

Собирать Python-ом 3.12 (в 3.14 py2app пока не работает):

    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 setup.py py2app -A

Готовое приложение появится в dist/Scribe.app
"""
from setuptools import setup

APP = ["whisper_gui.py"]
DATA_FILES = []
OPTIONS = {
    "argv_emulation": False,          # на Apple Silicon включать нельзя, и это мешает нашему флагу воркера
    "iconfile": "scribe.icns",
    "includes": ["dictaphone", "dictaphone_worker"],
    "plist": {
        "CFBundleName": "Scribe",
        "CFBundleDisplayName": "Scribe",
        "CFBundleIdentifier": "com.andrey.scribe",
        "CFBundleShortVersionString": "2.0",
        "CFBundleVersion": "2.0",
        "NSMicrophoneUsageDescription":
            "Scribe использует микрофон для диктовки речи в текст.",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    },
}

setup(
    app=APP,
    name="Scribe",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
