# -*- mode: python ; coding: utf-8 -*-
# Сборка Scribe для Windows (PyInstaller, режим «папка»): Scribe.exe (окно) и scribe-cli.exe (консоль)
# в одной папке с общими библиотеками. Собирается на GitHub Actions: .github/workflows/release.yml
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
datas, binaries, hidden = [], [], []
for pkg in ("customtkinter", "tkinterdnd2", "faster_whisper", "ctranslate2", "onnxruntime",
            "_sounddevice_data", "docx", "tokenizers", "av",
            "pyannote", "lightning", "lightning_fabric", "pytorch_lightning", "asteroid_filterbanks",
            "pytorch_metric_learning", "torch_audiomentations", "torchmetrics", "torchaudio"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hidden += h
    except Exception as e:
        print("пропущено:", pkg, e)
for dist in ("pyannote.audio", "pyannote.core", "pyannote.pipeline", "pyannote.database", "pyannote.metrics",
             "lightning", "pytorch-lightning", "lightning-utilities", "torch", "torchaudio", "torchmetrics",
             "huggingface-hub", "tokenizers", "faster-whisper", "ctranslate2", "customtkinter", "safetensors",
             "einops", "asteroid-filterbanks", "pytorch-metric-learning", "torch-audiomentations", "rich",
             "opentelemetry-api", "opentelemetry-sdk", "numpy", "onnxruntime", "av", "python-docx"):
    try:
        datas += copy_metadata(dist)
    except Exception as e:
        print("без метаданных:", dist, e)
datas += [(os.path.join(ROOT, "assets"), "assets"), (os.path.join(ROOT, "scribe.png"), ".")]
hidden += ["dictaphone", "dictaphone_worker", "engine", "multilang", "vad", "summary", "theme", "models",
           "core", "platform_util", "scribe_cli"]
excludes = ["mlx", "mlx_whisper", "torchcodec", "IPython", "jupyter", "notebook", "pytest"]

common = dict(pathex=[ROOT], binaries=binaries, datas=datas, hiddenimports=hidden, excludes=excludes,
              hookspath=[], runtime_hooks=[], noarchive=False)
a_gui = Analysis([os.path.join(ROOT, "whisper_gui.py")], **common)
a_cli = Analysis([os.path.join(ROOT, "scribe_cli.py")], **common)
pyz_gui = PYZ(a_gui.pure)
pyz_cli = PYZ(a_cli.pure)
icon = os.path.join(SPECPATH, "scribe.ico")
exe_gui = EXE(pyz_gui, a_gui.scripts, [], exclude_binaries=True, name="Scribe", console=False,
              icon=icon, upx=False)
exe_cli = EXE(pyz_cli, a_cli.scripts, [], exclude_binaries=True, name="scribe-cli", console=True,
              icon=icon, upx=False)
coll = COLLECT(exe_gui, exe_cli, a_gui.binaries, a_gui.datas, a_cli.binaries, a_cli.datas,
               strip=False, upx=False, name="Scribe")
