#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Синтез речи для тестов на любой системе -> float32, 16 кГц, моно.

  Mac      - say (Milena - ru, Samantha - en)
  Linux    - espeak-ng (ru, en)
  Windows  - System.Speech (встроенный голос; на чистой Windows обычно только en)
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SR = 16000
MAC_VOICES = {"ru": "Milena", "en": "Samantha"}


def _decode(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-i", str(path), "-ar", str(SR), "-ac", "1", "-f", "f32le",
                          "-loglevel", "error", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32).copy()
    if a.size == 0:
        return a
    idx = np.where(np.abs(a) > np.abs(a).max() * 0.02)[0]
    return a[idx[0]:idx[-1]] * 0.5 / max(1e-6, float(np.abs(a).max()))


def _win_voices():
    ps = ("Add-Type -AssemblyName System.Speech; "
          "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
          "ForEach-Object { $_.VoiceInfo.Culture.TwoLetterISOLanguageName + '|' + $_.VoiceInfo.Name }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        if "|" in line:
            lang, name = line.strip().split("|", 1)
            out.setdefault(lang, name)
    return out


def available():
    """Языки, на которых можно синтезировать речь на этой машине."""
    if sys.platform == "darwin":
        return ["ru", "en"]
    if sys.platform.startswith("win"):
        return [l for l in ("ru", "en") if l in _win_voices()]
    return ["ru", "en"] if shutil.which("espeak-ng") else []


def speech(text, lang="ru", rate=None):
    with tempfile.TemporaryDirectory() as td:
        if sys.platform == "darwin":
            f = Path(td) / "x.aiff"
            cmd = ["say", "-v", MAC_VOICES[lang], "-o", str(f)]
            if rate:
                cmd += ["-r", str(rate)]
            subprocess.run(cmd + [text], check=True)
        elif sys.platform.startswith("win"):
            f = Path(td) / "x.wav"
            tf = Path(td) / "t.txt"
            tf.write_text(text, encoding="utf-8")
            voice = _win_voices().get(lang)
            ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  + (f"$s.SelectVoice('{voice}'); " if voice else "")
                  + (f"$s.Rate = {int((rate - 180) / 20)}; " if rate else "")
                  + f"$s.SetOutputToWaveFile('{f}'); $s.Speak([IO.File]::ReadAllText('{tf}')); $s.Dispose()")
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
        else:
            f = Path(td) / "x.wav"
            cmd = ["espeak-ng", "-v", lang, "-w", str(f)]
            if rate:
                cmd += ["-s", str(int(rate * 0.9))]
            subprocess.run(cmd + [text], check=True)
        return _decode(f)
