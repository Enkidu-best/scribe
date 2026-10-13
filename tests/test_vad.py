#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автотесты нарезки на фразы (vad.py) - для диктофона и режима «Несколько».

Речь синтезируется системным `say` (macOS), поэтому тесты самодостаточны.
Запуск:  python3 tests/test_vad.py      (или pytest tests/test_vad.py)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vad  # noqa: E402

SR = vad.SR
_rng = np.random.default_rng(7)


def speech(text, voice="Milena", rate=180):
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(f), text], check=True)
        raw = subprocess.run(["ffmpeg", "-nostdin", "-i", str(f), "-ar", str(SR), "-ac", "1", "-f", "f32le",
                              "-loglevel", "error", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32).copy()
    idx = np.where(np.abs(a) > np.abs(a).max() * 0.02)[0]
    return a[idx[0]:idx[-1]] * 0.5


def silence(sec):
    return (_rng.standard_normal(int(sec * SR)) * 1e-3).astype(np.float32)   # тихий фон, не ноль


def stream_phrases(audio):
    """Прогнать запись через диктофонную нарезку кадрами по 32 мс."""
    seg = vad.StreamSegmenter()
    out = []
    for k in range(0, len(audio) - vad.CHUNK + 1, vad.CHUNK):
        ph = seg.feed(audio[k:k + vad.CHUNK])
        if ph is not None:
            out.append(ph)
    tail = seg.flush()
    if tail is not None:
        out.append(tail)
    return out


SENTENCES = ["Добрый день, начинаем запись.", "Сегодня обсудим три вопроса.",
             "Первый вопрос касается договора аренды.", "На этом всё, спасибо."]


def test_pauses_give_phrases():
    """Четыре предложения, между ними паузы по 1 с -> четыре фразы (и в файле, и в потоке)."""
    parts = [silence(0.8)]
    for t in SENTENCES:
        parts += [speech(t), silence(1.0)]
    a = np.concatenate(parts)
    assert len(vad.split_phrases(a)) == 4
    ph = stream_phrases(a)
    assert len(ph) == 4, [len(p["audio"]) / SR for p in ph]
    assert all(p["speech"] > 0.5 for p in ph)


def test_short_pause_inside_phrase():
    """Пауза 0,2 с внутри фразы диктофона не рвёт её (порог 0,5 с)."""
    a = np.concatenate([silence(0.5), speech("Первая часть фразы"), silence(0.2),
                        speech("и её продолжение"), silence(1.0)])
    assert len(stream_phrases(a)) == 1


def test_long_speech_cut_in_quiet_place():
    """Сплошная речь ~45 с без пауз дольше 0,5 с -> фразы не длиннее 25 с,
    разрез - в тихом месте, а не посреди слова."""
    words = ("мы продолжаем говорить без остановки чтобы проверить как диктофон режет длинную речь "
             "в самом тихом месте а не посреди слова").split()
    parts = [silence(0.5)]
    while sum(len(p) for p in parts) < 45 * SR:
        for w in words:
            parts += [speech(w, rate=200), silence(0.12)]
    a = np.concatenate(parts)
    ph = stream_phrases(a)
    assert len(ph) >= 2
    lens = [len(p["audio"]) / SR for p in ph]
    assert max(lens) <= vad.MAX_PHRASE + 0.5, lens
    # энергия в точке разреза - ниже медианной энергии речи
    pos = 0
    win = int(0.05 * SR)
    energy = np.convolve(a.astype(np.float64) ** 2, np.ones(win) / win, mode="same")
    speech_med = np.median(energy[energy > np.percentile(energy, 30)])
    for p in ph[:-1]:
        # найти конец фразы в исходной записи по совпадению хвоста
        tail = p["audio"][-vad.CHUNK:]
        i = pos + int(np.argmin([np.abs(a[j:j + vad.CHUNK] - tail).sum()
                                 for j in range(pos, len(a) - vad.CHUNK, vad.CHUNK)])) * vad.CHUNK + vad.CHUNK
        assert energy[min(i, len(a) - 1)] < speech_med, "разрез посреди слова"
        pos = i


def test_file_split_long_phrase():
    """В файле фраза длиннее 25 с режется (режим «Несколько»)."""
    parts = [silence(0.5)]
    while sum(len(p) for p in parts) < 40 * SR:
        parts += [speech("сплошная речь без длинных пауз", rate=200), silence(0.1)]
    a = np.concatenate(parts + [silence(0.5)])
    ph = vad.split_phrases(a)
    assert max(b - x for x, b in ph) <= vad.MAX_PHRASE + 0.01
    assert len(ph) >= 2


if __name__ == "__main__":
    ok = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
            print("OK  ", t.__name__)
            ok += 1
        except AssertionError as e:
            print("FAIL", t.__name__, e)
    print(f"{ok} из {len(tests)}")
    sys.exit(0 if ok == len(tests) else 1)
