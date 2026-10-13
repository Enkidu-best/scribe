#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Детектор речи Silero VAD (ONNX, ~2 МБ, лежит в assets/models, без интернета) -
один на всё приложение:

  • файлы: speech_probs + split_phrases - запись режется на фразы по паузам;
    длина фразы определяется записью, а не сеткой; слишком длинная фраза
    режется в самой тихой точке;
  • диктофон: StreamSegmenter - то же самое на лету, кадрами по 32 мс.

Модель: github.com/snakers4/silero-vad (MIT). Вход - 512 отсчётов 16 кГц плюс 64
отсчёта контекста из предыдущего кадра, состояние сети переносится между кадрами.
"""
from pathlib import Path

import numpy as np

SR = 16000
CHUNK = 512                     # 32 мс
CTX = 64
HOP = CHUNK / SR
MODEL_PATH = Path(__file__).with_name("assets") / "models" / "silero_vad.onnx"

THRESHOLD = 0.5                 # начало речи
NEG_THRESHOLD = 0.35            # конец речи (гистерезис)
MAX_PHRASE = 25.0               # длиннее - режем в самой тихой точке

_session = None


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        _session = ort.InferenceSession(str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


class _Stream:
    """Состояние модели для одного или нескольких параллельных потоков (batch)."""

    def __init__(self, batch=1):
        self.sess = _get_session()
        self.state = np.zeros((2, batch, 128), dtype=np.float32)
        self.ctx = np.zeros((batch, CTX), dtype=np.float32)
        self.sr = np.array(SR, dtype=np.int64)

    def step(self, chunks):
        """chunks: (batch, 512) -> вероятность речи (batch,)."""
        x = np.concatenate([self.ctx, chunks], axis=1).astype(np.float32)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.ctx = chunks[:, -CTX:]
        return out[:, 0]


def speech_probs(audio, lanes=16, warm=1.0):
    """Вероятность речи для каждого кадра 32 мс. Запись режется на `lanes` участков,
    которые идут одним пакетом (в разы быстрее, чем подряд); каждый участок
    начинается на `warm` секунд раньше, чтобы сеть успела «прогреться»."""
    n = len(audio) // CHUNK
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    a = audio[:n * CHUNK].reshape(n, CHUNK).astype(np.float32)
    lanes = max(1, min(lanes, n // 200 or 1))
    per = -(-n // lanes)
    w = int(warm / HOP)
    starts = [max(0, k * per - w) for k in range(lanes)]
    length = per + w
    st = _Stream(lanes)
    probs = np.zeros(n, dtype=np.float32)
    for t in range(length):
        idx = [s + t for s in starts]
        batch = np.stack([a[i] if i < n else np.zeros(CHUNK, np.float32) for i in idx])
        p = st.step(batch)
        for k, i in enumerate(idx):
            own_from = k * per
            if own_from <= i < min(n, (k + 1) * per):
                probs[i] = p[k]
    return probs


def quietest_point(audio, lo, hi, win=0.1):
    """Самая тихая точка (сек) на отрезке [lo, hi] - по RMS в окне 100 мс."""
    a, b = int(lo * SR), int(hi * SR)
    seg = audio[a:b].astype(np.float64)
    wl = max(1, int(win * SR))
    if len(seg) <= wl:
        return (lo + hi) / 2
    energy = np.convolve(seg ** 2, np.ones(wl) / wl, mode="valid")
    return lo + (int(np.argmin(energy)) + wl // 2) / SR


def split_phrases(audio, probs=None, min_silence=0.3, min_speech=0.25, pad=0.1, max_phrase=MAX_PHRASE):
    """Фразы [(start, end)] в секундах: речь между паузами от min_silence.
    Длиннее max_phrase - режем в самой тихой точке последних 10 с допустимого."""
    if probs is None:
        probs = speech_probs(audio)
    total = len(audio) / SR
    spans, on, start, quiet = [], False, 0.0, 0
    need_quiet = int(round(min_silence / HOP))
    for i, p in enumerate(probs):
        if not on:
            if p >= THRESHOLD:
                on, start, quiet = True, i * HOP, 0
        else:
            if p < NEG_THRESHOLD:
                quiet += 1
                if quiet >= need_quiet:
                    end = (i - quiet + 1) * HOP
                    if end - start >= min_speech:
                        spans.append([start, end])
                    on = False
            else:
                quiet = 0
    if on:
        spans.append([start, len(probs) * HOP])
    # поля по краям, не залезая в соседнюю фразу
    for k, s in enumerate(spans):
        prev_end = spans[k - 1][1] if k else 0.0
        nxt = spans[k + 1][0] if k + 1 < len(spans) else total
        s[0] = max(prev_end, s[0] - pad, 0.0)
        s[1] = min(nxt, s[1] + pad, total)
    out = []
    for a, b in spans:
        while b - a > max_phrase:
            cut = quietest_point(audio, a + max_phrase - 10.0, a + max_phrase)
            out.append((a, cut))
            a = cut
        out.append((a, b))
    return out


class StreamSegmenter:
    """Нарезка живого звука на фразы для диктофона.

    feed(кадр 512 отсчётов) -> None или dict(audio, speech) - законченная фраза
    и сколько в ней речи по детектору (сек). Фраза закрывается паузой от
    `silence` секунд; без пауз - режется в самой тихой точке последних 3 с, но не
    позже max_phrase. Начало речи не теряется: держим 0,3 с звука до неё."""

    def __init__(self, silence=0.5, min_speech=0.3, max_phrase=MAX_PHRASE, preroll=0.3, tail=3.0):
        self.silence, self.min_speech, self.max_phrase = silence, min_speech, max_phrase
        self.preroll_n = int(preroll / HOP)
        self.tail = tail
        self.reset()

    def reset(self):
        self._st = _Stream(1)
        self._pre = []
        self._buf = []                 # кадры текущей фразы
        self._probs = []
        self._on = False
        self._quiet = 0

    def _phrase(self, frames, probs):
        speech = sum(1 for p in probs if p >= THRESHOLD) * HOP
        if speech < self.min_speech:
            return None
        return dict(audio=np.concatenate(frames).astype(np.float32), speech=speech)

    def feed(self, frame):
        frame = np.asarray(frame, dtype=np.float32)
        if len(frame) != CHUNK:
            frame = np.pad(frame, (0, max(0, CHUNK - len(frame))))[:CHUNK]
        p = float(self._st.step(frame[None])[0])
        if not self._on:
            self._pre.append((frame, p))
            if len(self._pre) > self.preroll_n:
                self._pre.pop(0)
            if p >= THRESHOLD:
                self._on, self._quiet = True, 0
                self._buf = [f for f, _ in self._pre]
                self._probs = [q for _, q in self._pre]
                self._pre = []
            return None
        self._buf.append(frame)
        self._probs.append(p)
        self._quiet = self._quiet + 1 if p < NEG_THRESHOLD else 0
        if self._quiet * HOP >= self.silence:
            return self._close()
        if len(self._buf) * HOP >= self.max_phrase:
            return self._cut()
        return None

    def _close(self):
        out = self._phrase(self._buf, self._probs)
        self._buf, self._probs, self._on, self._quiet = [], [], False, 0
        return out

    def _cut(self):
        """Длинная речь без пауз: режем в самой тихой точке последних `tail` секунд,
        остаток становится началом следующей фразы."""
        audio = np.concatenate(self._buf)
        dur = len(audio) / SR
        cut = quietest_point(audio, max(0.0, dur - self.tail), dur)
        k = max(1, min(len(self._buf) - 1, int(round(cut / HOP))))
        out = self._phrase(self._buf[:k], self._probs[:k])
        self._buf, self._probs = self._buf[k:], self._probs[k:]
        self._quiet = 0
        return out

    def flush(self):
        if not self._on or not self._buf:
            return None
        return self._close()
