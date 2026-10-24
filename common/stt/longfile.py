# -*- coding: utf-8 -*-
"""Длинный файл - режим «файл» общего блока распознавания (стандарт 7.1, README).

По практике WhisperX (Bain и др., Interspeech 2023): запись режется по паузам (VAD) на куски до 30 с («cut &
merge»), куски распознаются пачкой, связь «текст прошлого куска → следующий» не используется. Готовые куски
сразу пишутся на диск (`Store`) - после сбоя или остановки распознавание продолжается с первого не готового куска.

Движок подключается функцией `decode(clips) -> [Piece]` (MLX - `mlx_decoder`, faster-whisper - `fw_decoder`):
    Piece = {"segments": [{"start", "end", "text"}] - время от начала куска,
             "rest": секунда от начала куска, с которой текст не дописан (упёрся в предел токенов) или None,
             "repeat": True - кусок зациклился (сжатие текста > 2,4), "silent": True - в куске нет речи}
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import deque
from pathlib import Path

SR = 16000
CHUNK = 30.0          # кусок - не длиннее окна Whisper
REST_MIN = 0.5        # недописанный хвост короче - не распознаётся заново
REPEAT_RATIO = 2.4    # сжатие текста выше - зациклился (как compression_ratio_threshold в Whisper)
FALLBACK_T = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class Cancelled(Exception):
    pass


def merge_chunks(spans, limit=CHUNK):
    """Фразы VAD [(начало, конец) в секундах] - в куски до limit секунд; граница куска всегда в паузе.
    Фраза длиннее limit (VAD сам режет фразы до 25 с) - отдельный кусок."""
    out = []
    for a, b in spans:
        if out and b - out[-1][0] <= limit:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(round(a, 3), round(b, 3)) for a, b in out]


def file_key(path, **params) -> dict:
    """Что делает продолжение законным: тот же файл (путь, размер, время изменения) и те же настройки."""
    st = Path(path).stat()
    return {"file": str(Path(path).resolve()), "size": st.st_size, "mtime": int(st.st_mtime), **params}


class Store:
    """Готовые куски одного файла на диске: `<root>/<ключ>/meta.json` (ключ и куски) и `done.jsonl` (строка на
    готовый кусок). Другой ключ или другие куски - начать заново."""

    def __init__(self, root, key: dict):
        self.key = key
        digest = hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        self.dir = Path(root) / digest
        self.n = 0
        self.done: dict[int, list] = {}

    def open(self, chunks) -> dict[int, list]:
        meta = {"key": self.key, "chunks": [list(c) for c in chunks]}
        self.n = len(chunks)
        self.done = {}
        try:
            old = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if old != meta:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            return self.done
        try:
            for line in (self.dir / "done.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:          # строка, оборванная сбоем, - кусок не готов
                    continue
                self.done[int(rec["i"])] = rec["segments"]
        except OSError:
            pass
        return self.done

    def add(self, i: int, segments: list):
        self.done[i] = segments
        with open(self.dir / "done.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"i": i, "segments": segments}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def share(self) -> float:
        return len(self.done) / self.n if self.n else 0.0

    def clear(self):
        shutil.rmtree(self.dir, ignore_errors=True)


def saved_share(root, key: dict) -> float | None:
    """Сколько уже готово у прерванной расшифровки (0..1) - для кнопки «Продолжить (готово 64 %)»; None - нечего
    продолжать."""
    st = Store(root, key)
    try:
        meta = json.loads((st.dir / "meta.json").read_text(encoding="utf-8"))
        n = len(meta["chunks"])
        lines = (st.dir / "done.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError, KeyError):
        return None
    done = set()
    for line in lines:
        try:
            done.add(json.loads(line)["i"])
        except (ValueError, KeyError):
            pass
    return len(done) / n if n and done and meta.get("key") == key else None


def prune(root, days=30):
    """Прерванные и забытые расшифровки старше days дней - убрать (служебные файлы не копятся)."""
    import time
    cutoff = time.time() - days * 86400
    try:
        dirs = [d for d in Path(root).iterdir() if d.is_dir()]
    except OSError:
        return
    for d in dirs:
        try:
            newest = max(f.stat().st_mtime for f in [d, *d.iterdir()])
        except (OSError, ValueError):
            continue
        if newest < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def _shift(segments, x, b):
    out = []
    for s in segments:
        text = s.get("text") or ""
        if not text.strip():
            continue
        out.append({"start": round(x + float(s.get("start", 0.0)), 3),
                    "end": round(min(b, x + float(s.get("end", 0.0))), 3), "text": text})
    return out


def transcribe(audio, spans, decode, *, fallback=None, sr=SR, limit=CHUNK, batch=16, store=None,
               prog=None, cancel=None, on_chunk=None):
    """Распознать запись кусками по паузам, пачками по batch кусков. -> сегменты [{start, end, text}] по порядку.

    spans - фразы VAD в секундах; fallback(clip) - обычный вызов движка с запасными температурами для куска,
    который зациклился; store - `Store` (продолжение); prog(доля готовых кусков); cancel() -> True - остановить
    (готовое остаётся в store); on_chunk(готово, всего) - после каждой пачки."""
    chunks = merge_chunks(spans, limit)
    done = dict(store.open(chunks)) if store is not None else {}
    total = len(chunks)
    pending = deque((i, x, b, b) for i, (x, b) in enumerate(chunks) if i not in done)
    partial: dict[int, list] = {}
    if prog:
        prog(len(done) / total if total else 1.0)
    while pending:
        if cancel and cancel():
            raise Cancelled()
        group = [pending.popleft() for _ in range(min(batch, len(pending)))]
        pieces = decode([audio[int(x * sr):int(e * sr)] for _, x, e, _ in group])
        again = []
        for (i, x, e, b), p in zip(group, pieces):
            segs = partial.setdefault(i, [])
            rest = None
            if p.get("repeat") and fallback is not None:
                segs += _shift(fallback(audio[int(x * sr):int(e * sr)]), x, e)
            elif not p.get("silent"):
                segs += _shift(p.get("segments") or [], x, e)
                rest = p.get("rest")
            if rest is not None and e - (x + rest) > REST_MIN:
                again.append((i, round(x + rest, 3), e, b))      # хвост куска - в следующую пачку
                continue
            done[i] = partial.pop(i)
            if store is not None:
                store.add(i, done[i])
        pending.extendleft(reversed(again))
        if prog:
            prog(len(done) / total)
        if on_chunk:
            on_chunk(len(done), total)
    return [s for i in sorted(done) for s in done[i]]


# --- движки -------------------------------------------------------------------------------------------------

def _timestamp_segments(tokens, tb, eot, decode_text, prec=0.02):
    """Токены Whisper с отметками времени -> (сегменты, rest). Так же, как mlx_whisper / openai-whisper режут
    ответ модели: пара отметок подряд - граница сегмента; ответ не закончился одиночной отметкой - хвост после
    последней пары не дописан (rest - его начало)."""
    toks = [t for t in tokens if t != eot]
    cut = [k + 1 for k in range(len(toks) - 1) if toks[k] >= tb and toks[k + 1] >= tb]
    finished = len(toks) >= 2 and toks[-1] >= tb and toks[-2] < tb
    if finished:
        cut.append(len(toks))
    if not cut:                                 # отметок нет - весь кусок одним сегментом
        text = decode_text([t for t in toks if t < tb])
        return ([{"start": 0.0, "end": None, "text": text}] if text.strip() else []), None
    segs, last = [], 0
    for c in cut:
        piece = toks[last:c]
        text = decode_text([t for t in piece if t < tb])
        if text.strip():
            segs.append({"start": (piece[0] - tb) * prec, "end": (piece[-1] - tb) * prec, "text": text})
        last = c
    rest = None if finished else (toks[last - 1] - tb) * prec
    return segs, rest


def mlx_decoder(repo, lang, prompt=None):
    """Пачка кусков на MLX: кодер и декодер mlx_whisper - один вызов на пачку (как `DecodingTask` для n_audio > 1),
    температура 0. Вызывать в нити MLX (стандарт: одна нить MLX)."""
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions
    from mlx_whisper.tokenizer import get_tokenizer
    from mlx_whisper.transcribe import ModelHolder

    model = ModelHolder.get_model(repo, mx.float16)
    tok = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language=lang, task="transcribe")
    opts = DecodingOptions(language=lang, task="transcribe", temperature=0.0, fp16=True,
                           prompt=tok.encode(" " + prompt.strip()) if prompt else None)

    def decode(clips):
        mel = mx.stack([pad_or_trim(log_mel_spectrogram(c, n_mels=model.dims.n_mels, padding=N_SAMPLES), N_FRAMES,
                                    axis=-2) for c in clips]).astype(mx.float16)
        out = []
        for c, r in zip(clips, model.decode(mel, opts)):
            if r.no_speech_prob > 0.6 and r.avg_logprob < -1.0:
                out.append({"silent": True})
                continue
            if r.compression_ratio > REPEAT_RATIO:
                out.append({"repeat": True})
                continue
            segs, rest = _timestamp_segments(list(r.tokens), tok.timestamp_begin, tok.eot, tok.decode)
            for s in segs:
                if s["end"] is None:
                    s["end"] = len(c) / SR
            out.append({"segments": segs, "rest": rest})
        return out
    return decode


def fw_decoder(whisper_model, lang, prompt=None):
    """Пачка кусков на faster-whisper: `BatchedInferencePipeline` с нашими кусками (`clip_timestamps`) - его VAD не
    нужен. Недописанный хвост faster-whisper в пачке не возвращает (rest = None)."""
    import numpy as np
    import faster_whisper
    from faster_whisper import BatchedInferencePipeline

    pipe = BatchedInferencePipeline(model=whisper_model)
    # границы кусков: faster-whisper 1.2+ - в секундах, 1.1 - в отсчётах звука
    unit = 1 / SR if tuple(int(x) for x in faster_whisper.__version__.split(".")[:2]) >= (1, 2) else 1

    def decode(clips):
        offs, pos = [], 0
        for c in clips:
            offs.append(pos)
            pos += len(c)
        audio = np.concatenate(clips).astype(np.float32) if clips else np.zeros(0, np.float32)
        ts = [{"start": o * unit, "end": (o + len(c)) * unit} for o, c in zip(offs, clips)]
        segs, _ = pipe.transcribe(audio, language=lang, clip_timestamps=ts, batch_size=len(clips), beam_size=1,
                                  temperature=0.0, initial_prompt=prompt, condition_on_previous_text=False,
                                  vad_filter=False, without_timestamps=False)
        out = [{"segments": []} for _ in clips]
        for s in segs:
            k = max(j for j, o in enumerate(offs) if o / SR <= s.start + 1e-3)
            if s.compression_ratio > REPEAT_RATIO:
                out[k] = {"repeat": True}
            elif "repeat" not in out[k] and not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0):
                out[k]["segments"].append({"start": s.start - offs[k] / SR, "end": s.end - offs[k] / SR, "text": s.text})
        return out
    return decode
