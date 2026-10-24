"""Задание 06, Б6 - проверка до внедрения: длинный файл одним проходом (как сейчас) против нарезки по паузам
(WhisperX «cut & merge»: Silero VAD, куски до 30 с строго по паузам, связь между кусками не используется).

    HEAVY_LOCK_WHO=background python tests/ab_long.py <файл> --out <папка вне проекта> [--lang ru] [--model turbo]

Меряет: время каждого способа (звук готовится один раз - он общий), долю совпадающих слов между способами, повторы
(одна и та же фраза 3+ раза подряд - типичная выдумка Whisper на тишине), слова на участках без речи (по VAD),
пиковую память MLX. Тексты обоих способов - в --out (личные записи - только локально, не в git). Под настоящей
очередью видеокарты, фоном.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core  # noqa: E402
import engine  # noqa: E402
import vad  # noqa: E402
from common.lock import heavy_lock as hl  # noqa: E402

CHUNK = 30.0


def chunks(spans, limit=CHUNK):
    """Фразы VAD (до 25 с) склеиваются в куски до limit секунд - граница всегда в паузе."""
    out = []
    for a, b in spans:
        if out and b - out[-1][0] <= limit:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def batched(eng, audio, parts, model, lang, batch=16):
    """В - как WhisperX на MLX: куски по паузам дополняются до 30 с и распознаются пачкой (кодер и декодер - один вызов
    на пачку, температура 0 - как сейчас в Scribe). Куски с повторами (сжатие > 2,4) - заново по одному обычным
    вызовом; тишина (no_speech > 0,6 при низкой уверенности) - пропуск, как в mlx_whisper.transcribe."""
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions
    from mlx_whisper.tokenizer import get_tokenizer
    from mlx_whisper.transcribe import ModelHolder
    m = ModelHolder.get_model(eng.repo(model), mx.float16)
    tok = get_tokenizer(m.is_multilingual, num_languages=m.num_languages, language=lang, task="transcribe")
    opts = DecodingOptions(language=lang, task="transcribe", temperature=0.0, fp16=True)
    tb, prec, out, again, rest = tok.timestamp_begin, 0.02, [], 0, 0
    todo = [tuple(p) for p in parts]
    while todo:
        group, todo, nxt = todo[:batch], todo[batch:], []
        mel = mx.stack([pad_or_trim(log_mel_spectrogram(audio[int(x * engine.SR):int(b * engine.SR)], n_mels=m.dims.n_mels,
                                                        padding=N_SAMPLES), N_FRAMES, axis=-2) for x, b in group]).astype(mx.float16)
        for (x, b), r in zip(group, m.decode(mel, opts)):
            if r.no_speech_prob > 0.6 and r.avg_logprob < -1.0:
                continue
            if r.compression_ratio > 2.4:
                again += 1
                for s in eng.transcribe(audio[int(x * engine.SR):int(b * engine.SR)], model, lang):
                    out.append({**s, "start": s.get("start", 0) + x, "end": s.get("end", 0) + x})
                continue
            toks = list(r.tokens)
            cut = [k + 1 for k in range(len(toks) - 1) if toks[k] >= tb and toks[k + 1] >= tb]
            finished = len(toks) >= 2 and toks[-1] >= tb and toks[-2] < tb
            if finished:
                cut.append(len(toks))
            if not cut:
                out.append({"start": x, "end": b, "text": tok.decode([t for t in toks if t < tok.eot])})
                continue
            last = 0
            for c in cut:
                piece = toks[last:c]
                text = tok.decode([t for t in piece if t < tb])
                if text.strip():
                    out.append({"start": x + (piece[0] - tb) * prec, "end": min(b, x + (piece[-1] - tb) * prec), "text": text})
                last = c
            # не договорил (упёрся в предел токенов на кусок): остаток куска - в следующую пачку, как делает
            # обычный проход (seek к последней отметке времени)
            if not finished:
                end_ts = x + (toks[last - 1] - tb) * prec
                if b - end_ts > 0.5:
                    nxt.append((end_ts, b))
                    rest += 1
        todo = nxt + todo
    return out, again, rest


def words(segs):
    return [w for s in segs for w in re.findall(r"\w+", (s.get("text") or "").lower())]


def repeats(segs):
    t = [re.sub(r"\W+", " ", (s.get("text") or "").lower()).strip() for s in segs]
    return sum(1 for i in range(2, len(t)) if t[i] and t[i] == t[i - 1] == t[i - 2])


def outside_speech(segs, spans):
    """Слова в сегментах, которые целиком лежат вне речи по VAD."""
    n = 0
    for s in segs:
        a, b = s.get("start", 0), s.get("end", 0)
        if not any(min(b, y) - max(a, x) > 0.2 for x, y in spans):
            n += len(re.findall(r"\w+", s.get("text") or ""))
    return n


def lost(segs, spans, gap=10.0):
    """Речь (по VAD), попавшая в провалы текста длиннее gap секунд: Whisper одним проходом иногда пропускает окно
    в 30 с целиком («Вот так.» на 30 секунд разговора)."""
    def dense(s):           # «Вот так.» на 30 с не покрывает эти 30 с
        d = s["end"] - s["start"]
        return d <= gap or len(re.findall(r"\w+", s.get("text") or "")) / d >= 0.5
    covered = sorted((s["start"], s["end"]) for s in segs if re.search(r"\w{2}", s.get("text") or "") and dense(s))
    holes, prev = [], 0.0
    for a, b in covered:
        if a - prev > gap:
            holes.append((prev, a))
        prev = max(prev, b)
    holes.append((prev, float("inf")))
    return round(sum(max(0.0, min(b, y) - max(a, x)) for a, b in holes for x, y in spans
                     if min(b, y) - max(a, x) > 0 and (y - x) > 0) / 60, 1)


def load(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[([\d.]+)-([\d.]+)\] ?(.*)", line)
        if m:
            out.append({"start": float(m[1]), "end": float(m[2]), "text": m[3]})
    return out


def peak_gb():
    try:
        import mlx.core as mx
        f = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
        return round(f() / 2**30, 2)
    except Exception:  # noqa: BLE001
        return None


def reset_peak():
    try:
        import mlx.core as mx
        (getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory)()
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--ways", default="АБВ", help="какие способы мерить; А, которого нет в списке, берётся из --out")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = hl.HeavyLock("Scribe (проверка 06 Б)", "длинный файл: один проход и по паузам",
                        who=os.environ.get("HEAVY_LOCK_WHO") or "background", path=hl.default_path(), idle_after=None)
    if not lock.acquire(poll=10, on_wait=lambda h, n: print(f"жду видеокарту: {h.program if h else 'очередь'}", flush=True)):
        return 75
    res = {"файл, мин": None}
    try:
        t0 = time.time()
        audio = core._decode_to_array(a.file)
        dur = len(audio) / engine.SR
        res["файл, мин"] = round(dur / 60, 1)
        res["подготовка звука, с"] = round(time.time() - t0, 1)
        eng = engine.get_engine()
        eng.warmup(a.model)
        # VAD - нужен и для оценки «слов вне речи» у обоих способов
        t0 = time.time()
        spans = vad.split_phrases(audio)
        res["VAD, с"] = round(time.time() - t0, 1)
        parts = chunks(spans)
        res["кусков"] = len(parts)
        res["речь, мин"] = round(sum(b - x for x, b in spans) / 60, 1)
        segs = {}
        if "А" in a.ways:            # А - один проход
            reset_peak()
            t0 = time.time()
            segs["А"] = eng.transcribe(audio, a.model, a.lang)
            res["А один проход, с"] = round(time.time() - t0, 1)
            res["А пик памяти, ГБ"] = peak_gb()
        else:
            segs["А"] = load(out / f"{Path(a.file).stem} - A.txt")
        if "Б" in a.ways:            # Б - по паузам, по одному куску
            reset_peak()
            t0 = time.time()
            segs["Б"] = []
            for x, b in parts:
                for s in eng.transcribe(audio[int(x * engine.SR):int(b * engine.SR)], a.model, a.lang):
                    segs["Б"].append({**s, "start": s.get("start", 0) + x, "end": s.get("end", 0) + x})
            res["Б по паузам, с"] = round(time.time() - t0 + res["VAD, с"], 1)
            res["Б пик памяти, ГБ"] = peak_gb()
        if "В" in a.ways:            # В - по паузам пачками
            reset_peak()
            t0 = time.time()
            segs["В"], again, rest = eng._call(batched, eng, audio, parts, a.model, a.lang, a.batch)   # в нити MLX
            res[f"В пачками по {a.batch}, с"] = round(time.time() - t0 + res["VAD, с"], 1)
            res["В пик памяти, ГБ"] = peak_gb()
            res["В кусков заново по одному"], res["В остатков кусков в следующей пачке"] = again, rest
        wa = words(segs["А"])
        for k, sg in segs.items():
            res[f"{k}: слов"] = len(words(sg))
            res[f"{k}: повторов 3+ подряд"] = repeats(sg)
            res[f"{k}: слов вне речи"] = outside_speech(sg, spans)
            res[f"{k}: потерянной речи (пропуски > 10 с), мин"] = lost(sg, spans)
            if k != "А":
                wk = words(sg)
                sm = difflib.SequenceMatcher(None, wa, wk, autojunk=False)
                res[f"{k}: совпадающих слов с А, %"] = round(200 * sum(m.size for m in sm.get_matching_blocks()) / max(1, len(wa) + len(wk)), 1)
            if k in a.ways:
                (out / f"{Path(a.file).stem} - {'A' if k == 'А' else k}.txt").write_text(
                    "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {(s.get('text') or '').strip()}" for s in sg), encoding="utf-8")
    finally:
        lock.release()
        try:
            engine.get_engine().unload()
        except Exception:  # noqa: BLE001
            pass
    (out / f"{Path(a.file).stem} - итог.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
