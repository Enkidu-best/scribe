#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Выбор определителя языка фразы для режима «Несколько» - по замеру, а не по мнению.

Кандидаты (оба офлайн):
  whisper     - detect_language самой модели распознавания (пакетом), вероятности ru/en;
  speechbrain - VoxLingua107 ECAPA (нужен пакет speechbrain; в приложение не входит,
                ставится только для этого замера).

Фразы режет Silero VAD (vad.py), как в приложении. Эталон языка фразы - из JSON
тестового набора. Для своей записи (--real) эталона нет: ориентир - двойное
распознавание фразы (ru и en) и выбор варианта с лучшим avg_logprob.

    python3 tests/compare_langid.py [--real "~/Downloads/запись.m4a"] [--no-speechbrain]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import vad  # noqa: E402

SR = 16000
MODEL = "mlx-community/whisper-large-v3-turbo"


def load(path):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-i", str(path), "-ar", str(SR), "-ac", "1", "-f", "f32le",
                          "-loglevel", "error", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def phrases_with_truth(name):
    a = load(HERE / "audio" / f"{name}.m4a")
    gt = json.loads((HERE / "audio" / f"{name}.json").read_text(encoding="utf-8"))
    out = []
    for s, e in vad.split_phrases(a):
        ov = {}
        for p in gt["phrases"]:
            o = min(e, p["end"]) - max(s, p["start"])
            if o > 0:
                ov[p["lang"]] = ov.get(p["lang"], 0) + o
        if not ov:
            continue
        lang, o = max(ov.items(), key=lambda x: x[1])
        if lang in ("ru", "en") and o >= 0.8 * sum(ov.values()):
            out.append(dict(src=name, start=s, end=e, audio=a[int(s * SR):int(e * SR)], truth=lang))
    return out


class WhisperLID:
    def __init__(self):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        from mlx_whisper.tokenizer import get_tokenizer
        self.mx = mx
        self.model = ModelHolder.get_model(MODEL, mx.float16)
        self.tok = get_tokenizer(self.model.is_multilingual, num_languages=self.model.num_languages)
        self.ru = self.tok.to_language_token("ru")
        self.en = self.tok.to_language_token("en")

    def p_en(self, clips, batch=16):
        from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
        mx, out = self.mx, []
        n_mels = self.model.dims.n_mels
        for i in range(0, len(clips), batch):
            mels = [pad_or_trim(log_mel_spectrogram(c[:30 * SR], n_mels), N_FRAMES, axis=0) for c in clips[i:i + batch]]
            mel = mx.stack(mels).astype(mx.float16)
            feats = self.model.encoder(mel)
            x = mx.array([[self.tok.sot]] * len(mels))
            logits = self.model.logits(x, feats)[:, 0].astype(mx.float32)
            two = mx.stack([logits[:, self.ru], logits[:, self.en]], axis=1)
            out += np.array(mx.softmax(two, axis=1))[:, 1].tolist()
        return out


class SpeechBrainLID:
    def __init__(self):
        from speechbrain.inference.classifiers import EncoderClassifier
        self.clf = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                                  savedir=str(Path.home() / ".cache" / "speechbrain-voxlingua107"))
        labels = self.clf.hparams.label_encoder
        self.ru = labels.lab2ind[[k for k in labels.lab2ind if k.startswith("ru")][0]]
        self.en = labels.lab2ind[[k for k in labels.lab2ind if k.startswith("en")][0]]

    def p_en(self, clips):
        import torch
        out = []
        for c in clips:
            with torch.no_grad():
                lp = self.clf.classify_batch(torch.from_numpy(c).unsqueeze(0))[0][0]
            two = torch.stack([lp[self.ru], lp[self.en]])
            out.append(float(torch.softmax(two, 0)[1]))
        return out


def dual_decode_truth(clips):
    """Ориентир для своей записи: распознать как ru и как en, взять лучший avg_logprob."""
    import mlx_whisper
    out = []
    for c in clips:
        best = None
        for lang in ("ru", "en"):
            r = mlx_whisper.transcribe(c, path_or_hf_repo=MODEL, language=lang, temperature=0.0,
                                       condition_on_previous_text=False, verbose=None)
            segs = r.get("segments", [])
            lp = np.mean([s["avg_logprob"] for s in segs]) if segs else -9.0
            if best is None or lp > best[0]:
                best = (lp, lang)
        out.append(best[1])
    return out


def report(name, items, probs, secs):
    rows = []
    for it, p in zip(items, probs):
        pred = "en" if p >= 0.5 else "ru"
        conf = max(p, 1 - p)
        rows.append((it, pred, conf))
    acc = np.mean([pred == it["truth"] for it, pred, _ in rows])
    res = dict(lid=name, n=len(rows), accuracy=round(float(acc), 3), seconds=round(secs, 2))
    for lo, hi, lab in ((0, 1.5, "<1.5s"), (1.5, 3, "1.5-3s"), (3, 99, ">3s")):
        sub = [r for r in rows if lo <= r[0]["end"] - r[0]["start"] < hi]
        if sub:
            res[lab] = f"{sum(r[1] == r[0]['truth'] for r in sub)}/{len(sub)}"
    for thr in (0.6, 0.7, 0.8, 0.9, 0.95):
        sure = [r for r in rows if r[2] >= thr]
        wrong = sum(r[1] != r[0]["truth"] for r in sure)
        res[f"conf>={thr}"] = f"ошибок {wrong} из {len(sure)} уверенных, неуверенных {len(rows) - len(sure)}"
    res["errors"] = [dict(src=r[0]["src"], start=round(r[0]["start"], 1), dur=round(r[0]["end"] - r[0]["start"], 1),
                          truth=r[0]["truth"], conf=round(r[2], 2)) for r in rows if r[1] != r[0]["truth"]]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="своя запись без эталона (ориентир - двойное распознавание)")
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--no-speechbrain", action="store_true")
    a = ap.parse_args()
    if a.real:
        audio = load(Path(a.real).expanduser())
        ph = vad.split_phrases(audio)
        step = max(1, len(ph) // a.sample)
        ph = ph[::step][:a.sample]
        items = [dict(src="real", start=s, end=e, audio=audio[int(s * SR):int(e * SR)]) for s, e in ph]
        t = time.time()
        for it, lang in zip(items, dual_decode_truth([it["audio"] for it in items])):
            it["truth"] = lang
        print(f"ориентир (двойное распознавание) {time.time() - t:.0f} c, en: {sum(i['truth'] == 'en' for i in items)}/{len(items)}")
    else:
        items = phrases_with_truth("lesson") + phrases_with_truth("long_insert")
    clips = [it["audio"] for it in items]
    results = []
    w = WhisperLID()
    w.p_en(clips[:4])                                   # прогрев
    t = time.time(); pw = w.p_en(clips); results.append(report("whisper", items, pw, time.time() - t))
    if not a.no_speechbrain:
        sb = SpeechBrainLID()
        sb.p_en(clips[:2])
        t = time.time(); ps = sb.p_en(clips); results.append(report("speechbrain", items, ps, time.time() - t))
    for r in results:
        brief = {k: v for k, v in r.items() if k != "errors"}
        print(json.dumps(brief, ensure_ascii=False))
        for e in r["errors"][:12]:
            print("    ошибка:", e)
    dest = HERE / "results" / f"langid{'-real' if a.real else ''}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
