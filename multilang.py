#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Режим «Несколько» (RU/EN), Scribe 2.2: язык определяется для каждой фразы.

  1. Фразы - Silero VAD (vad.py): речь между паузами от 0,3 с. Человек меняет язык
     на стыке предложений, а там пауза - смена языка совпадает с границей фразы.
  2. Язык фразы - детектор самой модели Whisper (вероятности ru/en), пакетом.
     Вход дополняется до 10 с, а не до 30: в ~3,5 раза быстрее при тех же решениях
     (до 5 с - уже ошибается). SpeechBrain VoxLingua107 проиграл на реальной записи
     (английский с русским акцентом принимал за русский) - tests/compare_langid.py.
  3. Сглаживание: очень короткая неуверенная фраза между двумя фразами одного
     языка получает язык соседей и распознаётся вместе с ними.
  4. Остальные неуверенные или короткие фразы - двойное распознавание (ru и en)
     на полном 30-секундном входе (на коротком декодер зацикливается); побеждает вариант с лучшей средней лог-вероятностью, кроме
     зацикленных (compression_ratio > 2,4 - критерий самого Whisper): ru-вариант
     английской речи бывает «переводом с повтором», и его вероятность обманчиво высока.
     Если оба варианта написаны одним алфавитом, Whisper сам «услышал» этот язык.
  5. Распознавание - блоками: соседние фразы одного языка до ~30 с одним вызовом
     с явным языком; подсказка - последнее предложение предыдущего блока того же языка
     (и только если оно написано алфавитом этого языка - иначе Whisper начинает переводить).
     Зациклившийся блок повторяется с небольшой температурой; фразы, которые Whisper
     пропустил внутри блока (бывает: заканчивает блок раньше времени), распознаются ещё раз.

Никаких словарей и правил под конкретную запись - только общие критерии.
"""
import re
import time
import zlib

import numpy as np

import vad

SR = 16000
LANGS = ("ru", "en")
SURE = 0.90            # уверенность детектора, выше которой фразе верим сразу
SHORT = 1.5            # короче - даже уверенный ответ перепроверяем двойным распознаванием
TINY = 1.0             # «одно-два слова»: кандидат на язык соседей
BLOCK_MAX = 28.0       # блок одного языка - до ~30 с (одно окно Whisper)
BLOCK_GAP = 3.0        # длинная пауза разрывает блок: меньше выдуманного текста на тишине
LID_MIN_SEC = 10.0     # до скольких секунд дополнять вход детектора (5 с - мало: 12 ошибок на 116 фраз)

_SENT = re.compile(r"[^.!?…]*[.!?…]")


def _script(text):
    cyr = len(re.findall(r"[а-яё]", text, re.I))
    lat = len(re.findall(r"[a-z]", text, re.I))
    if cyr + lat < 2:
        return None
    return "ru" if cyr > lat else "en"


def _compression(text):
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


class PhraseLID:
    """Детектор языка и быстрое двойное распознавание на модели Whisper (MLX).
    Модель - та же, что для распознавания (ModelHolder), второй раз не грузится."""

    def __init__(self, model_repo, own=False):
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder
        from mlx_whisper.tokenizer import get_tokenizer
        self.mx = mx
        if own:        # своя копия (маленькая модель-сито) - не вытесняет модель распознавания из ModelHolder
            from mlx_whisper.load_models import load_model
            self.model = load_model(model_repo, dtype=mx.float16)
        else:
            self.model = ModelHolder.get_model(model_repo, mx.float16)
        nl = self.model.num_languages
        self.tok = get_tokenizer(self.model.is_multilingual, num_languages=nl)
        self.lang_id = {l: self.tok.to_language_token(l) for l in LANGS}
        self.sot = {l: list(get_tokenizer(True, num_languages=nl, language=l, task="transcribe")
                            .sot_sequence_including_notimestamps) for l in LANGS}
        self.eot = self.tok.eot

    # --- кодировщик на коротком входе ---
    def _features(self, clips, min_sec=None):
        import mlx.nn as nn
        from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
        mx = self.mx
        min_sec = LID_MIN_SEC if min_sec is None else min_sec
        T = int(min(N_FRAMES, max(min_sec * 100, max(len(c) for c in clips) / 160 + 2)))
        T += T % 2
        mel = mx.stack([pad_or_trim(log_mel_spectrogram(c[:30 * SR], self.model.dims.n_mels), T, axis=0)
                        for c in clips]).astype(mx.float16)
        enc = self.model.encoder
        if T == N_FRAMES:
            return enc(mel)
        x = nn.gelu(enc.conv1(mel))
        x = nn.gelu(enc.conv2(x))
        x = x + enc._positional_embedding[: x.shape[1]]
        for b in enc.blocks:
            x, _, _ = b(x)
        return enc.ln_post(x)

    def p_en(self, clips, batch=16, prog=None, cancel=lambda: False):
        """Вероятность английского для каждой фразы (среди ru/en)."""
        mx = self.mx
        out = np.zeros(len(clips))
        order = np.argsort([len(c) for c in clips])       # похожие длины - в один пакет
        for i in range(0, len(clips), batch):
            if cancel():
                from summary import Cancelled
                raise Cancelled()
            idx = order[i:i + batch]
            feats = self._features([clips[j] for j in idx])
            x = mx.array([[self.tok.sot]] * len(idx))
            lg = self.model.logits(x, feats)[:, 0].astype(mx.float32)
            two = mx.stack([lg[:, self.lang_id["ru"]], lg[:, self.lang_id["en"]]], axis=1)
            out[idx] = np.array(mx.softmax(two, axis=1))[:, 1]
            if prog:
                prog(min(1.0, (i + len(idx)) / len(clips)))
        return out

    def top_lang(self, clips, batch=16, cancel=lambda: False):
        """Самый вероятный язык фразы среди ВСЕХ языков Whisper и его вероятность."""
        mx = self.mx
        codes = list(self.tok.all_language_codes)
        ids = list(self.tok.all_language_tokens)
        out = [None] * len(clips)
        order = np.argsort([len(c) for c in clips])
        for i in range(0, len(clips), batch):
            if cancel():
                from summary import Cancelled
                raise Cancelled()
            idx = order[i:i + batch]
            feats = self._features([clips[j] for j in idx])
            x = mx.array([[self.tok.sot]] * len(idx))
            lg = self.model.logits(x, feats)[:, 0].astype(mx.float32)
            pr = np.array(mx.softmax(lg[:, mx.array(ids)], axis=1))
            for k, j in enumerate(idx):
                b = int(pr[k].argmax())
                out[j] = (codes[b], float(pr[k, b]))
        return out

    def lang_probs(self, clips, langs=("ru", "en"), batch=32, cancel=lambda: False):
        """Вероятности выбранных языков (среди всех языков Whisper) для каждой фразы."""
        mx = self.mx
        codes = list(self.tok.all_language_codes)
        ids = list(self.tok.all_language_tokens)
        cols = [codes.index(l) for l in langs]
        out = np.zeros((len(clips), len(langs)))
        order = np.argsort([len(c) for c in clips])
        for i in range(0, len(clips), batch):
            if cancel():
                from summary import Cancelled
                raise Cancelled()
            idx = order[i:i + batch]
            feats = self._features([clips[j] for j in idx])
            x = mx.array([[self.tok.sot]] * len(idx))
            lg = self.model.logits(x, feats)[:, 0].astype(mx.float32)
            pr = np.array(mx.softmax(lg[:, mx.array(ids)], axis=1))
            out[idx] = pr[:, cols]
        return out

    def dual(self, clip):
        """Распознать фразу и как ru, и как en (жадно, одним пакетом на двоих).
        -> {lang: (avg_logprob, compression_ratio, text)}"""
        mx = self.mx
        feats = self._features([clip], min_sec=30)
        feats = mx.concatenate([feats, feats], axis=0)
        max_new = int(min(120, 12 + len(clip) / SR * 9))
        res = {}
        toks = [self.sot[l] for l in LANGS]
        x = mx.array(toks)
        kv = None
        out = [[] for _ in LANGS]
        lp = [0.0 for _ in LANGS]
        done = [False for _ in LANGS]
        for step in range(max_new):
            logits, kv, _ = self.model.decoder(x, feats, kv)
            logits = logits[:, -1].astype(mx.float32)
            # только обычные токены текста и конец
            mask = np.zeros(logits.shape[-1], dtype=np.float32)
            mask[self.eot + 1:] = -np.inf
            if step == 0:
                mask[self.eot] = -np.inf
            logits = logits + mx.array(mask)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            nxt = np.array(mx.argmax(logp, axis=-1))
            lpn = np.array(logp)[np.arange(len(LANGS)), nxt]
            for k in range(len(LANGS)):
                if done[k]:
                    continue
                lp[k] += float(lpn[k])
                if int(nxt[k]) == self.eot:
                    done[k] = True
                else:
                    out[k].append(int(nxt[k]))
            if all(done):
                break
            x = mx.array(nxt.reshape(-1, 1))
        for k, l in enumerate(LANGS):
            text = self.tok.decode(out[k]).strip()
            res[l] = (lp[k] / (len(out[k]) + 1), _compression(text) if text else 0.0, text)
        return res


def _choose(dual, p_en):
    """Выбор языка неуверенной фразы по двойному распознаванию."""
    # 1) оба варианта одним алфавитом - Whisper сам «услышал» этот язык
    scripts = {_script(dual[l][2]) for l in LANGS}
    if len(scripts) == 1 and None not in scripts:
        return scripts.pop()
    # 2) лучший avg_logprob среди незацикленных
    ok = {l: v for l, v in dual.items() if v[2] and v[1] <= 2.4}
    if ok:
        return max(ok, key=lambda l: ok[l][0])
    return "en" if p_en >= 0.5 else "ru"


def _last_sentence(text, limit=120):
    """Подсказка для следующего блока: последнее настоящее предложение (не «.» и не «!»)."""
    parts = [x.strip() for x in _SENT.findall(text.strip()) if len(re.findall(r"\w{2,}", x)) >= 2]
    if not parts:
        return None
    return parts[-1][-limit:]


def _has_words(text):
    return bool(re.search(r"\w", text or ""))


def _looped(segs):
    """Зацикливание по критерию самого Whisper: текст сжимается слишком хорошо."""
    return any(s.get("compression_ratio", 0) > 2.4 for s in segs)


_LID = {}


def get_lid(model_repo):
    if model_repo not in _LID:
        _LID.clear()
        _LID[model_repo] = PhraseLID(model_repo)
    return _LID[model_repo]


# --- проверка для «Авто»: есть ли в записи второй язык ---
# 1) основной язык - turbo по нескольким фразам, равномерно по записи;
# 2) сито - маленькая модель whisper-base (~25 мс на фразу, в 15-20 раз дешевле turbo) - смотрит
#    ВСЕ фразы и отмечает подозрительные: высокая вероятность второго языка («сильные») или
#    сито не уверено даже в основном («слабые» - так оно видит английский с русским акцентом);
# 3) turbo перепроверяет сильные (до 40) и выборку слабых; если в выборке слабых нашёлся второй
#    язык - и остальные слабые.
# Замеры - tests/measure_auto.py, docs/Приёмка 2.3.md.
SIEVE_REPO = "mlx-community/whisper-base-mlx"
SIEVE_STRONG = 0.3     # у сита p(второй язык) от 0,3 - сильная отметка (настоящие русские фразы: от 0,77)
SIEVE_WEAK = 0.5       # у сита p(основной язык) ниже 0,5 - слабая отметка
VERIFY_MAX = 40        # больше фраз turbo не перепроверяет
VERIFY_STEP = 8        # порция перепроверки
WEAK_SAMPLE = 8
DOM_SAMPLE = 6         # фраз для основного языка
PROBE_MIN = 1.5        # фразы короче не смотрим: на них детектор неуверен
PROBE_CLIP = 10.0      # для языка хватает 10 с из середины фразы
PROBE_SURE = 0.9       # turbo подтверждает только уверенным ответом
MIX_MIN = 2            # смешанная: не меньше двух подтверждённых фраз второго языка
MIX_TIME = 0.02        # и они занимают не меньше 2 % времени речи (одна случайная фраза - нет)
# запасной путь, если whisper-base недоступна: выборка фраз и только turbo
PROBE_N, PROBE_PER_MIN, PROBE_N_MIN = 60, 8, 20

_SIEVE = {}


def _get_sieve():
    if "lid" not in _SIEVE:
        try:
            _SIEVE["lid"] = PhraseLID(SIEVE_REPO, own=True)
        except Exception:
            _SIEVE["lid"] = None            # нет сети и модели в кэше - пойдём запасным путём
    return _SIEVE["lid"]


def has_language(found, lang):
    """found - итог probe_languages: есть ли заметно речи на языке lang."""
    k = found["confirmed"].get(lang, 0)
    t = found["confirmed_time"].get(lang, 0.0)
    return k >= MIX_MIN and found["speech_time"] > 0 and t / found["speech_time"] >= MIX_TIME


def _sample(cands, n, total, shift=0.5):
    """n фраз равномерно по времени."""
    if len(cands) <= n:
        return list(cands)
    picked, used = [], set()
    for k in range(n):
        t = (k + shift) / n * total
        j = min(range(len(cands)), key=lambda q: 0 if cands[q][0] <= t <= cands[q][1]
                else min(abs(cands[q][0] - t), abs(cands[q][1] - t)))
        if j not in used:
            used.add(j)
            picked.append(cands[j])
    return picked


def _clip(audio, a, b):
    if b - a > PROBE_CLIP:
        m = (a + b) / 2
        a, b = m - PROBE_CLIP / 2, m + PROBE_CLIP / 2
    return audio[int(a * SR):int(b * SR)]


def probe_languages(audio, model_repo, cancel=lambda: False):
    """Дешёвая проверка перед «Авто»: какие языки звучат в записи.
    -> dict(mixed, dominant, other, confirmed{lang: фраз}, confirmed_time{lang: с}, speech_time, seconds, ...)"""
    t0 = time.monotonic()
    total = len(audio) / SR
    phrases = vad.split_phrases(audio)
    cands = [(a, b) for a, b in phrases if b - a >= PROBE_MIN] or phrases
    speech = sum(b - a for a, b in cands)
    turbo = get_lid(model_repo)
    checked = {}                                         # фраза -> (язык, вероятность) по turbo

    def verify(items):
        items = [c for c in items if c not in checked]
        if items:
            checked.update(zip(items, turbo.top_lang([_clip(audio, a, b) for a, b in items], cancel=cancel)))

    # 1) основной язык: если сито уверенно слышит один язык почти во всей речи - берём его,
    #    иначе (например, английский с русским акцентом сито не узнаёт) - turbo по нескольким фразам
    sieve = _get_sieve()
    P = None
    votes = {}
    if sieve is not None:
        P = sieve.lang_probs([_clip(audio, a, b) for a, b in cands], ("ru", "en"), cancel=cancel)
        for lang, col in (("ru", 0), ("en", 1)):
            t = sum(b - a for (a, b), pr in zip(cands, P) if pr[col] >= 0.5)
            if t >= 0.8 * speech:
                votes = {lang: t}
    dom_by = "сито"
    if not votes:
        dom_by = "turbo"
        verify(_sample(cands, DOM_SAMPLE, total))
        for (a, b), (lang, pr) in checked.items():
            if pr >= PROBE_SURE:
                votes[lang] = votes.get(lang, 0.0) + (b - a)
    dom = max(votes, key=votes.get) if votes else None
    other = {"ru": "en", "en": "ru"}.get(dom)
    strong = weak = []
    if other and sieve is not None:
        # 2) сито по всем фразам (уже посчитано): вероятности основного и второго языка
        P = P[:, [("ru", "en").index(dom), ("ru", "en").index(other)]]
        strong = [c for c, (pd, po) in zip(cands, P) if po >= SIEVE_STRONG]
        weak = [c for c, (pd, po) in zip(cands, P) if po < SIEVE_STRONG and pd < SIEVE_WEAK]
        strong.sort(key=lambda c: -P[cands.index(c)][1])
        method = f"сито whisper-base + turbo (основной язык - {dom_by})"
        # 3) перепроверка turbo: сильные - порциями от самых подозрительных; порция без единой
        #    фразы второго языка - дальше (ещё менее подозрительные) не смотрим
        for i in range(0, min(len(strong), VERIFY_MAX), VERIFY_STEP):
            part = strong[i:i + VERIFY_STEP]
            verify(part)
            if not any(checked[c][0] == other and checked[c][1] >= PROBE_SURE for c in part):
                break
        #    слабые - выборкой по времени, тоже порциями: пока порция находит второй язык
        first = _sample(weak, WEAK_SAMPLE, total)
        rest = [c for c in _sample(weak, VERIFY_MAX, total) if c not in first]
        parts = [first] + [rest[i:i + WEAK_SAMPLE] for i in range(0, len(rest), WEAK_SAMPLE)]
        for part in parts:
            verify(part)
            if not any(checked[c][0] == other and checked[c][1] >= PROBE_SURE for c in part):
                break
    elif other:
        method = "выборка turbo (сито недоступно)"
        if dom_by == "сито":
            verify(_sample(cands, DOM_SAMPLE, total))
        verify(_sample(cands, int(min(PROBE_N, max(PROBE_N_MIN, total / 60 * PROBE_PER_MIN))), total))
    else:
        method = "turbo (основной язык не ru/en)"
    confirmed, ctime, hits = {}, {}, []
    for (a, b), (lang, pr) in sorted(checked.items()):
        if pr >= PROBE_SURE and lang != dom:
            confirmed[lang] = confirmed.get(lang, 0) + 1
            ctime[lang] = ctime.get(lang, 0.0) + (b - a)
            if lang == other:
                hits.append((round(a, 1), round(b - a, 1), lang, round(pr, 2)))
    dom_share = votes.get(dom, 0.0) / max(1e-9, sum(votes.values())) if dom else 0.0
    found = dict(dominant=dom, dominant_share=round(dom_share, 3), other=other, confirmed=confirmed,
                 confirmed_time={k: round(v, 1) for k, v in ctime.items()}, speech_time=round(speech, 1),
                 phrases=len(cands), flagged=len(strong) + len(weak), verified=len(checked), method=method, hits=hits)
    found["mixed"] = bool(other) and has_language(found, other)
    found["seconds"] = round(time.monotonic() - t0, 1)
    return found


def plan(audio, model_repo, prog=lambda f: None, cancel=lambda: False, stats=None):
    """Фразы и их язык -> блоки [(start, end, lang)]."""
    t0 = time.monotonic()
    probs = vad.speech_probs(audio)
    phrases = vad.split_phrases(audio, probs)
    if not phrases:
        return []
    t_vad = time.monotonic() - t0
    lid = get_lid(model_repo)
    clips = [audio[int(a * SR):int(b * SR)] for a, b in phrases]
    t0 = time.monotonic()
    p = lid.p_en(clips, prog=lambda f: prog(0.7 * f), cancel=cancel)
    t_lid = time.monotonic() - t0
    lang = ["en" if x >= 0.5 else "ru" for x in p]
    conf = [max(x, 1 - x) for x in p]
    dur = [b - a for a, b in phrases]
    sure = [c >= SURE and d >= SHORT for c, d in zip(conf, dur)]
    n_smooth = n_dual = 0
    # сглаживание: очень короткая неуверенная фраза между соседями одного (уверенного) языка
    for i in range(1, len(phrases) - 1):
        if not sure[i] and dur[i] < TINY and conf[i] < SURE and sure[i - 1] and sure[i + 1] \
                and lang[i - 1] == lang[i + 1]:
            lang[i], sure[i] = lang[i - 1], True
            n_smooth += 1
    # короткая фраза, которая по детектору того же языка, что и обе уверенные соседки,
    # всё равно распознается в их блоке - перепроверять её незачем
    for i in range(len(phrases)):
        if sure[i] or conf[i] < SURE:
            continue
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(phrases)]
        if nb and all(sure[j] and lang[j] == lang[i] for j in nb):
            sure[i] = True
    # остальные неуверенные - двойное распознавание
    unsure = [i for i in range(len(phrases)) if not sure[i]]
    t0 = time.monotonic()
    for k, i in enumerate(unsure):
        if cancel():
            from summary import Cancelled
            raise Cancelled()
        lang[i] = _choose(lid.dual(clips[i]), p[i])
        n_dual += 1
        prog(0.7 + 0.3 * (k + 1) / len(unsure))
    prog(1.0)
    if stats is not None:
        stats.update(phrases=len(phrases), smoothed=n_smooth, dual=n_dual,
                     en=sum(1 for x in lang if x == "en"), t_vad=round(t_vad, 1), t_lid=round(t_lid, 1),
                     t_dual=round(time.monotonic() - t0, 1))
    # блоки: соседние фразы одного языка, до BLOCK_MAX и без длинных пауз внутри
    blocks = []
    for (a, b), l in zip(phrases, lang):
        if blocks and blocks[-1][2] == l and b - blocks[-1][0] <= BLOCK_MAX and a - blocks[-1][1] <= BLOCK_GAP:
            blocks[-1][1] = b
            blocks[-1][3].append((a, b))
        else:
            blocks.append([a, b, l, [(a, b)]])
    return [tuple(x) for x in blocks]


def transcribe(audio, model_repo, transcribe_block, prog=lambda f: None, cancel=lambda: False,
               stage=lambda k, d="": None, stats=None):
    """Весь режим «Несколько». transcribe_block(clip, lang, prompt, prog[, temperature]) -> segments."""
    stage("lang")
    prog(0)
    blocks = plan(audio, model_repo, prog, cancel, stats)
    t0 = time.monotonic()
    stage("asr")
    prog(0)
    total = sum(b - a for a, b, _, _ in blocks) or 1.0
    segments, done, last = [], 0.0, {}
    n_loop = n_gap = 0

    def run(a, b, lang, prompt, bprog, temperature=None):
        clip = np.ascontiguousarray(audio[int(a * SR):int(b * SR)])
        segs = transcribe_block(clip, lang, prompt, bprog, temperature) if temperature is not None \
            else transcribe_block(clip, lang, prompt, bprog)
        out = []
        for x in segs:
            if not _has_words(x.get("text")):
                continue                       # «!», «.» и пустые сегменты - не речь
            x = dict(x)
            x["start"] += a
            x["end"] += a
            x["lang"] = lang
            out.append(x)
        return out

    for a, b, lang, phr in blocks:
        if cancel():
            from summary import Cancelled
            raise Cancelled()

        def block_prog(f, a=a, b=b, done=done):
            prog(min(1.0, (done + f * (b - a)) / total))

        segs = run(a, b, lang, last.get(lang), block_prog)
        if _looped(segs):
            # зациклился - повтор с небольшой температурой (запасной ход самого Whisper)
            alt = run(a, b, lang, None, lambda f: None, temperature=(0.2, 0.4, 0.6))
            if not _looped(alt):
                segs = alt
                n_loop += 1
        # Whisper иногда заканчивает блок раньше времени и пропускает остаток:
        # фразы, где детектор слышал речь, а текста нет, распознаём ещё раз
        covered = [(x["start"], x["end"]) for x in segs]
        gaps = []
        for pa, pb in phr:
            ov = sum(max(0.0, min(pb, e) - max(pa, st)) for st, e in covered)
            if ov < 0.4 * (pb - pa):
                if gaps and pa - gaps[-1][1] <= BLOCK_GAP and pb - gaps[-1][0] <= BLOCK_MAX:
                    gaps[-1][1] = pb
                else:
                    gaps.append([pa, pb])
        for ga, gb in gaps:
            extra = run(ga, gb, lang, None, lambda f: None)
            if extra:
                segs += extra
                n_gap += 1
        segs.sort(key=lambda x: x["start"])
        text = " ".join(x.get("text", "").strip() for x in segs).strip()
        sent = _last_sentence(text)
        # подсказка тянет язык вывода: английская фраза в подсказке русского блока
        # заставляет Whisper переводить. Берём только предложение «своим» алфавитом.
        if sent and _script(sent) == lang:
            last[lang] = sent
        segments += segs
        done += b - a
        prog(min(1.0, done / total))
    if stats is not None:
        stats.update(blocks=len(blocks), loops_fixed=n_loop, gaps_filled=n_gap,
                     t_asr=round(time.monotonic() - t0, 1))
    return segments
