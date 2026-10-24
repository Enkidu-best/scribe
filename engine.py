#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Движок распознавания Scribe: один интерфейс, две реализации (2.4).

  • MLX (Mac на Apple Silicon) - mlx_whisper на видеоядре. Код перенесён без
    изменений: на Mac скорость и результат те же, что в 2.3.
  • faster-whisper (CTranslate2) - Windows и Linux: видеокарта NVIDIA (float16)
    или процессор (int8). Та же модель large-v3-turbo.

Выбор автоматический: Apple Silicon - MLX; есть NVIDIA с CUDA - faster-whisper на
видеокарте; иначе - на процессоре. Пользователь движок не выбирает. Для проверок
можно заставить: SCRIBE_ENGINE=mlx | fw, SCRIBE_DEVICE=cpu | cuda.

Интерфейс (Engine):
  transcribe(audio, model, lang, prompt, temperature, prog, cancel) -> [сегменты]
  transcribe_long(audio, spans, model, lang, store, prog, cancel) -> [сегменты]
                                    (файл: куски по паузам пачками, продолжение после сбоя - common/stt)
  transcribe_text(audio, model, lang, prompt) -> текст          (для диктофона)
  lid(model, own) -> детектор языка фраз (p_en, top_lang, lang_probs, dual)
  warmup(model), unload(), is_local(model), repo(model)

Модуль импортирует только numpy и стандартную библиотеку (тяжёлое - лениво):
его загружает и процесс диктофона, который запускает Legal Knowledge Desk.
"""
import contextlib
import gc
import io
import os
import queue
import re
import threading
import zlib

import numpy as np

import platform_util as pu
from summary import Cancelled

SR = 16000
LANGS = ("ru", "en")
LID_MIN_SEC = 10.0     # до скольких секунд дополнять вход детектора (MLX; 5 с - уже ошибается)
LONG_BATCH = 8         # файл (06 Б): кусков до 30 с в пачке MLX (4-16 - та же скорость, 8 - меньше памяти; Приёмка 2.7)
FW_LONG_BATCH = 8      # то же для faster-whisper (процессор, NVIDIA)

# Модели: ключ -> (MLX, CTranslate2, размер для faster-whisper, МБ). Все - лицензия MIT.
MODELS = {
    "turbo": ("mlx-community/whisper-large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", 1622),
    "large-v3": ("mlx-community/whisper-large-v3-mlx", "Systran/faster-whisper-large-v3", 3087),
    "medium": ("mlx-community/whisper-medium-mlx", "Systran/faster-whisper-medium", 1531),
    "small": ("mlx-community/whisper-small-mlx", "Systran/faster-whisper-small", 486),
    "base": ("mlx-community/whisper-base-mlx", "Systran/faster-whisper-base", 148),
}
DEFAULT = "turbo"


def model_key(model):
    """Ключ модели по ключу или по имени репозитория (MLX или CTranslate2)."""
    if model in MODELS:
        return model
    for k, (mlx, ct2, _) in MODELS.items():
        if model in (mlx, ct2):
            return k
    return model


def _compression(text):
    b = text.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


def _is_local(repo):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


class _ProgressCatcher(io.TextIOBase):
    """Ловит строки «[00:12.000 --> 00:15.000] текст», которые mlx_whisper печатает
    при verbose=True, и превращает их в прогресс. Заодно - точка отмены: если
    нажали «Остановить», исключение из print прерывает распознавание сразу."""

    def __init__(self, duration, cb, cancel=lambda: False):
        self.duration = duration or 0
        self.cb = cb or (lambda f: None)
        self.cancel = cancel or (lambda: False)
        self.buf = ""

    def write(self, s):
        if self.cancel():
            raise Cancelled()
        self.buf += s
        if "\n" in self.buf:
            *lines, self.buf = self.buf.split("\n")
            for line in lines:
                m = re.search(r'-->\s*([\d:.]+)\s*\]', line)
                if m and self.duration > 0:
                    self.cb(min(1.0, _parse_ts(m.group(1)) / self.duration))
        return len(s)


def _parse_ts(text):
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return 0.0
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


# =====================================================================  MLX
class MLXExecutor:
    """Одна долгоживущая нить для всех вызовов MLX (2.7).

    Потоки вычислений MLX привязаны к нити: массив, созданный в одной нити и не вычисленный до
    конца, в другой нити даёт «There is no Stream(gpu, N) in current thread». Раньше Scribe грузил
    модель в одной нити, а распознавал в новой нити на каждый запуск - и после сбоя (например,
    нехватки памяти) в кэшах mlx_whisper оставались такие массивы, повторный запуск падал.
    Теперь прогрев, распознавание, детектор языка и выгрузка идут здесь; остальные нити только
    ставят задачу и ждут результат. Прогресс и отмена работают как раньше (колбэки потокобезопасны).
    Практика: vllm-mlx bind_generation_streams, mlx-lm #1179, mlx-audio #744."""

    def __init__(self):
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="scribe-mlx", daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            fn, args, kw, box, done = self._q.get()
            try:
                box["r"] = fn(*args, **kw)
            except BaseException as e:          # noqa: BLE001 - исключение уходит вызывающей нити
                box["e"] = e
            finally:
                done.set()

    def run(self, fn, *args, **kw):
        if threading.current_thread() is self._thread:      # вызов изнутри задачи - сразу
            return fn(*args, **kw)
        box, done = {}, threading.Event()
        self._q.put((fn, args, kw, box, done))
        done.wait()
        if "e" in box:
            raise box["e"]
        return box.get("r")


_MLX_EXEC = {}


def mlx_executor():
    if "x" not in _MLX_EXEC:
        _MLX_EXEC["x"] = MLXExecutor()
    return _MLX_EXEC["x"]


class _OnExecutor:
    """Обёртка объекта MLX: каждый его метод выполняется в нити-исполнителе."""

    def __init__(self, obj, ex):
        self._obj, self._ex = obj, ex

    def __getattr__(self, name):
        val = getattr(self._obj, name)
        if callable(val):
            return lambda *a, **k: self._ex.run(val, *a, **k)
        return val


class MLXLID:
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

    def _lang_logits(self, clips):
        mx = self.mx
        feats = self._features(clips)
        x = mx.array([[self.tok.sot]] * len(clips))
        return self.model.logits(x, feats)[:, 0].astype(mx.float32)

    def p_en(self, clips, batch=16, prog=None, cancel=lambda: False):
        """Вероятность английского для каждой фразы (среди ru/en)."""
        mx = self.mx
        out = np.zeros(len(clips))
        order = np.argsort([len(c) for c in clips])       # похожие длины - в один пакет
        for i in range(0, len(clips), batch):
            if cancel():
                raise Cancelled()
            idx = order[i:i + batch]
            lg = self._lang_logits([clips[j] for j in idx])
            two = mx.stack([lg[:, self.lang_id["ru"]], lg[:, self.lang_id["en"]]], axis=1)
            out[idx] = np.array(mx.softmax(two, axis=1))[:, 1]
            if prog:
                prog(min(1.0, (i + len(idx)) / len(clips)))
        return out

    def _all_probs(self, clips, batch, cancel):
        mx = self.mx
        ids = list(self.tok.all_language_tokens)
        out = np.zeros((len(clips), len(ids)))
        order = np.argsort([len(c) for c in clips])
        for i in range(0, len(clips), batch):
            if cancel():
                raise Cancelled()
            idx = order[i:i + batch]
            lg = self._lang_logits([clips[j] for j in idx])
            out[idx] = np.array(mx.softmax(lg[:, mx.array(ids)], axis=1))
        return out

    def top_lang(self, clips, batch=16, cancel=lambda: False):
        """Самый вероятный язык фразы среди ВСЕХ языков Whisper и его вероятность."""
        codes = list(self.tok.all_language_codes)
        pr = self._all_probs(clips, batch, cancel)
        return [(codes[int(r.argmax())], float(r.max())) for r in pr]

    def lang_probs(self, clips, langs=("ru", "en"), batch=32, cancel=lambda: False):
        """Вероятности выбранных языков (среди всех языков Whisper) для каждой фразы."""
        codes = list(self.tok.all_language_codes)
        pr = self._all_probs(clips, batch, cancel)
        return pr[:, [codes.index(l) for l in langs]]

    def dual(self, clip):
        """Распознать фразу и как ru, и как en (жадно, одним пакетом на двоих).
        -> {lang: (avg_logprob, compression_ratio, text)}"""
        mx = self.mx
        feats = self._features([clip], min_sec=30)
        feats = mx.concatenate([feats, feats], axis=0)
        max_new = int(min(120, 12 + len(clip) / SR * 9))
        res = {}
        x = mx.array([self.sot[l] for l in LANGS])
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


class MLXEngine:
    name = "mlx"
    label = "Apple MLX (видеоядро)"
    device = "mlx"

    def __init__(self):
        self._x = mlx_executor()
        self._x.run(lambda: __import__("mlx_whisper"))   # проверка, что MLX есть
        self._lids = {}

    def _call(self, fn, *args, **kw):
        """Задача MLX в нити-исполнителе. Сбой (не отмена) - полная очистка: следующий запуск
        начинается с чистого состояния, а не с остатков упавшего."""
        try:
            return self._x.run(self._guarded, fn, *args, **kw)
        except Cancelled:
            raise
        except Exception:
            self._x.run(self._reset)
            raise

    def _guarded(self, fn, *args, **kw):
        """Модель в кэше mlx_whisper загружена не здесь (в другой нити) - выгрузить: у неё есть
        невычисленные поля (_positional_embedding), привязанные к той нити."""
        from mlx_whisper.transcribe import ModelHolder
        if ModelHolder.model is not None and id(ModelHolder.model) != getattr(self, "_own_model", None):
            self._reset()
        try:
            return fn(*args, **kw)
        finally:
            self._own_model = id(ModelHolder.model) if ModelHolder.model is not None else None

    @staticmethod
    def repo(model):
        k = model_key(model)
        return MODELS[k][0] if k in MODELS else model

    def is_local(self, model):
        return _is_local(self.repo(model))

    def size_mb(self, model):
        k = model_key(model)
        return MODELS[k][2] if k in MODELS else 0

    def transcribe(self, audio, model, lang=None, prompt=None, temperature=0.0, prog=None, cancel=None):
        return self._call(self._transcribe, audio, model, lang, prompt, temperature, prog, cancel)

    def _transcribe(self, audio, model, lang=None, prompt=None, temperature=0.0, prog=None, cancel=None):
        """Один вызов mlx_whisper по массиву в памяти, с прогрессом и отменой."""
        import mlx_whisper
        catcher = _ProgressCatcher(len(audio) / SR, prog, cancel)
        with contextlib.redirect_stdout(catcher):
            res = mlx_whisper.transcribe(
                audio, path_or_hf_repo=self.repo(model), language=lang, initial_prompt=prompt,
                temperature=temperature, condition_on_previous_text=False, verbose=True)
        return res.get("segments", [])

    def transcribe_long(self, audio, spans, model, lang=None, store=None, prog=None, cancel=None, batch=None,
                        on_chunk=None):
        return self._call(self._transcribe_long, audio, spans, model, lang, store, prog, cancel, batch or LONG_BATCH,
                          on_chunk)

    def _transcribe_long(self, audio, spans, model, lang, store, prog, cancel, batch, on_chunk=None):
        """Файл - кусками по паузам, пачками (06 Б, common/stt): в нити MLX, как всё MLX."""
        from common.stt import longfile as lf
        try:
            return lf.transcribe(audio, spans, lf.mlx_decoder(self.repo(model), lang),
                                 fallback=lambda c: self._transcribe(c, model, lang, temperature=lf.FALLBACK_T),
                                 batch=batch, store=store, prog=prog, cancel=cancel, on_chunk=on_chunk)
        except lf.Cancelled:
            raise Cancelled() from None

    def transcribe_text(self, audio, model, lang=None, prompt=None):
        return self._call(self._transcribe_text, audio, model, lang, prompt)

    def _transcribe_text(self, audio, model, lang=None, prompt=None):
        """Для диктофона - ровно тот же вызов, что в 2.3 (Legal Knowledge Desk на него рассчитывает)."""
        import mlx_whisper
        res = mlx_whisper.transcribe(
            np.ascontiguousarray(audio), path_or_hf_repo=self.repo(model),
            language=lang, temperature=0.0, condition_on_previous_text=False,
            initial_prompt=prompt, verbose=False)
        return (res.get("text") or "").strip()

    def warmup(self, model):
        return self._call(self._warmup, model)

    def _warmup(self, model):
        import mlx_whisper
        mlx_whisper.transcribe(np.zeros(SR, dtype=np.float32), path_or_hf_repo=self.repo(model),
                               language="ru", verbose=False)

    def detect_language(self, clips, model):
        """Язык каждого клипа (16 кГц): [(язык, вероятность)] - детектором самой модели Whisper."""
        return self.lid(model).top_lang(clips)

    def lid(self, model, own=False):
        key = (self.repo(model), own)
        if key not in self._lids:
            if not own:
                for k in [k for k in self._lids if not k[1]]:
                    del self._lids[k]
            # создаётся и работает в нити-исполнителе; сбой - очистка, как у распознавания
            obj = self._call(MLXLID, self.repo(model), own=own)
            self._lids[key] = _OnExecutor(obj, _Guard(self))
        return self._lids[key]

    def unload(self):
        self._x.run(self._unload)

    def _reset(self):
        """После сбоя: модели, кэш MLX и кэши mlx_whisper (фильтры спектрограммы и окно), где мог
        остаться недовычисленный массив."""
        self._unload()
        try:
            from mlx_whisper import audio as _a
            _a.mel_filters.cache_clear()
            _a.hanning.cache_clear()
        except Exception:
            pass

    def _unload(self):
        self._lids.clear()
        try:
            from mlx_whisper.transcribe import ModelHolder
            ModelHolder.model = None
            ModelHolder.model_path = None
        except Exception:
            pass
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass


class _Guard:
    """Исполнитель для детектора языка: методы - в нити MLX, сбой - очистка движка."""

    def __init__(self, eng):
        self._eng = eng

    def run(self, fn, *a, **k):
        return self._eng._call(fn, *a, **k)


# ========================================================  faster-whisper
class FWLID:
    """Детектор языка фраз и двойное распознавание на faster-whisper (CTranslate2).
    CTranslate2 принимает только полные 30-секундные окна - короткий вход, как в MLX,
    здесь недоступен; зато окна идут пакетом."""

    def __init__(self, eng, model):
        self.eng = eng
        self.model = model
        self.wm = eng._model(model)

    def _probs(self, clips, batch, cancel):
        """-> [{язык: вероятность}] по всем языкам Whisper."""
        from faster_whisper.audio import pad_or_trim
        wm = self.wm
        out = []
        for i in range(0, len(clips), batch):
            if cancel():
                raise Cancelled()
            feats = np.stack([pad_or_trim(wm.feature_extractor(np.ascontiguousarray(c[:30 * SR])))
                              for c in clips[i:i + batch]])
            enc = wm.encode(feats)
            for res in wm.model.detect_language(enc):
                out.append({tok[2:-2]: float(p) for tok, p in res})
        return out

    def p_en(self, clips, batch=8, prog=None, cancel=lambda: False):
        out = np.zeros(len(clips))
        for i in range(0, len(clips), batch):
            pr = self._probs(clips[i:i + batch], batch, cancel)
            for k, d in enumerate(pr):
                ru, en = d.get("ru", 0.0), d.get("en", 0.0)
                out[i + k] = en / (ru + en) if ru + en > 0 else 0.5
            if prog:
                prog(min(1.0, (i + batch) / len(clips)))
        return out

    def top_lang(self, clips, batch=8, cancel=lambda: False):
        res = []
        for d in self._probs(clips, batch, cancel):
            lang = max(d, key=d.get) if d else "en"
            res.append((lang, d.get(lang, 0.0)))
        return res

    def lang_probs(self, clips, langs=("ru", "en"), batch=8, cancel=lambda: False):
        return np.array([[d.get(l, 0.0) for l in langs] for d in self._probs(clips, batch, cancel)])

    def dual(self, clip):
        res = {}
        for l in LANGS:
            segs, _ = self.wm.transcribe(np.ascontiguousarray(clip), language=l, beam_size=1, temperature=0.0,
                                         condition_on_previous_text=False, without_timestamps=True, vad_filter=False)
            segs = list(segs)
            text = " ".join(s.text.strip() for s in segs).strip()
            lp = float(np.mean([s.avg_logprob for s in segs])) if segs else -9.0
            res[l] = (lp, _compression(text) if text else 0.0, text)
        return res


class FWEngine:
    name = "fw"

    def __init__(self, device=None):
        pu.setup_cuda()
        import ctranslate2
        import faster_whisper  # noqa: F401
        want = device or os.environ.get("SCRIBE_DEVICE")
        n_gpu = 0
        try:
            n_gpu = ctranslate2.get_cuda_device_count()
        except Exception:
            pass
        self.device = want or ("cuda" if n_gpu > 0 else "cpu")
        self.cuda_failed = False
        self._models = {}
        self._lids = {}

    @property
    def label(self):
        return "NVIDIA (видеокарта)" if self.device == "cuda" else "процессор - медленнее"

    @staticmethod
    def repo(model):
        k = model_key(model)
        return MODELS[k][1] if k in MODELS else model

    def is_local(self, model):
        return _is_local(self.repo(model))

    def size_mb(self, model):
        k = model_key(model)
        return MODELS[k][2] if k in MODELS else 0

    def _to_cpu(self):
        """Видеокарта есть, но библиотек CUDA нет (или не подошли) - работаем на процессоре."""
        self.device, self.cuda_failed = "cpu", True
        self._models.clear()
        self._lids.clear()

    def _model(self, model):
        from faster_whisper import WhisperModel
        repo = self.repo(model)
        if repo not in self._models:
            local = _is_local(repo)
            try:
                self._models[repo] = WhisperModel(repo, device=self.device,
                                                  compute_type="float16" if self.device == "cuda" else "int8",
                                                  local_files_only=local)
            except Exception as e:
                if self.device != "cuda":
                    raise
                self._to_cpu()
                self._models[repo] = WhisperModel(repo, device="cpu", compute_type="int8", local_files_only=local)
                self._cuda_error = str(e)
        return self._models[repo]

    def _run(self, fn):
        """Ошибки нехватки библиотек CUDA часто всплывают только при первом расчёте."""
        try:
            return fn()
        except RuntimeError as e:
            msg = str(e).lower()
            if self.device == "cuda" and any(w in msg for w in ("cuda", "cublas", "cudnn", "dll", ".so")):
                self._to_cpu()
                return fn()
            raise

    def transcribe(self, audio, model, lang=None, prompt=None, temperature=0.0, prog=None, cancel=None):
        cancel = cancel or (lambda: False)
        dur = len(audio) / SR

        def go():
            wm = self._model(model)
            segs, _ = wm.transcribe(
                np.ascontiguousarray(audio, dtype=np.float32), language=lang, initial_prompt=prompt,
                temperature=list(temperature) if isinstance(temperature, (tuple, list)) else temperature,
                beam_size=1, condition_on_previous_text=False, vad_filter=False,
                compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6)
            out = []
            for s in segs:                      # генератор: распознаёт по мере чтения
                if cancel():
                    raise Cancelled()
                out.append(dict(start=s.start, end=s.end, text=s.text, avg_logprob=s.avg_logprob,
                                compression_ratio=s.compression_ratio, no_speech_prob=s.no_speech_prob))
                if prog and dur:
                    prog(min(1.0, s.end / dur))
            return out
        return self._run(go)

    def transcribe_long(self, audio, spans, model, lang=None, store=None, prog=None, cancel=None, batch=None,
                        on_chunk=None):
        """Файл - кусками по паузам, пачкой `BatchedInferencePipeline` (06 Б, common/stt)."""
        from common.stt import longfile as lf

        def go():
            return lf.transcribe(audio, spans, lf.fw_decoder(self._model(model), lang),
                                 fallback=lambda c: self.transcribe(c, model, lang, temperature=lf.FALLBACK_T),
                                 batch=batch or FW_LONG_BATCH, store=store, prog=prog, cancel=cancel,
                                 on_chunk=on_chunk)
        try:
            return self._run(go)
        except lf.Cancelled:
            raise Cancelled() from None

    def transcribe_text(self, audio, model, lang=None, prompt=None):
        segs = self.transcribe(audio, model, lang, prompt)
        return " ".join(s["text"].strip() for s in segs).strip()

    def warmup(self, model):
        self.transcribe(np.zeros(SR, dtype=np.float32), model, "ru")

    def detect_language(self, clips, model):
        """Язык каждого клипа (16 кГц): [(язык, вероятность)] - детектором самой модели Whisper."""
        return self.lid(model).top_lang(clips)

    def lid(self, model, own=False):
        repo = self.repo(model)
        if repo not in self._lids:
            self._lids[repo] = FWLID(self, model)
        return self._lids[repo]

    def unload(self):
        self._lids.clear()
        self._models.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ============================================================  выбор
_ENGINE = {}
_FORCED = threading.local()     # движок, подставленный для этой нити (файл «на процессоре», 2.7)


@contextlib.contextmanager
def using(eng):
    """На время блока get_engine() в этой нити возвращает eng (например, процессор вместо MLX,
    когда видеокарте не хватило памяти). Остальные нити не затрагиваются."""
    prev = getattr(_FORCED, "eng", None)
    _FORCED.eng = eng
    try:
        yield eng
    finally:
        _FORCED.eng = prev


def cpu_engine():
    """faster-whisper на процессоре - запасной путь, когда видеокарте не хватает памяти."""
    if "cpu" not in _ENGINE:
        _ENGINE["cpu"] = FWEngine(device="cpu")
    return _ENGINE["cpu"]


def get_engine():
    """Apple Silicon - MLX; иначе faster-whisper (видеокарта NVIDIA или процессор)."""
    forced = getattr(_FORCED, "eng", None)
    if forced is not None:
        return forced
    if "e" not in _ENGINE:
        want = os.environ.get("SCRIBE_ENGINE", "")
        eng = None
        if want != "fw" and pu.IS_APPLE_SILICON:
            try:
                eng = MLXEngine()
            except Exception:
                eng = None
        _ENGINE["e"] = eng or FWEngine()
    return _ENGINE["e"]


def engine_info():
    """Справочная строка для Настроек."""
    try:
        e = get_engine()
        return f"Распознавание: {e.label}"
    except Exception as ex:
        return f"Распознавание недоступно: {ex}"
