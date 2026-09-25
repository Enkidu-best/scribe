#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Процесс распознавания для диктофона Scribe.

Запускается как отдельный процесс из dictaphone.py. Держит модель MLX Whisper
и распознаёт присланные фразы. Вынесен в отдельный процесс специально: тяжёлое
распознавание держит Python-поток (GIL), и если бы оно шло в основном процессе,
захват микрофона «замерзал» бы на время распознавания и речь терялась. В своём
процессе оно не мешает захвату — микрофон пишет непрерывно.

Связь с основным процессом — multiprocessing.connection по локальному сокету.
Протокол:
  воркер -> основной:  ("ready", None)  — подключился
                       ("warmed", None) — модель загружена (прогрев сделан)
                       ("text", str)    — распознанный текст фразы
                       ("error", str)   — ошибка распознавания
  основной -> воркер:  (audio_float32, lang, prompt)  — фраза на распознавание;
                       prompt — «хвост» уже набранного текста, чтобы Whisper
                       продолжал мысль в том же регистре и с пунктуацией
                       None                   — команда завершения
"""
import sys

PUNCT_PROMPT = "Привет! Как дела? Хорошо, спасибо. Let's continue, okay?"


def main():
    from multiprocessing.connection import Client
    addr = sys.argv[1]
    authkey = bytes.fromhex(sys.argv[2])
    model_repo = sys.argv[3]

    conn = Client(addr, authkey=authkey)
    import numpy as np
    import mlx_whisper

    conn.send(("ready", None))

    # Прогрев: заранее загружаем модель на коротком молчании, чтобы первая
    # реальная фраза не ждала загрузку модели.
    try:
        mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32),
                               path_or_hf_repo=model_repo, language="ru",
                               verbose=False)
    except Exception:
        pass
    try:
        conn.send(("warmed", None))
    except Exception:
        return

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if msg is None:
            break
        audio, lang, prompt = msg
        try:
            res = mlx_whisper.transcribe(
                np.ascontiguousarray(audio), path_or_hf_repo=model_repo,
                language=lang, temperature=0.0, condition_on_previous_text=False,
                initial_prompt=(prompt or PUNCT_PROMPT), verbose=False)
            conn.send(("text", (res.get("text") or "").strip()))
        except Exception as e:
            try:
                conn.send(("error", str(e)))
            except Exception:
                break

    try:
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
