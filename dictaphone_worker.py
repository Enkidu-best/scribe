#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Процесс распознавания для диктофона Scribe.

Запускается как отдельный процесс из dictaphone.py. Держит модель Whisper (engine.py:
MLX на Mac - тот же вызов mlx_whisper, что раньше; faster-whisper на Windows и Linux)
и распознаёт присланные фразы. Вынесен в отдельный процесс специально: тяжёлое
распознавание держит Python-поток (GIL), и если бы оно шло в основном процессе,
захват микрофона «замерзал» бы на время распознавания и речь терялась. В своём
процессе оно не мешает захвату — микрофон пишет непрерывно.

Связь с основным процессом — multiprocessing.connection: на Mac и Linux — локальный
сокет, на Windows — именованный канал (pipe). Сообщения те же.
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

    conn = Client(addr, authkey=authkey)        # адрес сам говорит, сокет это или канал Windows
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import engine                              # MLX на Mac (тот же вызов, что раньше), faster-whisper - иначе
    eng = engine.get_engine()

    conn.send(("ready", None))

    # Прогрев: заранее загружаем модель на коротком молчании, чтобы первая
    # реальная фраза не ждала загрузку модели.
    try:
        eng.warmup(model_repo)
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
            conn.send(("text", eng.transcribe_text(audio, model_repo, lang, prompt or PUNCT_PROMPT)))
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
