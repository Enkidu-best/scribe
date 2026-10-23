#!/usr/bin/env bash
# Очередь видеокарты .heavy-lock для shell и ночных скриптов (описание - README.md, 6.2).
#   heavy_lock.sh take "<программа>" "<что делаю>" [--who human|background] [--wait-min 330] [--pid $$]
#   heavy_lock.sh release "<программа>" [--pid $$]
#   heavy_lock.sh yield "<программа>" [--pid $$]     - контрольная точка фона: ждёт человек - отдать и снова взять
#   heavy_lock.sh status [--json]
#   heavy_lock.sh run "<программа>" "<что делаю>" -- команда аргументы…
# По умолчанию владелец - процесс, вызвавший скрипт; кто - фон. Логика одна - heavy_lock.py.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HEAVY_LOCK_PY:-python3}" "$HERE/heavy_lock.py" "$@"
