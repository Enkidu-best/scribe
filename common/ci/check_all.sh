#!/usr/bin/env bash
# Проверка перед push одной командой (стандарт 6.8, «GitHub 01»): ruff → юнит-тесты без моделей → check_repo
# [→ окно Playwright по --ui]. Та же команда идёт в облаке (.github/workflows/linux.yml, там CI=true).
#   scripts/check_all.sh [--ui] [--only ruff|unit|repo|ui]
# Что запускать - scripts/check_all.conf проекта (bash): PY, RUFF_PATHS, RUFF_SELECT, функции ruff_step / unit / repo / ui.
# Код 0 - всё прошло, 1 - что-то не прошло, 2 - неверный ключ.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # <проект>/common/ci
ROOT="${CHECK_ROOT:-$(cd "$HERE/../.." && pwd)}"
cd "$ROOT" || exit 2
UI=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --ui) UI=1 ;;
    --only) ONLY="${2:-}"; shift ;;
    -h|--help) sed -n 2,6p "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "неизвестный ключ: $1"; exit 2 ;;
  esac
  shift
done

# Python: в облаке - python раннера, на Mac - .venv проекта (check_all.conf может указать свой)
if [ -n "${CI:-}" ]; then
  PY=python
else
  PY=python3
  for v in .venv .venv312; do [ -x "$ROOT/$v/bin/python" ] && { PY="$ROOT/$v/bin/python"; break; }; done
fi
RUFF_PATHS=(.)
RUFF_SELECT="F821,F841,B023"   # стандарт 6.11
ruff_step() {
  if "$PY" -m ruff --version >/dev/null 2>&1; then "$PY" -m ruff check --select "$RUFF_SELECT" "${RUFF_PATHS[@]}"
  elif command -v ruff >/dev/null 2>&1; then ruff check --select "$RUFF_SELECT" "${RUFF_PATHS[@]}"
  else echo "ruff не установлен: $PY -m pip install ruff"; return 1
  fi
}
unit() { "$PY" -m pytest -q -p no:cacheprovider -m "not model and not ui"; }
repo() {
  if [ -f scripts/check_repo.py ]; then "$PY" scripts/check_repo.py; else echo "нет scripts/check_repo.py - пропуск"; fi
}
ui() { echo "проверка окна не задана в scripts/check_all.conf"; return 1; }
# shellcheck source=/dev/null
[ -f scripts/check_all.conf ] && . scripts/check_all.conf

RES=()
FAIL=0
step() {
  local name="$1"; shift
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then return 0; fi
  printf '\n── %s\n' "$name"
  local t0=$SECONDS rc
  "$@"; rc=$?
  if [ $rc = 0 ]; then RES+=("  $name - да, $((SECONDS - t0)) с"); else RES+=("  $name - НЕТ (код $rc), $((SECONDS - t0)) с"); FAIL=1; fi
}

T0=$SECONDS
step ruff ruff_step
step unit unit
step repo repo
if [ $UI = 1 ] || [ "$ONLY" = ui ]; then step ui ui; fi

# облачная проверка приватного проекта должна совпадать с шаблоном общего блока
if [ -f .github/workflows/linux.yml ] && ! cmp -s .github/workflows/linux.yml "$HERE/linux.yml"; then
  RES+=("  предупреждение: .github/workflows/linux.yml отличается от common/ci/linux.yml - common/ci/install.sh --workflow")
fi

printf '\n── итог (%s, %d с)\n' "$(basename "$ROOT")" "$((SECONDS - T0))"
printf '%s\n' "${RES[@]}"
exit $FAIL
