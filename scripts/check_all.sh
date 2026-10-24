#!/usr/bin/env bash
# Проверка перед push одной командой - общий блок «_общие модули/ci» (что запускать - scripts/check_all.conf).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/ci/check_all.sh" "$@"
