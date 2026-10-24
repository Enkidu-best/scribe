#!/usr/bin/env bash
# Подключить общий блок ci в проекте (после «sync_common.sh ci <проект>»):
#   common/ci/install.sh [--workflow]
# - scripts/check_all.sh - запуск common/ci/check_all.sh (если файла нет);
# - .git/hooks/pre-push - ссылка на common/ci/pre-push; чужой хук не удаляется, а переименовывается в pre-push.old;
# - --workflow: .github/workflows/linux.yml из шаблона (приватные репозитории, правило «GitHub 01»).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
WF=0
[ "${1:-}" = "--workflow" ] && WF=1

mkdir -p scripts
if [ ! -f scripts/check_all.sh ]; then
  cat > scripts/check_all.sh <<'EOF'
#!/usr/bin/env bash
# Проверка перед push одной командой - общий блок «_общие модули/ci» (что запускать - scripts/check_all.conf).
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common/ci/check_all.sh" "$@"
EOF
  chmod +x scripts/check_all.sh
  echo "scripts/check_all.sh - создан"
fi

HOOKS="$(git rev-parse --git-path hooks)"
mkdir -p "$HOOKS"
STUB='#!/usr/bin/env bash
# общий блок ci - см. common/ci/pre-push
exec "$(git rev-parse --show-toplevel)/common/ci/pre-push" "$@"'
if [ -f "$HOOKS/pre-push" ] && [ "$(cat "$HOOKS/pre-push")" != "$STUB" ]; then
  mv "$HOOKS/pre-push" "$HOOKS/pre-push.old"
  echo "прежний хук pre-push - в $HOOKS/pre-push.old"
fi
printf '%s\n' "$STUB" > "$HOOKS/pre-push"
chmod +x "$HOOKS/pre-push" "$HERE/pre-push" "$HERE/check_all.sh"
echo "хук pre-push - подключён"

if [ $WF = 1 ]; then
  mkdir -p .github/workflows
  cp "$HERE/linux.yml" .github/workflows/linux.yml
  echo ".github/workflows/linux.yml - из шаблона"
fi
