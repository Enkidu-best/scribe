# Проверки перед push и в облаке - общий модуль

Версия 1.0.1 · 25.10.2026 · задание «_GitHub 01»; 1.0.1 - «_Единый вид 02», п. 3. Стандарт: «Единый стандарт программ», 6.8 - общий блок `common/ci/`.

## 1. Зачем

План GitHub - бесплатный (решение Андрея 24.10). В октябре 2026 бесплатные 2 000 минут приватных репозиториев кончились
к 24.10, дальше проверки в приватных репозиториях блокируются до 1 ноября. Минуты съели Windows-проверки (минута Windows
дороже Linux), запуск на каждый push в любую ветку и правки `docs/`.

## 2. Практика

| Что | Источник | Что берём |
|---|---|---|
| GitHub Free: 2 000 минут в месяц на приватные репозитории; в публичных стандартные машины бесплатны; цена минуты: Linux 2 ядра $0,006, Windows $0,010, macOS $0,062 | docs.github.com, «How use of GitHub Actions is measured» | приватные - только Linux; Windows - в публичных или вручную |
| `paths-ignore`, `concurrency: cancel-in-progress`, кэш pip в `setup-python` | документация GitHub Actions | правка `docs/` не запускает проверку, новый push отменяет старый запуск |
| Проверка перед отправкой - git-хук `pre-push`: код не 0 - push не уходит | git, «githooks» | ruff, юнит-тесты, check_repo - до GitHub, на Mac |

## 3. Правило (стандарт 6.8)

1. **Приватные репозитории - только Linux**, одна задача (`linux.yml` из этого модуля): `ruff` + юнит-тесты без
   моделей + `check_repo`, та же команда, что на Mac перед push. Без macOS и Windows; окно (Playwright) - локально.
2. Запуск - на push в `main` и на метки `v*`, кроме правок `docs/`, `*.md`, `*.docx` в корне; `concurrency` отменяет
   прошлый запуск той же ветки; кэш pip; таймаут 10 мин.
3. **Windows** - только в публичных (Scribe, Herald, Anonymizer, Markdown) или вручную (`windows.yml`,
   `workflow_dispatch`) перед релизом.
4. **Перед push** - `scripts/check_all.sh` (ruff, юнит, check_repo; окно - `--ui`), хук `pre-push` в каждом проекте.
   Пропустить один раз - `git push --no-verify` (только если проверка уже прошла вручную).
5. В конце месяца - строка в «Очередь проектов.md»: сколько минут ушло (по `gh run list`, число запусков ×
   длительность задач).

## 4. Как устроено

| Файл | Что делает |
|---|---|
| `check_all.sh` | шаги `ruff` → `unit` → `repo` [→ `ui` по `--ui`], итог с временем каждого шага; код 1, если не прошёл хоть один (остальные всё равно идут). `--only <шаг>` - один шаг. Python: в облаке (`CI=true`) - `python` раннера, на Mac - `.venv` проекта. Предупреждает, если `.github/workflows/linux.yml` проекта отличается от шаблона |
| `pre-push` | хук: перед отправкой - `scripts/check_all.sh`; удаление ветки или метки на GitHub не проверяется |
| `install.sh [--workflow]` | `scripts/check_all.sh` (если нет), `.git/hooks/pre-push` - короткая ссылка на `common/ci/pre-push` (чужой хук - в `pre-push.old`), `--workflow` - `linux.yml` из шаблона |
| `linux.yml` | облачная проверка приватного репозитория: Python 3.12, `requirements-ci.txt` проекта, `scripts/ci-apt.txt` (если есть), `scripts/check_all.sh` |

Что проверять - в проекте, `scripts/check_all.conf` (bash):

```bash
RUFF_PATHS=(cleardoc app scripts tests)        # по умолчанию (.); правила - F821, F841, B023 (стандарт 6.11)
unit() { "$PY" -m pytest -q -m "not model and not ui"; }   # по умолчанию так
repo() { "$PY" scripts/check_repo.py; }        # по умолчанию так, если файл есть
ui()   { "$PY" scripts/check_12_ui.py; }       # только своё окно: свой сервер и порт, своя папка данных
# PY - свой Python (Scribe: Python.framework 3.12); ruff_step() - свой шаг ruff, если нужен
```

`requirements-ci.txt` проекта - зависимости без моделей, torch, Playwright и Apple Vision: облачная проверка ставит
только его.

## 5. Подключить в проекте

```bash
"_общие модули/scripts/sync_common.sh" ci "<проект>"
cd "<проект>" && bash common/ci/install.sh --workflow    # приватный; публичный - без --workflow
# написать scripts/check_all.conf и requirements-ci.txt; старый tests.yml с Windows - в windows.yml (workflow_dispatch)
scripts/check_all.sh
```

Хук живёт в `.git/hooks` и в git не попадает: на новой машине после клона - `bash common/ci/install.sh`.

## 6. Проверки

`python3 "_общие модули/ci/tests/test_ci.py"` - проект-пробник во временной папке с git и «GitHub» рядом (пустой
репозиторий; нет ruff у этого Python - шаг ruff заглушкой, тест F821 пропускается): все шаги прошли; один шаг не прошёл - код 1, остальные идут; ruff находит F821; `--ui` без окна в
настройке - ошибка; неверный ключ - код 2; в облаке - `python` раннера, на Mac - `.venv`; шаблон облака изменён в
проекте - предупреждение; хук не пускает push с ошибкой и пускает с `--no-verify`; хорошая проверка - push уходит;
удаление метки не проверяется; чужой хук сохраняется в `pre-push.old`, повторная установка ничего не меняет.

## 7. Где подключён (25.10.2026)

| Проект | GitHub | `linux.yml` | Windows | `--ui` |
|---|---|---|---|---|
| Понятный документ | clear-doc, приватный | да | `windows.yml` вручную | `scripts/check_12_ui.py` |
| Interview Coach | interview-coach, приватный | да | `windows.yml` вручную | `scripts/ui_check.py` (порт 8041) |
| Legal Doc Builder | legal-doc-builder, приватный | да | `windows.yml` вручную | на запущенной программе - вручную |
| Legal Knowledge Desk | legal-knowledge-desk, приватный | да | `windows.yml` вручную (установщик, скорость - галочкой) | на запущенной программе - вручную |
| Настольный помощник | tabletop-assistant, приватный | да (раньше проверок не было) | - | `tests/test_ui.py` |
| Scribe | scribe, публичный | - (свои `ci.yml`, `release.yml`) | на push, бесплатно | `tests/scenario_gpu_lock.py` |
| Herald | Herald, публичный | - (свои `ci.yml`, `release.yml`) | на push, бесплатно | «Инструменты разработки/test_gui.py» |
| Anonymizer | Anonymizer, публичный | да (25.10, ветка `master`; бесплатно) | - (не добавлять) | `pytest -m ui` + `scripts/ui_check.py` (порт 8074) |

Не подключён: Markdown (папка на Mac - не клон репозитория), Legal Ops Hub (нет репозитория на GitHub), ascend-english
(нет копии на Mac, проверок в репозитории нет).

## Журнал

- 1.0.1 (25.10.2026) - `linux.yml`: запуск и на ветке `master` (Anonymizer), шапка - не только для приватных.
- 1.0.0 (24.10.2026) - первая версия.
