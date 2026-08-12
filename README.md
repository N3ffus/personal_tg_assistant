# Telegram AI Assistant

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/types-mypy-blue)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)

Telegram-бот на Python, который передаёт сообщения пользователя в LLM и превращает её ответ в структурированное действие: обычный ответ, задачу, событие или заметку.

Интеграции с календарём, трекером задач и хранилищем заметок пока не подключены: бот определяет нужное действие и формирует ответ пользователю.

## Требования

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- токен Telegram-бота и ключ совместимого с OpenAI LLM API

## Быстрый старт

```bash
git clone <repository-url>
cd personal_ai_assistent
make install
```

Создайте `.env` на основе примера:

```bash
cp .env.example .env
```

На Windows вместо этого:

```powershell
Copy-Item .env.example .env
```

Заполните значения:

```env
TELEGRAM_BOT_TOKEN=your-telegram-token
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5.6
```

Запустите бота:

```bash
make run
```

## Docker

Соберите образ:

```bash
make docker-build
```

Запустите бота в фоне, передав настройки из `.env`:

```bash
make docker-run
```

При повторном запуске предыдущий контейнер с тем же именем будет заменён. Для просмотра логов и остановки используйте:

```bash
make docker-logs
make docker-stop
```

Для запуска нужен работающий Docker Engine (Docker Desktop на Windows).

## Разработка

`Makefile` использует `uv`, поэтому команды работают одинаково в Windows, Linux и macOS.

| Команда | Назначение |
| --- | --- |
| `make install` | Установить зависимости, включая инструменты разработки |
| `make run` | Запустить бота |
| `make lint` | Проверить стиль и ошибки Ruff |
| `make format` | Отформатировать код и применить безопасные исправления Ruff |
| `make typecheck` | Запустить строгую проверку типов mypy |
| `make test` | Запустить тесты pytest |
| `make check` | Запустить все проверки: Ruff, mypy и pytest |
| `make docker-build` | Собрать Docker-образ `personal-ai-assistant` |
| `make docker-run` | Собрать и запустить бота в Docker с `.env` |
| `make docker-logs` | Следить за логами контейнера |
| `make docker-stop` | Остановить контейнер |

Перед отправкой изменений выполните:

```bash
make check
```

## Структура

```text
src/
├── application/    # use cases, сервисы и порты
├── domain/         # модели и типы действий
├── infrastructure/ # реализации LLM- и Telegram-адаптеров
├── config.py       # настройки окружения
└── main.py         # запуск приложения
tests/              # автоматические тесты
```
