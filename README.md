# Telegram AI Assistant

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/types-mypy-blue)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)

Telegram-бот на Python, который превращает естественный текст в
структурированные действия. Бот создаёт задачи в Linear, работает с
Google Calendar, отвечает на обычные сообщения и получает личные сообщения
через Telegram Business.

## Требования

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- токен Telegram-бота и ключ совместимого с OpenAI LLM API
- персональный Linear API-ключ и UUID команды
- Google OAuth credentials и публичный HTTPS callback URL

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
TELEGRAM_ALLOWED_USER_ID=your-numeric-telegram-user-id
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5.6
GOOGLE_OAUTH_CLIENT_ID=your-client-id
GOOGLE_OAUTH_CLIENT_SECRET=your-client-secret
GOOGLE_OAUTH_REDIRECT_URI=https://your-domain.example/oauth/google/callback
GOOGLE_TOKEN_ENCRYPTION_KEY=your-fernet-key
LINEAR_API_KEY=your-linear-api-key
LINEAR_TEAM_ID=your-team-uuid
```

## Linear

Создайте [персональный API-ключ](https://linear.app/developers/graphql) в
**Settings → Account → Security & Access** и
разрешите ему создание issues. UUID команды можно скопировать в Linear
через `Cmd/Ctrl+K` → **Copy model UUID**. Запишите значения в
`LINEAR_API_KEY` и `LINEAR_TEAM_ID`.

Когда бот распознаёт действие `create_task`, он сразу создаёт issue в
настроенной команде и возвращает его идентификатор и ссылку. Один
ключ и команда общие для экземпляра бота, поэтому свободный текст, LLM и
создание Linear-задач доступны только Telegram-пользователю с ID из
`TELEGRAM_ALLOWED_USER_ID`.

## Telegram Business

1. Включите **Business Mode** для бота через `@BotFather`.
2. В Telegram откройте **Settings → Chat Automation** (в старых клиентах —
   **Telegram Business → Chatbots**), подключите бота и выберите чаты, к которым
   он получит доступ.
3. Разрешите боту чтение сообщений. ID подключаемого аккаунта должен совпадать
   с `TELEGRAM_ALLOWED_USER_ID`.

Приложение подписывается на `business_connection`, `business_message` и
`edited_business_message`. Оно
проверяет владельца подключения, получает входящие личные сообщения и помечает
их прочитанными. Исходящие сообщения владельца и бизнес-бота игнорируются.
Автоответы и запуск команд Linear/Calendar из клиентских сообщений намеренно
отключены.

## Google Calendar

Создайте OAuth Client ID типа **Web application** в Google Cloud Console, включите
Google Calendar API и добавьте `GOOGLE_OAUTH_REDIRECT_URI` в список Authorized
redirect URIs. Для шифрования токенов создайте ключ:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Приложение запускает HTTP-сервис на `HTTP_HOST:HTTP_PORT` (по умолчанию
`0.0.0.0:8080`), который должен быть доступен снаружи по HTTPS через reverse
proxy. В Telegram используйте `/calendar_connect`, затем `/calendar` для
ближайших событий. Кнопка «Изменить» запоминает выбранное событие и ожидает
сообщение с новым названием и временем; удаление требует подтверждения. Для
отключения аккаунта используйте `/calendar_disconnect`.

Reverse proxy не должен сохранять query string OAuth callback в access logs.
Одноразовые состояния и кнопки календаря привязаны к
`TELEGRAM_ALLOWED_USER_ID`, имеют TTL и хранят event ID только на сервере.

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

Команда публикует порт `8080` для OAuth callback и хранит SQLite-базу в
именованном Docker volume, поэтому подключения календарей переживают пересоздание
контейнера.

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
| `make test` | Запустить pytest с branch coverage и обязательным порогом 90% |
| `make check` | Запустить все проверки: Ruff, mypy и pytest |
| `make docker-build` | Собрать Docker-образ `personal-ai-assistant` |
| `make docker-run` | Собрать и запустить бота в Docker с `.env` |
| `make docker-logs` | Следить за логами контейнера |
| `make docker-stop` | Остановить контейнер |

Перед отправкой изменений выполните:

```bash
make check
```

Тот же gate запускается в GitHub Actions; дополнительно CI выполняет Bandit и
pip-audit. Результаты последнего предрелизного ревью находятся в
[`docs/audit/CODE_AUDIT_REPORT_2026-09-04.md`](docs/audit/CODE_AUDIT_REPORT_2026-09-04.md).

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
