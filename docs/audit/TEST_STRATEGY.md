# Test Strategy

## Цель

Не допустить релиза, который нарушает single-user boundary, теряет или неверно
изменяет Calendar/Linear data, раскрывает OAuth credentials, зависает при shutdown
или падает на недоверенном внешнем payload.

## Risk-to-test matrix

| Риск | Уровень теста | Покрытие |
| --- | --- | --- |
| Owner authorization bypass | Router unit/negative | Все Telegram commands, text и calendar callbacks |
| OAuth CSRF/replay/cross-user use | Storage + ASGI contract | Random, TTL, user binding, atomic single use, callback outcomes |
| Token disclosure | Storage/security | Fernet ciphertext, wrong-key failure, ignored `.env`, secret scan |
| Wrong Calendar mutation | Service/adapter contract | Event selection token, user binding, confirmation, Google request shape |
| Linear duplicate after uncertain response | Adapter/use-case | Timeout/read/write failure gets distinct uncertain outcome |
| External schema drift | Adapter negative matrix | Empty, non-object, malformed and provider-error payloads |
| Telegram platform limits | Boundary unit | 4096-char chunks and 64-byte callback data |
| Async lifecycle leak/hang | Supervisor concurrency | Normal child exit, exception and parent cancellation |
| Dependency vulnerability | CI SCA | pip-audit/OSV and weekly Dependabot |

## Автоматические слои

- Domain/application unit tests без сети.
- Adapter contract tests с mocked HTTP/Google/Telegram transports.
- SQLite integration tests на временных реальных database files.
- FastAPI ASGI tests для `/health` и OAuth callback.
- Concurrency tests для single-use state/operations, Business cache и service supervisor.
- Static gates: Ruff format/lint, strict mypy, Bandit.
- Supply-chain gate: locked `uv` environment, pip-audit и SHA-pinned Actions.
- Prompt/integration eval: 23 сценария с фиксированными датами и записываемыми
  вызовами тестовых портов; DeepEval ToolCorrectness + G-Eval с DeepInfra
  GLM-5.3-Flash. Живой платный запуск включается через `make eval`, см.
  [`evals/README.md`](../../evals/README.md).

## Release gate

Обязательные команды:

```bash
uv lock --check --offline
make check
uvx bandit==1.9.4 -r src
```

Критерии выхода:

- 0 failed tests;
- branch coverage не ниже 90%;
- 0 Ruff/mypy/Bandit findings без документированного triage;
- 0 известных dependency vulnerabilities;
- 0 verified secrets среди Git candidates;
- Docker image build и live smoke выполнены в release environment.

Текущий результат: первые пять критериев выполнены. Docker/live smoke остаются
внешним release step, потому что локальный Docker Engine и sandbox credentials не
доступны.

## Test data и безопасность

Тесты используют только временные SQLite files, fake API keys, mocked transports и
synthetic user IDs. Настоящие `.env`, tokens и production data не должны читаться,
логироваться или попадать в fixtures/snapshots.

Исключение: явно запущенные LLM-evals читают ключи модели приложения и DeepInfra
из `.env` / `.env.eval` для авторизованных API-вызовов. В тестовые запросы и отчёты
попадают только синтетические сценарии, ответы и оценки, без credentials.

## Следующее усиление

- Sandbox E2E для Google OAuth/Calendar и Linear mutation с cleanup.
- Telegram staging bot smoke после каждого deployment.
- Property-based tests для arbitrary Unicode/long titles и malformed JSON.
- Load/soak test polling + callback service и проверка graceful shutdown.
- Проверяемая backup/restore drill для SQLite volume.
