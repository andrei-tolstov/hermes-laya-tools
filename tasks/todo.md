# План: bridge-тулзы Hermes, синхронизация valid_tool_names, фикс looks_trivial

Утверждённый план: `implementation_plan.md` (артефакт сессии).

- [x] 1. `hygiene.py`: `looks_trivial` понимает ролевые префиксы (`USER:`/`ASSISTANT:`), берёт последнюю реплику пользователя (и не считает тривиальным короткий ответ на вопрос ассистента со знаком `?`)
- [x] 2. `tool_collector.py`: `bridge_tool_names()`, `discovery_tool_names()`, `get_bridge_tools(..., fallback_sources, rebuild)` (tools → сохранённый список → сессионный кэш → `tools.tool_search.bridge_tool_schemas`), `verify_bridge_tools()`, `messages_used_bridge_tools()`
- [x] 3. `router.py`: `route_tools` всегда добавляет bridge-тулзы (`tool_search`, `tool_describe`, `tool_call`) во всех активных режимах сверх `max_tools`, проверяет их наличие с выводом `ERROR` в лог/аудит и `fail-open` при отсутствии в Hermes, раскрывает полный список (`bridge_unlocked`) после вызова поисковых тулзов
- [x] 4. `__init__.py`: `_apply_effective_tools` (синхронизация `agent.tools` и `agent.valid_tool_names`), `_on_post_tool_call` мгновенно раскрывает полный список после `tool_search`/`tool_describe`/`tool_call`/`manage_catalog`/`manage_connections` (и вызовов скрытых тулзов через мост), `/laya status` показывает состояние `Hermes bridge tools`
- [x] 5. Тесты: `tests/test_bridge_tools.py` (22 теста), обновлены `tests/test_hygiene.py` и `tests/test_enrich_top3.py` (убран мок `looks_trivial`)
- [x] 6. Локально `python run_tests.py` — 88/88 passed
- [x] 7. Деплой на `192.168.50.38` (бэкап в `/root/.hermes/backups/`), тесты на сервере и в `.venv` Hermes — passed, `hermes plugins validate /root/.hermes/plugins/hermes-laya-tools` — Validation passed
- [x] 8. Обновлены `README.md` и `tasks/lessons.md`

## Review
- Устранена первопричина ошибки `Tool '...' does not exist` (рассинхронизация `agent.tools` и `agent.valid_tool_names` в Hermes).
- Стандартные инструменты прогрессивного раскрытия Hermes (`tool_search`, `tool_describe`, `tool_call`) теперь гарантированно сохраняются на каждом ходу и во всех режимах маршрутизации.
- Добавлена самодиагностика `verify_bridge_tools()` на случай будущих изменений в Hermes: при исчезновении или переименовании мостовых инструментов пишется `ERROR` и включается `fail-open` (отдаётся полный каталог инструментов).
- Вызов `tool_search` / `tool_describe` / `tool_call` (как внутри хода через `post_tool_call`, так и между ходами через историю сообщений и состояние сессии) немедленно открывает модели полный список всех специализированных и базовых инструментов.
