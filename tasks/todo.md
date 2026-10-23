# План выполнения: Двухуровневое распределение и отправка полного списка инструментов (88 инструментов) в Laya

## Задачи
- [x] 1. Исследовать и реализовать сбор полного каталога всех доступных инструментов (hot + catalog/deferred/MCP) в `tool_collector.py` / `__init__.py` <!-- id: 0 -->
- [x] 2. Обновить `categorizer.py`: добавить поддержку категорий `marketplace_and_commerce`, `desktop_and_computer_use` и расширить паттерны для всех 88 инструментов <!-- id: 1 -->
- [x] 3. Обновить двухуровневую маршрутизацию в `router.py`: Level 1 (Категории из всех 88 инструментов) и Level 2 (Top-N выборка из полного пула инструментов категории) <!-- id: 2 -->
- [x] 4. Связать сбор полного списка инструментов с хуками `pre_llm_call`, `pre_api_request`, командами `/laya status` и `/laya recategorize` в `__init__.py` <!-- id: 3 -->
- [x] 5. Расширить тестовый набор `tests/` новыми тестами на сбор 88 инструментов, двухуровневую маршрутизацию MCP и запустить `python run_tests.py` <!-- id: 4 -->
- [x] 6. Синхронизировать обновления на сервер `192.168.50.38` и проверить работу `hermes laya recategorize` и `hermes laya status` (проверить отображение 88 инструментов) <!-- id: 5 -->
- [x] 7. Зафиксировать изменения в git (commit & push в GitHub `andrei-tolstov/hermes-laya-tools`) <!-- id: 6 -->
- [x] 8. Документировать результаты в `tasks/todo.md` <!-- id: 7 -->

## Review

### Выполненные работы:
1. **Сбор полного каталога инструментов (`tool_collector.py`)**:
   - Реализована функция `collect_all_tools()`, извлекающая полный каталог инструментов без сворачивания прогрессивного раскрытия (`skip_tool_search_assembly=True`), включая центральный реестр (`tools.registry`), зарегистрированные MCP-серверы (включая `marketplace-mcp` с 36-40 инструментами) и `agent.tools`.
   - Изолированы сервисные bridge-инструменты (`tool_search`, `tool_describe`, `tool_call`), чтобы Laya оценивала и маршрутизировала запросы исключительно по реальным возможностям инструментов, сохраняя bridge-инструменты в финальном наборе активного режима для совместимости с progressive disclosure.

2. **Семантическая категоризация (`categorizer.py`)**:
   - Добавлены новые семантические категории:
     - `marketplace_and_commerce` (Ozon, Wildberries, Yandex Market, Avito, AliExpress, Taobao, Lamoda, Detmir, Drom, Citilink, DNS, Cian, MPStats, корзины, заказы, сравнение цен).
     - `desktop_and_computer_use` (Computer Use, OS GUI, перемещение мыши, клики, оконный менеджмент, preview).
   - Расширены регулярные выражения для распознавания инструментов всех категорий с кэшированием распределения на 24 часа.

3. **Двухуровневая маршрутизация (`router.py`)**:
   - Level 1: классификация намерения (`direct_chat` vs `execute_tools`) и выбор наиболее релевантной категории из полного пула (77–88 инструментов).
   - Level 2: Top-N ранжирование инструментов внутри победившей категории с автоматическим добором из смежных категорий и безусловным включением `always_include`.
   - В Shadow Mode: отправка полного каталога в Laya для аудита и логирования с сохранением исходного набора `tools` для основной LLM.

4. **Хуки и CLI команды (`__init__.py`, `config.py`)**:
   - Хуки `pre_llm_call` и `pre_api_request` передают полный каталог и ссылку на live agent в роутер.
   - Команды `/laya status` и `/laya recategorize` (а также CLI `hermes laya status` и `hermes laya recategorize`) выводят точную сводку распределения по всем категориям.
   - Реализован сброс одноразового триггера `recategorize_now` в `config.yaml` (`clear_yaml_setting`) без нарушения форматирования и комментариев.

5. **Тестирование (`tests/test_two_level_tools.py`, `run_tests.py`)**:
   - Написаны тесты на синтетическом каталоге из 88 инструментов (21 hot, 10 fs/code, 10 desktop, 7 system, 40 marketplace).
   - Проверена изоляция тестов от хостового `~/.hermes/config.yaml`.
   - Все 41 тест успешно проходят на локальной машине и на удаленном сервере `192.168.50.38`.

6. **Проверка на сервере `192.168.50.38`**:
   - Выполнена команда `hermes laya recategorize`:
     `⚡ Laya: Всего тулзов: 77, категорий: 7. Распределение: marketplace_and_commerce: 36, desktop_and_computer_use: 15, filesystem: 10, meta_and_skills: 6, web_and_search: 5, terminal_execution: 4, multimedia_and_vision: 1`
   - Команда `hermes laya status` подтвердила корректное сохранение и отображение распределения.
