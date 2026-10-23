# Интеграция с плагином kerpopule/hermes-jev-skills и методология дообучения решений

## 1. Обзор репозитория `kerpopule/hermes-jev-skills`

Репозиторий `kerpopule/hermes-jev-skills` представляет собой подключаемый плагин (`hermes-jev`) и набор навыков для агента Hermes. Его ключевая идея:
> *"Give your agent a fast, cheap second brain for the small decisions: LLMs think, Jev decides, tools act."*

TypeSafe Jev выступает как **System 1 Decision Engine** (аналогично Laya), беря на себя высокочастотные структурированные решения:
- **Model routing** (`jev-model-routing`): динамический выбор оптимальной модели (tier/specialty) под сложность текущего хода.
- **Skill selection** (`skillpick`): выбор релевантного навыка из каталога без раздувания контекста.
- **Memory & context filtering**: фильтрация фрагментов памяти (`jev_memory_filter`).
- **Transcript compaction**: отбор критических ходов при компактификации диалога (`compact_select`).
- **Action selection**: выбор дискретного действия в GUI/Browser (`choose_action`).
- **Command-risk gate**: анализ безопасности выполнения команд в терминале (`logs/jev-gate.jsonl`).

---

## 2. Структура логов и артефактов `hermes-jev` в Hermes

Плагин сохраняет телеметрию и решения в домашней директории агента (`~/.hermes/`):

1. **`~/.hermes/logs/jev-decisions.jsonl`**:
   Записи каждого принятого решения в реальном и теневом (Shadow) режимах.
   ```json
   {
     "ts": 1792255462.552,
     "profile": "default",
     "kind": "route",
     "mode": "shadow",
     "from": "custom:qwen38/qwen3.8-27b",
     "routed": false,
     "model": "custom:qwen38/qwen3.8-27b",
     "tier": "fast",
     "confidence": 0.92,
     "latency_ms": 38.4,
     "reason": "code editing detected, requires standard tier"
   }
   ```
2. **`~/.hermes/logs/jev-shadow.jsonl`**:
   Параллельные теневые предсказания для сравнения с фактическим поведением агента.
3. **`~/.hermes/logs/jev-gate.jsonl`**:
   Решения по валидации и рискам запуска команд.
4. **`~/.hermes/jev/state.json`** и **`~/.hermes/jev/routing.json`**:
   Конфигурация профилей, весов, пулов моделей и порогов уверенности.

---

## 3. Методология дообучения и калибровки моделей решений в Jev

Анализ исходных кодов (`scripts/calibrate_choose.py`, `docs/shadow-to-live.md`, `docs/measuring-a-router.md`) выявил следующие ключевые принципы:

### А. Цикл "Shadow-to-Live"
1. **Сбор фактической истории**:
   Извлечение триплетов: `(state, facts, truth)` из реальных сессий агента.
2. **Пакетный прогон гипотез**:
   Прогон решений на исторических данных (`jev batch --policy ...`).
3. **Оценка калибровки (Brier Score, ECE, Reliability Table)**:
   Проверка соответствия выданной моделью уверенности $p$ реальной частоте успеха.
4. **Калибровка порога уверенности (Confidence Floor)**:
   Критический компромисс между:
   - **Stalls** (модель занизила уверенность и отказалась от шага/вызова инструмента, запросив подтверждение человека или старшей LLM);
   - **Wrong Actions** (модель была уверена, но совершила ошибочное действие).
   Главный критерий оптимизации: минимизация `Wrong Actions` при контроле доли `Stalls`.

### Б. Использование Teacher LLM для разрешения неопределенностей
Если в истории сессий действие неочевидно или отсутствует прямое сравнение, привлекается **основная модель Hermes (Primary LLM Judge)** для присвоения эталонной метки (Ground Truth).

---

## 4. Совместное дообучение Laya на основе сессий Hermes и логов Jev

При наличии установленного плагина `kerpopule/hermes-jev-skills`:
1. Скилл `laya-train` выполняет поиск и слияние данных:
   - Сессии пользователя из `~/.hermes/state.db`;
   - Логи обращений к Laya из `~/.hermes/logs/laya_requests.jsonl`;
   - Логи решений Jev из `~/.hermes/logs/jev-decisions.jsonl` и `jev-shadow.jsonl`.
2. Извлеченные решения Jev по сложности хода и классификации намерений обогащают разметку Laya:
   - Если Jev классифицировал ход как требующий выполнения команд или инструментов (`kind: route` с высокими требованиями или вызов специфических утилит), Laya обучается на этом сэмпле с соответствующей категорией (`terminal_execution`, `filesystem`, `desktop_and_computer_use`).
   - Если в логах Jev зафиксирована неоднозначность, вызывается Primary LLM Teacher для формирования эталона.
3. Обученная модель Laya проходит оценку точности и калибровку порогов уверенности до и после дообучения.
