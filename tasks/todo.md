# План: Создание скила обучения и калибровки Laya на основе сессий Hermes и логов плагина

## Обзор задачи
Создать автономный, полнофункциональный скилл `laya-train` (соответствующий лучшим практикам создания скилов для Hermes и Antigravity) для дистилляции, обучения и оценки точности Laya System 1 Decision Engine на основе реальных сессий Hermes (`state.db`) и логов плагина (`laya_requests.jsonl`).
При отсутствии фактических вызовов инструментов или для разрешения неоднозначностей в обучающих данных скилл выполняет запрос к основной языковой модели (Primary LLM Judge) для получения эталонной разметки (Ground Truth).

---

## Задачи

- [x] 1. Исследовать и зафиксировать лучшие практики создания скилов (структура `SKILL.md`, progressive disclosure, `scripts/`, `references/`, `examples/`, валидация и отчёты) <!-- id: 0 -->
- [x] 2. Разработать скрипт извлечения и разметки данных `extract_and_label.py`: парсинг `state.db` и `laya_requests.jsonl`, эвристика вызовов, fallback-запрос к основной модели (Primary Model Teacher) при отсутствии эталона, генерация `train.jsonl` и `eval.jsonl` <!-- id: 1 -->
- [x] 3. Разработать скрипт бенчмаркинга и тестирования точности `evaluate_accuracy.py`: расчёт метрик (Accuracy, Brier score, Mean Confidence, ECE, по категорийный срез) до и после обучения через live-эндпоинт Laya (`/ask` / `/v1/systemone`) <!-- id: 2 -->
- [x] 4. Разработать оркестратор обучения и компиляции `train_pipeline.py`: выполнение пре-теста, запуск `laya-train`, компиляция GGUF через `ggmlc`, горячий reload Laya (`POST /reload`), выполнение пост-теста и генерация сравнительного отчёта <!-- id: 3 -->
- [x] 5. Создать спецификацию навыка `skills/laya-train/SKILL.md` с YAML frontmatter, пошаговыми инструкциями, командами запуска и описанием структуры отчёта <!-- id: 4 -->
- [x] 6. Создать документацию в `references/` и примеры датасетов в `examples/` для соблюдения progressive disclosure <!-- id: 5 -->
- [x] 7. Протестировать работу скриптов локально и на удалённом сервере `192.168.50.38` (сбор реальных данных, пре-тест точности, формирование отчёта) <!-- id: 6 -->
- [x] 8. Синхронизировать скилл в каталог навыков Hermes (`~/.hermes/skills/laya-train`) на сервере `192.168.50.38` и зафиксировать изменения в git (commit & push) <!-- id: 7 -->
- [x] 9. Сформировать итоговый обзор в `tasks/todo.md` и представить результаты пользователю <!-- id: 8 -->

## Review

### Результаты разработки скила `laya-train`
1. **Структура и размещение**:
   - Все компоненты, скрипты, документация и данные изолированы внутри каталога `skills/laya-train/`:
     - `skills/laya-train/SKILL.md` — манифест скила с YAML frontmatter и Runbook для Hermes / Antigravity.
     - `skills/laya-train/scripts/`:
       - `extract_and_label.py` — извлечение логов `state.db` и `laya_requests.jsonl`, эвристика вызовов, fallback-запрос к Primary LLM Teacher (`Omniroute/Hermes 127.0.0.1:20128`) для получения Ground Truth при неоднозначностях.
       - `evaluate_accuracy.py` — расчет метрик точности (Tool Decision Accuracy, Category Accuracy, Strict Combined Accuracy, Brier score, Mean Confidence, ECE, срез по категориям) через HTTP API Laya (`/ask` / `/v1/systemone`).
       - `train_pipeline.py` — сквозной конвейер (Pre-Test $\rightarrow$ Training / Calibration $\rightarrow$ GGUF Export $\rightarrow$ Hot Reload Laya $\rightarrow$ Post-Test $\rightarrow$ Markdown & JSON Reports).
     - `skills/laya-train/data/` — рабочие и итоговые датасеты (`train.jsonl`, `eval.jsonl`), метрики (`eval_before.json`, `eval_after.json`) и отчёты (`report.md`, `report.json`).
     - `skills/laya-train/references/` — архитектура Laya, руководство по RLCD-обучению и лучшие практики скилов.
     - `skills/laya-train/examples/` — демонстрационные датасеты для быстрой проверки.

2. **Валидация на реальном сервере (192.168.50.38)**:
   - Скилл развернут в `/root/.hermes/skills/laya-train`.
   - `hermes skills list` успешно зарегистрировал и включил скилл (`laya-train | local | enabled`).
   - `extract_and_label.py` извлек 35 уникальных запросов из реальных логов и базы Hermes, обратившись к основной модели Hermes за эталоном при необходимости.
   - `train_pipeline.py` отработал цикл пре-теста и пост-теста на живом демоне Laya (`http://127.0.0.1:8787`).
   - Итоговый отчёт сохранен в `skills/laya-train/data/report.md` и `report.json`.
