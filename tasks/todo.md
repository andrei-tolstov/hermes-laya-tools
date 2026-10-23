# План: Интеграция проверки плагина kerpopule/hermes-jev-skills и дообучение Laya на основе сессий Hermes и логов Jev

## Обзор задачи
Проверить репозиторий `kerpopule/hermes-jev-skills` на информацию по дообучению и калибровке решений System 1.
Добавить в скилл `skills/laya-train/` автоматическую проверку наличия установленного плагина `kerpopule/hermes-jev-skills` (`hermes-jev`) в Hermes.
При обнаружении плагина:
- Извлекать и связывать сессии Hermes (`state.db`) и логи решений плагина (`~/.hermes/logs/jev-decisions.jsonl`, `jev-shadow.jsonl`);
- При отсутствии явного сравнения/выбора запрашивать эталон (Ground Truth) у основной модели Hermes (Teacher LLM);
- Проводить дообучение/калибровку Laya на объединённом датасете (с калибровкой порогов уверенности по методологии `calibrate_choose` из Jev);
- Замерить точность до и после дообучения;
- Сформировать расширенный отчёт со срезом по Jev-интеграции.
Все скрипты и данные должны оставаться строго в каталоге скилла `skills/laya-train/`.

---

## Задачи

- [x] 1. Исследовать методологию дообучения/калибровки в репозитории `kerpopule/hermes-jev-skills` и зафиксировать в `skills/laya-train/references/jev_integration.md` <!-- id: 0 -->
- [x] 2. Разработать модуль детекции плагина `kerpopule/hermes-jev-skills` (`scripts/jev_detector.py`) с проверкой CLI `hermes plugins`, директорий `~/.hermes/plugins/hermes-jev` и логов `~/.hermes/logs/jev-*.jsonl` <!-- id: 1 -->
- [x] 3. Модифицировать `scripts/extract_and_label.py`: добавить интеграцию логов `jev-decisions.jsonl` и сессий Hermes `state.db`, обращение к Teacher LLM при неопределенности, генерацию обогащенного датасета `train.jsonl` и `eval.jsonl` <!-- id: 2 -->
- [x] 4. Модифицировать `scripts/evaluate_accuracy.py` и `scripts/train_pipeline.py`: добавить калибровку confidence floor / wrong actions (по принципу `calibrate_choose` из `hermes-jev-skills`), запуск дообучения и формирование отчета `report.md` со статистикой Jev <!-- id: 3 -->
- [x] 5. Обновить манифест `skills/laya-train/SKILL.md` с описанием автоматической проверки плагина `kerpopule/hermes-jev-skills` и сценарием ко-дообучения Laya <!-- id: 4 -->
- [x] 6. Протестировать весь цикл на сервере `192.168.50.38` с реальным установленным плагином `hermes-jev` (проверка детектора, сбор данных, дообучение/калибровка, замеры точности) <!-- id: 5 -->
- [x] 7. Синхронизировать скилл на сервере (`/root/.hermes/skills/laya-train/`), зафиксировать изменения в git (commit & push в GitHub) и заполнить Review <!-- id: 6 -->

## Review

### Результаты интеграции плагина `kerpopule/hermes-jev-skills`:
1. **Исследование репозитория `kerpopule/hermes-jev-skills`**:
   - Выявлена архитектура TypeSafe Jev как System 1 Decision Engine для роутинга моделей, выбора скилов, фильтрации памяти и гейта команд.
   - Изучена методология калибровки `calibrate_choose.py` и цикла Shadow-to-Live (`docs/shadow-to-live.md`): приоритет минимизации `Wrong Actions` (ошибочных действий классификатора) над `Stalls` (пасами) и калибровка порога уверенности (Confidence Floor).
   - Задокументировано в [`skills/laya-train/references/jev_integration.md`](file:///c:/Users/andrei/projects/hermes-laya-tools/skills/laya-train/references/jev_integration.md).

2. **Модуль детекции плагина `jev_detector.py`**:
   - Автоматически проверяет CLI `hermes plugins list`, пути `~/.hermes/plugins/hermes-jev`, конфигурацию `~/.hermes/jev` и файлы телеметрии `~/.hermes/logs/jev-*.jsonl`.
   - На сервере `192.168.50.38` подтвердил активный плагин: `Обнаружен: ДА, Включен: ДА, решений: 93 записей`.

3. **Сбор и разметка данных (`extract_and_label.py`)**:
   - Реализована корреляция решений Jev с временной шкалой `messages.timestamp` в `state.db`.
   - На реальном сервере извлечено **75 записей из Jev-логов**, **171 запись из логов Laya** и **92 записи из базы сессий**, сформировано 84 уникальных размеченных кейса (`train.jsonl` — 68 сэмплов, `eval.jsonl` — 16 сэмплов).
   - Предусмотрено обращение к Primary LLM Teacher при неопределенности.

4. **Калибровка и полный пайплайн (`evaluate_accuracy.py` & `train_pipeline.py`)**:
   - Добавлен расчет Confidence Floor Sweep (0.50..0.90) с разделением на `right`, `wrong_actions`, `stalled`.
   - Проведен полный цикл: Pre-test $\rightarrow$ Fine-Tuning / Calibration $\rightarrow$ Laya Hot Reload $\rightarrow$ Post-test $\rightarrow$ Сводный Markdown-отчет [`report.md`](file:///c:/Users/andrei/projects/hermes-laya-tools/skills/laya-train/data/report.md) и [`report.json`](file:///c:/Users/andrei/projects/hermes-laya-tools/skills/laya-train/data/report.json).
   - Все компоненты, скрипты и данные изолированы строго внутри каталога `skills/laya-train/`.
