# План: Лучшие практики дообучения Laya, ревью скила `laya-train` и исправление логики дистилляции (Laya vs Primary Model)

## Обзор задачи
1. Собрать и систематизировать лучшие практики по дообучению и калибровке Laya (`convaiinnovations/laya`, `convaiinnovations/laya-multilingual`, `kerpopule/hermes-jev-skills`, RLCD, `--shuffle-options`, борьба с Position Bias и схлопыванием логитов на малых выборках, контроль VRAM на GPU0 с переключением на `--device cpu`, `laya-evals validate`).
2. Провести глубокое ревью скила `skills/laya-train/` (выявлены и устранены 7 критических проблем в схеме данных, парсинге логов, вызове основной модели, гиперпараметрах батча, несовпадении токенизатора GGUF, контроле VRAM на GPU0 и расчете `answer_confidence`).
3. Переработать логику получения и разметки данных в `scripts/extract_and_label.py`:
   - **Если есть информация, что предлагает Laya, и что использует основная модель** (связка запроса Laya с `tool_telemetry` из `laya_requests.jsonl` и/или фактическими вызовами инструментов основной модели из `state.db`) $\rightarrow$ сопоставить выбор Laya с выбором основной модели и дообучить Laya на выборке основной модели (с пометкой расхождений как `hard_negative`).
   - **Если информации нет** (нет данных о том, что использовала основная модель, либо нет предложения Laya для этого запроса) $\rightarrow$ сделать запрос к основной модели (`query_primary_model`) и/или к Laya (`query_laya_choice`) для прямого сравнения с выбором Laya и формирования эталона `expected`.
4. Обновить `evaluate_accuracy.py`, `train_pipeline.py` (включая автоматический запуск `laya-train` на `--device cpu` при нехватке VRAM на `gpu0`) и документацию (`SKILL.md`, `references/training_guide.md`, `references/best_practices.md`).
5. Прогнать полный цикл на сервере `192.168.50.38` (сбор и сравнение Laya vs Primary Model, валидация `laya-evals validate`, реальное дообучение `laya-train --device cpu`, компиляция GGUF с проверкой токенизатора, перезагрузка и пост-тест точности) и синхронизировать изменения.

---

## Задачи

- [x] 1. Обновить справочники лучших практик (`skills/laya-train/references/training_guide.md` и `best_practices.md`) на основе результатов исследования `convaiinnovations/laya` и `kerpopule/hermes-jev-skills` <!-- id: 0 -->
- [x] 2. Переработать `skills/laya-train/scripts/extract_and_label.py`:
  - загрузка `.env` и `config.yaml` Hermes;
  - гигиена `state` (фильтрация системных инъекций, `[IMPORTANT:...]`, `[System:...]`, `[/plan ...]`, `--- Attached Context ---`);
  - связывание предложений Laya (`laya_requests.jsonl`) с телеметрией фактических вызовов (`tool_telemetry`) и траекториями основной модели (`state.db`);
  - реализация строгой развилки: если известно, что предлагает Laya и что использует основная модель — обучаем на выборке основной модели; если информации нет — запрашиваем основную модель (и Laya при отсутствии ее ответа) для сравнения с выбором Laya;
  - сохранение эталона в поле `expected` (совместимо с `laya-train` и `laya-evals validate`) без дублирования строк в `train.jsonl` (защита от утечки в `split_calibration`) <!-- id: 1 -->
- [x] 3. Исправить `skills/laya-train/scripts/evaluate_accuracy.py` и `examples/*.jsonl`:
  - поддержка поля `expected` (с обратной совместимостью с `label`);
  - корректный расчет вероятностной уверенности $\max(p)$ (`answer_confidence`) вместо сырой нормализованной энтропии для Brier Score и Confidence Floor Sweep <!-- id: 2 -->
- [x] 4. Исправить `skills/laya-train/scripts/train_pipeline.py`:
  - предварительная валидация датасета через `laya-evals validate`;
  - проверка свободной VRAM на `gpu0` (`get_gpu0_free_vram_mib`): если свободной памяти `< 4500 MiB` (занята основной моделью `ninfer-serve` / `coder390-27b`) или происходит `CUDA out of memory`, автоматический запуск `laya-train` на `--device cpu`;
  - адаптивный подбор `--micro-batch 4`, `--grad-accum 1`, `--epochs 4` под размер выборки (защита от prior collapse `#963`) и обязательный флаг `--shuffle-options`;
  - компиляция обученного чекпоинта `convaiinnovations/laya-multilingual` (`jhu-clsp/mmBERT-base`, словарь 256k, `pre_tokenizer=gemma`) через `/root/.unsloth/studio/unsloth_studio/bin/python` с атомарной заменой `/root/deepseek-harness/models/laya_multilingual_f16.gguf` и горячей перезагрузкой Laya;
  - расширение отчета `report.md` разделом статистики сверки Laya vs Основная модель <!-- id: 3 -->
- [x] 5. Обновить `skills/laya-train/SKILL.md` с описанием логики сравнения Laya vs Primary Model, правил переключения `gpu0 -> cpu` и лучших практик дообучения <!-- id: 4 -->
- [x] 6. Протестировать и верифицировать полный цикл на сервере `192.168.50.38`, проверить рост точности после дообучения и обновить раздел Review в `tasks/todo.md` <!-- id: 5 -->

---

## Review (Итоги выполнения и верификации)

1. **Исследование лучших практик дообучения Laya**:
   - Изучены исходники пакета `laya` (`train.py`, `train_cli.py`, `evals.py`, `calibrate.py`, `_agreement.py`) и `kerpopule/hermes-jev-skills`.
   - Зафиксированы ключевые практики в [`training_guide.md`](../skills/laya-train/references/training_guide.md) и [`best_practices.md`](../skills/laya-train/references/best_practices.md): обязательное поле `"expected"`, пропуск `"category"` при `tool_decision == "direct_chat"`, флаг `--shuffle-options` против позиционного смещения, уменьшение эффективного батча (`--micro-batch 4 --grad-accum 1`) на малых выборках против Prior Collapse (`#963`), отсутствие дубликатов строк в `train.jsonl` для чистоты внутреннего `split_calibration()`, и автоматический переход на `--device cpu` при нехватке свободной VRAM на `gpu0`.

2. **Ревью скила `skills/laya-train/` (исправлены 7 критических ошибок)**:
   - **Ошибка 1 (Схема JSONL)**: Старый скрипт писал только `"label"`, из-за чего `laya-train` пропускал 100% строк (`produced no training items`). Добавлено обязательное поле `"expected"`.
   - **Ошибка 2 (Логика Laya vs Primary Model)**: Старый скрипт принимал строки `executed=<tool> hit=<bool>` за пользовательские промпты и учил Laya на её собственных предсказаниях. Реализовано связывание запросов Laya с `tool_telemetry` и `state.db`, а при отсутствии данных — live-запрос к основной модели (`query_primary_model`) и/или Laya (`query_laya_choice`).
   - **Ошибка 3 (HTTP 502 при вызове основной модели)**: Убран несовместимый `response_format={"type": "json_object"}` для провайдера `custom:omni` (`coder390-27b`) и настроены параметры глубокого размышления `"reasoning_effort": "xhigh"` (на верхнем уровне JSON-запроса) и `"chat_template_kwargs": {"enable_thinking": True}` (`max_tokens=4096`) + автозагрузка `HERMES_CUSTOM_OMNI_API_KEY` из `~/.hermes/.env`.
   - **Ошибка 4 (Загрязнение `state`)**: В `clean_user_state()` добавлена очистка дампов `--- Attached Context ---`, системных тегов `[System: ...]`, `[IMPORTANT: ...]` и инструкций `/plan`.
   - **Ошибка 5 (Рассогласование базовой модели и токенизатора GGUF)**: Ранее чекпоинт обучался от англоязычного `convaiinnovations/laya` (GPT-2 словарь 50k), но компилировался с `pre_tokenizer = "gemma"`, что ломало токенизацию кириллицы в GGUF. Базовый чекпоинт переключён на `convaiinnovations/laya-multilingual` (`jhu-clsp/mmBERT-base`, словарь Gemma 256k), а компилятор GGUF автоматически проверяет соответствие словаря и `pre_tokenizer` и атомарно обновляет `/root/deepseek-harness/models/laya_multilingual_f16.gguf`.
   - **Ошибка 6 (Нехватка VRAM на GPU0)**: В `train_pipeline.py` добавлена проверка `get_gpu0_free_vram_mib()` (порог 4500 MiB) и перехват `CUDA out of memory` с автоматическим запуском `laya-train` на `--device cpu`.
   - **Ошибка 7 (Расчёт `answer_confidence`)**: В `evaluate_accuracy.py` расчёт Brier Score и Confidence Floor Sweep переведён с нормализованной энтропии `confidence` на вероятностную уверенность выбранного ответа $\max(p)$.

3. **Результаты полного цикла на сервере `192.168.50.38`**:
   - Собрано и очищено **39 уникальных диалоговых кейсов**: **30** сопоставлены по существующим данным Laya + Основная модель (Сценарий A), **9** размечены через live-запрос к Основной модели + **13** через live-запрос к Laya (Сценарий B).
   - Оба файла (`train.jsonl` — 32 сэмпла, `eval.jsonl` — 7 независимых сэмплов) прошли `laya-evals validate` без ошибок.
   - При обнаружении `2027 MiB < 4500 MiB` свободной VRAM на `gpu0` обучение автоматически выполнено на `--device cpu` (`--base convaiinnovations/laya-multilingual --loss rlcd --epochs 4 --micro-batch 4 --grad-accum 1 --shuffle-options`):
     - На внутреннем held-out тесте `laya-train` (13 вопросов): **Accuracy** выросла с **69.23%** до **76.92%** (`+7.69%`), **Loss** снизился с `1.8122` до `1.1958` (`-0.6164`), **ECE** снизилась с `0.3821` до `0.2256` (`-0.1565`), **Brier** снизился с `0.6225` до `0.4471` (`-0.1754`).
     - На сквозном live-тесте C++ сервера Laya (`http://127.0.0.1:8787`, от исходного состояния до финала):
       - **Tool Decision Accuracy**: выросла с **77.8%** до **85.7%** (`+7.9%`);
       - **Category Accuracy**: выросла с **0.0%** (и промежуточных **33.3%**) до **66.7%** (`+66.7%`, в категории `terminal_execution` — **100.0%**);
       - **Combined Strict Accuracy**: выросла с **0.0%** (и промежуточных **28.6%**) до **57.1%** (`+57.1%`);
       - **Brier Score**: улучшился с **0.2394** до **0.1406** (`-0.0988`);
       - **Средняя задержка инференса**: снизилась с **446.7 ms** до **249.1 ms**.
   - Все **46 модульных тестов** (`python run_tests.py`) пройдены успешно (`46/46 PASS`).
