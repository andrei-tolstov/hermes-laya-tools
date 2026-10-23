# Руководство по обучению, лучшим практикам, функциям потерь и компиляции Laya

Данное руководство объединяет официальные лучшие практики дообучения **Laya System 1** (`convaiinnovations/laya`, `laya-train`, `laya-evals`) и методологию калибровки решений `kerpopule/hermes-jev-skills`.

---

## 1. Спецификация данных для `laya-train` и `laya-evals`

Утилиты `laya-train` (`laya.train:items_from_rows`) и `laya-evals validate` (`laya.evals:Example.from_dict`) требуют строгого соблюдения JSONL-схемы. Каждая строка обязана содержать:
- **`state`** (`str` или `dict`): входной запрос пользователя, очищенный от системных промптов, `[System: ...]`, `[IMPORTANT: ...]` и громоздких вложений (`--- Attached Context ---`).
- **`questions`** (`dict`): словарь типизированных вопросов (`choice`, `noul`, `score`).
- **`expected`** (`dict`) или **`gold`** (`dict`):
  - В **`expected`** передаются дискретные эталонные ответы основной модели по каждому `question_id` (строка из `criteria` для `choice`, `bool` для `noul`, строка или индекс уровня для `score`).
  - В **`gold`** могут передаваться полные распределения вероятностей от модели-учителя (`{"probabilities": {...}}`).
  - **Важно**: использование нестандартного ключа `label` вместо `expected`/`gold` приводит к пропуску 100% строк при обучении в `laya-train`!

---

## 2. Логика получения данных: Laya vs Основная модель (Primary Model)

При сборе датасета для каждого уникального пользовательского запроса (`state`) формируются два вектора решений:
1. **Предложение Laya (`laya_choice`)**:
   - Извлекается из лога `laya_requests.jsonl` (`raw_answers` / `decision` / `winning_category` / `selected_tools`).
   - Если запрос поступил из `state.db` или `jev-decisions.jsonl` и для него нет записи в `laya_requests.jsonl`, выполняется прямой запрос к инференс-серверу Laya (`/ask` или `/v1/systemone`).
2. **Выбор основной модели (`primary_choice` — Ground Truth)**:
   - **Случай А (Есть информация, что предлагает Laya и что использует основная модель)**:
     - В `laya_requests.jsonl` есть связанные записи `tool_telemetry` (`decision: "tool_exec:hit" | "tool_exec:miss"`, `raw_answers.tool_name`) для того же `session_id`/`turn_id`, ИЛИ в `state.db` зафиксированы фактические вызовы инструментов (`tool_calls`, `tool_name`) основной моделью на этом ходу (либо завершённый чисто диалоговый ответ без инструментов на тривиальном запросе).
     - В этом случае **эталоном (`expected`) назначается фактический выбор основной модели**, и Laya дообучается непосредственно на выборке основной модели.
   - **Случай Б (Информации нет или действие не зафиксировано)**:
     - Если в логах есть только предложение Laya без телеметрии выполнения и нет подтверждённых вызовов тулов в `state.db` (или есть только запись из `state.db` / `jev-decisions.jsonl` без явного выбора инструментов):
     - Выполняется **прямой запрос к основной модели Hermes (`query_primary_model`)** (и к Laya, если её предложение ещё не известно).
     - Ответ основной модели сравнивается с выбором Laya (`match = (laya_choice == primary_choice)`), и Laya дообучается на эталоне основной модели.
3. **Hard Negative Mining (Приоритизация ошибок Laya)**:
   - Если предложение Laya разошлось с выбором основной модели (`match == False`) или имело промах по тулу (`tool_exec:miss`), сэмпл получает тег `priority:hard_negative` и дублируется (oversampling $\times 2$) в `train.jsonl` для ускоренного исправления ошибок роутера.

---

## 3. Параметры `laya-train` и лучшие практики гиперпараметров

```bash
/root/.unsloth/studio/unsloth_studio/bin/laya-train \
  --data data/train.jsonl \
  --eval data/eval.jsonl \
  --base /root/deepseek-harness/finetune-checkpoint \
  --out data/finetune_checkpoint \
  --loss rlcd \
  --epochs 8 \
  --micro-batch 4 \
  --grad-accum 2 \
  --encoder-lr 2.5e-5 \
  --head-lr 1.0e-4 \
  --shuffle-options \
  --target-error 0.10 \
  --min-abstain-n 10
```

### Ключевые рекомендации (`convaiinnovations/laya`):
1. **Борьба с позиционным смещением (`--shuffle-options`)**:
   - Энкодеры (`ModernBERT` / `mmBERT`) подвержены позиционному смещению (первая опция получает паразитный бонус вероятности).
   - Всегда включайте флаг `--shuffle-options`: на каждой эпохе порядок вариантов `criteria` в вопросах `choice` случайно перемешивается, а целевой вектор синхронно переставляется. Это снижает чувствительность к порядку вариантов в 5–6 раз.
2. **Бюджет шагов оптимизатора и защита от Prior Collapse (`#963`)**:
   - Число шагов оптимизатора рассчитывается как:
     $$\text{updates} = \text{epochs} \times \lceil \lceil \text{train\_items} / \text{micro\_batch} \rceil / \text{grad\_accum} \rceil$$
   - Дефолтные параметры (`--micro-batch 8 --grad-accum 8`, эффективный батч = 64) рассчитаны на выборки в **5000+ примеров** (~350 шагов за 4 эпохи).
   - На малых и средних выборках (50–500 примеров) эффективный батч 64 даёт всего 4–12 обновлений весов, из-за чего модель не успевает сойти с плато $\ln K$ и схлопывается к равномерному распределению (`prior collapse`).
   - **Правило для выборок `< 500` элементов**: уменьшать `--micro-batch 4 --grad-accum 1..2` (эффективный батч 4–8) и увеличивать `--epochs 6..10`, обеспечивая **не менее 80–150 обновлений оптимизатора**.
3. **Раздельные скорости обучения (Differential Learning Rates)**:
   - `--encoder-lr 2.5e-5`: плавная доводка весов трансформера без катастрофического забывания.
   - `--head-lr 1.0e-4`: ускоренная адаптация классификационных голов решений.
4. **Выбор функции потерь (`--loss rlcd` vs `soft-ce`)**:
   - `rlcd` (Reinforcement Learning for Calibrated Decisions): совмещает мягкую кросс-энтропию и policy-gradient по 4 зашумлённым сэмплам логитов со строго собственными правилами оценивания (spherical score + ranked probability score). Обеспечивает честную калибровку вероятностей (низкий Brier Score и ECE).
   - `soft-ce`: чистая кросс-энтропия (может комбинироваться с `--label-smoothing 0.05..0.10` на жёстких метках).
5. **Обязательная пред-валидация (`laya-evals validate` и `--dry-run`)**:
   - Перед обучением запускать `laya-evals validate data/train.jsonl` и `laya-train --dry-run` для проверки схемы, бюджета токенов (`options_collapsed`, `options_beyond_max_len`) и числа шагов оптимизатора.
6. **Контроль VRAM на GPU0 и автоматический запуск на CPU (`--device cpu`)**:
   - Полное дообучение `convaiinnovations/laya-multilingual` (`jhu-clsp/mmBERT-base`, словарь 256 тыс. токенов) требует $\ge 4500\text{ MiB}$ свободной видеопамяти на `gpu0` (для весов, градиентов и буферов `AdamW`).
   - Если на `gpu0` одновременно запущена основная LLM (`ninfer-serve` / `coder390-27b`) и свободной VRAM меньше $4500\text{ MiB}$ (или возникает `CUDA out of memory`), дообучение `laya-train` **обязательно запускается на CPU** с флагом `--device cpu` (оркестратор `train_pipeline.py` выполняет эту проверку и переключение автоматически).

---

## 4. Температурная калибровка и различие `answer_confidence` vs `confidence`

1. **Автоматическая подгонка температур (`laya.calibrate`)**:
   - До начала обучения `laya-train` отделяет 10% выборки (`calib_frac=0.1`, до 400 элементов) в независимый калибровочный срез.
   - После обучения LBFGS-оптимизатор подбирает температуру $T \in [0.1, 10.0]$ для каждого типа вопросов (при наличии $\ge 10$ калибровочных элементов, `MIN_TYPE_N = 10`) и пороги воздержания (`fit_abstention_thresholds`).
2. **Метрика уверенности при оценке (`answer_confidence` / $\max(p)$)**:
   - В ответах нативного C++ сервера Laya поле `confidence` содержит нормализованную энтропию, зависящую от числа опций $K$, которая на широких списках категорий численно мала (`0.01..0.15`).
   - Для расчёта калибровки (Brier Score, ECE) и подбора порога отсечения **Confidence Floor Sweep** (`Wrong Actions` vs `Stalls` по методологии `kerpopule/hermes-jev-skills`) необходимо использовать вероятностную уверенность выбранного ответа:
     $$p_{\max} = \max_{c} P(c)$$
     (или поле `answer_confidence`, если оно присутствует в ответе).

---

## 5. Компиляция весов в GGUF (`ggmlc`) и горячая перезагрузка

После сохранения обученного чекпоинта (`model.safetensors`, `rl_agent_config.json`) веса компилируются в формат `.gguf` с помощью окружения `/root/.unsloth/studio/unsloth_studio/bin/python`:

```bash
/root/.unsloth/studio/unsloth_studio/bin/python /root/deepseek-harness/compile_custom_laya.py \
  --checkpoint data/finetune_checkpoint \
  --output /root/deepseek-harness/models/laya_multilingual_f16.gguf
```

И активируются без даунтайма:
```bash
curl -X POST http://127.0.0.1:8787/reload
```
