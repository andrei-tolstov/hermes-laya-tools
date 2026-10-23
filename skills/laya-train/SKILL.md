---
name: laya-train
description: >-
  Use this skill to extract training data from Hermes sessions and plugin logs,
  compare Laya proposals against Primary Model actual tool usage (or query the Primary Model
  when ground truth is missing), validate JSONL datasets with laya-evals, fine-tune
  convaiinnovations/laya-multilingual with RLCD and option shuffling, compile GGUF weights,
  hot-reload Laya, and generate before/after comparative evaluation reports.
version: 1.2.0
license: MIT
metadata:
  hermes:
    tags: [laya, system1, fine-tuning, distillation, evaluation, router, benchmark, jev, rlcd]
---

# Laya Training & Calibration Skill (`laya-train`)

Скилл предназначен для полного замкнутого цикла дистилляции, дообучения и калибровки движка быстрых решений **Laya System 1 Decision Engine** (`convaiinnovations/laya-multilingual`) с поддержкой плагина **`kerpopule/hermes-jev-skills`** (`hermes-jev`). Все данные, скрипты и отчёты инкапсулированы внутри директории скилла: `skills/laya-train/`.

---

## Структура каталога скила

```text
skills/laya-train/
├── SKILL.md                   # Данное руководство (Runbook)
├── scripts/                   # Исполняемые скрипты автоматизации
│   ├── jev_detector.py        # Проверка статуса плагина kerpopule/hermes-jev-skills и логов
│   ├── extract_and_label.py   # Сверка выбора Laya и Основной модели + live-запросы к Primary Model
│   ├── evaluate_accuracy.py   # Замер точности (Accuracy, Brier Score, Confidence Floor Sweep)
│   └── train_pipeline.py      # Полный замкнутый оркестратор цикла обучения и GGUF-компиляции
├── data/                      # Датасеты и аналитические отчеты
│   ├── train.jsonl            # Обучающая выборка (со строгим полем expected)
│   ├── eval.jsonl             # Стратифицированная валидационная выборка
│   ├── distill_comparison.json# Подробный аудит сравнения выбора Laya и Основной модели
│   ├── eval_before.json       # Результаты теста до обучения
│   ├── eval_after.json        # Результаты теста после обучения
│   ├── report.json            # Машиночитаемый отчет с дельтами метрик
│   └── report.md              # Сводный аналитический отчет в Markdown
├── references/                # Справочные материалы и архитектурные руководства
│   ├── best_practices.md      # Лучшие практики дообучения Laya и инженерии скилов
│   ├── jev_integration.md     # Интеграция с kerpopule/hermes-jev-skills и калибровка
│   ├── laya_architecture.md   # Архитектура System 1 и спецификация категорий
│   └── training_guide.md      # Гиперпараметры RLCD, схема expected и компиляция GGUF
└── examples/                  # Образцы валидированных сэмплов (.jsonl)
    ├── train_sample.jsonl     # Эталонные примеры для обучения
    └── eval_sample.jsonl      # Эталонные примеры для валидации
```

---

## Логика получения и сверки данных (`extract_and_label.py`)

Скрипт [`scripts/extract_and_label.py`](scripts/extract_and_label.py) реализует строгую двухфазную сверку **Laya (Student)** и **Основной модели (Primary Model / Teacher)**:

1. **Сбор и сопоставление источников**:
   - Из `~/.hermes/logs/laya_requests.jsonl` извлекаются предсказания Laya (`laya_choice`) и связываются с последующими записями `tool_telemetry` (`executed=<tool> hit=<bool>`).
   - Из `~/.hermes/state.db` извлекаются реальные вызовы инструментов (`role='assistant', tool_calls`) или прямые текстовые ответы основной модели (`primary_choice`).
   - Из `~/.hermes/logs/jev-decisions.jsonl` и `jev-shadow.jsonl` извлекаются решения плагина `hermes-jev`.
   - Применяется очистка состояния (`clean_user_state`): удаляются гигантские блоки `--- Attached Context ---`, системные теги `[System: ...]` и внутренние промпты маршрутизатора.

2. **Два сценария формирования эталона (`expected`)**:
   - **Сценарий A (Известно и что предлагает Laya, и что реально использует Основная модель)**:
     - Laya дообучается на реальной выборке Основной модели (`expected = primary_choice`).
     - Если выбор Laya расходится с выбором Основной модели (`laya_choice != primary_choice`), кейс помечается как **`hard_negative`** и дублируется (`2x` oversampling) в `train.jsonl` для приоритетного исправления ошибки классификатора.
   - **Сценарий B (Информация о выборе Основной модели или Laya отсутствует)**:
     - Если неизвестен выбор Основной модели — автоматически выполняется live-запрос к Основной модели (`POST /v1/chat/completions` с `"reasoning_effort": "xhigh"`, `"chat_template_kwargs": {"enable_thinking": true}` и автозагрузкой ключа из `~/.hermes/.env`) для глубокого анализа и сравнения с выбором Laya.
     - Если неизвестен выбор самой Laya (например, сессия из `state.db` прошла мимо роутера) — выполняется live-запрос к Laya (`POST /v1/systemone`) для фиксации расхождения `Laya vs Primary`.

3. **Формирование поля `expected` для `laya-train`**:
   - Каждая строка содержит обязательное поле `"expected"`.
   - Если `tool_decision == "direct_chat"`, в `"expected"` передаётся только `{"tool_decision": "direct_chat"}` (без `"category"`), чтобы голова `category` не размывалась на приветствиях и общих вопросах.

---

## Быстрый запуск полного цикла (One-Command)

Запуск всего конвейера одной командой:
1. Проверка наличия плагина `kerpopule/hermes-jev-skills` (`hermes-jev`) и обнаружение его логов;
2. Сбор сессий Hermes и логов Laya / Jev со сверкой `Laya vs Primary Model` и live-запросами при отсутствии данных;
3. Валидация схемы датасетов (`laya-evals validate`);
4. Пре-тест точности текущей модели на валидационном наборе;
5. Дообучение `convaiinnovations/laya-multilingual` (`--loss rlcd --shuffle-options --micro-batch 4 --grad-accum 1 --epochs 8`);
6. Компиляция GGUF с проверкой совместимости токенизатора (`gemma` / `gpt2`) и атомарным обновлением `/root/deepseek-harness/models/laya_multilingual_f16.gguf`;
7. Горячая перезагрузка весов (`POST /reload`);
8. Пост-тест точности новой модели и калибровка Confidence Floor;
9. Генерация сводного отчёта (`data/report.md`, `data/report.json`, `data/distill_comparison.json`).

```bash
# Автономный прогон полного конвейера:
python scripts/train_pipeline.py --extract

# Быстрый контрольный запуск (без GPU-переобучения):
python scripts/train_pipeline.py --dry-run
```

---

## Пошаговый рабочий процесс (Step-by-Step Runbook)

### Шаг 1: Проверка плагина `kerpopule/hermes-jev-skills`
```bash
python scripts/jev_detector.py
```

### Шаг 2: Извлечение данных и сверка `Laya vs Primary Model`
```bash
python scripts/extract_and_label.py \
  --logs-path ~/.hermes/logs/laya_requests.jsonl \
  --db-path ~/.hermes/state.db \
  --laya-url http://127.0.0.1:8787 \
  --out-dir data/
```

### Шаг 3: Валидация датасетов и пре-тест точности (Pre-Training Test)
```bash
/root/.unsloth/studio/unsloth_studio/bin/laya-evals validate data/train.jsonl
/root/.unsloth/studio/unsloth_studio/bin/laya-evals validate data/eval.jsonl

python scripts/evaluate_accuracy.py \
  --eval-data data/eval.jsonl \
  --phase before \
  --laya-url http://127.0.0.1:8787
```

### Шаг 4: Дообучение модели (Fine-Tuning / RLCD) и контроль VRAM на GPU0
**Правило выбора устройства (`GPU0 VRAM` vs `CPU`)**:
- Полное дообучение `convaiinnovations/laya-multilingual` (`jhu-clsp/mmBERT-base`, словарь 256k) требует не менее **4500 MiB** свободной видеопамяти на `gpu0`.
- Если на `gpu0` запущена основная модель (`ninfer-serve` / `coder390-27b`) и свободной VRAM меньше **4500 MiB** (или возникает `CUDA out of memory`), дообучение **обязательно запускается на CPU** (`--device cpu`). Оркестратор `train_pipeline.py` проверяет свободную VRAM на `gpu0` через `nvidia-smi` автоматически и переключает `laya-train` на `--device cpu`.

```bash
# Если на GPU0 достаточно свободной VRAM (>= 4500 MiB):
/root/.unsloth/studio/unsloth_studio/bin/laya-train \
  --data data/train.jsonl \
  --eval data/eval.jsonl \
  --base convaiinnovations/laya-multilingual \
  --out /root/deepseek-harness/finetune-checkpoint \
  --loss rlcd \
  --epochs 4 \
  --micro-batch 4 \
  --grad-accum 1 \
  --shuffle-options

# Если на GPU0 не хватает VRAM (< 4500 MiB) — запуск на CPU:
/root/.unsloth/studio/unsloth_studio/bin/laya-train \
  --data data/train.jsonl \
  --eval data/eval.jsonl \
  --base convaiinnovations/laya-multilingual \
  --out /root/deepseek-harness/finetune-checkpoint \
  --loss rlcd \
  --epochs 4 \
  --micro-batch 4 \
  --grad-accum 1 \
  --device cpu \
  --shuffle-options
```

### Шаг 5: Резервные копии, компиляция GGUF и перезагрузка
`train_pipeline.py` перед заменой автоматически копирует рядом с оригиналом:
- чекпоинт `/root/deepseek-harness/finetune-checkpoint` → `finetune-checkpoint.<YYYYmmdd-HHMMSS>.bak` (до запуска `laya-train`, который перезаписывает `--out`);
- рабочий GGUF `/root/deepseek-harness/models/laya_multilingual_f16.gguf` → `laya_multilingual_f16.gguf.<YYYYmmdd-HHMMSS>.bak` (до атомарной подмены).

Хранятся 3 последние копии (`--keep-backups N`), отключение — `--no-backup`. Ручные бэкапы без метки времени (`*.gguf.bak`) не удаляются. Пути копий пишутся в `report.json` (`train_metadata.checkpoint_backup`, `gguf_backup`).

Откат вручную:
```bash
cp /root/deepseek-harness/models/laya_multilingual_f16.gguf.<метка>.bak /root/deepseek-harness/models/laya_multilingual_f16.gguf
curl -X POST http://127.0.0.1:8787/reload
```

### Шаг 6: Пост-тест точности и калибровка Confidence Floor
```bash
python scripts/evaluate_accuracy.py \
  --eval-data data/eval.jsonl \
  --phase after \
  --laya-url http://127.0.0.1:8787
```
