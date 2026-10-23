---
name: laya-train
description: >-
  Use this skill to extract training data from Hermes sessions and plugin logs,
  label ground truth using the primary model as Teacher, evaluate Laya accuracy
  before and after fine-tuning, run the training pipeline, and generate comparative reports.
version: 1.0.0
license: MIT
metadata:
  hermes:
    tags: [laya, system1, fine-tuning, distillation, evaluation, router, benchmark]
---

# Laya Training & Calibration Skill (`laya-train`)

Скилл предназначен для полного цикла дистилляции, дообучения и калибровки движка быстрых решений **Laya System 1 Decision Engine**. Все данные, скрипты и отчеты инкапсулированы внутри директории скилла: `skills/laya-train/`.

---

## Структура каталога скила

```text
skills/laya-train/
├── SKILL.md                 # Данное руководство
├── scripts/                 # Исполняемые скрипты автоматизации
│   ├── extract_and_label.py # Сбор сессий, эвристика и запрос к основной модели
│   ├── evaluate_accuracy.py # Замер метрик точности (Accuracy, Brier, Confidence)
│   └── train_pipeline.py    # Полный замкнутый оркестратор цикла обучения
├── data/                    # Датасеты и аналитические отчеты
│   ├── train.jsonl          # Обучающая выборка
│   ├── eval.jsonl           # Валидационная выборка
│   ├── eval_before.json     # Результаты теста до обучения
│   ├── eval_after.json      # Результаты теста после обучения
│   ├── report.json          # Машиночитаемый отчет с дельтами метрик
│   └── report.md            # Сводный аналитический отчет в Markdown
├── references/              # Справочные материалы и архитектурные руководства
│   ├── best_practices.md    # Лучшие практики создания агентных скилов
│   ├── laya_architecture.md # Архитектура System 1 и спецификация категорий
│   └── training_guide.md    # Гиперпараметры, функции потерь и компиляция GGUF
└── examples/                # Образцы валидированных сэмплов (.jsonl)
    ├── train_sample.jsonl   # Эталонные примеры для обучения
    └── eval_sample.jsonl    # Эталонные примеры для валидации
```

---

## Быстрый запуск полного цикла (One-Command)

Запуск всего конвейера одной командой (с автоматическим пре-тестом, дообучением, компиляцией GGUF, горячей перезагрузкой и пост-тестом):

```bash
# Автономный прогон полного конвейера:
python scripts/train_pipeline.py --extract

# Быстрый контрольный запуск (с верификацией live-инференса без GPU-переобучения):
python scripts/train_pipeline.py --dry-run
```

---

## Пошаговый рабочий процесс (Step-by-Step Runbook)

### Шаг 1: Извлечение и экспертная разметка данных
Скрипт анализирует обращения из `laya_requests.jsonl` и сессии из `state.db`:
- Если ход содержал фактический запуск инструмента (`read_file`, `terminal`, `mcp__marketplace__ozon_search`, `computer_use`), инструмент принимается за эталон.
- **Разметка с привлечением основной модели**: если в сессии не было вызова инструмента или ситуация неоднозначна, скрипт автоматически направляет запрос к **основной модели Hermes** (`Primary LLM Teacher`) для определения эталонного решения (`execute_tools` vs `direct_chat`) и целевой категории.

```bash
python scripts/extract_and_label.py \
  --logs-path ~/.hermes/logs/laya_requests.jsonl \
  --db-path ~/.hermes/state.db \
  --out-dir data/
```

### Шаг 2: Тестирование точности до обучения (Pre-Training Test)
Замер базовых метрик текущей рабочей модели на валидационной выборке:
- **Tool Decision Accuracy** (разделение инструментов и диалога).
- **Category Accuracy** (точность выбора семантической категории).
- **Combined Strict Accuracy** (совместное совпадение решения и категории).
- **Brier Score** (показатель калибровки вероятностей, ниже — лучше).
- **Mean Confidence** (средняя уверенность классификатора).

```bash
python scripts/evaluate_accuracy.py \
  --eval-data data/eval.jsonl \
  --phase before \
  --laya-url http://127.0.0.1:8787
```

### Шаг 3: Дообучение модели (Fine-Tuning / RLCD)
Обучение чекпоинта на собранных данных:

```bash
/root/.unsloth/studio/unsloth_studio/bin/laya-train \
  --data data/train.jsonl \
  --eval data/eval.jsonl \
  --base /root/deepseek-harness/finetune-checkpoint \
  --out data/finetune_checkpoint \
  --loss rlcd \
  --epochs 4
```

### Шаг 4: Компиляция GGUF и горячая перезагрузка
1. Компиляция весов в квантованный GGUF через `ggmlc`:
   ```bash
   python3 /root/deepseek-harness/compile_custom_laya.py
   ```
2. Отправка сигнала на горячую перезагрузку весов без простоя:
   ```bash
   curl -X POST http://127.0.0.1:8787/reload
   ```

### Шаг 5: Тестирование точности после обучения (Post-Training Test)
Повторный запуск замера на том же валидационном наборе `eval.jsonl`:

```bash
python scripts/evaluate_accuracy.py \
  --eval-data data/eval.jsonl \
  --phase after \
  --laya-url http://127.0.0.1:8787
```

### Шаг 6: Формирование сравнительного отчета
Скрипт автоматически сравнивает метрики `eval_before.json` и `eval_after.json` и формирует отчеты:
- `data/report.md`: Наглядная сводка с таблицей дельт ($\Delta$), по категорийной разбивкой и списком исправленных расхождений.
- `data/report.json`: Полная машиночитаемая структура для дашбордов и CI/CD.

---

## Ссылки на справочные материалы

- [Лучшие практики создания скилов](file:///c:/Users/andrei/projects/hermes-laya-tools/skills/laya-train/references/best_practices.md)
- [Архитектура Laya System 1 и категории](file:///c:/Users/andrei/projects/hermes-laya-tools/skills/laya-train/references/laya_architecture.md)
- [Руководство по гиперпараметрам и компиляции](file:///c:/Users/andrei/projects/hermes-laya-tools/skills/laya-train/references/training_guide.md)
