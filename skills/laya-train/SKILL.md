---
name: laya-train
description: >-
  Use this skill to extract training data from Hermes sessions and plugin logs,
  detect kerpopule/hermes-jev-skills plugin and incorporate Jev decision logs,
  label ground truth using the primary model as Teacher, evaluate Laya accuracy
  before and after fine-tuning, run the training pipeline, and generate comparative reports.
version: 1.1.0
license: MIT
metadata:
  hermes:
    tags: [laya, system1, fine-tuning, distillation, evaluation, router, benchmark, jev]
---

# Laya Training & Calibration Skill (`laya-train`)

Скилл предназначен для полного цикла дистилляции, дообучения и калибровки движка быстрых решений **Laya System 1 Decision Engine** с поддержкой плагина **`kerpopule/hermes-jev-skills`** (`hermes-jev`). Все данные, скрипты и отчеты инкапсулированы внутри директории скилла: `skills/laya-train/`.

---

## Структура каталога скила

```text
skills/laya-train/
├── SKILL.md                 # Данное руководство (Runbook)
├── scripts/                 # Исполняемые скрипты автоматизации
│   ├── jev_detector.py      # Проверка статуса плагина kerpopule/hermes-jev-skills и логов
│   ├── extract_and_label.py # Сбор сессий, Jev-логов, эвристика и запрос к основной модели
│   ├── evaluate_accuracy.py # Замер точности (Accuracy, Brier, Confidence Floor Sweep)
│   └── train_pipeline.py    # Полный замкнутый оркестратор цикла обучения
├── data/                    # Датасеты и аналитические отчеты
│   ├── train.jsonl          # Обучающая выборка (Hermes + Laya + Jev)
│   ├── eval.jsonl           # Валидационная выборка
│   ├── eval_before.json     # Результаты теста до обучения
│   ├── eval_after.json      # Результаты теста после обучения
│   ├── report.json          # Машиночитаемый отчет с дельтами метрик
│   └── report.md            # Сводный аналитический отчет в Markdown
├── references/              # Справочные материалы и архитектурные руководства
│   ├── best_practices.md    # Лучшие практики создания агентных скилов
│   ├── jev_integration.md   # Интеграция с kerpopule/hermes-jev-skills и калибровка
│   ├── laya_architecture.md # Архитектура System 1 и спецификация категорий
│   └── training_guide.md    # Гиперпараметры, функции потерь и компиляция GGUF
└── examples/                # Образцы валидированных сэмплов (.jsonl)
    ├── train_sample.jsonl   # Эталонные примеры для обучения
    └── eval_sample.jsonl    # Эталонные примеры для валидации
```

---

## Быстрый запуск полного цикла (One-Command)

Запуск всего конвейера одной командой:
1. Проверка наличия плагина `kerpopule/hermes-jev-skills` (`hermes-jev`) и обнаружение его логов;
2. Извлечение сессий Hermes и логов Laya / Jev с fallback-запросами к Primary Model Teacher;
3. Пре-тест точности текущей модели на валидационном наборе;
4. Запуск дообучения / калибровки весов Laya и компиляция GGUF;
5. Горячая перезагрузка весов (`POST /reload`);
6. Пост-тест точности новой модели и калибровка Confidence Floor;
7. Генерация сводного отчёта со срезом по Jev (`report.md` и `report.json`).

```bash
# Автономный прогон полного конвейера с автоматической проверкой Jev:
python scripts/train_pipeline.py --extract

# Быстрый контрольный запуск (с верификацией live-инференса без GPU-переобучения):
python scripts/train_pipeline.py --dry-run
```

---

## Пошаговый рабочий процесс (Step-by-Step Runbook)

### Шаг 1: Проверка плагина `kerpopule/hermes-jev-skills`
Проверка регистрации плагина в Hermes, статуса `enabled` и наличия логов решений:
```bash
python scripts/jev_detector.py
```

### Шаг 2: Извлечение и экспертная разметка данных
Скрипт объединяет данные из трёх источников:
- Аудит-логи Laya: `~/.hermes/logs/laya_requests.jsonl`;
- Сессии и действия диалогов: `~/.hermes/state.db`;
- Логи решений и теневых прогонов Jev: `~/.hermes/logs/jev-decisions.jsonl`, `jev-shadow.jsonl`.

**Запрос к основной модели (Teacher LLM)**:
Если в сессии или логах Jev отсутствует однозначный выбор или есть расхождение, скрипт автоматически обращается к **основной модели Hermes** (`Primary LLM Teacher`) для определения эталона:
```bash
python scripts/extract_and_label.py \
  --logs-path ~/.hermes/logs/laya_requests.jsonl \
  --db-path ~/.hermes/state.db \
  --check-jev \
  --out-dir data/
```

### Шаг 3: Тестирование точности до обучения (Pre-Training Test)
Замер базовых метрик текущей рабочей модели на валидационной выборке:
```bash
python scripts/evaluate_accuracy.py \
  --eval-data data/eval.jsonl \
  --phase before \
  --laya-url http://127.0.0.1:8787
```

### Шаг 4: Дообучение модели (Fine-Tuning / RLCD)
Обучение чекпоинта на объединенных данных:
```bash
/root/.unsloth/studio/unsloth_studio/bin/laya-train \
  --data data/train.jsonl \
  --eval data/eval.jsonl \
  --base /root/deepseek-harness/finetune-checkpoint \
  --out data/finetune_checkpoint \
  --loss rlcd \
  --epochs 4
```

### Шаг 5: Компиляция GGUF и горячая перезагрузка
```bash
python /root/deepseek-harness/compile_custom_laya.py
curl -X POST http://127.0.0.1:8787/reload
```

### Шаг 6: Пост-тест точности и калибровка Confidence Floor
Замер метрик точности после обучения и подбор оптимального порога уверенности (Confidence Floor) по стандарту `calibrate_choose` из `hermes-jev-skills` (минимизация ошибочных действий `Wrong Actions` при контроле доли отказов `Stalls`):
```bash
python scripts/evaluate_accuracy.py \
  --eval-data data/eval.jsonl \
  --phase after \
  --laya-url http://127.0.0.1:8787
```

---

## Формат итогового отчета (`data/report.md`)

Сформированный отчет включает:
1. **Сводную таблицу метрик**: точность решений, точность категорий, Combined Strict Accuracy, Brier score, задержка инференса;
2. **Срез по категориям инструментов**: детальный анализ точности в разрезе Marketplace, Shell, Desktop, Filesystem и др.;
3. **Интеграцию с плагином `kerpopule/hermes-jev-skills`**: статус детекции, объемы логов и таблица калибровки Confidence Floor Sweep;
4. **Анализ оставшихся расхождений (Discrepancies)** с указанием точных запросов и предсказаний;
5. **Итоговое заключение и рекомендацию** к использованию в production.
