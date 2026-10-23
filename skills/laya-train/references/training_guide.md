# Руководство по обучению, функциям потерь и компиляции Laya

## 1. Конвейер обучения

Дообучение Laya состоит из трёх этапов:
1. **Подготовка датасета**: парсинг сессий и логов с разметкой через Teacher LLM (`train.jsonl` и `eval.jsonl`).
2. **Обучение чекпоинта**: запуск утилиты `laya-train` на GPU/CPU.
3. **Компиляция GGUF**: упаковка весов, калибровочных температур и метаданных через библиотеку `ggmlc`.

---

## 2. Параметры утилиты `laya-train`

```bash
/root/.unsloth/studio/unsloth_studio/bin/laya-train \
  --data data/train.jsonl \
  --eval data/eval.jsonl \
  --base /root/deepseek-harness/finetune-checkpoint \
  --out data/finetune_checkpoint \
  --loss rlcd \
  --epochs 4 \
  --micro-batch 8 \
  --encoder-lr 2.5e-5 \
  --head-lr 1.0e-4
```

### Основные аргументы:
- `--data`: Путь к тренировочному файлу в формате JSONL.
- `--eval`: Путь к валидационному файлу для автоматической оценки до/после.
- `--loss`:
  - `rlcd` (Reinforcement Learning from Confidence Distillation): оптимизирует не только argmax-выбор, но и калибровку вероятностей (минимизирует Brier score и ECE). Рекомендуется по умолчанию.
  - `soft-ce`: классическая кросс-энтропия со сглаживанием меток (label smoothing).
- `--epochs`: Количество эпох (обычно 3–5 достаточно для адаптации под домен без переобучения).
- `--encoder-lr`: Скорость обучения основы (2e-5 – 3e-5).
- `--head-lr`: Скорость обучения голов решений (1e-4).

---

## 3. Компиляция весов в GGUF (`ggmlc`)

После завершения `laya-train` создаётся директория с весами PyTorch (`model.safetensors`, `rl_agent_config.json`, `questions.json`). Для работы быстрого C++ инференс-сервера они компилируются в файл `.gguf`:

```python
import ggmlc
from compile_laya import LayaCleanTrunk, DecisionPipelineSpec, _pipeline_tokenizer

# 1. Загрузка чекпоинта
agent = laya.load("data/finetune_checkpoint")

# 2. Построение вычислительного графа (Trunk + Decision Heads)
trunk = LayaCleanTrunk(agent, max_len=512).eval()

# 3. Компиляция в GGUF с сохранением температурных калибровок
ggmlc.compile(
    trunk,
    example_batch,
    output="models/laya_multilingual_f16.gguf",
    quantize="f16",
    tasks=["classification"],
    extra_metadata=metadata,
)
```

---

## 4. Горячая перезагрузка (Zero-Downtime Reload)

После генерации нового файла `laya_multilingual_f16.gguf` нет необходимости перезапускать весь Docker-стек:

```bash
# Сигнал через REST API Sidecar:
curl -X POST http://127.0.0.1:8787/reload

# Или через атомарный маркер в файловой системе контейнера:
touch /tmp/laya_restart_requested
```

C++ сервер `laya` подхватывает новые веса за ~1–2 секунды, сохраняя сессии и очередь запросов.
