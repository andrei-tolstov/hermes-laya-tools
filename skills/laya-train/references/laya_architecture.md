# Архитектура Laya System 1 Decision Engine

## 1. Концепция System 1
В классической когнитивной психологии (Канеман) «Система 1» отвечает за быстрые, автоматические и инстинктивные реакции, в то время как «Система 2» — за медленные, энергоёмкие рассуждения.

В Hermes Agent:
- **Основная LLM (Система 2)**: большая генеративная модель (DeepSeek / Qwen / Claude / GPT), выполняющая логический синтез, генерацию кода и работу с контекстом.
- **Laya Engine (Система 1)**: легковесный энкодер на базе PyTorch / GGUF, выполняющий **один прямой проход (encoder pass)** без авторегрессионной генерации токенов. Принимает решения за **15–90 мс**.

---

## 2. Формат типизированных вопросов (Typed Decisions)

Laya оценивает запрос по словарю типизированных вопросов:

### Тип `choice` (Дискретный выбор с распределением вероятностей)
```json
{
  "tool_decision": {
    "type": "choice",
    "instructions": "Determine whether external tools are required or if direct conversational response is appropriate.",
    "criteria": {
      "direct_chat": "Pure conversational chat, greeting, text explanation",
      "execute_tools": "Requires executing tools, commands, reading files, searching web"
    }
  }
}
```

### Формат ответа Laya:
```json
{
  "answers": {
    "tool_decision": {
      "type": "choice",
      "choice": "execute_tools",
      "confidence": 0.94,
      "probabilities": {
        "execute_tools": 0.97,
        "direct_chat": 0.03
      }
    }
  },
  "latency_ms": 22.4
}
```

---

## 3. Семантические категории инструментов

Laya распределяет полный пул (88+ инструментов) по 13 устойчивым категориям:

1. `marketplace_and_commerce`: Ozon, Wildberries, Yandex Market, Avito, AliExpress, Taobao, Lamoda, Detmir, Drom, Citilink, DNS, Cian, MPStats, корзины, заказы, сравнение цен.
2. `desktop_and_computer_use`: Computer Use, OS GUI, перемещение курсора, клики мыши, переключение окон, preview.
3. `filesystem`: чтение, запись, редактирование, поиск файлов, дерево каталогов (`read_file`, `write_file`, `list_dir`, `edit_file`).
4. `terminal_execution`: выполнение bash, powershell, системных команд, процессов (`terminal`, `execute_code`).
5. `web_and_search`: поисковые системы, скачивание страниц, HTTP API (`web_search`, `web_extract`).
6. `browser_automation`: навигация по сайтам, DOM-клики, снапшоты страниц (`browser_exec`, `browser_navigate`).
7. `system_and_tasks`: cron-расписания, vault-хранилище, управление процессами, todo (`process_manage`, `cronjob_manage`).
8. `meta_and_skills`: управление навыками, делегирование subagent, долговременная память (`skill_manage`, `delegate_task`, `memory`).
9. `multimedia_and_vision`: анализ изображений, генерация видео, распознавание речи, TTS (`vision_analyze`, `text_to_speech`).
10. `code_and_git`: Git-контроль версий, диффы, коммиты, анализ кода (`git_status`, `git_commit`).
11. `devops_containers`: Docker, контейнеры, Kubernetes, системные службы (`docker`, `k8s`).
12. `external_messaging`: отправка уведомлений в Telegram, Discord, Slack, подтверждения пользователя (`telegram`, `ask_user`).
13. `general_utility`: калькулятор, форматирование, преобразование времени и данных.

---

## 4. Эндпоинты сервиса

| Эндпоинт | Порт | Метод | Описание |
|---|---|---|---|
| `/ask` | `8787` | POST | Основной фасад Laya Sidecar. Принимает `state` и `questions`. |
| `/v1/systemone` | `8080` | POST | Прямой C++ инференс-сервер `ggmlc`. |
| `/health` | `8787` / `8080` | GET | Проверка состояния движка (статус `ok`, используемая модель). |
| `/reload` | `8787` | POST | Горячая перезагрузка весов модели после компиляции нового GGUF. |
