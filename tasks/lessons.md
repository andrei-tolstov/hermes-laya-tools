# Lessons Learned

## Hermes Laya Tools
- Always load configuration from environment variables (.env) with sane defaults.
- Zero external dependencies: use Python standard library (`urllib.request`, `json`, etc.) so it works out of the box in Hermes Python environment without requiring extra pip packages.
- Shadow mode must ensure zero mutation of `agent.tools` and zero prompt injection, while performing full Laya inference and detailed logging.
- When executing remote SSH commands, always specify proper timeouts (`-o ConnectTimeout=...`) and avoid piping unbounded network calls.
- **Hermes WebUI Settings Integration**: To render plugin settings in the Hermes Desktop / WebUI under "Настройки → Плагины", `plugin.yaml` must declare `config_schema` (Manifest v2). Saving settings in UI writes to `plugins.entries.<id>.settings` in `config.yaml`, which the plugin should read via `ctx.get_config(key)` for instant zero-restart updates.
- **Encoder Position Bias & Safe Decision Thresholds**: Encoder models like `laya-multilingual` suffer from position bias on binary choices (the first option can receive a spurious +3-5% probability boost). Never strip tools on raw argmax (`> 0.50`) with near-zero confidence. Enforce high safety thresholds (e.g. `direct_chat_threshold >= 0.75` and `min_confidence >= 0.15`) before stripping tools in an agent environment.
- **State Snippet Hygiene**: Strip system prompts, role headers, and slash command preambles (like `[/plan — plan mode]`) before sending state to Laya to avoid prompt pollution and tokenizer truncation.
