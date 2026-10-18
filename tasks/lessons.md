# Lessons Learned

## Hermes Laya Tools
- Always load configuration from environment variables (.env) with sane defaults.
- Zero external dependencies: use Python standard library (`urllib.request`, `json`, etc.) so it works out of the box in Hermes Python environment without requiring extra pip packages.
- Shadow mode must ensure zero mutation of `agent.tools` and zero prompt injection, while performing full Laya inference and detailed logging.
- When executing remote SSH commands, always specify proper timeouts (`-o ConnectTimeout=...`) and avoid piping unbounded network calls.
