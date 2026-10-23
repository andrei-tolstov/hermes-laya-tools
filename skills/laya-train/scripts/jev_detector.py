#!/usr/bin/env python3
"""Модуль и CLI-утилита для проверки наличия и состояния плагина kerpopule/hermes-jev-skills в Hermes.

Проверяет:
1. Регистрацию в CLI Hermes (`hermes plugins list`);
2. Наличие директорий плагина в `~/.hermes/plugins/hermes-jev` или `kerpopule/hermes-jev-skills`;
3. Наличие директории настроек `~/.hermes/jev/` (state.json, routing.json);
4. Наличие и объем логов решений Jev (`~/.hermes/logs/jev-decisions.jsonl`, `jev-shadow.jsonl`, `jev-gate.jsonl`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def get_default_hermes_home() -> Path:
    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        return Path(env_home).resolve()
    return (Path.home() / ".hermes").resolve()


def check_cli_plugin_status() -> Dict[str, Any]:
    """Проверяет статус плагина через CLI `hermes plugins list`."""
    hermes_bin = shutil.which("hermes")
    if not hermes_bin:
        return {"cli_available": False, "found": False, "status": "unknown"}

    try:
        proc = subprocess.run(
            [hermes_bin, "plugins", "list", "--plain", "--no-bundled"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        output = proc.stdout + "\n" + proc.stderr
        if proc.returncode != 0:
            # Fallback к обычному plugins list
            proc = subprocess.run(
                [hermes_bin, "plugins", "list"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            output = proc.stdout + "\n" + proc.stderr

        for line in output.splitlines():
            low = line.lower()
            if "hermes-jev" in low or "hermes_jev" in low or "hermes-jev-skills" in low:
                is_enabled = "enabled" in low
                is_disabled = "disabled" in low
                return {
                    "cli_available": True,
                    "found": True,
                    "line": line.strip(),
                    "enabled": is_enabled and not is_disabled,
                }
    except Exception as e:
        return {"cli_available": True, "found": False, "error": str(e)}

    return {"cli_available": True, "found": False, "enabled": False}


def detect_jev_plugin(hermes_home: Optional[Path] = None) -> Dict[str, Any]:
    """Комплексная проверка наличия плагина kerpopule/hermes-jev-skills и его логов."""
    home = hermes_home or get_default_hermes_home()
    plugins_dir = home / "plugins"
    logs_dir = home / "logs"
    jev_dir = home / "jev"

    result: Dict[str, Any] = {
        "is_detected": False,
        "is_enabled": False,
        "hermes_home": str(home),
        "plugin_dir": None,
        "plugin_yaml": None,
        "manifest": {},
        "jev_state_dir": str(jev_dir) if jev_dir.exists() else None,
        "logs": {},
        "log_counts": {},
        "cli_check": check_cli_plugin_status(),
    }

    # 1. Поиск папки плагина
    candidate_names = [
        "hermes-jev",
        "kerpopule/hermes-jev-skills",
        "hermes-jev-skills",
        "jev",
    ]
    found_plugin_path: Optional[Path] = None

    if plugins_dir.exists():
        for cand in candidate_names:
            p = plugins_dir / cand
            if p.exists() and p.is_dir():
                found_plugin_path = p
                break

    if found_plugin_path:
        result["plugin_dir"] = str(found_plugin_path)
        manifest_file = found_plugin_path / "plugin.yaml"
        if manifest_file.exists():
            result["plugin_yaml"] = str(manifest_file)
            try:
                # Простой парсинг yaml ключей без внешних зависимостей pyyaml
                manifest_text = manifest_file.read_text(encoding="utf-8", errors="replace")
                manifest_data: Dict[str, str] = {}
                for line in manifest_text.splitlines():
                    if ":" in line and not line.strip().startswith("#"):
                        k, v = line.split(":", 1)
                        manifest_data[k.strip()] = v.strip().strip("\"'")
                result["manifest"] = manifest_data
            except Exception:
                pass

    # 2. Проверка логов Jev
    log_candidates = {
        "decisions": logs_dir / "jev-decisions.jsonl",
        "shadow": logs_dir / "jev-shadow.jsonl",
        "gate": logs_dir / "jev-gate.jsonl",
        "ledger": logs_dir / "jev-ledger.jsonl",
    }

    total_jev_records = 0
    for key, path in log_candidates.items():
        if path.exists():
            result["logs"][key] = str(path)
            try:
                count = sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
            except Exception:
                count = 0
            result["log_counts"][key] = count
            total_jev_records += count

    # 3. Принятие решения об обнаружении
    cli_found = result["cli_check"].get("found", False)
    cli_enabled = result["cli_check"].get("enabled", False)
    fs_found = found_plugin_path is not None
    has_jev_dir = jev_dir.exists()
    has_logs = total_jev_records > 0

    if fs_found or cli_found or has_jev_dir or has_logs:
        result["is_detected"] = True
        # Плагин включен, если подтверждено CLI или есть активные логи / директории
        result["is_enabled"] = cli_enabled or (fs_found and not result["cli_check"].get("found") and has_logs) or cli_found

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка плагина kerpopule/hermes-jev-skills")
    parser.add_argument("--hermes-home", default=None, help="Путь к домашней директории Hermes")
    parser.add_argument("--json", action="store_true", help="Вывод в формате JSON")
    args = parser.parse_args()

    home = Path(args.hermes_home).resolve() if args.hermes_home else None
    info = detect_jev_plugin(home)

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    print("=" * 60)
    print(" Hermes Jev Skills Plugin Detection ")
    print("=" * 60)
    print(f"Обнаружен: {'ДА' if info['is_detected'] else 'НЕТ'}")
    print(f"Активен/Включен: {'ДА' if info['is_enabled'] else 'НЕТ'}")
    print(f"Директория плагина: {info['plugin_dir'] or 'Не найдена'}")
    print(f"Конфигурация Jev: {info['jev_state_dir'] or 'Не найдена'}")
    print("Логи решений:")
    for k, p in info["logs"].items():
        print(f"  - {k}: {p} ({info['log_counts'].get(k, 0)} записей)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
