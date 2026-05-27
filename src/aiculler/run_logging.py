from __future__ import annotations

import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class RunLogger:
    """Small structured logger for CLI runs.

    Writes append-only JSONL events plus optional CSV tables inside a per-run
    directory. It avoids Python logging configuration so host GUI apps can keep
    their own logging stack separate.
    """

    def __init__(
        self,
        command: str,
        *,
        log_dir: str | Path | None,
        run_id: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ):
        self.enabled = bool(enabled and log_dir)
        self.command = command
        self.started_at = time.perf_counter()
        self.run_id = sanitize_name(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
        self.run_dir: Path | None = None
        self.events_path: Path | None = None
        if self.enabled:
            self.run_dir = Path(log_dir) / f"{self.run_id}_{sanitize_name(command)}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.events_path = self.run_dir / "events.jsonl"
            self.event("run_start", {"command": command, "metadata": metadata or {}})

    def event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if not self.enabled or self.events_path is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - self.started_at, 6),
            "type": event_type,
            "payload": normalize_value(payload or {}),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def table(self, name: str, rows: Iterable[dict[str, Any]]) -> Path | None:
        if not self.enabled or self.run_dir is None:
            return None
        row_list = [normalize_value(row) for row in rows]
        path = self.run_dir / f"{sanitize_name(name)}.csv"
        fieldnames: list[str] = []
        for row in row_list:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            if fieldnames:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(row_list)
        self.event("table_written", {"name": name, "path": str(path), "rows": len(row_list)})
        return path

    def summary(self, payload: dict[str, Any]) -> Path | None:
        if not self.enabled or self.run_dir is None:
            return None
        path = self.run_dir / "summary.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(normalize_value(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        self.event("summary_written", {"path": str(path), "summary": payload})
        return path

    def close(self, status: str = "ok") -> None:
        self.event("run_end", {"status": status})


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "run"


def normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
