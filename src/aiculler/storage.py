"""The score cache: every scorer's results per capture, so no frame is scored twice."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Sequence


class ScoreCache:
    """Per-capture score cache: ``scores(capture, scorer, revision, json)``.

    Keyed by capture identity (not path) and scorer revision, so copies and sidecars are never scored
    twice and re-running with new settings never re-runs a scorer.
    """

    def __init__(self, path: str | Path):
        self.lock = threading.RLock()
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        with self.lock:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    capture TEXT NOT NULL,
                    scorer TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (capture, scorer, revision)
                );
                """
            )
            self.connection.commit()

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def get_many(self, captures: Sequence[str], scorer: str, revision: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        with self.lock:
            for start in range(0, len(captures), 500):
                chunk = list(captures[start : start + 500])
                placeholders = ", ".join("?" for _ in chunk)
                rows = self.connection.execute(
                    f"SELECT capture, value_json FROM scores WHERE scorer = ? AND revision = ? AND capture IN ({placeholders})",
                    [scorer, revision, *chunk],
                ).fetchall()
                for capture, value in rows:
                    out[capture] = json.loads(value)
        return out

    def put_many(self, rows: Iterable[tuple[str, str, str, dict]]) -> None:
        with self.lock:
            self.connection.executemany(
                "INSERT OR REPLACE INTO scores (capture, scorer, revision, value_json) VALUES (?, ?, ?, ?)",
                [(c, s, r, json.dumps(v, sort_keys=True, default=_plain)) for c, s, r, v in rows],
            )
            self.connection.commit()


def _plain(value):
    """numpy scalars and arrays (from model engines) as plain JSON values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")
