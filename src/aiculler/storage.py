from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class SQLiteFeatureStore:
    """Thread-safe SQLite contract for image metadata, embeddings, and feedback."""

    def __init__(self, path: str | Path | sqlite3.Connection):
        self.lock = threading.RLock()
        if isinstance(path, sqlite3.Connection):
            self.path: Path | None = None
            self.connection = path
        else:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def _ensure_schema(self) -> None:
        with self.lock:
            self.connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL UNIQUE,
                    preview_path TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    width INTEGER,
                    height INTEGER,
                    technical_score REAL,
                    aesthetic_prior REAL,
                    pointwise_score REAL,
                    prompt_score REAL,
                    prompt_text TEXT,
                    learned_user_score REAL,
                    profile_score REAL,
                    profile_name TEXT,
                    tag_base_score REAL,
                    tag_penalty REAL,
                    tag_flags TEXT,
                    final_score REAL,
                    error TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    image_id INTEGER PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    dtype TEXT NOT NULL,
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    label INTEGER NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
                CREATE INDEX IF NOT EXISTS idx_images_final_score ON images(final_score);
                """
            )
            self._ensure_column("images", "prompt_score", "REAL")
            self._ensure_column("images", "prompt_text", "TEXT")
            self._ensure_column("images", "learned_user_score", "REAL")
            self._ensure_column("images", "profile_score", "REAL")
            self._ensure_column("images", "profile_name", "TEXT")
            self._ensure_column("images", "tag_base_score", "REAL")
            self._ensure_column("images", "tag_penalty", "REAL")
            self._ensure_column("images", "tag_flags", "TEXT")
            self.connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def upsert_image(
        self,
        source_path: str | Path,
        *,
        preview_path: str | Path | None = None,
        status: str = "pending",
        width: int | None = None,
        height: int | None = None,
        metadata: dict | None = None,
        error: str | None = None,
    ) -> int:
        source = str(Path(source_path))
        preview = str(Path(preview_path)) if preview_path is not None else None
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        with self.lock:
            cur = self.connection.execute(
                """
                INSERT INTO images (
                    source_path, preview_path, status, width, height, error, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    preview_path = COALESCE(excluded.preview_path, images.preview_path),
                    status = excluded.status,
                    width = COALESCE(excluded.width, images.width),
                    height = COALESCE(excluded.height, images.height),
                    error = excluded.error,
                    metadata_json = excluded.metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """,
                (source, preview, status, width, height, error, metadata_json),
            )
            image_id = int(cur.fetchone()["id"])
            self.connection.commit()
            return image_id

    def mark_error(self, image_id: int, error: str) -> None:
        with self.lock:
            self.connection.execute(
                """
                UPDATE images
                SET status = 'error', error = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (error, image_id),
            )
            self.connection.commit()

    def save_features(
        self,
        image_id: int,
        embedding: Sequence[float] | np.ndarray,
        *,
        technical_score: float | None = None,
        aesthetic_prior: float | None = None,
        pointwise_score: float | None = None,
        prompt_score: float | None = None,
        prompt_text: str | None = None,
        learned_user_score: float | None = None,
        profile_score: float | None = None,
        profile_name: str | None = None,
        tag_base_score: float | None = None,
        tag_penalty: float | None = None,
        tag_flags: str | None = None,
        final_score: float | None = None,
        status: str = "ready",
    ) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        with self.lock:
            self.connection.execute(
                """
                INSERT INTO embeddings (image_id, embedding, dim, dtype)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    dim = excluded.dim,
                    dtype = excluded.dtype
                """,
                (image_id, vector.tobytes(), int(vector.size), str(vector.dtype)),
            )
            self.connection.execute(
                """
                UPDATE images
                SET technical_score = COALESCE(?, technical_score),
                    aesthetic_prior = COALESCE(?, aesthetic_prior),
                    pointwise_score = COALESCE(?, pointwise_score),
                    prompt_score = COALESCE(?, prompt_score),
                    prompt_text = COALESCE(?, prompt_text),
                    learned_user_score = COALESCE(?, learned_user_score),
                    profile_score = COALESCE(?, profile_score),
                    profile_name = COALESCE(?, profile_name),
                    tag_base_score = COALESCE(?, tag_base_score),
                    tag_penalty = COALESCE(?, tag_penalty),
                    tag_flags = COALESCE(?, tag_flags),
                    final_score = COALESCE(?, final_score),
                    status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    technical_score,
                    aesthetic_prior,
                    pointwise_score,
                    prompt_score,
                    prompt_text,
                    learned_user_score,
                    profile_score,
                    profile_name,
                    tag_base_score,
                    tag_penalty,
                    tag_flags,
                    final_score,
                    status,
                    image_id,
                ),
            )
            self.connection.commit()

    def update_tag_scores(
        self,
        scores: dict[int, tuple[float, float, str, float]],
    ) -> None:
        with self.lock:
            self.connection.executemany(
                """
                UPDATE images
                SET tag_base_score = ?,
                    tag_penalty = ?,
                    tag_flags = ?,
                    final_score = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                [
                    (float(base_score), float(tag_penalty), tag_flags, float(final_score), int(image_id))
                    for image_id, (base_score, tag_penalty, tag_flags, final_score) in scores.items()
                ],
            )
            self.connection.commit()

    def update_composite_scores(self, scores: dict[int, dict]) -> None:
        with self.lock:
            self.connection.executemany(
                """
                UPDATE images
                SET prompt_score = ?,
                    prompt_text = ?,
                    profile_score = ?,
                    profile_name = ?,
                    learned_user_score = ?,
                    tag_base_score = ?,
                    tag_penalty = ?,
                    tag_flags = ?,
                    final_score = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                [
                    (
                        float(values["prompt_score"]),
                        values["prompt_text"],
                        float(values["profile_score"]),
                        values["profile_name"],
                        float(values["learned_user_score"]),
                        float(values["tag_base_score"]),
                        float(values["tag_penalty"]),
                        values["tag_flags"],
                        float(values["final_score"]),
                        int(image_id),
                    )
                    for image_id, values in scores.items()
                ],
            )
            self.connection.commit()

    def update_profile_scores(
        self,
        scores: dict[int, tuple[float, float]],
        *,
        profile_name: str,
    ) -> None:
        with self.lock:
            self.connection.executemany(
                """
                UPDATE images
                SET profile_score = ?,
                    profile_name = ?,
                    final_score = ?,
                    tag_base_score = NULL,
                    tag_penalty = NULL,
                    tag_flags = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                [
                    (float(profile_score), profile_name, float(final_score), int(image_id))
                    for image_id, (profile_score, final_score) in scores.items()
                ],
            )
            self.connection.commit()

    def update_scores(self, scores: dict[int, float]) -> None:
        with self.lock:
            self.connection.executemany(
                """
                UPDATE images
                SET final_score = ?,
                    tag_base_score = NULL,
                    tag_penalty = NULL,
                    tag_flags = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                [(float(score), int(image_id)) for image_id, score in scores.items()],
            )
            self.connection.commit()

    def update_prompt_scores(
        self,
        scores: dict[int, tuple[float, float]],
        *,
        prompt: str,
    ) -> None:
        with self.lock:
            self.connection.executemany(
                """
                UPDATE images
                SET prompt_score = ?,
                    prompt_text = ?,
                    final_score = ?,
                    tag_base_score = NULL,
                    tag_penalty = NULL,
                    tag_flags = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                [
                    (float(prompt_score), prompt, float(final_score), int(image_id))
                    for image_id, (prompt_score, final_score) in scores.items()
                ],
            )
            self.connection.commit()

    def update_learned_scores(self, scores: dict[int, tuple[float, float]]) -> None:
        with self.lock:
            self.connection.executemany(
                """
                UPDATE images
                SET learned_user_score = ?,
                    final_score = ?,
                    tag_base_score = NULL,
                    tag_penalty = NULL,
                    tag_flags = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                [
                    (float(learned_user_score), float(final_score), int(image_id))
                    for image_id, (learned_user_score, final_score) in scores.items()
                ],
            )
            self.connection.commit()

    def add_feedback(self, image_id: int, label: int, note: str | None = None) -> None:
        with self.lock:
            self.connection.execute(
                "INSERT INTO feedback (image_id, label, note) VALUES (?, ?, ?)",
                (int(image_id), int(label), note),
            )
            self.connection.commit()

    def get_embedding(self, image_id: int) -> np.ndarray:
        with self.lock:
            row = self.connection.execute(
                "SELECT embedding, dim, dtype FROM embeddings WHERE image_id = ?",
                (int(image_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"No embedding stored for image id {image_id}")
        return np.frombuffer(row["embedding"], dtype=np.dtype(row["dtype"])).copy()

    def get_all_embeddings(self) -> np.ndarray:
        rows = self.list_images(require_embedding=True)
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([self.get_embedding(int(row["id"])) for row in rows])

    def get_all_embedding_ids(self) -> list[int]:
        return [int(row["id"]) for row in self.list_images(require_embedding=True)]

    def get_image(self, image_id: int) -> sqlite3.Row | None:
        with self.lock:
            return self.connection.execute(
                "SELECT * FROM images WHERE id = ?",
                (int(image_id),),
            ).fetchone()

    def list_images(
        self,
        *,
        statuses: Iterable[str] | None = None,
        require_embedding: bool = False,
    ) -> list[sqlite3.Row]:
        query = "SELECT images.* FROM images"
        params: list[str] = []
        clauses: list[str] = []
        if require_embedding:
            query += " INNER JOIN embeddings ON embeddings.image_id = images.id"
        if statuses:
            status_values = list(statuses)
            placeholders = ", ".join("?" for _ in status_values)
            clauses.append(f"images.status IN ({placeholders})")
            params.extend(status_values)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY COALESCE(images.final_score, images.pointwise_score, 0.0) DESC, images.id ASC"
        with self.lock:
            return list(self.connection.execute(query, params).fetchall())


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
                [(c, s, r, json.dumps(v, sort_keys=True)) for c, s, r, v in rows],
            )
            self.connection.commit()
