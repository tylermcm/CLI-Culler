from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aiculler.storage import SQLiteFeatureStore
from aiculler.text_scoring import CLIPTextEncoder, cosine_similarity, normalize_scores


@dataclass(frozen=True)
class ProfilePromptAtom:
    profile: str
    kind: str
    weight: float
    prompt: str


@dataclass(frozen=True)
class ProfileScoreRecord:
    image_id: int
    filename: str
    source_path: str
    technical_score: float
    profile_score: float
    normalized_profile_score: float
    final_score: float


class ProfileScorer:
    """Score images from weighted positive/negative CLIP prompt atoms."""

    def __init__(
        self,
        store: SQLiteFeatureStore,
        text_encoder: CLIPTextEncoder,
        *,
        technical_weight: float = 0.35,
        profile_weight: float = 0.65,
    ):
        self.store = store
        self.text_encoder = text_encoder
        self.technical_weight = float(technical_weight)
        self.profile_weight = float(profile_weight)

    def score_profile(self, profile_name: str, atoms: list[ProfilePromptAtom]) -> list[ProfileScoreRecord]:
        selected = [atom for atom in atoms if atom.profile == profile_name]
        if not selected:
            raise ValueError(f"No prompt atoms found for profile {profile_name!r}")

        rows = self.store.list_images(require_embedding=True)
        if not rows:
            return []
        prompt_vectors = [
            (atom, self.text_encoder.encode(atom.prompt))
            for atom in selected
        ]

        raw_profile_scores: dict[int, float] = {}
        technical_scores: dict[int, float] = {}
        for row in rows:
            image_id = int(row["id"])
            image_embedding = self.store.get_embedding(image_id)
            raw_profile_scores[image_id] = profile_similarity(image_embedding, prompt_vectors)
            technical_scores[image_id] = float(row["technical_score"] or 0.0)

        normalized_profile_scores = normalize_scores(raw_profile_scores, mode="minmax")
        updates: dict[int, tuple[float, float]] = {}
        records: list[ProfileScoreRecord] = []
        for row in rows:
            image_id = int(row["id"])
            final_score = (
                self.technical_weight * technical_scores[image_id]
                + self.profile_weight * normalized_profile_scores[image_id]
            )
            updates[image_id] = (raw_profile_scores[image_id], final_score)
            records.append(
                ProfileScoreRecord(
                    image_id=image_id,
                    filename=Path(row["source_path"]).name,
                    source_path=row["source_path"],
                    technical_score=technical_scores[image_id],
                    profile_score=raw_profile_scores[image_id],
                    normalized_profile_score=normalized_profile_scores[image_id],
                    final_score=final_score,
                )
            )

        self.store.update_profile_scores(updates, profile_name=profile_name)
        return sorted(records, key=lambda record: record.final_score, reverse=True)


def profile_similarity(image_embedding: np.ndarray, prompt_vectors: list[tuple[ProfilePromptAtom, np.ndarray]]) -> float:
    total = 0.0
    total_abs_weight = 0.0
    for atom, text_embedding in prompt_vectors:
        sign = 1.0 if atom.kind == "positive" else -1.0
        total += sign * atom.weight * cosine_similarity(image_embedding, text_embedding)
        total_abs_weight += abs(atom.weight)
    return total / total_abs_weight if total_abs_weight else 0.0


def load_profile_atoms(path: str | Path) -> list[ProfilePromptAtom]:
    atoms: list[ProfilePromptAtom] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("profiles CSV must include headers")
        for line_number, row in enumerate(reader, start=2):
            profile = (row.get("profile") or "").strip()
            kind = (row.get("type") or row.get("kind") or "").strip().lower()
            prompt = (row.get("prompt") or "").strip()
            if not profile or not kind or not prompt:
                raise ValueError(f"profiles row {line_number} requires profile, type, and prompt")
            if kind not in {"positive", "negative"}:
                raise ValueError(f"profiles row {line_number} type must be positive or negative")
            try:
                weight = float(row.get("weight") or 1.0)
            except ValueError as exc:
                raise ValueError(f"profiles row {line_number} has invalid weight") from exc
            atoms.append(ProfilePromptAtom(profile=profile, kind=kind, weight=weight, prompt=prompt))
    return atoms


def list_profile_names(atoms: list[ProfilePromptAtom]) -> list[str]:
    return sorted({atom.profile for atom in atoms})
