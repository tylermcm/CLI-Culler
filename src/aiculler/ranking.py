from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.decomposition import IncrementalPCA

from aiculler.storage import SQLiteFeatureStore

QueryCallback = Callable[[int, int, dict], int | bool | str]


@dataclass(frozen=True)
class PreferenceResult:
    left_id: int
    right_id: int
    probability_left: float
    margin: float
    indifference: bool


class GlobalRanker:
    """Global preference function using IPCA, a linear vector, and BTL probabilities."""

    def __init__(
        self,
        *,
        projected_dim: int = 64,
        epsilon: float = 0.05,
        technical_threshold: float = 0.25,
    ):
        self.projected_dim = int(projected_dim)
        self.epsilon = float(epsilon)
        self.technical_threshold = float(technical_threshold)
        self.projector: IncrementalPCA | None = None
        self.weights: np.ndarray | None = None
        self.fitted = False

    def fit_projection(self, embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("GlobalRanker requires at least one embedding")
        n_components = min(self.projected_dim, embeddings.shape[0], embeddings.shape[1])
        self.projector = IncrementalPCA(n_components=n_components)
        self.projector.partial_fit(embeddings)
        self.weights = np.zeros(n_components, dtype=np.float32)
        self.fitted = True

    def update_weights(self, weights: Sequence[float]) -> None:
        weights_arr = np.asarray(weights, dtype=np.float32).reshape(-1)
        if self.weights is None:
            self.weights = weights_arr
        elif weights_arr.size != self.weights.size:
            raise ValueError(f"Expected {self.weights.size} weights, got {weights_arr.size}")
        else:
            self.weights = weights_arr

    def project(self, embedding: np.ndarray) -> np.ndarray:
        if not self.fitted or self.projector is None:
            raise RuntimeError("GlobalRanker must be fitted before scoring")
        return self.projector.transform(np.atleast_2d(np.asarray(embedding, dtype=np.float32)))[0]

    def score(self, embedding: np.ndarray, *, aesthetic_prior: float = 0.0) -> float:
        projected = self.project(embedding)
        weights = self.weights
        if weights is None:
            weights = np.zeros(projected.size, dtype=np.float32)
        return float(np.dot(weights, projected) + aesthetic_prior)

    def preference_probability(self, left_score: float, right_score: float) -> float:
        delta = float(left_score - right_score)
        if abs(delta) <= self.epsilon:
            return 0.5
        return 1.0 / (1.0 + math.exp(-delta))

    def compare(self, left_id: int, left_score: float, right_id: int, right_score: float) -> PreferenceResult:
        margin = float(left_score - right_score)
        probability = self.preference_probability(left_score, right_score)
        return PreferenceResult(left_id, right_id, probability, abs(margin), abs(margin) <= self.epsilon)


class ActiveQuicksortCuller:
    """Noisy active quicksort with CLI or GUI-ready query callbacks."""

    def __init__(
        self,
        store: SQLiteFeatureStore,
        *,
        ranker: GlobalRanker | None = None,
        query_callback: QueryCallback | None = None,
        active_threshold: float = 0.10,
        technical_threshold: float = 0.25,
        rng: random.Random | None = None,
    ):
        self.store = store
        self.ranker = ranker or GlobalRanker(technical_threshold=technical_threshold)
        self.query_callback = query_callback
        self.active_threshold = float(active_threshold)
        self.technical_threshold = float(technical_threshold)
        self.rng = rng or random.Random()
        self.lock = threading.RLock()

    def prepare(self) -> None:
        embeddings = self.store.get_all_embeddings()
        if embeddings.size == 0:
            raise ValueError("No embeddings are available for sorting")
        if not self.ranker.fitted:
            self.ranker.fit_projection(embeddings)

    def sort(self, image_ids: Sequence[int] | None = None) -> list[int]:
        with self.lock:
            self.prepare()
            ids = list(image_ids) if image_ids is not None else self.store.get_all_embedding_ids()
            candidates: list[int] = []
            rejected: list[int] = []
            for image_id in ids:
                row = self.store.get_image(image_id)
                technical_score = float(row["technical_score"] or 0.0) if row is not None else 0.0
                if technical_score < self.technical_threshold:
                    rejected.append(image_id)
                else:
                    candidates.append(image_id)
            ranked = self._quicksort(candidates) + rejected
            scores = {image_id: float(len(ranked) - idx) for idx, image_id in enumerate(ranked)}
            self.store.update_scores(scores)
            return ranked

    def _quicksort(self, image_ids: list[int]) -> list[int]:
        if len(image_ids) <= 1:
            return image_ids
        pivot_id = self.rng.choice(image_ids)
        pivot_score = self._score_image(pivot_id)
        higher: list[int] = []
        lower: list[int] = []
        rejected: list[int] = []

        for image_id in image_ids:
            if image_id == pivot_id:
                continue
            score = self._score_image(image_id)
            margin = abs(score - pivot_score)
            if margin > self.active_threshold:
                if score >= pivot_score:
                    higher.append(image_id)
                else:
                    lower.append(image_id)
                continue

            label = self._query(image_id, pivot_id, {"score": score, "pivot_score": pivot_score, "margin": margin})
            if label:
                higher.append(image_id)
            else:
                lower.append(image_id)

        return self._quicksort(higher) + [pivot_id] + self._quicksort(lower) + rejected

    def _score_image(self, image_id: int) -> float:
        row = self.store.get_image(image_id)
        if row is None:
            raise KeyError(f"Unknown image id {image_id}")
        prior = row["aesthetic_prior"]
        if prior is None:
            prior = row["technical_score"] or 0.0
        return self.ranker.score(self.store.get_embedding(image_id), aesthetic_prior=float(prior))

    def _query(self, image_id: int, pivot_id: int, context: dict) -> bool:
        if self.query_callback is None:
            return self._cli_query(image_id, pivot_id, context)
        result = self.query_callback(image_id, pivot_id, context)
        return self._normalize_query_result(result)

    @staticmethod
    def _normalize_query_result(result: int | bool | str) -> bool:
        if isinstance(result, str):
            normalized = result.strip().lower()
            if normalized in {"k", "keep", "1", "true", "yes", "y"}:
                return True
            if normalized in {"r", "reject", "0", "false", "no", "n"}:
                return False
            raise ValueError(f"Unsupported query result: {result!r}")
        return bool(result)

    def _cli_query(self, image_id: int, pivot_id: int, context: dict) -> bool:
        while True:
            answer = input(
                f"Image {image_id} vs pivot {pivot_id} "
                f"(margin={context['margin']:.4f}) [K]eep/[R]eject? "
            )
            try:
                return self._normalize_query_result(answer)
            except ValueError:
                print("Please enter K to keep or R to reject.")
