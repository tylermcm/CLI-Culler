from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass(frozen=True)
class AdapterOutput:
    embedding: np.ndarray
    pointwise_score: float


class RankingAwareAdapter:
    """Deterministic NumPy layout for the ranking-aware visual adapter.

    This module mirrors the architecture shape from the technical design without
    training a heavyweight transformer inside the local culling engine. It is
    useful as a stable offline adapter and as an integration target for future
    learned weights.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        projected_dim: int = 512,
        text_dim: int = 512,
        relation_tokens: int = 16,
        hidden_dim: int = 256,
        seed: int = 13,
    ):
        self.input_dim = int(input_dim)
        self.projected_dim = int(projected_dim)
        self.text_dim = int(text_dim)
        self.relation_tokens = int(relation_tokens)
        self.hidden_dim = int(hidden_dim)
        rng = np.random.default_rng(seed)

        self.visual_projection = self._init_matrix(rng, self.input_dim, self.projected_dim)
        self.text_projection = self._init_matrix(rng, self.text_dim, self.projected_dim)
        self.cross_gate = self._init_matrix(rng, self.projected_dim, self.projected_dim)

        self.point_w1 = self._init_matrix(rng, self.projected_dim, self.hidden_dim)
        self.point_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.point_w2 = self._init_matrix(rng, self.hidden_dim, 1)
        self.point_b2 = np.zeros(1, dtype=np.float32)

        self.rel_tokens = rng.normal(0.0, 0.02, (self.relation_tokens, self.projected_dim * 2)).astype(
            np.float32
        )
        self.rel_w1 = self._init_matrix(rng, self.projected_dim * 2, self.hidden_dim)
        self.rel_b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.rel_w2 = self._init_matrix(rng, self.hidden_dim, self.hidden_dim // 2)
        self.rel_b2 = np.zeros(self.hidden_dim // 2, dtype=np.float32)
        self.rel_w3 = self._init_matrix(rng, self.hidden_dim // 2, 1)
        self.rel_b3 = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _init_matrix(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
        scale = np.sqrt(2.0 / max(1, rows + cols))
        return rng.normal(0.0, scale, (rows, cols)).astype(np.float32)

    def encode_text_query(self, query: str | None) -> np.ndarray:
        """Hash a text query into a deterministic dense vector."""

        if not query:
            return np.zeros(self.text_dim, dtype=np.float32)
        vector = np.zeros(self.text_dim, dtype=np.float32)
        for token in query.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % self.text_dim
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def text_conditioned_embedding(
        self,
        embedding: np.ndarray,
        *,
        text_query: str | None = None,
        text_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        visual = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if visual.size != self.input_dim:
            raise ValueError(f"Expected embedding dimension {self.input_dim}, got {visual.size}")
        text = (
            self.encode_text_query(text_query)
            if text_embedding is None
            else np.asarray(text_embedding, dtype=np.float32).reshape(-1)
        )
        if text.size != self.text_dim:
            raise ValueError(f"Expected text dimension {self.text_dim}, got {text.size}")

        visual_tokens = visual @ self.visual_projection
        text_tokens = text @ self.text_projection
        cross_scores = (visual_tokens * text_tokens) / np.sqrt(self.projected_dim)
        gate = 1.0 / (1.0 + np.exp(-(cross_scores + visual_tokens @ self.cross_gate / self.projected_dim)))
        conditioned = (gate * visual_tokens) + ((1.0 - gate) * text_tokens)
        norm = np.linalg.norm(conditioned)
        return (conditioned / norm if norm > 0 else conditioned).astype(np.float32)

    def pointwise_score(self, projected_embedding: np.ndarray) -> float:
        z = np.asarray(projected_embedding, dtype=np.float32).reshape(-1)
        hidden = gelu(z @ self.point_w1 + self.point_b1)
        return float((hidden @ self.point_w2 + self.point_b2).reshape(-1)[0])

    def adapt(self, embedding: np.ndarray, *, text_query: str | None = None) -> AdapterOutput:
        projected = self.text_conditioned_embedding(embedding, text_query=text_query)
        return AdapterOutput(projected, self.pointwise_score(projected))

    def pairwise_distance(
        self,
        left_projected: np.ndarray,
        right_projected: np.ndarray,
    ) -> float:
        left = np.asarray(left_projected, dtype=np.float32).reshape(-1)
        right = np.asarray(right_projected, dtype=np.float32).reshape(-1)
        if left.size != self.projected_dim or right.size != self.projected_dim:
            raise ValueError("Pairwise inputs must already be projected adapter embeddings")

        pair = np.concatenate([left, right])
        token_scores = (self.rel_tokens @ pair) / np.sqrt(pair.size)
        attention = softmax(token_scores, axis=0)
        attended = np.sum(self.rel_tokens * attention[:, None], axis=0)
        left_ctx, right_ctx = np.split(attended, 2)
        difference = np.concatenate([left - right_ctx, right - left_ctx])
        h1 = gelu(difference @ self.rel_w1 + self.rel_b1)
        h2 = gelu(h1 @ self.rel_w2 + self.rel_b2)
        return float((h2 @ self.rel_w3 + self.rel_b3).reshape(-1)[0])
