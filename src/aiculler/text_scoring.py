from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from aiculler.storage import SQLiteFeatureStore


@dataclass(frozen=True)
class TextScoreRecord:
    image_id: int
    source_path: str
    technical_score: float
    prompt_score: float
    normalized_prompt_score: float
    final_score: float


class CLIPTextEncoder:
    """Offline CLIP text encoder backed by local tokenizer.json and ONNX files."""

    def __init__(
        self,
        text_onnx_path: str | Path,
        tokenizer_json_path: str | Path,
        *,
        sequence_length: int = 77,
        providers: list[str] | None = None,
    ):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("onnxruntime and tokenizers are required for prompt scoring") from exc

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = providers or self._available_providers(ort)
        self.session = ort.InferenceSession(str(text_onnx_path), opts, providers=providers)
        self.input_name = self._select_input_name(self.session)
        self.output_name = self._select_output_name(self.session)

        self.tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
        self.sequence_length = int(sequence_length)
        pad_id = self.tokenizer.token_to_id("<|endoftext|>")
        if pad_id is None:
            raise ValueError("Tokenizer does not contain CLIP <|endoftext|> pad token")
        self.tokenizer.enable_truncation(max_length=self.sequence_length)
        self.tokenizer.enable_padding(length=self.sequence_length, pad_id=pad_id, pad_token="<|endoftext|>")

    @staticmethod
    def _available_providers(ort) -> list[str]:
        preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        return [provider for provider in preferred if provider in available] or ["CPUExecutionProvider"]

    @staticmethod
    def _select_input_name(session) -> str:
        for input_meta in session.get_inputs():
            if input_meta.name == "input_ids":
                return input_meta.name
        if len(session.get_inputs()) != 1:
            names = ", ".join(input_meta.name for input_meta in session.get_inputs())
            raise ValueError(f"Expected a text model with one input_ids input, got: {names}")
        return session.get_inputs()[0].name

    @staticmethod
    def _select_output_name(session) -> str:
        for output_meta in session.get_outputs():
            if "embed" in output_meta.name.lower():
                return output_meta.name
        return session.get_outputs()[0].name

    def encode(self, prompt: str) -> np.ndarray:
        encoding = self.tokenizer.encode(prompt)
        input_ids = np.asarray([encoding.ids], dtype=np.int64)
        output = self.session.run([self.output_name], {self.input_name: input_ids})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)


class TextConditionedScorer:
    """Blend TOPIQ technical quality with CLIP prompt alignment."""

    def __init__(
        self,
        store: SQLiteFeatureStore,
        text_encoder: CLIPTextEncoder,
        *,
        technical_weight: float = 0.45,
        prompt_weight: float = 0.55,
        normalize_prompt: str = "minmax",
    ):
        self.store = store
        self.text_encoder = text_encoder
        self.technical_weight = float(technical_weight)
        self.prompt_weight = float(prompt_weight)
        if normalize_prompt not in {"minmax", "none"}:
            raise ValueError("normalize_prompt must be 'minmax' or 'none'")
        self.normalize_prompt = normalize_prompt

    def score_prompt(self, prompt: str) -> list[TextScoreRecord]:
        text_embedding = self.text_encoder.encode(prompt)
        rows = self.store.list_images(require_embedding=True)
        if not rows:
            return []

        raw_prompt_scores: dict[int, float] = {}
        technical_scores: dict[int, float] = {}
        for row in rows:
            image_id = int(row["id"])
            image_embedding = self.store.get_embedding(image_id)
            raw_prompt_scores[image_id] = cosine_similarity(image_embedding, text_embedding)
            technical_scores[image_id] = float(row["technical_score"] or 0.0)

        normalized_prompt_scores = normalize_scores(raw_prompt_scores, mode=self.normalize_prompt)
        records: list[TextScoreRecord] = []
        updates: dict[int, tuple[float, float]] = {}
        for row in rows:
            image_id = int(row["id"])
            prompt_score = raw_prompt_scores[image_id]
            normalized_prompt_score = normalized_prompt_scores[image_id]
            final_score = (
                self.technical_weight * technical_scores[image_id]
                + self.prompt_weight * normalized_prompt_score
            )
            updates[image_id] = (prompt_score, final_score)
            records.append(
                TextScoreRecord(
                    image_id=image_id,
                    source_path=row["source_path"],
                    technical_score=technical_scores[image_id],
                    prompt_score=prompt_score,
                    normalized_prompt_score=normalized_prompt_score,
                    final_score=final_score,
                )
            )

        self.store.update_prompt_scores(updates, prompt=prompt)
        return sorted(records, key=lambda record: record.final_score, reverse=True)


def cosine_similarity(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float32).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float32).reshape(-1)
    if left_arr.size != right_arr.size:
        raise ValueError(f"Embedding dimensions differ: {left_arr.size} vs {right_arr.size}")
    left_norm = float(np.linalg.norm(left_arr))
    right_norm = float(np.linalg.norm(right_arr))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_arr, right_arr) / (left_norm * right_norm))


def normalize_scores(scores: dict[int, float], *, mode: str) -> dict[int, float]:
    if mode == "none":
        return dict(scores)
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high == low:
        return {image_id: 0.5 for image_id in scores}
    return {image_id: (score - low) / (high - low) for image_id, score in scores.items()}
