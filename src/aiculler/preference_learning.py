from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from aiculler.storage import SQLiteFeatureStore
from aiculler.text_scoring import normalize_scores


@dataclass(frozen=True)
class FeedbackExample:
    image_id: int
    label: int
    source_path: str
    filename: str


@dataclass(frozen=True)
class LearnedScoreRecord:
    image_id: int
    filename: str
    source_path: str
    technical_score: float
    prompt_score: float
    learned_user_score: float
    normalized_learned_user_score: float
    final_score: float
    feedback_label: str


@dataclass(frozen=True)
class PreferenceDiagnosticRecord:
    image_id: int
    filename: str
    source_path: str
    label: str
    train_score: float
    predicted_label: str
    correct: bool
    leave_one_out_score: float | None
    leave_one_out_label: str | None
    leave_one_out_correct: bool | None


@dataclass(frozen=True)
class PreferenceDiagnostics:
    feedback_count: int
    keep_count: int
    reject_count: int
    train_accuracy: float
    leave_one_out_accuracy: float | None
    top_n_keep_count: int
    top_n_reject_count: int
    top_n: int
    records: list[PreferenceDiagnosticRecord]


@dataclass(frozen=True)
class PreferenceLearningResult:
    ranking: list[LearnedScoreRecord]
    diagnostics: PreferenceDiagnostics


class PreferenceLearningScorer:
    """Train a lightweight user preference model from keep/reject CSV labels."""

    def __init__(
        self,
        store: SQLiteFeatureStore,
        *,
        projected_dim: int = 64,
        technical_weight: float = 0.25,
        prompt_weight: float = 0.25,
        preference_weight: float = 0.50,
        alpha: float = 0.0001,
        random_state: int = 13,
    ):
        self.store = store
        self.projected_dim = int(projected_dim)
        self.technical_weight = float(technical_weight)
        self.prompt_weight = float(prompt_weight)
        self.preference_weight = float(preference_weight)
        self.alpha = float(alpha)
        self.random_state = int(random_state)

    def learn_from_csv(
        self,
        feedback_csv_path: str | Path,
        *,
        record_feedback: bool = True,
        top_n: int = 10,
    ) -> PreferenceLearningResult:
        examples = resolve_feedback_examples(self.store, feedback_csv_path)
        labels = np.asarray([example.label for example in examples], dtype=np.int64)
        if len(examples) < 2 or len(set(labels.tolist())) < 2:
            raise ValueError("feedback CSV must include at least one keep and one reject example")

        rows = self.store.list_images(require_embedding=True)
        if not rows:
            return []
        all_ids = [int(row["id"]) for row in rows]
        all_embeddings = np.vstack([self.store.get_embedding(image_id) for image_id in all_ids])
        feedback_index = {example.image_id: idx for idx, example in enumerate(examples)}
        train_indices = [all_ids.index(example.image_id) for example in examples]

        n_components = min(self.projected_dim, all_embeddings.shape[0], all_embeddings.shape[1])
        projector = IncrementalPCA(n_components=n_components)
        projected_all = projector.fit_transform(all_embeddings)
        scaler = StandardScaler()
        scaled_all = scaler.fit_transform(projected_all)
        scaled_train = scaled_all[train_indices]

        classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=self.alpha,
            max_iter=1000,
            tol=1e-3,
            random_state=self.random_state,
        )
        classifier.fit(scaled_train, labels)
        train_scores = classifier.decision_function(scaled_train)
        raw_learned_scores = classifier.decision_function(scaled_all)
        learned_scores = {image_id: float(score) for image_id, score in zip(all_ids, raw_learned_scores)}
        normalized_learned = normalize_scores(learned_scores, mode="minmax")
        prompt_scores = {
            int(row["id"]): float(row["prompt_score"]) if row["prompt_score"] is not None else 0.0
            for row in rows
        }
        normalized_prompt = normalize_scores(prompt_scores, mode="minmax")

        updates: dict[int, tuple[float, float]] = {}
        records: list[LearnedScoreRecord] = []
        for row in rows:
            image_id = int(row["id"])
            technical_score = float(row["technical_score"] or 0.0)
            prompt_score = prompt_scores[image_id]
            final_score = (
                self.technical_weight * technical_score
                + self.prompt_weight * normalized_prompt[image_id]
                + self.preference_weight * normalized_learned[image_id]
            )
            updates[image_id] = (learned_scores[image_id], final_score)
            feedback_label = ""
            if image_id in feedback_index:
                feedback_label = "keep" if examples[feedback_index[image_id]].label == 1 else "reject"
            records.append(
                LearnedScoreRecord(
                    image_id=image_id,
                    filename=Path(row["source_path"]).name,
                    source_path=row["source_path"],
                    technical_score=technical_score,
                    prompt_score=prompt_score,
                    learned_user_score=learned_scores[image_id],
                    normalized_learned_user_score=normalized_learned[image_id],
                    final_score=final_score,
                    feedback_label=feedback_label,
                )
            )

        self.store.update_learned_scores(updates)
        if record_feedback:
            for example in examples:
                self.store.add_feedback(example.image_id, example.label, note=f"learn-feedback:{Path(feedback_csv_path).name}")
        ranking = sorted(records, key=lambda record: record.final_score, reverse=True)
        diagnostics = build_diagnostics(
            examples=examples,
            ranking=ranking,
            train_scores=train_scores,
            projected_all=scaled_all,
            all_ids=all_ids,
            labels=labels,
            alpha=self.alpha,
            random_state=self.random_state,
            top_n=top_n,
        )
        return PreferenceLearningResult(ranking=ranking, diagnostics=diagnostics)


def build_diagnostics(
    *,
    examples: list[FeedbackExample],
    ranking: list[LearnedScoreRecord],
    train_scores: np.ndarray,
    projected_all: np.ndarray,
    all_ids: list[int],
    labels: np.ndarray,
    alpha: float,
    random_state: int,
    top_n: int,
) -> PreferenceDiagnostics:
    keep_count = int(np.sum(labels == 1))
    reject_count = int(np.sum(labels == 0))
    example_by_id = {example.image_id: example for example in examples}
    train_score_by_id = {
        example.image_id: float(score)
        for example, score in zip(examples, np.asarray(train_scores).reshape(-1))
    }
    loo_scores = leave_one_out_scores(
        examples=examples,
        projected_all=projected_all,
        all_ids=all_ids,
        labels=labels,
        alpha=alpha,
        random_state=random_state,
    )

    records: list[PreferenceDiagnosticRecord] = []
    train_correct = 0
    loo_correct_values: list[bool] = []
    for example in examples:
        train_score = train_score_by_id[example.image_id]
        predicted_label = score_to_label(train_score)
        actual_label = "keep" if example.label == 1 else "reject"
        correct = predicted_label == actual_label
        train_correct += int(correct)
        loo_score = loo_scores.get(example.image_id)
        loo_label = score_to_label(loo_score) if loo_score is not None else None
        loo_correct = (loo_label == actual_label) if loo_label is not None else None
        if loo_correct is not None:
            loo_correct_values.append(loo_correct)
        records.append(
            PreferenceDiagnosticRecord(
                image_id=example.image_id,
                filename=example.filename,
                source_path=example.source_path,
                label=actual_label,
                train_score=train_score,
                predicted_label=predicted_label,
                correct=correct,
                leave_one_out_score=loo_score,
                leave_one_out_label=loo_label,
                leave_one_out_correct=loo_correct,
            )
        )

    top_feedback = [record for record in ranking[: max(0, top_n)] if record.image_id in example_by_id]
    top_n_keep_count = sum(1 for record in top_feedback if example_by_id[record.image_id].label == 1)
    top_n_reject_count = sum(1 for record in top_feedback if example_by_id[record.image_id].label == 0)
    return PreferenceDiagnostics(
        feedback_count=len(examples),
        keep_count=keep_count,
        reject_count=reject_count,
        train_accuracy=train_correct / len(examples) if examples else 0.0,
        leave_one_out_accuracy=(
            sum(1 for value in loo_correct_values if value) / len(loo_correct_values)
            if loo_correct_values
            else None
        ),
        top_n_keep_count=top_n_keep_count,
        top_n_reject_count=top_n_reject_count,
        top_n=max(0, top_n),
        records=records,
    )


def leave_one_out_scores(
    *,
    examples: list[FeedbackExample],
    projected_all: np.ndarray,
    all_ids: list[int],
    labels: np.ndarray,
    alpha: float,
    random_state: int,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    if len(examples) < 3:
        return scores
    id_to_all_index = {image_id: idx for idx, image_id in enumerate(all_ids)}
    for holdout_index, example in enumerate(examples):
        train_mask = np.ones(len(examples), dtype=bool)
        train_mask[holdout_index] = False
        train_labels = labels[train_mask]
        if len(set(train_labels.tolist())) < 2:
            continue
        train_example_indices = [
            id_to_all_index[examples[idx].image_id]
            for idx in range(len(examples))
            if train_mask[idx]
        ]
        classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=1000,
            tol=1e-3,
            random_state=random_state,
        )
        classifier.fit(projected_all[train_example_indices], train_labels)
        holdout_vector = projected_all[[id_to_all_index[example.image_id]]]
        scores[example.image_id] = float(classifier.decision_function(holdout_vector).reshape(-1)[0])
    return scores


def score_to_label(score: float) -> str:
    return "keep" if float(score) >= 0.0 else "reject"


def resolve_feedback_examples(store: SQLiteFeatureStore, feedback_csv_path: str | Path) -> list[FeedbackExample]:
    rows = store.list_images(require_embedding=True)
    by_id = {int(row["id"]): row for row in rows}
    by_source = {normalize_key(row["source_path"]): row for row in rows}
    by_filename: dict[str, list] = {}
    for row in rows:
        by_filename.setdefault(normalize_key(Path(row["source_path"]).name), []).append(row)

    examples: list[FeedbackExample] = []
    with Path(feedback_csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("feedback CSV must include headers")
        for line_number, record in enumerate(reader, start=2):
            label = parse_feedback_label(record.get("label", ""))
            row = resolve_feedback_row(record, by_id=by_id, by_source=by_source, by_filename=by_filename)
            if row is None:
                raise ValueError(f"feedback row {line_number} did not match any image")
            examples.append(
                FeedbackExample(
                    image_id=int(row["id"]),
                    label=label,
                    source_path=row["source_path"],
                    filename=Path(row["source_path"]).name,
                )
            )
    return examples


def resolve_feedback_row(record: dict, *, by_id: dict, by_source: dict, by_filename: dict):
    image_id = (record.get("id") or record.get("image_id") or "").strip()
    if image_id:
        try:
            return by_id.get(int(image_id))
        except ValueError as exc:
            raise ValueError(f"invalid image id {image_id!r}") from exc

    source_path = (record.get("source_path") or record.get("path") or "").strip()
    if source_path:
        return by_source.get(normalize_key(source_path))

    filename = (record.get("filename") or "").strip()
    if filename:
        matches = by_filename.get(normalize_key(filename), [])
        if len(matches) > 1:
            raise ValueError(f"filename {filename!r} matched multiple images; use id or source_path")
        return matches[0] if matches else None
    return None


def parse_feedback_label(label: str) -> int:
    normalized = label.strip().lower()
    if normalized in {"keep", "k", "1", "true", "yes", "y"}:
        return 1
    if normalized in {"reject", "r", "0", "false", "no", "n", "remove"}:
        return 0
    raise ValueError(f"unsupported feedback label {label!r}")


def normalize_key(value: str) -> str:
    return str(Path(value)).replace("/", "\\").lower()
