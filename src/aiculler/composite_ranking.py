from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aiculler.preference_learning import PreferenceDiagnostics, PreferenceLearningScorer
from aiculler.profile_scoring import ProfilePromptAtom, profile_similarity
from aiculler.storage import SQLiteFeatureStore
from aiculler.technical_tags import TagPenaltyConfig, compute_tag_penalty, compute_technical_metrics
from aiculler.text_scoring import CLIPTextEncoder, cosine_similarity, normalize_scores


@dataclass(frozen=True)
class CompositeWeights:
    technical: float
    prompt: float
    profile: float
    preference: float
    penalty: float


@dataclass(frozen=True)
class CompositeRankRecord:
    rank: int
    image_id: int
    filename: str
    source_path: str
    technical_score: float
    prompt_score: float
    normalized_prompt_score: float
    profile_score: float
    normalized_profile_score: float
    learned_user_score: float
    normalized_learned_user_score: float
    pre_penalty_score: float
    tag_penalty: float
    triggered_tags: str
    final_score: float


@dataclass(frozen=True)
class CompositeRankResult:
    records: list[CompositeRankRecord]
    weights: CompositeWeights
    preference_diagnostics: PreferenceDiagnostics | None


class CompositeRanker:
    """Compose all scoring channels into one audited final ranking."""

    def __init__(
        self,
        store: SQLiteFeatureStore,
        *,
        weights: CompositeWeights,
    ):
        self.store = store
        self.weights = weights

    def rank(
        self,
        *,
        text_encoder: CLIPTextEncoder | None = None,
        prompt: str | None = None,
        profile_name: str | None = None,
        profile_atoms: list[ProfilePromptAtom] | None = None,
        feedback_csv: str | Path | None = None,
        tag_configs: list[TagPenaltyConfig] | None = None,
        avoid_tags: list[str] | None = None,
        use_existing_preference: bool = False,
        preference_projected_dim: int = 64,
        preference_alpha: float = 0.0001,
        record_feedback: bool = True,
        diagnostic_top_n: int = 10,
    ) -> CompositeRankResult:
        rows = self.store.list_images(require_embedding=True)
        if not rows:
            return CompositeRankResult([], self.weights, None)

        embeddings = {int(row["id"]): self.store.get_embedding(int(row["id"])) for row in rows}
        prompt_scores = self._prompt_scores(rows, embeddings, text_encoder, prompt)
        profile_scores = self._profile_scores(rows, embeddings, text_encoder, profile_name, profile_atoms or [])
        learned_scores, normalized_learned, diagnostics = self._preference_scores(
            rows,
            feedback_csv=feedback_csv,
            use_existing_preference=use_existing_preference,
            projected_dim=preference_projected_dim,
            alpha=preference_alpha,
            record_feedback=record_feedback,
            diagnostic_top_n=diagnostic_top_n,
        )
        normalized_prompt = normalize_scores(prompt_scores, mode="minmax") if prompt else {int(row["id"]): 0.0 for row in rows}
        normalized_profile = (
            normalize_scores(profile_scores, mode="minmax") if profile_name else {int(row["id"]): 0.0 for row in rows}
        )
        tag_penalties, tag_flags = self._tag_penalties(rows, tag_configs or [], avoid_tags or [])

        unranked: list[CompositeRankRecord] = []
        updates: dict[int, dict] = {}
        for row in rows:
            image_id = int(row["id"])
            technical_score = float(row["technical_score"] or 0.0)
            pre_penalty_score = (
                self.weights.technical * technical_score
                + self.weights.prompt * normalized_prompt[image_id]
                + self.weights.profile * normalized_profile[image_id]
                + self.weights.preference * normalized_learned[image_id]
            )
            final_score = pre_penalty_score - self.weights.penalty * tag_penalties[image_id]
            unranked.append(
                CompositeRankRecord(
                    rank=0,
                    image_id=image_id,
                    filename=Path(row["source_path"]).name,
                    source_path=row["source_path"],
                    technical_score=technical_score,
                    prompt_score=prompt_scores[image_id],
                    normalized_prompt_score=normalized_prompt[image_id],
                    profile_score=profile_scores[image_id],
                    normalized_profile_score=normalized_profile[image_id],
                    learned_user_score=learned_scores[image_id],
                    normalized_learned_user_score=normalized_learned[image_id],
                    pre_penalty_score=pre_penalty_score,
                    tag_penalty=tag_penalties[image_id],
                    triggered_tags=tag_flags[image_id],
                    final_score=final_score,
                )
            )
            updates[image_id] = {
                "prompt_score": prompt_scores[image_id],
                "prompt_text": prompt or "",
                "profile_score": profile_scores[image_id],
                "profile_name": profile_name or "",
                "learned_user_score": learned_scores[image_id],
                "tag_base_score": pre_penalty_score,
                "tag_penalty": tag_penalties[image_id],
                "tag_flags": tag_flags[image_id],
                "final_score": final_score,
            }

        ranked_records = [
            CompositeRankRecord(
                rank=index,
                image_id=record.image_id,
                filename=record.filename,
                source_path=record.source_path,
                technical_score=record.technical_score,
                prompt_score=record.prompt_score,
                normalized_prompt_score=record.normalized_prompt_score,
                profile_score=record.profile_score,
                normalized_profile_score=record.normalized_profile_score,
                learned_user_score=record.learned_user_score,
                normalized_learned_user_score=record.normalized_learned_user_score,
                pre_penalty_score=record.pre_penalty_score,
                tag_penalty=record.tag_penalty,
                triggered_tags=record.triggered_tags,
                final_score=record.final_score,
            )
            for index, record in enumerate(sorted(unranked, key=lambda item: item.final_score, reverse=True), start=1)
        ]
        self.store.update_composite_scores(updates)
        return CompositeRankResult(ranked_records, self.weights, diagnostics)

    def _prompt_scores(self, rows, embeddings, text_encoder, prompt: str | None) -> dict[int, float]:
        if not prompt:
            return {int(row["id"]): 0.0 for row in rows}
        if text_encoder is None:
            raise ValueError("text_encoder is required when prompt is supplied")
        text_embedding = text_encoder.encode(prompt)
        return {
            int(row["id"]): cosine_similarity(embeddings[int(row["id"])], text_embedding)
            for row in rows
        }

    def _profile_scores(self, rows, embeddings, text_encoder, profile_name: str | None, atoms: list[ProfilePromptAtom]) -> dict[int, float]:
        if not profile_name:
            return {int(row["id"]): 0.0 for row in rows}
        if text_encoder is None:
            raise ValueError("text_encoder is required when profile is supplied")
        selected = [atom for atom in atoms if atom.profile == profile_name]
        if not selected:
            raise ValueError(f"No prompt atoms found for profile {profile_name!r}")
        prompt_vectors = [(atom, text_encoder.encode(atom.prompt)) for atom in selected]
        return {
            int(row["id"]): profile_similarity(embeddings[int(row["id"])], prompt_vectors)
            for row in rows
        }

    def _preference_scores(
        self,
        rows,
        *,
        feedback_csv: str | Path | None,
        use_existing_preference: bool,
        projected_dim: int,
        alpha: float,
        record_feedback: bool,
        diagnostic_top_n: int,
    ) -> tuple[dict[int, float], dict[int, float], PreferenceDiagnostics | None]:
        if feedback_csv is not None:
            scorer = PreferenceLearningScorer(
                self.store,
                projected_dim=projected_dim,
                technical_weight=0.0,
                prompt_weight=0.0,
                preference_weight=1.0,
                alpha=alpha,
            )
            result = scorer.learn_from_csv(feedback_csv, record_feedback=record_feedback, top_n=diagnostic_top_n)
            learned_scores = {record.image_id: record.learned_user_score for record in result.ranking}
            normalized = {record.image_id: record.normalized_learned_user_score for record in result.ranking}
            return learned_scores, normalized, result.diagnostics

        if not use_existing_preference:
            return (
                {int(row["id"]): 0.0 for row in rows},
                {int(row["id"]): 0.0 for row in rows},
                None,
            )

        learned_scores = {
            int(row["id"]): float(row["learned_user_score"]) if row["learned_user_score"] is not None else 0.0
            for row in rows
        }
        if any(score != 0.0 for score in learned_scores.values()):
            return learned_scores, normalize_scores(learned_scores, mode="minmax"), None
        return learned_scores, {int(row["id"]): 0.0 for row in rows}, None

    def _tag_penalties(self, rows, configs: list[TagPenaltyConfig], avoid_tags: list[str]) -> tuple[dict[int, float], dict[int, str]]:
        if not avoid_tags:
            return ({int(row["id"]): 0.0 for row in rows}, {int(row["id"]): "" for row in rows})
        selected = [config for config in configs if config.tag in set(avoid_tags)]
        if not selected:
            raise ValueError(f"No matching tag configs for: {', '.join(avoid_tags)}")
        penalties: dict[int, float] = {}
        flags: dict[int, str] = {}
        for row in rows:
            image_id = int(row["id"])
            image_path = Path(row["preview_path"] or row["source_path"])
            metrics = compute_technical_metrics(image_path)
            penalty, triggered = compute_tag_penalty(metrics, selected)
            penalties[image_id] = penalty
            flags[image_id] = ",".join(triggered)
        return penalties, flags


def resolve_active_weights(
    *,
    technical_weight: float | None,
    prompt_weight: float | None,
    profile_weight: float | None,
    preference_weight: float | None,
    penalty_weight: float,
    prompt_active: bool,
    profile_active: bool,
    preference_active: bool,
) -> CompositeWeights:
    active = {
        "prompt": prompt_active,
        "profile": profile_active,
        "preference": preference_active,
    }
    explicit = {
        "prompt": prompt_weight,
        "profile": profile_weight,
        "preference": preference_weight,
    }
    if technical_weight is None:
        technical = 0.35 if any(active.values()) else 1.0
    else:
        technical = technical_weight
    remaining = max(0.0, 1.0 - technical)
    explicit_active_sum = sum(float(value) for name, value in explicit.items() if active[name] and value is not None)
    unspecified_active = [name for name, is_active in active.items() if is_active and explicit[name] is None]
    auto_weight = max(0.0, remaining - explicit_active_sum) / len(unspecified_active) if unspecified_active else 0.0
    prompt = float(prompt_weight) if prompt_active and prompt_weight is not None else (auto_weight if prompt_active else 0.0)
    profile = float(profile_weight) if profile_active and profile_weight is not None else (auto_weight if profile_active else 0.0)
    preference = (
        float(preference_weight)
        if preference_active and preference_weight is not None
        else (auto_weight if preference_active else 0.0)
    )
    return CompositeWeights(
        technical=float(technical),
        prompt=prompt,
        profile=profile,
        preference=preference,
        penalty=float(penalty_weight),
    )
