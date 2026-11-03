"""Scorers: one module per signal, one interface.

``Scorer.score(batch)`` takes candidates and returns one dict per candidate. Numbers become scores, which are
only meaningful within a comparison set (``decide`` normalises them per set); anything else (face lists,
embeddings) becomes a *detail*, kept out of ``decisions.json``'s scores. Both are cached per capture identity
and scorer revision.

Cheap, no models: ``exposure``, ``sharpness``, ``clipping``. Models (through the app's manifest): ``faces``
(MediaPipe + AdaFace), ``maniqa``, ``appeal`` (Q-ReAlign), ``tinyclip``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from numbers import Real
from typing import Callable, Protocol

from aiculler.ingest import Capture, PreviewCache, Shoot


@dataclass
class Candidate:
    capture: Capture
    _loader: Callable[[Capture], object] | None = None
    _model_loader: Callable[[Capture, int], object] | None = None
    _image: object | None = None
    _loaded: bool = False
    _model_images: dict[int, object] = field(default_factory=dict)

    @property
    def image(self):
        """Decoded preview (PIL RGB, ~1024 px) or None. Loaded once, on first use."""
        if not self._loaded:
            self._image = self._loader(self.capture) if self._loader else None
            self._loaded = True
        return self._image

    def model_image(self, long_edge: int):
        """The preview at a model's own input size (LANCZOS, as in the app), or None."""
        if long_edge not in self._model_images:
            self._model_images[long_edge] = self._model_loader(self.capture, long_edge) if self._model_loader else None
        return self._model_images[long_edge]


class Scorer(Protocol):
    name: str
    revision: str

    def score(self, batch: list[Candidate]) -> list[dict]: ...


def default_scorers() -> list[Scorer]:
    """The cheap scorers every candidate gets."""
    from aiculler.scorers.clipping import ClippingScorer
    from aiculler.scorers.exposure import ExposureScorer
    from aiculler.scorers.sharpness import SharpnessScorer

    return [ExposureScorer(), SharpnessScorer(), ClippingScorer()]


@dataclass
class ScoreStore:
    """Scores (numbers, per capture) and details (everything else, per capture) gathered across stages."""

    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    details: dict[str, dict[str, object]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)

    def add(self, capture: str, values: dict) -> None:
        for k, v in values.items():
            if isinstance(v, Real) and not isinstance(v, bool):
                self.scores.setdefault(capture, {})[k] = float(v)
            else:
                self.details.setdefault(capture, {})[k] = v


def run_scorers(
    store: ScoreStore,
    shoot: Shoot,
    capture_ids: list[str],
    scorers: list[Scorer],
    previews: PreviewCache,
    cache,
    progress: Callable[[str, int, int], None] | None = None,
    batch_size: int = 32,
    workers: int | None = None,
) -> None:
    """Run scorers on captures, reusing cached results (capture identity + scorer revision)."""
    progress = progress or (lambda stage, done, total: None)
    capture_ids = list(dict.fromkeys(capture_ids))
    for scorer in scorers:
        cached = cache.get_many(capture_ids, scorer.name, scorer.revision)
        todo = [cid for cid in capture_ids if cid not in cached]
        store.stats[f"{scorer.name}_frames"] = len(capture_ids)
        store.stats[f"{scorer.name}_computed"] = len(todo)
        store.stats[f"{scorer.name}_cached"] = len(cached)
        for cid, values in cached.items():
            store.add(cid, values)
        done, total = len(cached), len(capture_ids)
        progress(scorer.name, done, total)
        for start in range(0, len(todo), batch_size):
            ids = todo[start : start + batch_size]
            batch = [Candidate(shoot.captures[cid], previews.load, previews.load_for_model) for cid in ids]
            if getattr(scorer, "needs_pixels", False):
                with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 4)) as pool:
                    list(pool.map(lambda c: c.image, batch))  # decode previews in parallel (PIL releases the GIL)
            results = scorer.score(batch)
            cache.put_many([(cid, scorer.name, scorer.revision, res) for cid, res in zip(ids, results)])
            for cid, res in zip(ids, results):
                store.add(cid, res)
            done += len(ids)
            progress(scorer.name, done, total)


def score_captures(
    shoot: Shoot,
    capture_ids: list[str],
    previews: PreviewCache,
    cache,
    progress: Callable[[str, int, int], None] | None = None,
    scorers: list[Scorer] | None = None,
    batch_size: int = 32,
    workers: int | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Cheap scorers only: {capture: {metric: value}} and stats (kept for callers of the milestone-1 API)."""
    store = ScoreStore(scores={cid: {} for cid in capture_ids})
    run_scorers(store, shoot, capture_ids, scorers if scorers is not None else default_scorers(), previews, cache, progress, batch_size, workers)
    store.stats["scored_captures"] = len(capture_ids)
    return store.scores, store.stats
