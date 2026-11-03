"""Scorers: one module per signal, one interface.

``Scorer.score(batch)`` takes candidates and returns one dict of raw values per candidate. Raw values
are only meaningful within a comparison set; ``decide`` normalises them per set.
Milestone 1 has two: bracketeering's exposure stats and whole-frame sharpness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from aiculler.ingest import Capture, PreviewCache, Shoot


@dataclass
class Candidate:
    capture: Capture
    _loader: Callable[[Capture], object] | None = None
    _image: object | None = None
    _loaded: bool = False

    @property
    def image(self):
        """Decoded preview (PIL RGB, ~1024 px) or None. Loaded once, on first use."""
        if not self._loaded:
            self._image = self._loader(self.capture) if self._loader else None
            self._loaded = True
        return self._image


class Scorer(Protocol):
    name: str
    revision: str

    def score(self, batch: list[Candidate]) -> list[dict]: ...


def default_scorers() -> list[Scorer]:
    from aiculler.scorers.exposure import ExposureScorer
    from aiculler.scorers.sharpness import SharpnessScorer

    return [ExposureScorer(), SharpnessScorer()]


def score_captures(
    shoot: Shoot,
    capture_ids: list[str],
    previews: PreviewCache,
    cache,
    progress: Callable[[str, int, int], None] | None = None,
    scorers: list[Scorer] | None = None,
    batch_size: int = 32,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Scores per capture, merged across scorers: {capture: {metric: value}}. Cached by capture + revision."""
    progress = progress or (lambda stage, done, total: None)
    scorers = scorers if scorers is not None else default_scorers()
    out: dict[str, dict[str, float]] = {cid: {} for cid in capture_ids}
    stats: dict[str, int] = {"scored_captures": len(capture_ids)}
    for scorer in scorers:
        cached = cache.get_many(capture_ids, scorer.name, scorer.revision)
        todo = [cid for cid in capture_ids if cid not in cached]
        stats[f"{scorer.name}_computed"] = len(todo)
        stats[f"{scorer.name}_cached"] = len(cached)
        for cid, values in cached.items():
            out[cid].update(values)
        done = len(cached)
        total = len(capture_ids)
        progress(scorer.name, done, total)
        for start in range(0, len(todo), batch_size):
            ids = todo[start : start + batch_size]
            batch = [Candidate(shoot.captures[cid], previews.load) for cid in ids]
            results = scorer.score(batch)
            cache.put_many([(cid, scorer.name, scorer.revision, res) for cid, res in zip(ids, results)])
            for cid, res in zip(ids, results):
                out[cid].update(res)
            done += len(ids)
            progress(scorer.name, done, total)
    return out, stats
