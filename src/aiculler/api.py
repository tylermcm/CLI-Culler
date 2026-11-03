"""The app's entry point: ``cull(folder, settings) -> CullResult``.

In: a folder (or list of folders) and ``CullSettings``. Out: a decision for every file, with reasons
(``decisions.json``, schema ``aiculler-decisions/1``). No GUI; progress goes to an optional callback.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable

ProgressCallback = Callable[[str, int, int], None]  # (stage, done, total)

BRACKET_MODES = ("keep_set", "pick_one", "ignore")
SEQUENCE_MODES = ("keep_all", "cull")
FEATURELESS_MODES = ("review", "reject", "off")
NORMALISE_MODES = ("relative", "rank", "zscore")


@dataclass
class GateSettings:
    """Thresholds for clear failures only. Everything else is a score."""

    black_mean: float = 0.02  # mean preview luma below this: black frame / misfire (always a reject)
    # Highlights, by connected clipped region (never applied to bracket frames):
    clip_region_ignore: float = 0.05  # largest clipped region below this fraction of the frame: no penalty
    # Tuned on China 8-26: the closest final pick (_DSC1055) has a 24% region and 59% of its top third
    # clipped, so the reject thresholds sit well past it.
    clip_region_reject: float = 0.35  # ...at or above this: reject; in between, a score penalty
    clip_penalty: float = 0.15  # penalty (combined-score points) at clip_region_reject, linear from 0 at clip_region_ignore
    sky_blown: float = 0.75  # clipped fraction of the top third at or above this: reject ("sky blown out")
    washed_mean: float = 0.80  # mean luma at or above this...
    washed_spread: float = 0.15  # ...and tonal spread (p95 - p5) at or below this: reject ("washed out")
    shadows_crushed: float = 0.60  # fraction of crushed pixels above this: unusable
    # A featureless preview is a black misfire, a blank frame, or a moon in an empty sky; only the
    # first is a clear failure, so by default it goes to review instead of being rejected.
    featureless: str = "review"


@dataclass
class CullSettings:
    brackets: str = "keep_set"  # keep_set | pick_one | ignore
    bursts_keep: int = 2  # best N frames per burst; 0 = bursts left out of culling (all kept)
    scene_keep: int = 1  # best N units per near-duplicate scene; 0 = scenes not reduced
    focus_stacks: str = "keep_all"  # keep_all | cull
    timelapses: str = "keep_all"  # keep_all | cull
    review_below: float = 0.7  # group confidence below this, or warnings -> review
    require_eyes_open: bool = True  # milestone 2 (faces)
    keep_people_coverage: bool = True  # milestone 5 (AdaFace)
    gates: GateSettings = field(default_factory=GateSettings)
    # Combined-score weights over set-normalised scores. Milestone 1 has only these two signals.
    weights: dict[str, float] = field(default_factory=lambda: {"sharpness": 0.6, "exposure": 0.4})
    # How scores are made comparable within a set: "relative" (distance from the set's best, on an
    # absolute scale), "rank" (percentile) or "zscore".
    normalise: str = "relative"
    backend: str = "auto"  # bracketeering metadata backend: auto | exiftool | native
    exclude: list[str] = field(default_factory=list)  # folder names bracketeering should skip

    def __post_init__(self) -> None:
        if isinstance(self.gates, dict):
            gates = dict(self.gates)
            if "highlights_clipped" in gates:
                raise ValueError(
                    "gates.highlights_clipped was replaced by region rules: clip_region_ignore, clip_region_reject, "
                    "clip_penalty, sky_blown, washed_mean, washed_spread"
                )
            unknown = set(gates) - {f.name for f in fields(GateSettings)}
            if unknown:
                raise ValueError(f"unknown gate settings: {', '.join(sorted(unknown))}")
            self.gates = GateSettings(**gates)
        self.validate()

    def validate(self) -> None:
        def check(name: str, value, allowed) -> None:
            if value not in allowed:
                raise ValueError(f"{name} must be one of {', '.join(allowed)}; got {value!r}")

        check("brackets", self.brackets, BRACKET_MODES)
        check("focus_stacks", self.focus_stacks, SEQUENCE_MODES)
        check("timelapses", self.timelapses, SEQUENCE_MODES)
        check("gates.featureless", self.gates.featureless, FEATURELESS_MODES)
        check("normalise", self.normalise, NORMALISE_MODES)
        g = self.gates
        if not 0.0 <= g.clip_region_ignore < g.clip_region_reject <= 1.0:
            raise ValueError("gates: need 0 <= clip_region_ignore < clip_region_reject <= 1")
        if g.clip_penalty < 0:
            raise ValueError("gates.clip_penalty must be >= 0")
        for name in ("bursts_keep", "scene_keep"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
        if not 0.0 <= float(self.review_below) <= 1.0:
            raise ValueError(f"review_below must be within 0..1; got {self.review_below!r}")
        if not self.weights or any(float(w) < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative and not empty")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CullSettings":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
        return cls(**data)

    @classmethod
    def from_json(cls, path: str | Path) -> "CullSettings":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class FileDecision:
    decision: str  # keep | reject | review
    group: str
    kind: str
    role: str
    scene: int | None
    capture: str
    primary: bool
    comparison_set: str | None = None  # burst id or "scene:N"; None when the file was not ranked
    rank_in_set: int | None = None
    set_size: int | None = None
    score: float | None = None  # combined score, comparable only within comparison_set
    scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    gates: list[str] = field(default_factory=list)  # gates this capture failed, even if rescued or overridden

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CullResult:
    files: dict[str, FileDecision]
    settings: CullSettings
    bracketeering_version: str
    engine_version: str
    models: dict[str, str]
    timings: dict[str, float]
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, int]:
        out = {"files": len(self.files), "keep": 0, "reject": 0, "review": 0}
        for d in self.files.values():
            out[d.decision] += 1
        return out

    def to_dict(self) -> dict:
        from aiculler.outputs import decisions_document

        return decisions_document(self)


def cull(
    folder: str | Path | list[str | Path],
    settings: CullSettings | None = None,
    models=None,
    cache_db: str | Path | None = None,
    cache_dir: str | Path | None = None,
    on_progress: ProgressCallback | None = None,
    *,
    workers: int | None = None,
) -> CullResult:
    """Cull a shoot: keep / reject / review for every file, with reasons.

    ``models`` is reserved for the app's ``ModelRegistry`` (milestone 2 on); milestone 1 uses only
    bracketeering's exposure stats and whole-frame sharpness. ``cache_db`` (SQLite) caches scores by
    capture identity and scorer revision, so re-running with new settings recomputes nothing.
    ``cache_dir`` holds decoded previews (default: next to ``cache_db``, else ``.aiculler_cache``).
    """
    from aiculler import __version__
    from aiculler.decide import decide
    from aiculler.gates import apply_gates
    from aiculler.ingest import PreviewCache, ingest
    from aiculler.routing import route
    from aiculler.scorers import score_captures
    from aiculler.storage import ScoreCache

    settings = settings or CullSettings()
    progress = on_progress or (lambda stage, done, total: None)
    timings: dict[str, float] = {}

    t = time.perf_counter()
    progress("ingest", 0, 1)
    shoot = ingest(folder, backend=settings.backend, exclude=settings.exclude, workers=workers)
    progress("ingest", 1, 1)
    timings["ingest"] = time.perf_counter() - t

    t = time.perf_counter()
    routing = route(shoot, settings)
    timings["route"] = time.perf_counter() - t

    if cache_dir is None:
        cache_dir = Path(cache_db).with_suffix("").as_posix() + "_previews" if cache_db else ".aiculler_cache"
    previews = PreviewCache(cache_dir)
    cache = ScoreCache(cache_db) if cache_db else ScoreCache(":memory:")
    try:
        t = time.perf_counter()
        scores, score_stats = score_captures(shoot, routing.scored_captures(), previews, cache, progress, workers=workers)
        timings["score"] = time.perf_counter() - t
    finally:
        cache.close()

    t = time.perf_counter()
    gates = apply_gates(shoot, routing, settings, scores)
    timings["gates"] = time.perf_counter() - t

    t = time.perf_counter()
    files = decide(shoot, routing, gates, scores, settings)
    timings["decide"] = time.perf_counter() - t

    stats = {
        "captures": len(shoot.captures),
        "groups": len(shoot.groups),
        "comparison_sets": len(routing.bursts) + sum(1 for s in routing.scenes if len(s.members) > 1),
        **score_stats,
    }
    return CullResult(
        files=files,
        settings=settings,
        bracketeering_version=shoot.bracketeering_version,
        engine_version=__version__,
        models={},
        timings={k: round(v, 3) for k, v in timings.items()},
        stats=stats,
    )
