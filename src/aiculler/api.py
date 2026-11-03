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
REVIEW_WARNING_MODES = ("doubt", "all", "none")
# Optional models. The cull is complete without any of them: gates reject the junk, and bursts, brackets and
# near-duplicate scenes are culled on exposure, sharpness and clipping. Each one adds something:
#   faces   MediaPipe: blinks and face-crop sharpness count when ranking (contested frames only)
#   maniqa  MANIQA: technical quality when ranking, and the severe-blur gate (contested frames only)
#   appeal  Q-ReAlign: aesthetic appeal when ranking (each contested set's shortlist only)
#   search  TinyCLIP: an image embedding for every readable capture, written to the index for semantic search
#   people  MediaPipe + AdaFace: every readable capture's faces with identity vectors, clustered into people
#           and written to the index; also keeps one good frame of each person (keep_people_coverage). Implies faces.
MODEL_FEATURES = ("faces", "maniqa", "appeal", "search", "people")
FEATURE_MODELS: dict[str, tuple[str, ...]] = {
    "faces": ("face_landmarker",),
    "maniqa": ("maniqa",),
    "appeal": ("q_realign",),
    "search": ("tinyclip",),
    "people": ("face_landmarker", "adaface"),
}


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
    # Severe blur (needs MANIQA): MANIQA in the shoot's bottom maniqa_bottom fraction AND face (else whole-frame)
    # sharpness below blur_ratio x the median of its comparison set.
    maniqa_bottom: float = 0.02
    blur_ratio: float = 0.35
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
    review_below: float = 0.7  # group confidence below this -> review
    # Which bracketeering warnings send a group to review: "doubt" (only those that doubt the grouping itself, e.g.
    # a weak bracket candidate split into singles; "partial bracket", "frames don't register" and the like are
    # noted in the reasons but decided as usual), "all", or "none".
    review_warnings: str = "doubt"
    require_eyes_open: bool = True  # blinks count against a frame; in group shots, any prominent face
    blink_threshold: float = 0.6  # eyes-closed score at or above this is a blink (the app's BLINK_THRESHOLD)
    blink_penalty: float = 0.3  # combined-score points a blink costs
    keep_people_coverage: bool = True  # each recognised person keeps at least one acceptable frame
    face_cluster_threshold: float = 0.35  # AdaFace cosine for one person (the app's FACE_CLUSTER_THRESHOLD)
    gates: GateSettings = field(default_factory=GateSettings)
    # Optional models to use (MODEL_FEATURES, or "all"); empty: none, the cull runs on the cheap signals alone.
    models: list[str] = field(default_factory=list)
    # Combined-score weights over set-normalised scores (spec §8: appeal 0.4, technical 0.4 = MANIQA 0.25 +
    # face 0.15, preference low until the harness shows it helps). A signal nobody in a set has is left out of
    # that set's combination, so without models the cheap signals (sharpness, exposure) decide, as in milestone 1.
    weights: dict[str, float] = field(default_factory=lambda: {
        "appeal": 0.4, "maniqa": 0.25, "face_sharpness": 0.15, "preference": 0.1, "sharpness": 0.12, "exposure": 0.08,
    })
    appeal_shortlist: int = 3  # Q-ReAlign runs on at most this many per comparison set (at least keep + 1)
    refine_scenes: bool = False  # with "search": also split/merge bracketeering's scenes by TinyCLIP similarity
    split_cos: float = 0.80  # consecutive groups of one scene less similar than this: split
    merge_cos: float = 0.92  # adjacent scenes within merge_window this similar: merge (unless registration contradicts)
    merge_window: float = 120.0  # seconds
    preference_model: str | None = None  # path to a model from `ai-culler learn` (milestone 4)
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
        if isinstance(self.models, str):
            self.models = [m.strip() for m in self.models.split(",") if m.strip()]
        if "all" in self.models:
            self.models = list(MODEL_FEATURES)
        self.models = list(dict.fromkeys(self.models))
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
        check("review_warnings", self.review_warnings, REVIEW_WARNING_MODES)
        for m in self.models:
            check("models", m, MODEL_FEATURES + ("all",))
        g = self.gates
        if not 0.0 <= g.clip_region_ignore < g.clip_region_reject <= 1.0:
            raise ValueError("gates: need 0 <= clip_region_ignore < clip_region_reject <= 1")
        if g.clip_penalty < 0:
            raise ValueError("gates.clip_penalty must be >= 0")
        if self.appeal_shortlist < 1:
            raise ValueError("appeal_shortlist must be >= 1")
        if not -1.0 <= self.split_cos <= self.merge_cos <= 1.0:
            raise ValueError("need -1 <= split_cos <= merge_cos <= 1")
        for name in ("bursts_keep", "scene_keep"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
        if not 0.0 <= float(self.review_below) <= 1.0:
            raise ValueError(f"review_below must be within 0..1; got {self.review_below!r}")
        if not self.weights or any(float(w) < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative and not empty")

    def features(self) -> set[str]:
        """The model features in effect ("people" implies "faces")."""
        out = set(self.models)
        if "people" in out:
            out.add("faces")
        return out

    def model_keys(self) -> set[str]:
        """Registry keys these settings may load."""
        keys = {k for f in self.features() for k in FEATURE_MODELS[f]}
        if self.preference_model:
            keys.add("tinyclip")
        return keys

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
    people: list[int] = field(default_factory=list)  # person numbers in this capture (the "people" feature)

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
    warnings: list[str] = field(default_factory=list)
    # The search and people index (``outputs.write_index``): TinyCLIP embeddings per capture and clustered faces.
    # Empty unless the "search" / "people" model features ran. Not part of decisions.json (it is large).
    index: dict = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, int]:
        out = {"files": len(self.files), "keep": 0, "reject": 0, "review": 0}
        for d in self.files.values():
            out[d.decision] += 1
        return out

    def to_dict(self) -> dict:
        from aiculler.outputs import decisions_document

        return decisions_document(self)


def build_index(shoot, used: dict[str, str], embeddings: dict[str, list[float]], people, settings: CullSettings) -> dict:
    """The index for semantic and people search: what the "search" and "people" features found, per capture."""
    index: dict = {}
    if embeddings:
        index["search"] = {"model": used.get("tinyclip", ""), "embeddings": dict(embeddings)}
    if people:
        index["people"] = {
            "model": used.get("faces", ""),
            "threshold": settings.face_cluster_threshold,
            "faces": [
                {"capture": f.capture, "index": f.index, "person": f.person, "box": list(f.box), "area": f.area,
                 "eyes_closed": f.eyes_closed, "embedding": f.vector}
                for person in people for f in person
            ],
        }
    if index:
        index["captures"] = {cid: list(c.files) for cid, c in shoot.captures.items()}
    return index


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

    ``models``: a ``ModelRegistry`` (``ModelRegistry.from_app_manifest()``); None runs without models (exposure,
    sharpness, clipping only). A model that is unavailable or fails to load is skipped with a warning
    (``CullResult.warnings``); ``CullResult.models`` lists the ones actually used.
    ``cache_db`` (SQLite) caches every score by capture identity and scorer revision, so re-running with new
    settings re-runs no model. ``cache_dir`` holds decoded previews (default: next to ``cache_db``).

    Stages, cheap to expensive: ingest -> TinyCLIP embed + scene refinement -> route -> cheap scores -> gates ->
    faces -> MANIQA (contested frames only) -> blur gate -> Q-ReAlign (each contested set's shortlist only) ->
    preference -> decide -> people coverage.
    """
    from aiculler import __version__
    from aiculler.decide import decide
    from aiculler.gates import apply_gates
    from aiculler.ingest import PreviewCache, ingest
    from aiculler.models.registry import ModelRegistry
    from aiculler.routing import route
    from aiculler.scorers import ScoreStore, default_scorers, run_scorers
    from aiculler.stages import appeal_shortlist, blur_failures, contested
    from aiculler.storage import ScoreCache

    settings = settings or CullSettings()
    features = settings.features()
    wanted = settings.model_keys()
    source = models if models is not None else ModelRegistry.none()
    registry = source.restrict(wanted)
    progress = on_progress or (lambda stage, done, total: None)
    timings: dict[str, float] = {}
    warnings: list[str] = []
    used: dict[str, str] = {}
    store = ScoreStore()

    def stage(name: str):
        class _Timer:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timings[name] = timings.get(name, 0.0) + time.perf_counter() - self.t
                return False

        return _Timer()

    def unavailable(name: str, key: str) -> bool:
        """Warn (once per feature) when a requested model cannot be used."""
        if registry.available(key):
            return False
        problem = registry.entries[key].problem if key in registry.entries else "unknown model"
        if models is None:
            problem = "no models given"
        message = f"{name}: not used ({problem})"
        if message not in warnings:
            warnings.append(message)
        return True

    def run_model(name: str, key: str, make, captures) -> bool:
        """Run a model scorer on ``captures``; on any failure to load, warn and carry on without it."""
        if not captures or unavailable(name, key):
            return False
        try:
            scorer = make(registry)
            with stage(name):
                run_scorers(store, shoot, sorted(captures), [scorer], previews, cache, progress, batch_size=8, workers=workers)
        except Exception as exc:  # noqa: BLE001 - any failure to load or run a model skips that model, never the cull
            warnings.append(f"{name}: not used ({type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''})")
            return False
        used[name] = scorer.revision
        return True

    with stage("ingest"):
        progress("ingest", 0, 1)
        shoot = ingest(folder, backend=settings.backend, exclude=settings.exclude, workers=workers)
        progress("ingest", 1, 1)

    if cache_dir is None:
        cache_dir = Path(cache_db).with_suffix("").as_posix() + "_previews" if cache_db else ".aiculler_cache"
    previews = PreviewCache(cache_dir)
    cache = ScoreCache(cache_db) if cache_db else ScoreCache(":memory:")
    preference = None
    shortlist: set[str] = set()
    people_on = False
    try:
        from aiculler.scorers.models import AppealScorer, FacesScorer, ManiqaScorer, TinyClipScorer

        if settings.preference_model:
            from aiculler.preference import PreferenceModel

            preference = PreferenceModel.load(settings.preference_model)
        readable = [cid for cid, c in shoot.captures.items() if not c.error]

        # 1. TinyCLIP embeddings (every readable capture: the search index), optional scene refinement.
        if "search" in features or preference is not None:
            run_model("tinyclip", "tinyclip", TinyClipScorer, readable)
        embeddings = {cid: d["embedding"] for cid, d in store.details.items() if d.get("embedding")}
        scene_stats: dict[str, int] = {}
        if settings.refine_scenes and embeddings:
            from aiculler.scenes import refine_scenes

            with stage("scenes"):
                scene_stats = refine_scenes(shoot, embeddings, settings)

        # 2. Route.
        with stage("route"):
            routing = route(shoot, settings)
        candidates = routing.scored_captures()

        # 3. Cheap scores and clear-failure gates: the core cull needs nothing else.
        with stage("score"):
            run_scorers(store, shoot, candidates, default_scorers(), previews, cache, progress, workers=workers)
        with stage("gates"):
            gates = apply_gates(shoot, routing, settings, store.scores)
        live = contested(routing, gates)

        # 4. Faces: every readable capture for the people index (with AdaFace), else only contested frames (the
        # only ones a blink or a sharper face can still change).
        people_on = "people" in features and not unavailable("people", "adaface")
        face_frames = readable if people_on else (sorted(live) if "faces" in features else [])
        run_model("faces", "face_landmarker", FacesScorer, face_frames)

        # 5. MANIQA on contested frames, then the severe-blur gate.
        if "maniqa" in features and run_model("maniqa", "maniqa", ManiqaScorer, live):
            with stage("gates"):
                for cid, failure in blur_failures(shoot, routing, gates, store.scores, settings).items():
                    gates.failures.setdefault(cid, []).append(failure)
            live = contested(routing, gates)

        # 6. Appeal: Q-ReAlign only on each contested set's technical shortlist.
        if "appeal" in features and not unavailable("appeal", "q_realign"):
            shortlist = appeal_shortlist(routing, gates, store.scores, settings, live)
            run_model("appeal", "q_realign", AppealScorer, shortlist)

        # 7. Preference.
        if preference is not None:
            if used.get("tinyclip") != preference.embedding_revision:
                warnings.append(f"preference: not used (trained on {preference.embedding_revision}, embeddings are {used.get('tinyclip')})")
            else:
                with stage("preference"):
                    ids = [c for c in candidates if c in embeddings]
                    if ids:
                        import numpy as np

                        for cid, p in zip(ids, preference.score(np.asarray([embeddings[c] for c in ids]))):
                            store.scores.setdefault(cid, {})["preference"] = round(float(p), 4)
                used["preference"] = f"{preference.embedding_revision}+n{preference.examples}"
    finally:
        cache.close()
        registry.close()

    # 8. Decide; people (index and coverage).
    with stage("decide"):
        files = decide(shoot, routing, gates, store.scores, settings)
        people_stats: dict[str, int] = {}
        people = []
        if people_on and "faces" in used:
            from aiculler.people import apply_people_coverage, find_people, tag_people

            people = find_people(shoot, store.details, settings.face_cluster_threshold)
            if settings.keep_people_coverage:
                people_stats = apply_people_coverage(people, files, gates, settings)
            else:
                people_stats = {"people": len(people)}
            tag_people(people, files)
        index = build_index(shoot, used, embeddings if "tinyclip" in used else {}, people, settings)

    stats = {
        "captures": len(shoot.captures),
        "groups": len(shoot.groups),
        "comparison_sets": len(routing.bursts) + sum(1 for s in routing.scenes if len(s.members) > 1),
        "scored_captures": len(candidates),
        "contested_frames": len(live),
        "appeal_frames": len(shortlist),
        **scene_stats,
        **people_stats,
        **store.stats,
    }
    return CullResult(
        files=files,
        settings=settings,
        bracketeering_version=shoot.bracketeering_version,
        engine_version=__version__,
        models=used,
        timings={k: round(v, 3) for k, v in timings.items()},
        stats=stats,
        warnings=warnings,
        index=index,
    )
