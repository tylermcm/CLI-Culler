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
MOVING_BRACKET_MODES = ("as_burst", "bracket")
SEQUENCE_MODES = ("keep_all", "cull")
FEATURELESS_MODES = ("review", "reject", "off")
NORMALISE_MODES = ("relative", "rank", "zscore")
REVIEW_WARNING_MODES = ("doubt", "all", "none")
MANIQA_SCOPES = ("keepers", "all")
QUALITY_MODES = ("raw", "shoot", "category", "technique")
QUALITY_SIGNALS = ("maniqa", "qra_iqa", "either", "both", "agree")
QUALITY_UNIQUE = ("review", "reject")
FLOOR_FEATURE = {"maniqa": "maniqa", "qra_iqa": "appeal"}  # the model feature each floor signal needs
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
    # Highlights, by connected region of pixels blown to white (never applied to bracket frames). Saturated colour
    # (one channel clipped) only costs score: China 8-26's _DSC1055, a yellow light ring, has 44% of the frame
    # saturated but 0.8% blown.
    clip_region_ignore: float = 0.05  # largest clipped region below this fraction of the frame: no penalty
    clip_region_reject: float = 0.35  # largest blown region at or above this: reject; in between, a score penalty
    clip_penalty: float = 0.15  # penalty (combined-score points) at clip_region_reject, linear from 0 at clip_region_ignore
    sky_blown: float = 0.75  # blown fraction of the top third at or above this: reject ("sky blown out")
    washed_mean: float = 0.80  # mean luma at or above this...
    washed_spread: float = 0.15  # ...and tonal spread (p95 - p5) at or below this: reject ("washed out")
    shadows_crushed: float = 0.60  # fraction of crushed pixels above this: unusable
    # Quality floor: every frame about to be kept is scored by the floor's signal(s). Below a reject level the frame
    # fails a gate and its contest moves to the next-best frame (scored in turn); below a review level it goes to
    # review instead of keep. Never bracket frames or frames left out of culling. 0 turns a level off; all levels 0
    # (the default) is no floor. Set them from your own labels (ai-culler sample / calibrate).
    quality_reject: float = 0.0  # MANIQA levels (needs "maniqa" in models)
    quality_review: float = 0.0
    qra_reject: float = 0.0  # Q-ReAlign image-quality levels (needs "appeal" in models)
    qra_review: float = 0.0
    # Which signal decides: "maniqa", "qra_iqa", "either" (the stricter verdict of the two), "both" (the milder: a
    # frame fails only when both say so) or "agree" (reject only when both say reject; review when they disagree or
    # either says review).
    quality_signal: str = "maniqa"
    # A frame below a reject level that is its moment's best (a lone frame, or the best of a burst or scene whose
    # every candidate failed): "review" (default) or "reject". Redundant frames below it, with a better alternative
    # that passes, are rejected either way.
    quality_unique: str = "review"
    # What the levels compare against: "raw" (the MANIQA score itself), "shoot" (its percentile among this shoot's
    # competing frames, 0-1; MANIQA then scores all of them), "category" (its percentile among frames of the same
    # TinyCLIP category in a library baseline, 0-1) or "technique" (category, and category + EXIF technique cells
    # such as wildlife at high ISO). With several labels or cells the most favourable percentile counts.
    quality_mode: str = "raw"
    quality_baseline: str | None = None  # library baseline (ai-culler baseline) for "category" / "technique"
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
    # A bracket whose frames don't register (bracketeering: subject or camera moving, e.g. wildlife shot with
    # auto-bracketing on) is not an HDR set but alternatives of a moment: "as_burst" ranks its frames against each
    # other like a burst (bursts_keep survive; never gated, clipping still costs a penalty); "bracket" treats it
    # like any other bracket (the brackets setting).
    moving_brackets: str = "as_burst"
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
    # Which frames MANIQA scores: "keepers" (contested frames, for ranking, and every frame about to be kept, for
    # the quality floor) or "all" (every frame that competes, lone frames included, whether kept or not).
    maniqa_scope: str = "keepers"
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
        check("moving_brackets", self.moving_brackets, MOVING_BRACKET_MODES)
        check("focus_stacks", self.focus_stacks, SEQUENCE_MODES)
        check("timelapses", self.timelapses, SEQUENCE_MODES)
        check("gates.featureless", self.gates.featureless, FEATURELESS_MODES)
        check("normalise", self.normalise, NORMALISE_MODES)
        check("review_warnings", self.review_warnings, REVIEW_WARNING_MODES)
        check("maniqa_scope", self.maniqa_scope, MANIQA_SCOPES)
        check("gates.quality_mode", self.gates.quality_mode, QUALITY_MODES)
        check("gates.quality_signal", self.gates.quality_signal, QUALITY_SIGNALS)
        check("gates.quality_unique", self.gates.quality_unique, QUALITY_UNIQUE)
        if self.gates.quality_mode != "raw" and max(self.gates.quality_reject, self.gates.quality_review) > 1:
            raise ValueError("gates: in a percentile quality_mode the levels are percentiles, 0..1")
        if not 0 <= self.gates.qra_reject <= 1 or not 0 <= self.gates.qra_review <= 1 or (
                self.gates.qra_review and self.gates.qra_reject > self.gates.qra_review):
            raise ValueError("gates: need 0 <= qra_reject <= qra_review <= 1 (0 = off)")
        for m in self.models:
            check("models", m, MODEL_FEATURES + ("all",))
        g = self.gates
        if not 0.0 <= g.clip_region_ignore < g.clip_region_reject <= 1.0:
            raise ValueError("gates: need 0 <= clip_region_ignore < clip_region_reject <= 1")
        if g.quality_reject < 0 or g.quality_review < 0 or (g.quality_review and g.quality_reject > g.quality_review):
            raise ValueError("gates: need 0 <= quality_reject <= quality_review (0 = off)")
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

    def floor_signals(self) -> list[str]:
        """The metrics the quality floor judges ("maniqa", "qra_iqa")."""
        sig = self.gates.quality_signal
        return [sig] if sig in ("maniqa", "qra_iqa") else ["maniqa", "qra_iqa"]

    def floor_levels(self, metric: str) -> tuple[float, float]:
        g = self.gates
        return (g.quality_reject, g.quality_review) if metric == "maniqa" else (g.qra_reject, g.qra_review)

    def floor_on(self) -> bool:
        """Any floor level set for the floor's signal(s), and their models asked for."""
        return any(max(self.floor_levels(m)) > 0 and FLOOR_FEATURE[m] in self.features() for m in self.floor_signals())

    def category_floor(self) -> bool:
        """The quality floor judges scores against category baselines (needs TinyCLIP)."""
        return self.floor_on() and self.gates.quality_mode in ("category", "technique")

    def needs_tinyclip(self) -> bool:
        return "search" in self.features() or bool(self.preference_model) or self.category_floor() or self.refine_scenes

    def model_keys(self) -> set[str]:
        """Registry keys these settings may load."""
        keys = {k for f in self.features() for k in FEATURE_MODELS[f]}
        if self.needs_tinyclip():
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
    categories: list[str] = field(default_factory=list)  # what it shows (TinyCLIP, when it ran), most likely first
    techniques: list[str] = field(default_factory=list)  # EXIF technique flags (high_iso, long_lens, ...)
    # What the quality floor did: "reject", "review" (low quality, or the models disagree), "review_unique" (below the
    # floor but its moment's best frame: quality_unique review); None when the floor did not act on it.
    floor: str | None = None
    # Raw model outputs kept for reproduction and later fusion: MANIQA per-crop scores and boxes, Q-ReAlign's five
    # level probabilities per task (see scorers.models).
    outputs: dict = field(default_factory=dict)

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
    shoot: object = field(default=None, repr=False, compare=False)  # the ingested shoot (not serialised)

    @property
    def summary(self) -> dict[str, int]:
        out = {"files": len(self.files), "keep": 0, "reject": 0, "review": 0}
        for d in self.files.values():
            out[d.decision] += 1
        return out

    def to_dict(self) -> dict:
        from aiculler.outputs import decisions_document

        return decisions_document(self)


OUTPUT_DETAILS = ("maniqa_crops", "maniqa_boxes", "maniqa_size", "qra_probs")


def floor_applies(shoot, routing, capture: str) -> bool:
    """The quality floor judges frames that compete: not bracket frames (moving brackets' included), not frames left
    out of culling."""
    return capture not in routing.passthrough and shoot.group_of(capture).kind != "exposure"


METRIC_LABELS = {"maniqa": "MANIQA", "qra_iqa": "Q-ReAlign quality"}


FLOOR_ROLES = ("lone", "burst_winner", "burst_keep", "scene_winner", "scene_keep", "runner_up")


def floor_population(shoot, files: dict[str, "FileDecision"], settings: CullSettings) -> dict[str, str]:
    """The frames a quality floor judges in this cull, by role: what it keeps (a lone frame, a burst's or a scene's
    winner, another survivor) and each set's runner-up (promoted if a winner fails). Approximate for a burst that is
    itself a scene's runner-up (its frames carry only their burst ranking)."""
    from aiculler.routing import route

    routing = route(shoot, settings)
    out: dict[str, str] = {}
    for d in files.values():
        if not d.primary or not floor_applies(shoot, routing, d.capture):
            continue
        burst = d.kind == "burst" and d.comparison_set is not None and not d.comparison_set.startswith(("scene:", "solo:"))
        if d.decision == "keep":
            if d.comparison_set is None:
                out[d.capture] = "lone"
            elif burst:
                out[d.capture] = "burst_winner" if d.rank_in_set == 1 else "burst_keep"
            else:
                out[d.capture] = "scene_winner" if d.rank_in_set == 1 else "scene_keep"
        elif d.decision == "reject" and not d.gates and d.rank_in_set is not None:
            keep = settings.bursts_keep if burst else settings.scene_keep
            if keep and d.rank_in_set == keep + 1:
                out[d.capture] = "runner_up"
    return out


def make_floor_value(settings: CullSettings, shoot, routing, candidates, store, catinfo, used, warnings, metric: str = "maniqa"):
    """capture -> (value the floor levels compare, its description) for one signal, or None (floor off, with a warning)."""
    from bisect import bisect_left

    mode = settings.gates.quality_mode
    label = METRIC_LABELS[metric]

    def score(cid):
        return store.scores.get(cid, {}).get(metric)

    if mode == "raw":
        return lambda cid: (score(cid), "" if score(cid) is None else f"{label} {score(cid):.3f}")
    if mode == "shoot":
        ref = sorted(v for c in candidates if floor_applies(shoot, routing, c) and (v := score(c)) is not None)
        if len(ref) < 20:
            warnings.append(f"quality floor: not used (shoot mode needs {label} on at least 20 frames; {len(ref)} scored)")
            return None

        def by_shoot(cid):
            m = score(cid)
            if m is None:
                return None, ""
            pct = bisect_left(ref, m) / len(ref)
            return pct, f"{label} {m:.3f}, percentile {pct:.2f} of this shoot's {len(ref)} frames"
        return by_shoot
    # category / technique
    from aiculler.baseline import LibraryBaseline

    path = settings.gates.quality_baseline
    if not path:
        warnings.append(f"quality floor: not used ({mode} mode needs gates.quality_baseline, from ai-culler baseline)")
        return None
    try:
        lib = LibraryBaseline.load(path)
        lib.check(used.get("maniqa"), used.get("tinyclip"), used.get("appeal") if metric == "qra_iqa" else None)
    except (OSError, ValueError) as exc:
        warnings.append(f"quality floor: not used ({exc})")
        return None
    if metric not in lib.metrics:
        warnings.append(f"quality floor: not used (the library baseline has no {label} scores; build it with --qrealign)")
        return None
    if not catinfo:
        warnings.append(f"quality floor: not used ({mode} mode needs TinyCLIP categories)")
        return None
    from aiculler.categories import techniques

    def by_category(cid):
        m = score(cid)
        if m is None:
            return None, ""
        labels = catinfo.get(cid, {}).get("labels", [])
        pct, cell = lib.percentile(m, labels, techniques(shoot.captures[cid]) if mode == "technique" else [], metric)
        return pct, f"{label} {m:.3f}, percentile {pct:.2f} of {cell}"
    return by_category


def build_index(shoot, used: dict[str, str], embeddings: dict[str, list[float]], people, settings: CullSettings,
                catinfo: dict[str, dict] | None = None) -> dict:
    """The index for semantic and people search: what the "search" and "people" features found, per capture."""
    index: dict = {}
    if embeddings:
        index["search"] = {"model": used.get("tinyclip", ""), "embeddings": dict(embeddings)}
        if catinfo:
            from aiculler.categories import VERSION

            index["search"]["categories_version"] = VERSION
            index["search"]["categories"] = {cid: c["labels"] for cid, c in catinfo.items()}
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
    from aiculler.gates import QUALITY, apply_gates, floor_verdict
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

        # 1. TinyCLIP first, on every readable capture (every bracket and burst frame, kept or not), whatever needs it:
        # embeddings for search, categories for every capture, scene refinement, preference, the category floor.
        catinfo: dict[str, dict] = {}
        if settings.needs_tinyclip() and run_model("tinyclip", "tinyclip", TinyClipScorer, readable):
            try:
                from aiculler.categories import build_prototypes, classify

                with stage("categories"):
                    protos = build_prototypes(registry.engine("tinyclip"), used["tinyclip"])
                    catinfo = {cid: classify(d["embedding"], protos) for cid, d in store.details.items() if d.get("embedding")}
            except Exception as exc:  # noqa: BLE001 - no text encoder (tokenizers missing...): embeddings still stand
                warnings.append(f"categories: not used ({type(exc).__name__}: {exc})")
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

        # 5. MANIQA on contested frames (or every competing frame: maniqa_scope all), then the severe-blur gate.
        shoot_floor = settings.floor_on() and settings.gates.quality_mode == "shoot"  # a shoot percentile needs every frame scored
        every = settings.maniqa_scope == "all" or (shoot_floor and "maniqa" in settings.floor_signals())
        maniqa_frames = [c for c in candidates if not gates.failed(c)] if every else sorted(live)
        if "maniqa" in features and run_model("maniqa", "maniqa", ManiqaScorer, maniqa_frames):
            with stage("gates"):
                for cid, failure in blur_failures(shoot, routing, gates, store.scores, settings).items():
                    gates.failures.setdefault(cid, []).append(failure)
            live = contested(routing, gates)

        # 6. Appeal: Q-ReAlign only on each contested set's technical shortlist.
        if "appeal" in features and not unavailable("appeal", "q_realign"):
            shortlist = appeal_shortlist(routing, gates, store.scores, settings, live)
            if shoot_floor and "qra_iqa" in settings.floor_signals():
                shortlist |= {c for c in candidates if not gates.failed(c)}
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
        # 8. Decide, then the quality floor: its signal(s) score every frame about to be kept; a frame below a reject
        # level fails a gate and its contest is decided again, until no new keeper needs a score.
        floor_stats = {"quality_rejected": 0, "quality_review": 0, "floor_rounds": 0}
        floor_flagged: set[str] = set()
        with stage("decide"):
            files = decide(shoot, routing, gates, store.scores, settings)
        signals = settings.floor_signals()
        scorer_of = {"maniqa": ("maniqa", "maniqa", ManiqaScorer), "qra_iqa": ("appeal", "q_realign", AppealScorer)}
        floor = any(max(settings.floor_levels(m)) > 0 for m in signals)
        if floor:
            missing = [FLOOR_FEATURE[m] for m in signals if FLOOR_FEATURE[m] not in features]
            broken = [m for m in signals if any(w.startswith(f"{scorer_of[m][0]}: not used") for w in warnings)
                      or not registry.available(scorer_of[m][1])]
            if missing:
                warnings.append(f"quality floor: not used (quality_signal {settings.gates.quality_signal} needs --with {','.join(missing)})")
                floor = False
            elif broken:
                floor = False  # the model's own warning says why
        value_of = {}
        if floor:
            for m in signals:
                fn = make_floor_value(settings, shoot, routing, candidates, store, catinfo, used, warnings, m)
                if fn is None:
                    floor = False
                    break
                value_of[m] = fn
        checked: set[str] = set()
        while floor:
            keepers = [d.capture for d in files.values() if d.primary and d.decision == "keep" and d.capture not in checked
                       and floor_applies(shoot, routing, d.capture)]
            if not keepers:
                break
            floor_stats["floor_rounds"] += 1
            for m in signals:
                name, key, make = scorer_of[m]
                todo = [c for c in keepers if m not in store.scores.get(c, {})]
                if todo and not run_model(name, key, make, todo):
                    floor = False  # a model failed: no floor; the decisions so far stand
            if not floor:
                break
            changed = False
            for cid in keepers:
                checked.add(cid)
                failure, flag = floor_verdict({m: value_of[m](cid) for m in signals}, settings)
                if failure is not None:
                    gates.failures.setdefault(cid, []).append(failure)
                    floor_stats["quality_rejected"] += 1
                    changed = True
                elif flag is not None:
                    gates.flags.setdefault(cid, []).append(flag)
                    floor_flagged.add(cid)
                    floor_stats["quality_review"] += 1
                    changed = True
            if not changed:
                break
            with stage("decide"):
                files = decide(shoot, routing, gates, store.scores, settings)
        if floor_stats["floor_rounds"]:
            for d in files.values():
                if QUALITY in d.gates:
                    d.floor = "reject" if d.decision == "reject" else "review_unique"
                elif d.capture in floor_flagged and d.decision == "review":
                    d.floor = "review"
    finally:
        cache.close()
        registry.close()

    # 9. People (index and coverage).
    with stage("decide"):
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
        from aiculler.categories import techniques

        for d in files.values():
            d.categories = list(catinfo.get(d.capture, {}).get("labels", []))
            d.techniques = techniques(shoot.captures[d.capture])
            detail = store.details.get(d.capture, {})
            d.outputs = {k: detail[k] for k in OUTPUT_DETAILS if k in detail}
        index = build_index(shoot, used, embeddings if "tinyclip" in used else {}, people, settings, catinfo)

    stats = {
        "captures": len(shoot.captures),
        "groups": len(shoot.groups),
        "comparison_sets": len(routing.bursts) + sum(1 for s in routing.scenes if len(s.members) > 1),
        "scored_captures": len(candidates),
        "contested_frames": len(live),
        "appeal_frames": len(shortlist),
        **scene_stats,
        **(floor_stats if floor_stats["floor_rounds"] else {}),
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
        shoot=shoot,
    )
