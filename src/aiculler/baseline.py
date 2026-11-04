"""Baselines for the quality floor: ``ai-culler baseline`` (a library baseline: what MANIQA scores look like per
category, no labels needed), ``ai-culler sample`` and ``ai-culler calibrate`` (where your own labels put the levels).

A library baseline scores every competing frame of many shoots with MANIQA and TinyCLIP (categories) and records
each one with its EXIF technique flags. The floor then judges a frame's MANIQA score as a percentile among similar
frames: same shoot, same category, or same category and technique (``gates.quality_mode``).

``sample`` draws a random, seeded sample of frames across several shoot folders (equal shares per folder), scores
them (cheap signals and MANIQA), and writes a blind labelling page: the frames without their scores, each marked
keep, borderline or reject by the photographer, exported as ``labels.csv``. Only frames the floor can act on are
drawn: frames that compete (singles, burst frames, scene representatives), never bracket frames.

``calibrate`` joins the labels with the scores and reports, for a sweep of thresholds, how many frames of each
label fall below; it suggests ``quality_reject`` (below every keep and borderline frame, with a margin) and
``quality_review`` (at most ``review_share`` of keeps below it).
"""

from __future__ import annotations

import csv
import html
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_SCHEMA = "aiculler-sample/1"
LABELS = ("keep", "borderline", "reject")
PAGE_LABELS = ("keep", "borderline", "reject", "hero")  # hero: a select; counts as keep when calibrating
# Signals a floor can judge: MANIQA, Q-ReAlign image quality, Q-ReAlign aesthetics.
METRICS = ("maniqa", "qra_iqa", "appeal")
METRIC_NAMES = {"maniqa": "MANIQA", "qra_iqa": "Q-ReAlign quality", "appeal": "Q-ReAlign aesthetics"}


@dataclass
class SampleRow:
    id: str  # "001".."NNN": the thumbnail's name and the labels' key
    folder: str
    path: str  # primary file
    capture: str
    kind: str  # group kind (single, burst, ...)
    scores: dict[str, float] = field(default_factory=dict)
    role: str | None = None  # what the floor would judge it as (api.FLOOR_ROLES), with population "keepers"
    labels: list[str] = field(default_factory=list)  # TinyCLIP categories (with a library baseline)
    techniques: list[str] = field(default_factory=list)  # EXIF technique flags (with a library baseline)


def draw(candidates: dict[str, list[str]], n: int, seed: int) -> dict[str, list[str]]:
    """Equal shares per folder (a folder with too few gives its leftover to the others), each drawn at random."""
    rng = random.Random(seed)
    pools = {f: sorted(c) for f, c in candidates.items() if c}
    quota = {f: 0 for f in pools}
    left = n
    while left > 0:
        open_ = [f for f in pools if quota[f] < len(pools[f])]
        if not open_:
            break
        share = max(1, left // len(open_))
        for f in open_:
            take = min(share, len(pools[f]) - quota[f], left)
            quota[f] += take
            left -= take
            if left == 0:
                break
    return {f: rng.sample(pools[f], quota[f]) for f in pools}


def make_sample(
    folders: list[Path],
    out: Path,
    *,
    n: int = 100,
    seed: int = 1,
    models=None,
    cache_db: Path | None = None,
    cache_dir: Path | None = None,
    settings=None,
    workers: int | None = None,
    progress=None,
    library: "LibraryBaseline | None" = None,
    qrealign: bool = False,
    population: str = "keepers",
) -> dict:
    """Draw, score and write the labelling page. Returns the sample document (also ``out/sample.json``).

    ``population``: "keepers" (default) draws from the frames a quality floor would judge: each shoot is culled with
    ``settings`` (and ``models`` when they ask for any) with the floor off, and its would-be keeps (lone frames,
    burst and scene winners and survivors) and each set's runner-up are drawn, each row recording its role; "all"
    draws from every competing frame, kept or not.

    With a ``library`` baseline the draw is spread evenly across categories (each frame's most likely one) instead of
    folders, only frames the library holds are drawn, and each row carries its categories and technique flags."""
    from dataclasses import replace

    from aiculler.api import CullSettings, cull, floor_applies, floor_population
    from aiculler.ingest import PreviewCache, ingest
    from aiculler.routing import route
    from aiculler.scorers import ScoreStore, default_scorers, run_scorers
    from aiculler.scorers.models import AppealScorer, ManiqaScorer
    from aiculler.storage import ScoreCache

    settings = settings or CullSettings()
    progress = progress or (lambda stage, done, total: None)
    if cache_dir is None:
        cache_dir = Path(str(Path(cache_db).with_suffix("")) + "_previews") if cache_db else Path(".aiculler_cache")
    previews = PreviewCache(cache_dir)
    cache = ScoreCache(cache_db) if cache_db else ScoreCache(":memory:")
    if population not in ("keepers", "all"):
        raise ValueError(f"population must be keepers or all; got {population!r}")
    shoots, candidates, roles = {}, {}, {}
    no_floor = replace(settings, gates=replace(settings.gates, quality_reject=0.0, quality_review=0.0, qra_reject=0.0, qra_review=0.0))
    for folder in folders:
        key = str(folder)
        if population == "keepers":
            result = cull(folder, no_floor, models=models if no_floor.model_keys() else None, cache_db=cache_db,
                          cache_dir=cache_dir, workers=workers)
            shoot = result.shoot
            roles[key] = floor_population(shoot, result.files, no_floor)
            candidates[key] = [c for c in roles[key] if not shoot.captures[c].error]
        else:
            shoot = ingest(folder, backend=settings.backend, exclude=settings.exclude, workers=workers)
            routing = route(shoot, settings)
            roles[key] = {}
            candidates[key] = [c for c in routing.scored_captures() if floor_applies(shoot, routing, c) and not shoot.captures[c].error]
        shoots[key] = shoot
    records: dict[tuple[str, str], dict] = {}
    if library is not None:
        records = {(r["shoot"], r["capture"]): r for r in library.doc["records"]}
        strata: dict[str, list[str]] = {}
        for key, ids in candidates.items():
            for cid in ids:
                rec = records.get((key, cid))
                if rec is not None:
                    strata.setdefault(rec["labels"][0], []).append(f"{key}\x00{cid}")
        by_category = draw(strata, n, seed)
        picked = {key: [] for key in candidates}
        for ids in by_category.values():
            for joined in ids:
                key, cid = joined.split("\x00", 1)
                picked[key].append(cid)
        per_category = {k: len(v) for k, v in by_category.items()}
    else:
        picked = draw(candidates, n, seed)
        per_category = {}
    rows: list[SampleRow] = []
    warnings: list[str] = []
    maniqa_revision = None
    try:
        for key, ids in picked.items():
            shoot = shoots[key]
            store = ScoreStore()
            run_scorers(store, shoot, ids, default_scorers(), previews, cache, progress, workers=workers)
            if models is not None and models.available("maniqa"):
                try:
                    scorer = ManiqaScorer(models)
                    run_scorers(store, shoot, ids, [scorer], previews, cache, progress, batch_size=8, workers=workers)
                    maniqa_revision = scorer.revision
                except Exception as exc:  # noqa: BLE001 - the sample is still useful without MANIQA; say so
                    warnings.append(f"maniqa: not used ({type(exc).__name__}: {exc})")
            if qrealign and models is not None and models.available("q_realign"):
                try:
                    run_scorers(store, shoot, ids, [AppealScorer(models)], previews, cache, progress, batch_size=8, workers=workers)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"q_realign: not used ({type(exc).__name__}: {exc})")
            elif qrealign:
                warnings.append("q_realign: not used (not available)")
            for cid in ids:
                cap = shoot.captures[cid]
                row = SampleRow("", key, cap.primary, cid, shoot.group_of(cid).kind, dict(store.scores.get(cid, {})),
                                role=roles[key].get(cid))
                rec = records.get((key, cid))
                if rec is not None:
                    row.labels, row.techniques = list(rec["labels"]), list(rec["techniques"])
                    for metric in METRICS:  # the library's scores (same model revisions), where this run has none
                        if rec.get(metric) is not None and row.scores.get(metric) is None:
                            row.scores[metric] = rec[metric]
                rows.append(row)
    finally:
        cache.close()
        if models is not None:
            models.close()
    if maniqa_revision is None and not warnings:
        warnings.append("maniqa: not used (no models; pass --models): the sample has no quality scores to calibrate")

    rng = random.Random(seed + 1)
    rng.shuffle(rows)  # folders interleaved, so the page shows no shoot in a block
    width = len(str(len(rows)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "thumbs").mkdir(exist_ok=True)
    for i, row in enumerate(rows, 1):
        row.id = str(i).zfill(max(3, width))
        image = previews.load(shoots[row.folder].captures[row.capture])
        if image is not None:
            image.save(out / "thumbs" / f"{row.id}.jpg", quality=88)
    doc = {
        "schema": SAMPLE_SCHEMA,
        "seed": seed,
        "folders": [str(f) for f in folders],
        "requested": n,
        "drawn": len(rows),
        "per_folder": {k: len(v) for k, v in picked.items()},
        "per_category": per_category,
        "population": population,
        "per_role": {r: sum(1 for row in rows if row.role == r) for r in sorted({row.role for row in rows if row.role})},
        "candidates_per_folder": {k: len(v) for k, v in candidates.items()},
        "maniqa": maniqa_revision,
        "warnings": warnings,
        "rows": [row.__dict__ for row in rows],
    }
    (out / "sample.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (out / "label.html").write_text(_label_page(rows), encoding="utf-8")
    return doc


def _label_page(rows: list[SampleRow]) -> str:
    items = json.dumps([{"id": r.id, "name": Path(r.path).name} for r in rows])
    return _PAGE.replace("__ITEMS__", items).replace("__N__", str(len(rows))).replace("__TITLE__", html.escape("Quality baseline"))


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#fafaf9; --fg:#1c1917; --muted:#78716c; --line:#e7e5e4; --keep:#15803d; --border:#b45309; --reject:#b91c1c; }
@media (prefers-color-scheme: dark) { :root { --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --line:#44403c; --keep:#4ade80; --border:#fbbf24; --reject:#f87171; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.4 system-ui, sans-serif; }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; z-index:1; }
header h1 { font-size:16px; margin:0; }
.muted { color:var(--muted); }
button { font:inherit; padding:6px 12px; border-radius:6px; border:1px solid var(--line); background:transparent; color:var(--fg); cursor:pointer; }
main { max-width:1100px; margin:0 auto; padding:16px; }
.frame { display:grid; grid-template-columns:1fr; gap:8px; margin:0 0 28px; padding:12px; border:2px solid var(--line); border-radius:10px; }
.frame.current { border-color:var(--muted); }
.frame img { width:100%; max-height:78vh; object-fit:contain; background:#0002; border-radius:6px; }
.row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.row .id { margin-right:auto; }
.frame[data-label=keep], .frame[data-label=hero] { border-color:var(--keep); } .frame[data-label=hero] { border-width:4px; } .frame[data-label=borderline] { border-color:var(--border); } .frame[data-label=reject] { border-color:var(--reject); }
button.on.keep, button.on.hero { background:var(--keep); color:var(--bg); } button.on.borderline { background:var(--border); color:var(--bg); } button.on.reject { background:var(--reject); color:var(--bg); }
</style></head>
<body>
<header><h1>Quality baseline</h1><span id="count" class="muted"></span>
<span class="muted">Keys: 1 keep, 2 borderline, 3 reject, 4 hero (a select), ←/→ move</span>
<button id="export">Export labels.csv</button></header>
<main>
<p class="muted">Judge each frame on its own, as if it were the only frame of its moment: would you keep it? Scores are hidden so they do not steer you. Labels are saved in this browser as you go; export when done and run <code>ai-culler calibrate</code> on this folder.</p>
<div id="frames"></div></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "aiculler-baseline:" + location.pathname;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let current = 0;
const root = document.getElementById("frames");
function save() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function render() {
  root.innerHTML = "";
  ITEMS.forEach((it, i) => {
    const el = document.createElement("section");
    el.className = "frame" + (i === current ? " current" : "");
    if (labels[it.id]) el.dataset.label = labels[it.id];
    el.id = "f" + it.id;
    el.innerHTML = `<img loading="lazy" src="thumbs/${it.id}.jpg" alt="frame ${it.id}">
      <div class="row"><span class="id muted">#${it.id} · ${i + 1} of ${ITEMS.length}</span>
      ${["keep","borderline","reject","hero"].map(l => `<button class="${l}${labels[it.id] === l ? " on" : ""}" data-l="${l}">${l}</button>`).join("")}</div>`;
    el.querySelectorAll("button").forEach(b => b.onclick = () => { current = i; mark(b.dataset.l, false); });
    root.appendChild(el);
  });
  document.getElementById("count").textContent = Object.keys(labels).length + " of " + ITEMS.length + " labelled";
}
function mark(label, advance = true) {
  labels[ITEMS[current].id] = label; save();
  if (advance && current < ITEMS.length - 1) current++;
  render(); document.getElementById("f" + ITEMS[current].id).scrollIntoView({block: "start"});
}
document.addEventListener("keydown", e => {
  const m = {"1": "keep", "2": "borderline", "3": "reject", "4": "hero"}[e.key];
  if (m) return mark(m);
  if (e.key === "ArrowRight" || e.key === "ArrowDown") { current = Math.min(ITEMS.length - 1, current + 1); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowUp") { current = Math.max(0, current - 1); }
  else return;
  e.preventDefault(); render(); document.getElementById("f" + ITEMS[current].id).scrollIntoView({block: "start"});
});
document.getElementById("export").onclick = () => {
  const lines = ["id,name,label"].concat(ITEMS.filter(it => labels[it.id]).map(it => `${it.id},"${it.name.replace(/"/g, '""')}",${labels[it.id]}`));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], {type: "text/csv"}));
  a.download = "labels.csv"; a.click();
};
render();
</script></body></html>
"""


# --------------------------------------------------------------------------- calibrate


def read_labels(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("label") or "").strip().lower()
            if label not in PAGE_LABELS:
                raise ValueError(f"{path}: label must be one of {', '.join(PAGE_LABELS)}; got {label!r} for id {row.get('id')}")
            out[str(row["id"]).strip()] = "keep" if label == "hero" else label
    return out


FUSION_NAMES = {"either": "MANIQA or Q-ReAlign quality", "both": "MANIQA and Q-ReAlign quality",
                "agree": "MANIQA and Q-ReAlign quality, review on disagreement"}
ARM_NAMES = {"maniqa": "MANIQA", "qra_iqa": "Q-ReAlign quality", "appeal": "Q-ReAlign aesthetics",
             "either": "either", "both": "both", "agree": "agree (review)"}
MODE_NOTES = {"raw": "the score", "shoot": "percentile in its own shoot",
              "category": "percentile among its category", "technique": "percentile among its category (+ technique)"}


def signal_values(sample: dict, library: "LibraryBaseline | None") -> dict[str, dict[str, float]]:
    """"metric/mode" -> {row id: value}. raw for every signal the rows carry; with a library, also shoot, category and
    technique percentiles for each signal the library holds."""
    out: dict[str, dict[str, float]] = {}
    for r in sample["rows"]:
        for metric in METRICS:
            v = r["scores"].get(metric)
            if v is None:
                continue
            out.setdefault(f"{metric}/raw", {})[r["id"]] = v
            if library is None or metric not in library.metrics:
                continue
            pct = library.shoot_percentile(v, r["folder"], metric)
            if pct is not None:
                out.setdefault(f"{metric}/shoot", {})[r["id"]] = pct
            labels = r.get("labels") or []
            if labels:
                out.setdefault(f"{metric}/category", {})[r["id"]] = library.percentile(v, labels, (), metric)[0]
                out.setdefault(f"{metric}/technique", {})[r["id"]] = library.percentile(v, labels, r.get("techniques") or [], metric)[0]
    return out


def _stats(vs: list[float]) -> dict:
    if not vs:
        return {"n": 0}
    n = len(vs)
    return {"n": n, "min": vs[0], "p10": vs[n // 10], "median": vs[n // 2], "p90": vs[min(n - 1, (9 * n) // 10)], "max": vs[-1]}


def _safe_level(vals: dict[str, float], labels: dict[str, str], ids) -> float | None:
    """The highest level with no keep or borderline below it: the lowest such frame's value."""
    protected = [vals[i] for i in ids if labels[i] in ("keep", "borderline")]
    return min(protected) if protected else None


def _best_pair(a: dict[str, float], b: dict[str, float], labels: dict[str, str], ids) -> tuple[float, float, int]:
    """Levels (ta, tb) maximising rejects with a < ta AND b < tb, with no keep or borderline in both: (ta, tb, caught)."""
    protected = [i for i in ids if labels[i] in ("keep", "borderline")]
    rejects = [i for i in ids if labels[i] == "reject"]
    best = (float("-inf"), float("-inf"), 0)
    for ta in sorted({a[i] for i in ids}) + [float("inf")]:
        below = [b[i] for i in protected if a[i] < ta]
        tb = min(below) if below else float("inf")
        caught = sum(1 for i in rejects if a[i] < ta and b[i] < tb)
        if caught > best[2]:
            best = (ta, tb, caught)
    return best


def _breakdown(outcomes: dict[str, dict[str, dict[str, str]]], rows: dict[str, dict], labels: dict[str, str]) -> dict:
    """{mode: {dimension: {group: {arm: tallies}}}}: per role, most likely category and technique flag, how many
    rejects each arm catches and how many keeps it sends to review (it loses none at its fitted levels)."""
    out: dict = {}
    for mode, arms in outcomes.items():
        dims: dict[str, dict[str, dict[str, dict[str, int]]]] = {"role": {}, "category": {}, "technique": {}}
        for arm, verdicts in arms.items():
            for i, v in verdicts.items():
                r = rows.get(i, {})
                groups = {"role": [r.get("role") or "unknown"], "category": [(r.get("labels") or ["unknown"])[0]],
                          "technique": r.get("techniques") or ["none"]}
                for dim, names in groups.items():
                    for g in names:
                        t = dims[dim].setdefault(g, {}).setdefault(arm, {"keeps": 0, "rejects": 0, "caught": 0, "keeps_review": 0, "keeps_lost": 0})
                        lab = labels[i]
                        if lab == "reject":
                            t["rejects"] += 1
                            t["caught"] += v == "reject"
                        else:
                            t["keeps"] += 1
                            t["keeps_review"] += v == "review"
                            t["keeps_lost"] += v == "reject"
        out[mode] = dims
    return out


def calibrate(sample: dict, labels: dict[str, str], *, margin: float = 0.02, review_share: float = 0.10,
              library: "LibraryBaseline | None" = None) -> dict:
    """Every signal (MANIQA, Q-ReAlign quality, Q-ReAlign aesthetics) in every mode, and MANIQA + Q-ReAlign quality
    together, judged at the same keeper recall:

    * ``all_safe``: rejects caught at the highest level with none of your keeps (or borderlines) below it, fitted on
      all the labels (optimistic: the level is chosen on the frames it is scored on);
    * ``safe95``: the same, allowing 5% of keeps below;
    * ``held_out``: each shoot in turn, with the level fitted on the other shoots: keeps lost and rejects caught,
      summed (the honest estimate; needs labels from several shoots).

    Fusion without trained probabilities: ``either`` rejects a frame below either signal's safe level; ``both`` only
    below both, with the pair of levels chosen so no keep is below both; ``agree`` rejects a frame below both safe
    levels and sends one below only one of them to review (the models disagree). ``review`` counts what each arm sends
    to a person."""
    values = signal_values(sample, library)
    folder = {r["id"]: r["folder"] for r in sample["rows"]}
    rejects_total = sum(1 for i, lab in labels.items() if lab == "reject" and i in values.get("maniqa/raw", {}))
    signals: dict[str, dict] = {}
    for name, vals in values.items():
        ids = [i for i in vals if i in labels]
        by = {lab: sorted(vals[i] for i in ids if labels[i] == lab) for lab in LABELS}
        if not by["keep"]:
            continue
        safe = _safe_level(vals, labels, ids)
        keeps = by["keep"]
        k95 = keeps[int(0.05 * len(keeps))]
        reject_at = round(max(0.0, safe - margin), 3)
        allowed = int(review_share * len(keeps))
        review_at = max(round(keeps[allowed] if allowed < len(keeps) else keeps[-1], 3), reject_at)

        def caught(t: float, _by=by) -> dict[str, int]:
            return {lab: sum(1 for v in _by[lab] if v < t) for lab in LABELS}

        held = {"keeps_lost": 0, "borderline_lost": 0, "rejects_caught": 0, "folds": 0}
        for f in sorted({folder[i] for i in ids}):
            train = [i for i in ids if folder[i] != f]
            level = _safe_level(vals, labels, train)
            if level is None:
                continue
            held["folds"] += 1
            level -= margin
            for i in ids:
                if folder[i] == f and vals[i] < level:
                    held[{"keep": "keeps_lost", "borderline": "borderline_lost", "reject": "rejects_caught"}[labels[i]]] += 1
        lo, hi = min(min(v) for v in by.values() if v), max(max(v) for v in by.values() if v)
        step = 0.01 if name.endswith("/raw") else 0.02
        sweep, t = [], round(int(lo / step) * step, 2)
        while t <= hi + step:
            sweep.append({"threshold": round(t, 2), **caught(t)})
            t += step
        signals[name] = {
            "labelled": len(ids), "distribution": {lab: _stats(by[lab]) for lab in LABELS},
            "suggested": {"quality_reject": reject_at, "quality_review": review_at},
            "caught": {"quality_reject": caught(reject_at), "quality_review": caught(review_at)},
            "all_safe": caught(safe)["reject"], "safe95": caught(k95)["reject"], "held_out": held, "sweep": sweep,
        }
    # Outcomes per arm at its fitted levels (every keep safe): {mode: {arm: {id: "reject" | "review" | "pass"}}}.
    outcomes: dict[str, dict[str, dict[str, str]]] = {}
    for name, vals in values.items():
        metric, mode = name.split("/")
        if name not in signals:
            continue
        safe = _safe_level(vals, labels, [i for i in vals if i in labels])
        outcomes.setdefault(mode, {})[metric] = {i: ("reject" if vals[i] < safe else "pass") for i in vals if i in labels}
    fusion: dict[str, dict] = {}
    for mode in ("raw", "shoot", "category", "technique"):
        a, b = values.get(f"maniqa/{mode}"), values.get(f"qra_iqa/{mode}")
        if not a or not b:
            continue
        ids = [i for i in a if i in b and i in labels]
        if not any(labels[i] == "keep" for i in ids):
            continue
        sa, sb = _safe_level(a, labels, ids), _safe_level(b, labels, ids)
        ta, tb, both = _best_pair(a, b, labels, ids)
        out_e = {i: ("reject" if a[i] < sa or b[i] < sb else "pass") for i in ids}
        out_b = {i: ("reject" if a[i] < ta and b[i] < tb else "pass") for i in ids}
        # agree: each model at its own safe level; reject when both flag a frame, review when only one does (they disagree)
        out_a = {i: ("reject" if a[i] < sa and b[i] < sb else "review" if (a[i] < sa) != (b[i] < sb) else "pass") for i in ids}
        outcomes.setdefault(mode, {}).update({"either": out_e, "both": out_b, "agree": out_a})
        held = {arm: {"keeps_lost": 0, "rejects_caught": 0, "review_keeps": 0, "review_rejects": 0} for arm in ("either", "both", "agree")}
        for f in sorted({folder[i] for i in ids}):
            train, test = [i for i in ids if folder[i] != f], [i for i in ids if folder[i] == f]
            la, lb = _safe_level(a, labels, train), _safe_level(b, labels, train)
            if la is None:
                continue
            pa, pb, _ = _best_pair(a, b, labels, train)
            for i in test:
                rej = labels[i] == "reject"
                verdicts = {"either": "reject" if a[i] < la - margin or b[i] < lb - margin else "pass",
                            "both": "reject" if a[i] < pa and b[i] < pb else "pass",
                            "agree": "reject" if a[i] < la - margin and b[i] < lb - margin
                            else "review" if (a[i] < la - margin) != (b[i] < lb - margin) else "pass"}
                for arm, v in verdicts.items():
                    if v == "reject":
                        held[arm]["rejects_caught" if rej else "keeps_lost"] += 1
                    elif v == "review":
                        held[arm]["review_rejects" if rej else "review_keeps"] += 1

        def review(out):
            return {lab: sum(1 for i, v in out.items() if v == "review" and labels[i] == lab) for lab in LABELS}

        fusion[mode] = {"either": {"all_safe": sum(1 for i, v in out_e.items() if v == "reject" and labels[i] == "reject"),
                                   "held_out": held["either"], "levels": [sa, sb], "review": review(out_e)},
                        "both": {"all_safe": both, "held_out": held["both"], "levels": [ta, tb], "review": review(out_b)},
                        "agree": {"all_safe": sum(1 for i, v in out_a.items() if v == "reject" and labels[i] == "reject"),
                                  "held_out": held["agree"], "levels": [sa, sb], "review": review(out_a)}}
    breakdown = _breakdown(outcomes, {r["id"]: r for r in sample["rows"]}, labels)
    if not signals:
        raise ValueError("no frame labelled keep has a score: nothing to calibrate against")
    return {"rejects": rejects_total, "folders": len(set(folder.values())), "rules": {"margin": margin, "review_share": review_share},
            "signals": signals, "fusion": fusion, "breakdown": breakdown,
            "unscored": sum(1 for r in sample["rows"] if r["id"] in labels and r["scores"].get("maniqa") is None)}


def render_calibration(cal: dict) -> str:
    r = cal["rules"]
    lines = ["# Quality floor calibration", ""]
    if cal["unscored"]:
        lines += [f"{cal['unscored']} labelled frames have no MANIQA score and are left out.", ""]
    lines += [f"{cal['rejects']} frames labelled reject, from {cal['folders']} shoot(s). **All keeps safe**: rejects caught at the highest "
              "level with none of your keeps or borderlines below it (fitted on these same labels: optimistic). **95%**: "
              "allowing 5% of keeps below. **Held out**: each shoot judged with a level fitted on the others "
              "(keeps lost / rejects caught; the honest estimate).", "",
              "**Review** counts frames an arm sends to a person instead of deciding (keeps / rejects): it costs your time "
              "rather than your photos, so compare it too.", "",
              "| Signal | Compared as | All keeps safe | 95% | Review (keeps / rejects) | Held out: keeps lost | Held out: rejects caught | Held out: review |",
              "|---|---|---|---|---|---|---|---|"]
    for name, m in cal["signals"].items():
        metric, mode = name.split("/")
        h = m["held_out"]
        held = (str(h["keeps_lost"] + h["borderline_lost"]), str(h["rejects_caught"])) if h["folds"] > 1 else ("–", "–")
        lines.append(f"| {METRIC_NAMES[metric]} | {MODE_NOTES[mode]} | {m['all_safe']} | {m['safe95']} | 0 / 0 | {held[0]} | {held[1]} | – |")
    for mode, f in cal["fusion"].items():
        for rule, label in FUSION_NAMES.items():
            x = f[rule]
            h, rv = x["held_out"], x["review"]
            ok = cal["folders"] > 1
            held = (str(h["keeps_lost"]), str(h["rejects_caught"]), f"{h['review_keeps']} / {h['review_rejects']}") if ok else ("–", "–", "–")
            lines.append(f"| {label} | {MODE_NOTES[mode]} | {x['all_safe']} | – | {rv['keep'] + rv['borderline']} / {rv['reject']} | "
                         f"{held[0]} | {held[1]} | {held[2]} |")
    for mode, dims in cal.get("breakdown", {}).items():
        arms = list(next(iter(next(iter(dims.values())).values()), {}).keys()) if any(dims.values()) else []
        if not arms:
            continue
        lines += ["", f"## Breakdown, {MODE_NOTES[mode]}", "",
                  "Rejects caught / rejects in the group, at each arm's fitted levels (no keep lost); "
                  "for the agree arm, keeps it sends to review in brackets.", ""]
        for dim in ("role", "category", "technique"):
            groups = dims.get(dim, {})
            if not groups:
                continue
            names = [ARM_NAMES.get(a, a) for a in arms]
            lines += [f"**By {dim}**", "", "| " + dim.capitalize() + " | keeps | " + " | ".join(names) + " |",
                      "|---|---|" + "---|" * len(arms)]
            for g, by_arm in sorted(groups.items(), key=lambda kv: -max(t["keeps"] + t["rejects"] for t in kv[1].values())):
                first = next(iter(by_arm.values()))
                cells = []
                for a in arms:
                    t = by_arm.get(a)
                    if t is None:
                        cells.append("–")
                        continue
                    cell = f"{t['caught']} / {t['rejects']}"
                    if a == "agree" and t["keeps_review"]:
                        cell += f" ({t['keeps_review']})"
                    cells.append(cell)
                lines.append(f"| {g} | {first['keeps']} | " + " | ".join(cells) + " |")
            lines.append("")
    for name, m in cal["signals"].items():
        metric, mode = name.split("/")
        d, s, c = m["distribution"], m["suggested"], m["caught"]
        lines += ["", f"## {METRIC_NAMES[metric]}, {MODE_NOTES[mode]}", "", f"{m['labelled']} labelled frames.", "",
                  "| Label | n | min | p10 | median | p90 | max |", "|---|---|---|---|---|---|---|"]
        for lab in LABELS:
            x = d[lab]
            lines.append(f"| {lab} | {x['n']} | " + (" | ".join(f"{x[k]:.3f}" for k in ("min", "p10", "median", "p90", "max")) if x["n"] else "– | – | – | – | –") + " |")
        lines += ["",
                  f"- `quality_reject`: **{s['quality_reject']}** (lowest keep or borderline minus {r['margin']}). Below it: "
                  f"{c['quality_reject']['reject']} reject, {c['quality_reject']['borderline']} borderline, {c['quality_reject']['keep']} keep.",
                  f"- `quality_review`: **{s['quality_review']}** (at most {r['review_share']:.0%} of keeps below it). Below it: "
                  f"{c['quality_review']['reject']} reject, {c['quality_review']['borderline']} borderline, {c['quality_review']['keep']} keep."]
        if metric == "maniqa":
            lines += ["", "```json", json.dumps({"gates": {"quality_mode": mode, **s}}, indent=2), "```"]
        lines += ["", "| Below | keep | borderline | reject |", "|---|---|---|---|"]
        for row in m["sweep"]:
            lines.append(f"| {row['threshold']:.2f} | {row['keep']} | {row['borderline']} | {row['reject']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- library baseline

LIBRARY_SCHEMA = "aiculler-library/1"
MIN_CELL = 50  # a category (or category + technique) cell needs this many frames to be used; else broader ones


class LibraryBaseline:
    """Per-signal score distributions: the whole library, each shoot, each category, each category + technique.
    Each signal (MANIQA, Q-ReAlign quality, Q-ReAlign aesthetics) has its own; a record without a signal is left out
    of that signal's distributions."""

    def __init__(self, doc: dict, min_cell: int = MIN_CELL):
        if doc.get("schema") != LIBRARY_SCHEMA:
            raise ValueError(f"expected a library baseline ({LIBRARY_SCHEMA}), got {doc.get('schema')!r}")
        self.doc, self.min_cell = doc, min_cell
        self.cells: dict[str, dict[str, list[float]]] = {}
        for r in doc["records"]:
            for metric in METRICS:
                v = r.get(metric)
                if v is None:
                    continue
                cells = self.cells.setdefault(metric, {"all": []})
                cells["all"].append(v)
                cells.setdefault(f"shoot:{r['shoot']}", []).append(v)
                for label in r["labels"]:
                    cells.setdefault(label, []).append(v)
                    for flag in r["techniques"]:
                        cells.setdefault(f"{label}+{flag}", []).append(v)
        self.cells = {m: {k: sorted(v) for k, v in c.items()} for m, c in self.cells.items()}

    @property
    def metrics(self) -> list[str]:
        return [m for m in METRICS if m in self.cells]

    @classmethod
    def load(cls, path) -> "LibraryBaseline":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def check(self, maniqa: str | None, tinyclip: str | None, q_realign: str | None = None) -> None:
        """Refuse a baseline scored with another model, prompt or category version: its scores would not compare."""
        from aiculler.categories import VERSION

        d = self.doc
        for what, mine, theirs in (("MANIQA", maniqa, d["maniqa"]), ("TinyCLIP", tinyclip, d["tinyclip"]),
                                   ("Q-ReAlign", q_realign, d.get("q_realign")), ("categories", VERSION, d["categories_version"])):
            if mine is not None and theirs is not None and mine != theirs:
                raise ValueError(f"library baseline was built with {what} {theirs}, this run uses {mine}: rebuild it")

    def size(self, cell: str, metric: str = "maniqa") -> int:
        return len(self.cells.get(metric, {}).get(cell, ()))

    def _pct(self, cell: str, value: float, metric: str) -> float:
        from bisect import bisect_left

        ref = self.cells[metric][cell]
        return bisect_left(ref, value) / len(ref)

    def percentile(self, value: float, labels: list[str], flags: list[str] = (), metric: str = "maniqa") -> tuple[float, str]:
        """The most favourable percentile among the frame's usable cells (its categories; with flags, also category +
        technique), and that cell's name; the whole library when no cell has min_cell frames."""
        cells = [c for c in labels if self.size(c, metric) >= self.min_cell]
        cells += [f"{c}+{f}" for c in labels for f in flags if self.size(f"{c}+{f}", metric) >= self.min_cell]
        if not cells:
            return self._pct("all", value, metric), f"all {self.size('all', metric)} library frames"
        best = max(cells, key=lambda c: self._pct(c, value, metric))
        return self._pct(best, value, metric), f"{self.size(best, metric)} {best.replace('+', ' + ')} frames"

    def shoot_percentile(self, value: float, shoot: str, metric: str = "maniqa") -> float | None:
        cell = f"shoot:{shoot}"
        return self._pct(cell, value, metric) if self.size(cell, metric) >= self.min_cell else None

    def summary(self, metric: str = "maniqa") -> dict[str, int]:
        cells = self.cells.get(metric, {})
        return {k: len(v) for k, v in sorted(cells.items(), key=lambda kv: -len(kv[1])) if not k.startswith("shoot:")}


def build_library(folders: list[Path], models, *, cache_db: Path | None = None, cache_dir: Path | None = None,
                  settings=None, workers: int | None = None, progress=None, qrealign: bool = False) -> dict:
    """Score every competing frame of each shoot with MANIQA and TinyCLIP (TinyCLIP on every readable capture, as
    cull does, so later culls reuse the embeddings) and record maniqa, categories and technique flags; with
    ``qrealign``, also Q-ReAlign quality and aesthetics (and their level probabilities) for every competing frame."""
    from aiculler.api import CullSettings, floor_applies
    from aiculler.categories import VERSION, build_prototypes, classify, techniques
    from aiculler.ingest import PreviewCache, ingest
    from aiculler.routing import route
    from aiculler.scorers import ScoreStore, run_scorers
    from aiculler.scorers.models import AppealScorer, ManiqaScorer, TinyClipScorer
    from aiculler.storage import ScoreCache

    if models is None or not (models.available("maniqa") and models.available("tinyclip")):
        raise ValueError("a library baseline needs MANIQA and TinyCLIP (pass --models; install them with the app's Set Up AI)")
    settings = settings or CullSettings()
    progress = progress or (lambda stage, done, total: None)
    if cache_dir is None:
        cache_dir = Path(str(Path(cache_db).with_suffix("")) + "_previews") if cache_db else Path(".aiculler_cache")
    previews = PreviewCache(cache_dir)
    cache = ScoreCache(cache_db) if cache_db else ScoreCache(":memory:")
    if qrealign and not models.available("q_realign"):
        raise ValueError("Q-ReAlign was asked for but is not available (install it with the app's Set Up AI)")
    maniqa, tinyclip = ManiqaScorer(models), TinyClipScorer(models)
    appeal = AppealScorer(models) if qrealign else None
    records: list[dict] = []
    try:
        protos = build_prototypes(models.engine("tinyclip"), tinyclip.revision)
        for folder in folders:
            shoot = ingest(folder, backend=settings.backend, exclude=settings.exclude, workers=workers)
            routing = route(shoot, settings)
            readable = [c for c, cap in shoot.captures.items() if not cap.error]
            competing = [c for c in routing.scored_captures() if floor_applies(shoot, routing, c) and not shoot.captures[c].error]
            store = ScoreStore()
            run_scorers(store, shoot, readable, [tinyclip], previews, cache, progress, batch_size=8, workers=workers)
            run_scorers(store, shoot, competing, [maniqa], previews, cache, progress, batch_size=8, workers=workers)
            if appeal is not None:
                run_scorers(store, shoot, competing, [appeal], previews, cache, progress, batch_size=8, workers=workers)
            for cid in competing:
                m = store.scores.get(cid, {}).get("maniqa")
                emb = store.details.get(cid, {}).get("embedding")
                if m is None or not emb:
                    continue
                cat = classify(emb, protos)
                rec = {"shoot": str(folder), "capture": cid, "path": shoot.captures[cid].primary, "maniqa": m,
                       "maniqa_spread": store.scores.get(cid, {}).get("maniqa_spread"),
                       "labels": cat["labels"], "techniques": techniques(shoot.captures[cid])}
                values = store.scores.get(cid, {})
                if appeal is not None and values.get("qra_iqa") is not None:
                    rec.update({"qra_iqa": values["qra_iqa"], "appeal": values["appeal"],
                                "qra_probs": store.details.get(cid, {}).get("qra_probs")})
                records.append(rec)
    finally:
        cache.close()
        models.close()
    return {"schema": LIBRARY_SCHEMA, "maniqa": maniqa.revision, "tinyclip": tinyclip.revision,
            "q_realign": appeal.revision if appeal is not None else None,
            "categories_version": VERSION, "shoots": [str(f) for f in folders], "records": records}
