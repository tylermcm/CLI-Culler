"""A labelled baseline for the quality floor: ``ai-culler sample`` and ``ai-culler calibrate``.

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


@dataclass
class SampleRow:
    id: str  # "001".."NNN": the thumbnail's name and the labels' key
    folder: str
    path: str  # primary file
    capture: str
    kind: str  # group kind (single, burst, ...)
    scores: dict[str, float] = field(default_factory=dict)


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
) -> dict:
    """Draw, score and write the labelling page. Returns the sample document (also ``out/sample.json``)."""
    from aiculler.api import CullSettings, floor_applies
    from aiculler.ingest import PreviewCache, ingest
    from aiculler.routing import route
    from aiculler.scorers import ScoreStore, default_scorers, run_scorers
    from aiculler.scorers.models import ManiqaScorer
    from aiculler.storage import ScoreCache

    settings = settings or CullSettings()
    progress = progress or (lambda stage, done, total: None)
    if cache_dir is None:
        cache_dir = Path(str(Path(cache_db).with_suffix("")) + "_previews") if cache_db else Path(".aiculler_cache")
    previews = PreviewCache(cache_dir)
    cache = ScoreCache(cache_db) if cache_db else ScoreCache(":memory:")
    shoots, candidates = {}, {}
    for folder in folders:
        shoot = ingest(folder, backend=settings.backend, exclude=settings.exclude, workers=workers)
        routing = route(shoot, settings)
        key = str(folder)
        shoots[key] = shoot
        candidates[key] = [c for c in routing.scored_captures() if floor_applies(shoot, routing, c) and not shoot.captures[c].error]
    picked = draw(candidates, n, seed)
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
            for cid in ids:
                cap = shoot.captures[cid]
                rows.append(SampleRow("", key, cap.primary, cid, shoot.group_of(cid).kind, dict(store.scores.get(cid, {}))))
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
.frame[data-label=keep] { border-color:var(--keep); } .frame[data-label=borderline] { border-color:var(--border); } .frame[data-label=reject] { border-color:var(--reject); }
button.on.keep { background:var(--keep); color:var(--bg); } button.on.borderline { background:var(--border); color:var(--bg); } button.on.reject { background:var(--reject); color:var(--bg); }
</style></head>
<body>
<header><h1>Quality baseline</h1><span id="count" class="muted"></span>
<span class="muted">Keys: 1 keep, 2 borderline, 3 reject, ←/→ move</span>
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
      ${["keep","borderline","reject"].map(l => `<button class="${l}${labels[it.id] === l ? " on" : ""}" data-l="${l}">${l}</button>`).join("")}</div>`;
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
  const m = {"1": "keep", "2": "borderline", "3": "reject"}[e.key];
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
            if label not in LABELS:
                raise ValueError(f"{path}: label must be one of {', '.join(LABELS)}; got {label!r} for id {row.get('id')}")
            out[str(row["id"]).strip()] = label
    return out


def calibrate(sample: dict, labels: dict[str, str], *, metric: str = "maniqa", margin: float = 0.02,
              review_share: float = 0.10) -> dict:
    """Threshold sweep and suggested floor levels from labelled sample rows."""
    rows = [r for r in sample["rows"] if r["id"] in labels and r["scores"].get(metric) is not None]
    by = {lab: sorted(r["scores"][metric] for r in rows if labels[r["id"]] == lab) for lab in LABELS}
    if not by["keep"]:
        raise ValueError(f"no frame labelled keep has a {metric} score: nothing to calibrate against")
    values = sorted(v for vs in by.values() for v in vs)
    lo, hi = values[0], values[-1]
    step = 0.01
    sweep = []
    t = round(int(lo / step) * step, 2)
    while t <= hi + step:
        sweep.append({"threshold": round(t, 2), **{lab: sum(1 for v in by[lab] if v < t) for lab in LABELS}})
        t += step
    protected = by["keep"] + by["borderline"]
    reject_at = round(max(0.0, min(protected) - margin), 3)
    keeps = by["keep"]
    allowed = int(review_share * len(keeps))  # keeps that may land in review
    review_at = round(keeps[allowed] if allowed < len(keeps) else keeps[-1], 3)
    review_at = max(review_at, reject_at)

    def caught(t: float) -> dict[str, int]:
        return {lab: sum(1 for v in by[lab] if v < t) for lab in LABELS}

    def stats(vs: list[float]) -> dict:
        if not vs:
            return {"n": 0}
        n = len(vs)
        return {"n": n, "min": vs[0], "p10": vs[n // 10], "median": vs[n // 2], "p90": vs[min(n - 1, (9 * n) // 10)], "max": vs[-1]}

    return {
        "metric": metric,
        "labelled": len(rows),
        "unscored": sum(1 for r in sample["rows"] if r["id"] in labels and r["scores"].get(metric) is None),
        "distribution": {lab: stats(by[lab]) for lab in LABELS},
        "suggested": {"quality_reject": reject_at, "quality_review": review_at},
        "caught": {"quality_reject": caught(reject_at), "quality_review": caught(review_at)},
        "rules": {"margin": margin, "review_share": review_share},
        "sweep": sweep,
    }


def render_calibration(cal: dict) -> str:
    d, s, c = cal["distribution"], cal["suggested"], cal["caught"]
    lines = [f"# Quality floor calibration ({cal['metric']})", "",
             f"{cal['labelled']} labelled frames with a score" + (f"; {cal['unscored']} labelled but unscored" if cal["unscored"] else "") + ".", "",
             "| Label | n | min | p10 | median | p90 | max |", "|---|---|---|---|---|---|---|"]
    for lab in LABELS:
        x = d[lab]
        lines.append(f"| {lab} | {x['n']} | " + (" | ".join(f"{x[k]:.3f}" for k in ("min", "p10", "median", "p90", "max")) if x["n"] else "– | – | – | – | –") + " |")
    r = cal["rules"]
    lines += ["", "## Suggested", "",
              f"- `quality_reject`: **{s['quality_reject']}** (lowest keep or borderline frame minus {r['margin']}). "
              f"Below it: {c['quality_reject']['reject']} reject, {c['quality_reject']['borderline']} borderline, {c['quality_reject']['keep']} keep.",
              f"- `quality_review`: **{s['quality_review']}** (at most {r['review_share']:.0%} of keeps below it). "
              f"Below it: {c['quality_review']['reject']} reject, {c['quality_review']['borderline']} borderline, {c['quality_review']['keep']} keep.",
              "", "```json", json.dumps({"gates": s}, indent=2), "```", "",
              "## Sweep: frames of each label below each threshold", "",
              "| Threshold | keep | borderline | reject |", "|---|---|---|---|"]
    for row in cal["sweep"]:
        lines.append(f"| {row['threshold']:.2f} | {row['keep']} | {row['borderline']} | {row['reject']} |")
    return "\n".join(lines) + "\n"
