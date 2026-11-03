"""Evaluation harness: agreement with the user's own Lightroom culls.

Ground truth comes from XMP (sidecars, or XMP embedded in JPEG/TIFF/DNG) or from a read-only copy of a
Lightroom catalog (``.lrcat``). A capture counts as **kept** when it is picked, rated >= ``keep_min_rating``
or carries an accept label; everything else (rejected flag, unrated, low-rated) is **not kept**. The
truth of a capture is the union over its files (RAW, JPEG sidecar, virtual copies).

Metrics: precision / recall of ``keep``; per-set agreement (did the engine keep a frame the user kept, in
each burst and each near-duplicate scene); review rate; runtime per stage; the worst disagreements.

Note: Lightroom does not write pick/reject flags to XMP, only ratings and labels. Use the catalog when
your culls are flags.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
DESCRIPTION = f"{{{RDF_NS}}}Description"
RATING = f"{{{XMP_NS}}}Rating"
LABEL = f"{{{XMP_NS}}}Label"
# Same label vocabulary as the image-triage app (image_triage/xmp.py).
ACCEPTED_LABELS = frozenset({"accept", "accepted", "keep", "pick", "picked", "select", "selected", "winner"})
REJECTED_LABELS = frozenset({"reject", "rejected"})
EMBEDDED_XMP_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".dng", ".heic", ".heif", ".hif"}
_XMP_PACKET = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.S)


@dataclass
class Truth:
    rating: int | None = None  # 0-5; -1 = rejected (some tools write it)
    pick: int = 0  # 1 picked, -1 rejected, 0 none (catalog only)
    label: str = ""
    source: str = ""

    def kept(self, keep_min_rating: int) -> bool:
        if self.pick == 1 or self.label.strip().casefold() in ACCEPTED_LABELS:
            return True
        return self.rating is not None and self.rating >= keep_min_rating and self.pick != -1

    @property
    def rejected_flag(self) -> bool:
        return self.pick == -1 or (self.rating is not None and self.rating < 0) or self.label.strip().casefold() in REJECTED_LABELS

    def describe(self) -> str:
        parts = []
        if self.pick == 1:
            parts.append("picked")
        elif self.pick == -1:
            parts.append("rejected flag")
        if self.rating:
            parts.append(f"{self.rating}★" if self.rating > 0 else "rated reject")
        if self.label:
            parts.append(f"label {self.label}")
        return ", ".join(parts) or "unrated"


# --------------------------------------------------------------------------- XMP


def _field(description: ET.Element, name: str) -> str:
    value = description.get(name, "")
    if value:
        return value.strip()
    child = description.find(name)
    return (child.text or "").strip() if child is not None and child.text else ""


def parse_xmp(data: bytes | str) -> Truth | None:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    rating, label = None, ""
    for desc in root.iter(DESCRIPTION):
        r = _field(desc, RATING)
        if r and rating is None:
            try:
                rating = int(round(float(r)))
            except ValueError:
                pass
        label = label or _field(desc, LABEL)
    if rating is None and not label:
        return None
    return Truth(rating=rating, label=label)


def sidecar_candidates(path: str | Path) -> list[Path]:
    """Lightroom writes NAME.xmp next to a RAW; some tools write NAME.EXT.xmp."""
    p = Path(path)
    return [p.with_suffix(".xmp"), p.with_suffix(".XMP"), Path(f"{p}.xmp"), Path(f"{p}.XMP")]


def read_xmp_truth(path: str | Path, embedded_limit: int = 4 << 20) -> Truth | None:
    for cand in sidecar_candidates(path):
        if cand.is_file():
            t = parse_xmp(cand.read_bytes())
            if t is not None:
                t.source = cand.name
                return t
    p = Path(path)
    if p.suffix.lower() in EMBEDDED_XMP_SUFFIXES and p.is_file():
        with open(p, "rb") as fh:
            head = fh.read(embedded_limit)
        m = _XMP_PACKET.search(head)
        if m:
            t = parse_xmp(m.group(0))
            if t is not None:
                t.source = "embedded XMP"
                return t
    return None


def load_xmp_truth(paths: list[str]) -> dict[str, Truth]:
    out = {}
    for p in paths:
        t = read_xmp_truth(p)
        if t is not None:
            out[p] = t
    return out


# --------------------------------------------------------------------------- Lightroom catalog


def _norm(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").casefold()


def read_lrcat(catalog: str | Path) -> list[tuple[str, Truth]]:
    """(catalog path, truth) per file in the catalog, including JPEG sidecars. Opens the catalog read-only."""
    uri = Path(catalog).resolve().as_uri() + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            """
            SELECT i.rating, i.pick, i.colorLabels, rf.absolutePath, fo.pathFromRoot,
                   f.idx_filename, f.baseName, f.sidecarExtensions
            FROM Adobe_images i
            JOIN AgLibraryFile f ON i.rootFile = f.id_local
            JOIN AgLibraryFolder fo ON f.folder = fo.id_local
            JOIN AgLibraryRootFolder rf ON fo.rootFolder = rf.id_local
            """
        ).fetchall()
    finally:
        con.close()
    out = []
    for rating, pick, label, root, sub, filename, base, sidecars in rows:
        folder = (root or "") + (sub or "")
        t = Truth(
            rating=int(rating) if rating is not None else None,
            pick=int(round(float(pick))) if pick is not None else 0,
            label=label or "",
            source="catalog",
        )
        out.append((folder + filename, t))
        for ext in (sidecars or "").split(","):
            ext = ext.strip()
            if ext and ext.casefold() != "xmp":
                out.append((f"{folder}{base}.{ext}", t))
    return out


def match_catalog(catalog_rows: list[tuple[str, Truth]], paths: list[str]) -> dict[str, Truth]:
    """Match catalog files to shoot files: exact path, then parent folder + name, then a unique name."""
    by_path: dict[str, list[Truth]] = defaultdict(list)
    by_parent: dict[tuple[str, str], list[Truth]] = defaultdict(list)
    by_name: dict[str, list[Truth]] = defaultdict(list)
    for cpath, t in catalog_rows:
        n = _norm(cpath)
        parts = n.split("/")
        by_path[n].append(t)
        by_parent[(parts[-2] if len(parts) > 1 else "", parts[-1])].append(t)
        by_name[parts[-1]].append(t)
    out: dict[str, Truth] = {}
    for p in paths:
        n = _norm(str(p))
        parts = n.split("/")
        key = (parts[-2] if len(parts) > 1 else "", parts[-1])
        hits = by_path.get(n) or by_parent.get(key)
        if not hits and len(by_name.get(parts[-1], [])) == 1:  # the name alone, only when unambiguous
            hits = by_name[parts[-1]]
        if hits:
            out[p] = _merge(hits)  # virtual copies: kept if any copy is kept
    return out


def _merge(truths: list[Truth]) -> Truth:
    best = Truth(source=truths[0].source)
    ratings = [t.rating for t in truths if t.rating is not None]
    best.rating = max(ratings) if ratings else None
    picks = [t.pick for t in truths]
    best.pick = 1 if 1 in picks else (-1 if picks and all(p == -1 for p in picks) else 0)
    best.label = next((t.label for t in truths if t.label.strip().casefold() in ACCEPTED_LABELS), truths[0].label)
    return best


# --------------------------------------------------------------------------- metrics


@dataclass
class CaptureRow:
    capture: str
    primary: str
    decision: str
    group: str
    kind: str
    scene: int | None
    score: float | None
    reasons: list[str]
    files: list[str] = field(default_factory=list)
    truth: Truth | None = None
    kept: bool = False


@dataclass
class SetResult:
    id: str
    kind: str  # burst / scene
    size: int
    user_kept: int
    engine_kept: int
    outcome: str  # hit / miss / agree_none / extra


@dataclass
class Evaluation:
    rows: list[CaptureRow]
    sets: list[SetResult]
    keep_min_rating: int
    truth_source: str
    timings: dict[str, float]
    matched: int
    summary: dict


def evaluate(decisions: dict, truth: dict[str, Truth], *, keep_min_rating: int = 3, truth_source: str = "") -> Evaluation:
    """Compare a decisions.json document with per-file truth (any file of a capture)."""
    caps: dict[str, CaptureRow] = {}
    for path, d in decisions["files"].items():
        row = caps.get(d["capture"])
        if row is None:
            row = caps[d["capture"]] = CaptureRow(d["capture"], path, d["decision"], d["group"], d["kind"], d["scene"], d["score"], d["reasons"])
        row.files.append(path)
        if d["primary"]:
            row.primary, row.decision, row.score, row.reasons = path, d["decision"], d["score"], d["reasons"]
    matched = 0
    for row in caps.values():
        ts = [truth[f] for f in row.files if f in truth]
        if ts:
            matched += 1
            row.truth = _merge(ts) if len(ts) > 1 else ts[0]
        row.kept = bool(row.truth and row.truth.kept(keep_min_rating))
    rows = sorted(caps.values(), key=lambda r: r.primary)

    sets: list[SetResult] = []
    by_group: dict[str, list[CaptureRow]] = defaultdict(list)
    by_scene: dict[int, list[CaptureRow]] = defaultdict(list)
    for r in rows:
        by_group[r.group].append(r)
        if r.scene is not None:
            by_scene[r.scene].append(r)
    for gid, members in by_group.items():
        if members[0].kind == "burst" and len(members) > 1:
            sets.append(_set_result(gid, "burst", members))
    for scene, members in by_scene.items():
        if len({m.group for m in members}) > 1:
            sets.append(_set_result(f"scene {scene}", "scene", members))

    summary = _summary(rows, sets)
    return Evaluation(rows, sets, keep_min_rating, truth_source, dict(decisions.get("timings", {})), matched, summary)


def _set_result(sid: str, kind: str, members: list[CaptureRow]) -> SetResult:
    user = {m.capture for m in members if m.kept}
    engine = {m.capture for m in members if m.decision == "keep"}
    if user:
        outcome = "hit" if user & engine else "miss"
    else:
        outcome = "agree_none" if not engine else "extra"
    return SetResult(sid, kind, len(members), len(user), len(engine), outcome)


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def _summary(rows: list[CaptureRow], sets: list[SetResult]) -> dict:
    n = len(rows)
    kept = [r for r in rows if r.kept]
    engine_keep = [r for r in rows if r.decision == "keep"]
    tp = sum(1 for r in engine_keep if r.kept)
    confusion = {d: {"kept": 0, "not_kept": 0} for d in ("keep", "review", "reject")}
    for r in rows:
        confusion[r.decision]["kept" if r.kept else "not_kept"] += 1
    decided = [r for r in rows if r.decision != "review"]
    agree = sum(1 for r in decided if (r.decision == "keep") == r.kept)
    per_kind = {}
    for kind in sorted({r.kind for r in rows}):
        rk = [r for r in rows if r.kind == kind]
        ek = [r for r in rk if r.decision == "keep"]
        uk = [r for r in rk if r.kept]
        per_kind[kind] = {
            "captures": len(rk),
            "user_kept": len(uk),
            "engine_kept": len(ek),
            "precision": _ratio(sum(r.kept for r in ek), len(ek)),
            "recall": _ratio(sum(r.decision == "keep" for r in uk), len(uk)),
            "review": sum(r.decision == "review" for r in rk),
        }
    set_stats = {}
    for kind in ("burst", "scene"):
        ks = [s for s in sets if s.kind == kind]
        with_keep = [s for s in ks if s.user_kept]
        without = [s for s in ks if not s.user_kept]
        set_stats[kind] = {
            "sets": len(ks),
            "user_kept_some": len(with_keep),
            "hit": sum(s.outcome == "hit" for s in with_keep),
            "hit_rate": _ratio(sum(s.outcome == "hit" for s in with_keep), len(with_keep)),
            "user_kept_none": len(without),
            "agree_none": sum(s.outcome == "agree_none" for s in without),
        }
    return {
        "captures": n,
        "user_kept": len(kept),
        "engine_keep": len(engine_keep),
        "engine_review": sum(r.decision == "review" for r in rows),
        "engine_reject": sum(r.decision == "reject" for r in rows),
        "precision": _ratio(tp, len(engine_keep)),
        "recall": _ratio(tp, len(kept)),
        "recall_keep_or_review": _ratio(sum(1 for r in kept if r.decision != "reject"), len(kept)),
        "agreement_decided": _ratio(agree, len(decided)),
        "review_rate": _ratio(sum(r.decision == "review" for r in rows), n),
        "confusion": confusion,
        "per_kind": per_kind,
        "sets": set_stats,
    }


# --------------------------------------------------------------------------- report


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def render_report(ev: Evaluation, *, shoot: str = "", worst: int = 25) -> str:
    s = ev.summary
    lines = [f"# Cull evaluation{f': {shoot}' if shoot else ''}", ""]
    lines += [
        f"Truth: {ev.truth_source or 'n/a'}; kept = picked, accept label, or ≥ {ev.keep_min_rating}★. "
        f"Captures with any truth found: {ev.matched} of {s['captures']}.",
        "",
    ]
    if s["user_kept"] == 0:
        lines += ["> **Warning:** no kept captures found in the truth. Check the truth source (Lightroom writes no pick flags to XMP; use the catalog for flags).", ""]
    lines += [
        "## Headline", "",
        "| Metric | Value |", "|---|---|",
        f"| Captures | {s['captures']} |",
        f"| User kept | {s['user_kept']} |",
        f"| Engine keep / review / reject | {s['engine_keep']} / {s['engine_review']} / {s['engine_reject']} |",
        f"| Precision of keep | {_pct(s['precision'])} |",
        f"| Recall of keep | {_pct(s['recall'])} |",
        f"| Recall, keep or review | {_pct(s['recall_keep_or_review'])} |",
        f"| Agreement on auto-decided captures | {_pct(s['agreement_decided'])} |",
        f"| Review rate | {_pct(s['review_rate'])} |",
        "",
        "## Per-set agreement", "",
        "Did the engine keep a frame the user kept, in each set where the user kept one?", "",
        "| Set kind | Sets | User kept ≥ 1 | Engine kept one of them | Hit rate | User kept none | Engine also kept none |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, st in s["sets"].items():
        lines.append(f"| {kind} | {st['sets']} | {st['user_kept_some']} | {st['hit']} | {_pct(st['hit_rate'])} | {st['user_kept_none']} | {st['agree_none']} |")
    lines += ["", "## Confusion (captures)", "", "| Engine \\ User | kept | not kept |", "|---|---|---|"]
    for d, c in s["confusion"].items():
        lines.append(f"| {d} | {c['kept']} | {c['not_kept']} |")
    lines += ["", "## By group kind", "", "| Kind | Captures | User kept | Engine kept | Precision | Recall | Review |", "|---|---|---|---|---|---|---|"]
    for kind, k in s["per_kind"].items():
        lines.append(f"| {kind} | {k['captures']} | {k['user_kept']} | {k['engine_kept']} | {_pct(k['precision'])} | {_pct(k['recall'])} | {k['review']} |")
    if ev.timings:
        lines += ["", "## Runtime per stage", "", "| Stage | Seconds |", "|---|---|"]
        lines += [f"| {k} | {v:.2f} |" for k, v in ev.timings.items()]
        lines.append(f"| total | {sum(ev.timings.values()):.2f} |")

    missed = [r for r in ev.rows if r.kept and r.decision == "reject"]
    missed.sort(key=lambda r: (-(r.truth.rating or 0) if r.truth else 0, -(r.truth.pick if r.truth else 0), r.primary))
    extra = [r for r in ev.rows if not r.kept and r.decision == "keep"]
    extra.sort(key=lambda r: (not (r.truth and r.truth.rejected_flag), -(r.score or 0), r.primary))
    lines += _disagreements("User kept, engine rejected", missed, worst)
    lines += _disagreements("Engine kept, user did not", extra, worst)
    lines.append("")
    return "\n".join(lines)


def _disagreements(title: str, rows: list[CaptureRow], limit: int) -> list[str]:
    out = ["", f"## {title} ({len(rows)})", ""]
    if not rows:
        return out + ["None."]
    out += ["| File | Engine | User | Group | Reasons |", "|---|---|---|---|---|"]
    for r in rows[:limit]:
        user = r.truth.describe() if r.truth else "no truth (unrated)"
        reasons = "; ".join(r.reasons).replace("|", "\\|")
        out.append(f"| {Path(r.primary).name} | {r.decision} | {user} | {r.group} ({r.kind}{f', scene {r.scene}' if r.scene is not None else ''}) | {reasons} |")
    if len(rows) > limit:
        out.append(f"\n…and {len(rows) - limit} more.")
    return out


def load_truth(spec: str | Path, paths: list[str]) -> tuple[dict[str, Truth], str]:
    """``spec`` is "xmp" or a path to a .lrcat."""
    if str(spec).casefold() == "xmp":
        return load_xmp_truth(paths), "XMP (sidecars and embedded)"
    p = Path(spec)
    if p.suffix.casefold() != ".lrcat" or not p.is_file():
        raise ValueError(f"--truth must be 'xmp' or an existing .lrcat file; got {spec}")
    return match_catalog(read_lrcat(p), paths), f"Lightroom catalog {p.name} (read-only)"


__all__ = [
    "Evaluation",
    "Truth",
    "evaluate",
    "load_truth",
    "load_xmp_truth",
    "match_catalog",
    "parse_xmp",
    "read_lrcat",
    "render_report",
]
