"""Evaluation harness: agreement between ``decisions.json`` and the user's own culls.

Truth sources live in ``aiculler.truth`` (picks, csv, triage, xmp, lrcat). Metrics are per capture (the
truth of a capture is the union over its files): recall and, for full culls, precision of ``keep``;
per-set agreement (did the engine keep a frame the user kept, in each burst and near-duplicate scene);
review rate; runtime per stage; the worst disagreements.

In ``picks`` mode (final picks only) precision and agreement are reported as not meaningful: frames the
user kept but never exported count as "not picked".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from aiculler.truth import (  # noqa: F401  (re-exported for callers of the old API)
    Truth,
    TruthSet,
    load_truth,
    load_xmp_truth,
    match_catalog,
    merge_truths,
    parse_xmp,
    read_lrcat,
    read_xmp_truth,
)

NOT_MEANINGFUL = "not meaningful for picks"


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
    excluded: str | None = None  # why it is left out of the metrics
    gates: list[str] = field(default_factory=list)  # gates the capture failed (decisions.json)
    scores: dict[str, float] = field(default_factory=dict)
    floor: str | None = None  # what the quality floor did (decisions.json "floor")


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
    rows: list[CaptureRow]  # every capture, excluded ones included (``excluded`` set)
    sets: list[SetResult]
    keep_min_rating: int
    truth_source: str
    mode: str
    unlabelled: str
    notes: list[str]
    timings: dict[str, float]
    matched: int
    summary: dict

    @property
    def scored(self) -> list[CaptureRow]:
        return [r for r in self.rows if r.excluded is None]


def evaluate(
    decisions: dict,
    truth: TruthSet | dict[str, Truth],
    *,
    keep_min_rating: int = 3,
    truth_source: str = "",
) -> Evaluation:
    """Compare a decisions.json document with a truth set (per-file truth; any file of a capture counts)."""
    if not isinstance(truth, TruthSet):
        truth = TruthSet(dict(truth), truth_source or "truth")
    caps: dict[str, CaptureRow] = {}
    for path, d in decisions["files"].items():
        row = caps.get(d["capture"])
        if row is None:
            row = caps[d["capture"]] = CaptureRow(d["capture"], path, d["decision"], d["group"], d["kind"], d["scene"], d["score"], d["reasons"])
        row.files.append(path)
        if d["primary"]:
            row.primary, row.decision, row.score, row.reasons = path, d["decision"], d["score"], d["reasons"]
            row.gates, row.scores = list(d.get("gates") or []), dict(d.get("scores") or {})
            row.floor = d.get("floor")
    matched = 0
    for row in caps.values():
        ts = [truth.truth[f] for f in row.files if f in truth.truth]
        if ts:
            matched += 1
            row.truth = merge_truths(ts)
        row.kept = bool(row.truth and row.truth.kept(keep_min_rating))
        if row.capture in truth.excluded:
            row.excluded = truth.excluded[row.capture]
        elif truth.unlabelled == "exclude" and not (row.truth and row.truth.labelled(keep_min_rating)):
            row.excluded = "no label"
    rows = sorted(caps.values(), key=lambda r: r.primary)
    scored = [r for r in rows if r.excluded is None]

    sets: list[SetResult] = []
    by_group: dict[str, list[CaptureRow]] = defaultdict(list)
    by_scene: dict[int, list[CaptureRow]] = defaultdict(list)
    for r in scored:
        by_group[r.group].append(r)
        if r.scene is not None:
            by_scene[r.scene].append(r)
    for gid, members in by_group.items():
        if members[0].kind == "burst" and len(members) > 1:
            sets.append(_set_result(gid, "burst", members))
    for scene, members in by_scene.items():
        if len({m.group for m in members}) > 1:
            sets.append(_set_result(f"scene {scene}", "scene", members))

    summary = _summary(scored, sets, truth.mode)
    summary["excluded"] = len(rows) - len(scored)
    summary["gates_vs_kept"] = gate_margins(scored, decisions.get("settings", {}).get("gates", {}))
    summary["floor"] = floor_vs_truth(scored, decisions.get("settings", {}), truth.mode, every=rows)
    return Evaluation(
        rows, sets, keep_min_rating, truth.source, truth.mode, truth.unlabelled, list(truth.notes),
        dict(decisions.get("timings", {})), matched, summary,
    )


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


def _summary(rows: list[CaptureRow], sets: list[SetResult], mode: str) -> dict:
    picks = mode == "picks"
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
            "precision": None if picks else _ratio(sum(r.kept for r in ek), len(ek)),
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
            "agree_none": None if picks else sum(s.outcome == "agree_none" for s in without),
        }
    return {
        "mode": mode,
        "captures": n,
        "user_kept": len(kept),
        "engine_keep": len(engine_keep),
        "engine_review": sum(r.decision == "review" for r in rows),
        "engine_reject": sum(r.decision == "reject" for r in rows),
        "engine_keep_rate": _ratio(len(engine_keep), n),
        "precision": None if picks else _ratio(tp, len(engine_keep)),
        "recall": _ratio(tp, len(kept)),
        "recall_keep_or_review": _ratio(sum(1 for r in kept if r.decision != "reject"), len(kept)),
        "agreement_decided": None if picks else _ratio(agree, len(decided)),
        "review_rate": _ratio(sum(r.decision == "review" for r in rows), n),
        "confusion": confusion,
        "per_kind": per_kind,
        "sets": set_stats,
    }


# --------------------------------------------------------------------------- gates vs the user's keeps

# (gate, metric, direction): "max" = rejects at or above the threshold, "min" = below it.
GATE_METRICS = [
    ("clip_region", "clip_region", "clip_region_reject", "max"),
    ("sky_blown", "top_clip", "sky_blown", "max"),
    ("shadows", "shadows_crushed", "shadows_crushed", "max"),
    ("black", "preview_mean", "black_mean", "min"),
]


def floor_vs_truth(rows: list[CaptureRow], settings: dict, mode: str, every: list[CaptureRow] | None = None) -> dict:
    """What the quality floor did to your keeps (lost: rejected; reviewed: sent to review) and, when the truth has
    real rejects (not picks), how many of them it caught. ``acted`` counts the whole shoot (``every``), labelled or
    not. Empty when the floor was off."""
    g = settings.get("gates") or {}
    on = any(float(g.get(k) or 0) > 0 for k in ("quality_reject", "quality_review", "qra_reject", "qra_review"))
    if not on and not any(r.floor for r in rows):
        return {}
    kept = [r for r in rows if r.kept]
    others = [r for r in rows if not r.kept]
    out = {
        "signal": g.get("quality_signal", "maniqa"), "mode": g.get("quality_mode", "raw"), "unique": g.get("quality_unique", "review"),
        "levels": {k: g.get(k) for k in ("quality_reject", "quality_review", "qra_reject", "qra_review") if g.get(k)},
        "acted": {v: sum(1 for r in (every or rows) if r.floor == v) for v in ("reject", "review", "review_unique")},
        "keeps": len(kept),
        "keeps_lost": [Path(r.primary).name for r in kept if r.floor == "reject"],
        "keeps_reviewed": [Path(r.primary).name for r in kept if r.floor in ("review", "review_unique")],
        "others": len(others),
        "others_rejected": sum(1 for r in others if r.floor == "reject"),
        "others_reviewed": sum(1 for r in others if r.floor in ("review", "review_unique")),
        "real_rejects": mode != "picks",
    }
    return out


def gate_margins(rows: list[CaptureRow], gates: dict) -> dict:
    """For the frames the user kept: which a gate rejects, and how close the others come to each threshold.

    Only frames the gates look at (bursts and singles; never bracket frames) and that have the measurement.
    """
    kept = [r for r in rows if r.kept and r.kind != "exposure"]
    out: dict = {"kept_checked": 0, "gated": [], "metrics": {}}
    measured = [r for r in kept if r.scores]
    out["kept_checked"] = len(measured)
    out["gated"] = [
        {"file": Path(r.primary).name, "decision": r.decision, "gates": r.gates,
         **{k: r.scores.get(k) for k in ("clip_region", "top_clip", "luma_mean", "luma_spread", "preview_mean", "shadows_crushed")}}
        for r in measured if r.gates
    ]
    for gate, metric, setting, direction in GATE_METRICS:
        vals = [(r.scores[metric], r) for r in measured if r.scores.get(metric) is not None]
        threshold = gates.get(setting)
        if not vals or threshold is None:
            continue
        value, r = max(vals, key=lambda x: x[0]) if direction == "max" else min(vals, key=lambda x: x[0])
        passes = value < threshold if direction == "max" else value >= threshold
        out["metrics"][gate] = {"setting": setting, "threshold": threshold, "worst": value, "file": Path(r.primary).name, "passes": passes}
    if gates.get("washed_mean") is not None and gates.get("washed_spread") is not None:
        vals = [r for r in measured if r.scores.get("luma_mean") is not None and r.scores.get("luma_spread") is not None]
        if vals:
            # Washed out needs both: margin = how far the frame is from meeting either condition.
            def margin(r: CaptureRow) -> float:
                return max(gates["washed_mean"] - r.scores["luma_mean"], r.scores["luma_spread"] - gates["washed_spread"])

            r = min(vals, key=margin)
            out["metrics"]["washed_out"] = {
                "setting": "washed_mean / washed_spread",
                "threshold": f"{gates['washed_mean']:g} / {gates['washed_spread']:g}",
                "worst": f"{r.scores['luma_mean']:.2f} / {r.scores['luma_spread']:.2f}",
                "file": Path(r.primary).name,
                "passes": margin(r) > 0,
            }
    return out


# --------------------------------------------------------------------------- report


def _pct(x: float | None, picks_na: bool = False) -> str:
    if x is None:
        return NOT_MEANINGFUL if picks_na else "n/a"
    return f"{x:.1%}"


def render_report(ev: Evaluation, *, shoot: str = "", worst: int = 25) -> str:
    s = ev.summary
    picks = ev.mode == "picks"
    kept_word = "picked" if picks else "kept"
    lines = [f"# Cull evaluation{f': {shoot}' if shoot else ''}", ""]
    if picks:
        definition = "kept = a copy or edited export of the capture exists in another folder (final picks)"
    elif "ImageTriage" in ev.truth_source or "CSV" in ev.truth_source:
        definition = f"kept = Keep / Winner label (or ≥ {ev.keep_min_rating}★); rejected = Reject label"
    else:
        definition = f"kept = picked, accept label, or ≥ {ev.keep_min_rating}★"
    unl = "count as not kept" if ev.unlabelled == "not_kept" else "are left out"
    lines += [f"Truth: {ev.truth_source}; {definition}. Captures without a label {unl}.", ""]
    for note in ev.notes:
        lines.append(f"- {note}")
    lines.append(f"- captures in the metrics: {s['captures']} of {len(ev.rows)} ({s['excluded']} left out)")
    lines.append("")
    if picks:
        lines += [
            "> Picks are a subset of what you kept: a frame you kept but never exported counts as not picked. "
            "So precision is not meaningful; read recall and per-set agreement, next to the engine's keep rate.",
            "",
        ]
    if s["user_kept"] == 0:
        lines += [f"> **Warning:** no {kept_word} captures found in the truth. Check the truth source.", ""]
    lines += [
        "## Headline", "",
        "| Metric | Value |", "|---|---|",
        f"| Captures | {s['captures']} |",
        f"| User {kept_word} | {s['user_kept']} |",
        f"| Engine keep / review / reject | {s['engine_keep']} / {s['engine_review']} / {s['engine_reject']} |",
        f"| Engine keep rate | {_pct(s['engine_keep_rate'])} |",
        f"| Recall of {'picks' if picks else 'keep'} | {_pct(s['recall'])} |",
        f"| Recall, keep or review | {_pct(s['recall_keep_or_review'])} |",
        f"| Precision of keep | {_pct(s['precision'], picks)} |",
        f"| Agreement on auto-decided captures | {_pct(s['agreement_decided'], picks)} |",
        f"| Review rate | {_pct(s['review_rate'])} |",
        "",
        "## Per-set agreement", "",
        f"Did the engine keep a frame the user {kept_word}, in each set where the user {kept_word} one?", "",
        f"| Set kind | Sets | User {kept_word} ≥ 1 | Engine kept one of them | Hit rate | User {kept_word} none | Engine also kept none |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, st in s["sets"].items():
        none = NOT_MEANINGFUL if st["agree_none"] is None else st["agree_none"]
        lines.append(f"| {kind} | {st['sets']} | {st['user_kept_some']} | {st['hit']} | {_pct(st['hit_rate'])} | {st['user_kept_none']} | {none} |")
    lines += ["", "## Confusion (captures)", "", f"| Engine \\ User | {kept_word} | not {kept_word} |", "|---|---|---|"]
    for d, c in s["confusion"].items():
        lines.append(f"| {d} | {c['kept']} | {c['not_kept']} |")
    lines += [
        "", "## By group kind", "",
        f"| Kind | Captures | User {kept_word} | Engine kept | Precision | Recall | Review |", "|---|---|---|---|---|---|---|",
    ]
    for kind, k in s["per_kind"].items():
        lines.append(
            f"| {kind} | {k['captures']} | {k['user_kept']} | {k['engine_kept']} | {_pct(k['precision'], picks)} | {_pct(k['recall'])} | {k['review']} |"
        )
    lines += _gate_section(s.get("gates_vs_kept") or {}, kept_word)
    lines += _floor_section(s.get("floor") or {}, kept_word)
    if ev.timings:
        lines += ["", "## Runtime per stage", "", "| Stage | Seconds |", "|---|---|"]
        lines += [f"| {k} | {v:.2f} |" for k, v in ev.timings.items()]
        lines.append(f"| total | {sum(ev.timings.values()):.2f} |")

    scored = ev.scored
    missed = [r for r in scored if r.kept and r.decision == "reject"]
    missed.sort(key=lambda r: (-(r.truth.rating or 0) if r.truth else 0, -(r.truth.pick if r.truth else 0), r.primary))
    lines += _disagreements(f"User {kept_word}, engine rejected", missed, worst)
    missed_sets = [x for x in ev.sets if x.outcome == "miss"]
    if missed_sets:
        lines += ["", f"## Sets where the engine kept none of the {kept_word} frames ({len(missed_sets)})", ""]
        lines += ["| Set | Kind | Size | User " + kept_word + " | Engine kept |", "|---|---|---|---|---|"]
        lines += [f"| {x.id} | {x.kind} | {x.size} | {x.user_kept} | {x.engine_kept} |" for x in missed_sets[:worst]]
    extra = [r for r in scored if not r.kept and r.decision == "keep"]
    extra.sort(key=lambda r: (not (r.truth and r.truth.rejected_flag), -(r.score or 0), r.primary))
    title = "Engine kept, not picked (not necessarily wrong)" if picks else "Engine kept, user did not"
    lines += _disagreements(title, extra, worst)
    left_out = [r for r in ev.rows if r.excluded and r.excluded != "no label"]
    if left_out:
        lines += ["", f"## Left out of the metrics ({len(left_out)})", "", "| File | Why |", "|---|---|"]
        lines += [f"| {Path(r.primary).name} | {r.excluded} |" for r in left_out[:worst]]
        if len(left_out) > worst:
            lines.append(f"\n…and {len(left_out) - worst} more.")
    lines.append("")
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "–"
    return f"{v:.0%}" if isinstance(v, float) and v <= 1 else str(v)


def _gate_section(g: dict, kept_word: str) -> list[str]:
    if not g:
        return []
    out = ["", f"## Gates vs your {kept_word} frames", ""]
    gated = g.get("gated", [])
    out.append(f"{kept_word.capitalize()} frames checked: {g.get('kept_checked', 0)} (bursts and singles; bracket frames are never gated). "
               f"Rejected by a gate: **{len(gated)}**.")
    if gated:
        out += ["", "| File | Engine | Gates | Largest region | Top third | Mean | Spread |", "|---|---|---|---|---|---|---|"]
        for x in gated:
            out.append(f"| {x['file']} | {x['decision']} | {', '.join(x['gates'])} | {_fmt(x['clip_region'])} | {_fmt(x['top_clip'])} | "
                       f"{_fmt(x['luma_mean'])} | {_fmt(x['luma_spread'])} |")
    metrics = g.get("metrics", {})
    if metrics:
        out += ["", f"Closest {kept_word} frame to each threshold:", "", "| Gate | Setting | Threshold | Closest frame | Value | Passes |", "|---|---|---|---|---|---|"]
        for gate, m in metrics.items():
            out.append(f"| {gate} | `{m['setting']}` | {_fmt(m['threshold'])} | {m['file']} | {_fmt(m['worst'])} | {'yes' if m['passes'] else '**no**'} |")
    return out


def _floor_section(f: dict, kept_word: str) -> list[str]:
    if not f:
        return []
    levels = ", ".join(f"`{k}` {v}" for k, v in f["levels"].items()) or "none set"
    out = ["", f"## Quality floor vs your {kept_word} frames", "",
           f"Signal `{f['signal']}`, mode `{f['mode']}`, a moment's best frame: `{f['unique']}`; levels: {levels}. "
           f"Across the whole shoot the floor rejected {f['acted']['reject']}, sent {f['acted']['review']} to review (low quality "
           f"or the models disagree) and {f['acted']['review_unique']} more as their moment's best frame.", "",
           f"- **Your {kept_word} frames the floor rejected: {len(f['keeps_lost'])} of {f['keeps']}**" +
           (f" ({', '.join(f['keeps_lost'][:20])})" if f["keeps_lost"] else " (must be 0 before letting it reject)"),
           f"- Your {kept_word} frames it sent to review: {len(f['keeps_reviewed'])}" +
           (f" ({', '.join(f['keeps_reviewed'][:20])})" if f["keeps_reviewed"] else "")]
    other = "frames you rejected" if f["real_rejects"] else "frames you did not pick (not necessarily bad)"
    out.append(f"- Of the {f['others']} {other}: {f['others_rejected']} rejected by the floor, {f['others_reviewed']} sent to review")
    return out


def _disagreements(title: str, rows: list[CaptureRow], limit: int) -> list[str]:
    out = ["", f"## {title} ({len(rows)})", ""]
    if not rows:
        return out + ["None."]
    out += ["| File | Engine | User | Group | Reasons |", "|---|---|---|---|---|"]
    for r in rows[:limit]:
        user = r.truth.describe() if r.truth else "no label"
        reasons = "; ".join(r.reasons).replace("|", "\\|")
        out.append(f"| {Path(r.primary).name} | {r.decision} | {user} | {r.group} ({r.kind}{f', scene {r.scene}' if r.scene is not None else ''}) | {reasons} |")
    if len(rows) > limit:
        out.append(f"\n…and {len(rows) - limit} more.")
    return out


__all__ = ["Evaluation", "Truth", "TruthSet", "evaluate", "load_truth", "render_report"]
