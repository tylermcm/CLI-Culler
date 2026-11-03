"""Decide once: per-set normalisation, combined score, keep counts, review bucket, reasons.

Order of operations (each capture gets exactly one decision):

1. Unreadable captures -> review. Passthrough captures (left out of culling) -> keep.
2. Bursts: frames ranked against each other. Gated frames -> reject; beyond ``bursts_keep`` -> reject;
   the rest follow the burst's outcome in its scene. The best frame represents the burst.
3. Scenes: units ranked against each other on their representative frame; the top ``scene_keep`` win.
   If every unit in a scene failed a gate, the least-bad one is kept for review (never a black frame).
4. Review overrides: groups with confidence < ``review_below`` or a bracketeering warning that doubts the
   grouping (``review_warnings``), and frames flagged by a gate (featureless), go to review instead of an
   automatic keep/reject. Gate rejections stay. Other warnings are noted in the reasons.

Scores are normalised within the comparison set they compete in, never across the shoot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from aiculler.gates import GateResults, clip_penalty
from aiculler.ingest import Shoot
from aiculler.routing import ComparisonSet, Routing, Unit

if TYPE_CHECKING:
    from aiculler.api import CullSettings, FileDecision

# "relative" normalisation: distance from the best candidate in the set, on an absolute scale, so two
# near-identical frames get near-identical scores (rank normalisation would spread them to 0 and 1).
RELATIVE_SCALE = {
    "sharpness": ("log", math.log(4.0)),  # a quarter of the best frame's Laplacian variance -> 0
    "face_sharpness": ("log", math.log(4.0)),
    "exposure": ("linear", 0.5),  # half a quality point below the best -> 0
    "maniqa": ("linear", 0.2),  # KonIQ-scale quality (0..1): 0.2 below the best -> 0
    "appeal": ("linear", 0.25),  # Q-ReAlign level (0..1): one level (good -> fair) below the best -> 0
    "preference": ("linear", 0.5),  # probability the user keeps it
}
SHORTLIST_METRICS = frozenset({"appeal", "preference"})
# Bracketeering warnings that describe a group without doubting it: a bracket missing slots, frames that don't
# register (a moving subject: see moving_brackets), an auto-bracket frame whose partners were recomposed or deleted.
# Any other warning (a weak bracket candidate split into singles, a flagged auto-bracket frame with no partners
# found, or one this version does not know) doubts the grouping and sends the group to review.
INFORMATIONAL_WARNINGS = ("partial bracket", "possibly incomplete", "frames don't register", "auto-bracket frame")


def is_informational(warning: str) -> bool:
    return warning.startswith(INFORMATIONAL_WARNINGS)


MISSING = 0.5  # a candidate without a value (no preview) is scored as the set's middle


def normalise(values: dict[str, float | None], metric: str, method: str = "relative") -> dict[str, float]:
    present = {k: float(v) for k, v in values.items() if v is not None}
    if not present:
        return {k: MISSING for k in values}
    out: dict[str, float] = {}
    if method == "rank":
        ordered = sorted(present.values())
        n = len(ordered)
        for k, v in present.items():
            lo = ordered.index(v)
            hi = n - 1 - ordered[::-1].index(v)
            out[k] = 1.0 if n == 1 else ((lo + hi) / 2) / (n - 1)
    elif method == "zscore":
        vals = list(present.values())
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        for k, v in present.items():
            out[k] = 1.0 if std == 0 else min(1.0, max(0.0, 0.5 + (v - mean) / std / 4))
    else:
        mode, span = RELATIVE_SCALE.get(metric, ("linear", 1.0))
        tx = {k: (math.log1p(max(0.0, v)) if mode == "log" else v) for k, v in present.items()}
        best = max(tx.values())
        for k, v in tx.items():
            out[k] = min(1.0, max(0.0, 1.0 - (best - v) / span))
    for k in values:
        out.setdefault(k, MISSING)
    return {k: round(v, 4) for k, v in out.items()}


def combine(
    captures: list[str],
    scores: dict[str, dict[str, float]],
    weights: dict[str, float],
    method: str,
    penalties: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Combined score per capture within one set (minus penalties), plus each metric's set-normalised value.

    Only the signals at least one candidate in the set has take part, and their weights are renormalised, so
    a set scored without models is decided by the cheap signals on the same 0..1 scale. A candidate missing a
    signal the others have gets the set's middle (``MISSING``), except for the late-stage signals (appeal,
    preference): those run only on a set's shortlist, so a candidate without one was pruned and counts as worst.
    """
    rel: dict[str, dict[str, float]] = {c: {} for c in captures}
    active = [m for m in weights if any(scores.get(c, {}).get(m) is not None for c in captures)]
    for metric in active:
        values = {c: scores.get(c, {}).get(metric) for c in captures}
        normed = normalise(values, metric, method)
        for c in captures:
            rel[c][metric] = 0.0 if (values[c] is None and metric in SHORTLIST_METRICS) else normed[c]
    total = sum(weights[m] for m in active)
    pen = penalties or {}
    combined = {}
    for c in captures:
        base = sum(weights[m] * rel[c][m] for m in active) / total if total > 0 else 0.5
        combined[c] = round(max(0.0, base - pen.get(c, 0.0)), 4)
    return combined, rel


@dataclass
class _State:
    decision: str | None = None
    reasons: list[str] = field(default_factory=list)
    comparison_set: str | None = None
    rank: int | None = None
    size: int | None = None
    score: float | None = None
    rel: dict[str, float] = field(default_factory=dict)
    gate_rejected: bool = False

    def set(self, decision: str, reason: str | None = None) -> None:
        self.decision = decision
        if reason:
            self.reasons.append(reason)


def penalty_items(shoot: Shoot, scores: dict[str, dict[str, float]], settings: "CullSettings") -> dict[str, list[tuple[str, float, str]]]:
    """Score penalties per capture: (name, amount, reason). Never on bracket frames."""
    out: dict[str, list[tuple[str, float, str]]] = {}
    for cid, values in scores.items():
        if cid not in shoot.captures or shoot.group_of(cid).kind == "exposure":
            continue
        its = []
        pen = clip_penalty(values, settings)
        if pen > 0:
            its.append(("clip_penalty", pen, f"one clipped region covers {values['clip_region']:.0%} of the frame"))
        closed = values.get("eyes_closed")
        if settings.require_eyes_open and closed is not None and closed >= settings.blink_threshold and settings.blink_penalty > 0:
            who = "someone's eyes are closed" if values.get("face_count", 1) > 1 else "eyes closed"
            its.append(("blink_penalty", settings.blink_penalty, f"{who} ({closed:.2f})"))
        if its:
            out[cid] = its
    return out


def _why_better(winner: str, loser: str, rel: dict[str, dict[str, float]], pnames: dict[str, set[str]] | None = None) -> str:
    """Why ``winner`` beat ``loser``, as a predicate: "is sharper and more appealing, and has eyes open"."""
    pnames = pnames or {}
    is_: list[str] = []
    has: list[str] = []
    if "blink_penalty" in pnames.get(loser, set()) and "blink_penalty" not in pnames.get(winner, set()):
        has.append("eyes open")

    def ahead(metric: str) -> bool:
        return rel[winner].get(metric, 0) - rel[loser].get(metric, 0) >= 0.05

    if ahead("appeal"):
        is_.append("more appealing")
    if ahead("maniqa"):
        is_.append("higher quality (MANIQA)")
    if ahead("face_sharpness"):
        has.append("a sharper face")
    if ahead("sharpness"):
        is_.append("sharper")
    if ahead("exposure"):
        is_.append("better exposed")
    if ahead("preference"):
        is_.append("more like your past keeps")
    parts = []
    if is_:
        parts.append("is " + " and ".join(is_))
    if has:
        parts.append("has " + " and ".join(has))
    return ", and ".join(parts) if parts else "has a higher combined score"


def _name(shoot: Shoot, capture: str) -> str:
    return Path(shoot.captures[capture].primary).name


def _unit_label(shoot: Shoot, unit: Unit) -> str:
    if unit.kind == "single":
        return _name(shoot, unit.rep) if unit.rep else unit.id
    label = {"exposure": "bracket", "burst": "burst", "focus": "focus stack", "timelapse": "timelapse"}[unit.kind]
    return f"{label} {unit.id}"


def decide(
    shoot: Shoot,
    routing: Routing,
    gates: GateResults,
    scores: dict[str, dict[str, float]],
    settings: "CullSettings",
) -> dict[str, "FileDecision"]:
    from aiculler.api import FileDecision

    weights = {k: float(v) for k, v in settings.weights.items() if float(v) > 0}
    method = settings.normalise
    st: dict[str, _State] = {cid: _State() for cid in shoot.captures}
    items = penalty_items(shoot, scores, settings)
    penalties = {cid: round(sum(x[1] for x in its), 4) for cid, its in items.items()}
    pnames = {cid: {x[0] for x in its} for cid, its in items.items()}

    # 1. Unreadable and passthrough.
    for cid, err in routing.unreadable.items():
        st[cid].set("review", f"unreadable: {err}")
    for cid, reason in routing.passthrough.items():
        st[cid].set("keep", reason)

    # 2. Bursts.
    burst_survivors: dict[str, list[str]] = {}
    for b in routing.bursts:
        _rank_burst(shoot, b, routing.units[b.id], gates, scores, weights, method, st, burst_survivors, penalties, pnames)

    # 3. Scenes.
    for s in routing.scenes:
        _rank_scene(shoot, s, routing, gates, scores, weights, method, settings, st, burst_survivors, penalties, pnames)

    # 4. Review overrides.
    for g in shoot.groups.values():
        why = []
        if g.confidence < settings.review_below:
            why.append(f"grouping confidence {g.confidence:.2f} < {settings.review_below:g}")
        doubts = [w for w in g.warnings if settings.review_warnings == "all" or
                  (settings.review_warnings == "doubt" and not is_informational(w))]
        why.extend(f"bracketeering warning: {w}" for w in doubts)
        notes = [w for w in g.warnings if w not in doubts]
        for cid in g.captures if notes else ():
            st[cid].reasons.append("note: " + "; ".join(f"bracketeering: {w}" for w in notes))
        if not why:
            continue
        for cid in g.captures:
            s_ = st[cid]
            if s_.decision in ("keep", "reject") and not s_.gate_rejected:
                s_.reasons.append(f"review instead of auto-{s_.decision}: {'; '.join(why)}")
                s_.decision = "review"
    for cid, flags in gates.flags.items():
        s_ = st[cid]
        if s_.decision == "keep":
            s_.decision = "review"
            s_.reasons.extend(f"review instead of auto-keep: {f}" for f in flags)

    undecided = [cid for cid, s_ in st.items() if s_.decision is None]
    if undecided:
        raise RuntimeError(f"captures without a decision: {undecided[:5]}")

    # 5. Per file: every file gets its capture's decision.
    report_files = shoot.report.get("files", {})
    out: dict[str, FileDecision] = {}
    for path, f in shoot.files.items():
        cap = shoot.captures[f.capture]
        g = shoot.groups[cap.group]
        s_ = st[f.capture]
        raw = {k: v for k, v in scores.get(f.capture, {}).items()}
        if raw and cap.exposure is not None:  # the preview stats the black and shadow gates use
            raw["preview_mean"] = cap.exposure.mean
            raw["shadows_crushed"] = cap.exposure.shadows_crushed
        rel = {f"{k}_rel": v for k, v in s_.rel.items()}
        reasons = list(s_.reasons)
        for name, amount, why in items.get(f.capture, []):
            raw[name] = amount
            reasons.append(f"score -{amount:.2f}: {why}")
        if g.scene_note:
            reasons.append(g.scene_note)
        if not f.primary:
            reasons.append(f"follows its shot {Path(cap.primary).name}")
        out[path] = FileDecision(
            decision=s_.decision,
            group=g.id,
            kind=g.kind,
            role=report_files.get(path, {}).get("role", cap.role),
            scene=g.scene,
            capture=f.capture,
            primary=f.primary,
            comparison_set=s_.comparison_set,
            rank_in_set=s_.rank,
            set_size=s_.size,
            score=s_.score,
            scores={**raw, **rel},
            reasons=reasons,
            gates=[x.gate for x in gates.failures.get(f.capture, [])],
        )
    return out


def _rank_burst(shoot, b: ComparisonSet, unit: Unit, gates, scores, weights, method, st, survivors, penalties, pnames) -> None:
    combined, rel = combine(b.members, scores, weights, method, penalties)
    index = {c: i for i, c in enumerate(b.members)}
    order = sorted(b.members, key=lambda c: (gates.failed(c), -combined[c], index[c]))
    ok = [c for c in order if not gates.failed(c)]
    unit.rep = ok[0] if ok else order[0]
    unit.rep_reason = "best frame of the burst" if ok else "least-bad frame of the burst"
    keep = b.keep or len(b.members)
    survivors[b.id] = ok[:keep]
    n = len(b.members)
    for rank, c in enumerate(order, 1):
        s_ = st[c]
        s_.comparison_set, s_.rank, s_.size, s_.score, s_.rel = b.id, rank, n, combined[c], rel[c]
        if gates.failed(c):
            s_.gate_rejected = True
            s_.set("reject", "rejected: " + "; ".join(f.reason for f in gates.failures[c]) + f" (burst {b.id})")
        elif c in survivors[b.id]:
            if n <= keep:
                s_.reasons.append(f"burst of {n}: all frames kept (bursts_keep {keep})")
            else:
                s_.reasons.append(f"rank {ok.index(c) + 1} of {n} in burst {b.id} (keeping {keep})")
        else:
            better = ok.index(c)
            s_.set(
                "reject",
                f"rejected: {better} better frame{'s' if better != 1 else ''} in burst {b.id} (keeping {keep}); "
                f"{_name(shoot, ok[0])} {_why_better(ok[0], c, rel, pnames)}",
            )


def _rank_scene(shoot, s: ComparisonSet, routing, gates, scores, weights, method, settings, st, survivors, penalties, pnames) -> None:
    units = [routing.units[u] for u in s.members]
    reps = [u.rep for u in units]
    combined, rel = combine(reps, scores, weights, method, penalties)
    solo = len(units) == 1
    scene_name = f"scene {s.scene}" if s.scene is not None else "its scene"

    def gated(u: Unit) -> bool:
        return gates.failed(u.rep)

    index = {u.id: i for i, u in enumerate(units)}
    order = sorted(units, key=lambda u: (gated(u), -combined[u.rep], index[u.id]))
    ok = [u for u in order if not gated(u)]
    winners = ok if s.keep is None else ok[: s.keep]
    rescued = None
    if not ok:
        rescued = next((u for u in order if not gates.only_black(u.rep)), None)

    for rank, u in enumerate(order, 1):
        # Units other than bursts take the scene's ranking as their comparison set.
        if u.kind != "burst":
            for c in u.captures:
                s_ = st[c]
                s_.rel = rel[u.rep]
                if not solo:
                    s_.comparison_set, s_.rank, s_.size, s_.score = s.id, rank, len(units), combined[u.rep]

        if u in winners:
            if solo:
                scene_reason = "only unit in its scene: nothing to compare against"
            elif s.keep is None:
                scene_reason = f"{scene_name} not reduced (scene_keep 0)"
            elif rank == 1:
                scene_reason = f"best of {len(units)} in {scene_name}"
            else:
                scene_reason = f"rank {rank} of {len(units)} in {scene_name} (keeping {s.keep})"
            _apply(shoot, u, "keep", scene_reason, routing, st, survivors)
        elif u is rescued:
            fails = "; ".join(f.reason for f in gates.failures[u.rep])
            n_fail = len(units)
            reason = (
                f"every candidate in {scene_name} failed a gate ({n_fail}); least-bad kept for review: {fails}"
                if not solo
                else f"failed a gate but is the only unit in its scene; kept for review: {fails}"
            )
            _apply(shoot, u, "review", reason, routing, st, survivors, rescue=True)
        elif gated(u):
            if u.kind == "single":
                c = u.rep
                st[c].gate_rejected = True
                st[c].set("reject", "rejected: " + "; ".join(f.reason for f in gates.failures[c]))
            else:  # a burst whose frames all failed: already rejected frame by frame
                pass
        else:
            best = ok[0]
            better = ok.index(u)
            keep = s.keep or 0
            reason = (
                f"rejected: {better} better unit{'s' if better != 1 else ''} in {scene_name} (keeping {keep}); "
                f"{_unit_label(shoot, best)} {_why_better(best.rep, u.rep, rel, pnames)}"
            )
            _apply(shoot, u, "reject", reason, routing, st, survivors)


def _apply(shoot, u: Unit, outcome: str, scene_reason: str, routing: Routing, st, survivors, rescue: bool = False) -> None:
    """Apply a unit's scene outcome to the captures that follow it."""
    if u.kind == "burst":
        followers = [u.rep] if rescue else survivors.get(u.id, [])
        for c in followers:
            if rescue:
                st[c].gate_rejected = False
                st[c].decision = "review"
                st[c].reasons = [r for r in st[c].reasons if not r.startswith("rejected:")]
                st[c].reasons.append(scene_reason)
            elif outcome == "reject":
                st[c].set("reject", f"rejected: burst {u.id} lost in its scene: " + scene_reason.removeprefix("rejected: "))
            else:
                st[c].set(outcome, f"burst {u.id}: {scene_reason}")
        return
    if u.kind == "exposure":
        if routing.pick_one:
            for c in u.captures:
                if c == u.rep:
                    st[c].set(outcome, f"bracket judged on its {u.rep_reason} frame (pick_one); {scene_reason}")
                else:
                    st[c].set("reject", f"rejected: not the best-exposed frame of bracket {u.id} (pick_one)")
        else:
            for c in u.captures:
                verb = {"keep": "kept", "reject": "rejected", "review": "sent to review"}[outcome]
                st[c].set(outcome, f"bracket {verb} as a set (keep_set), judged on its {u.rep_reason} frame; {scene_reason}")
        return
    if u.kind in ("focus", "timelapse"):
        label = "focus stack" if u.kind == "focus" else "timelapse"
        for c in u.captures:
            st[c].set(outcome, f"{label} judged as one unit on its {u.rep_reason}; {scene_reason}")
        return
    c = u.rep  # single
    if rescue:
        st[c].gate_rejected = False
    st[c].set(outcome, scene_reason)
