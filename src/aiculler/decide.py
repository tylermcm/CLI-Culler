"""Decide once: per-set normalisation, combined score, keep counts, review bucket, reasons.

Order of operations (each capture gets exactly one decision):

1. Unreadable captures -> review. Passthrough captures (left out of culling) -> keep.
2. Bursts: frames ranked against each other. Gated frames -> reject; beyond ``bursts_keep`` -> reject;
   the rest follow the burst's outcome in its scene. The best frame represents the burst.
3. Scenes: units ranked against each other on their representative frame; the top ``scene_keep`` win.
   If every unit in a scene failed a gate, the least-bad one is kept for review (never a black frame).
4. Review overrides: groups with confidence < ``review_below`` or bracketeering warnings, and frames
   flagged by a gate (featureless), go to review instead of an automatic keep/reject. Gate rejections stay.

Scores are normalised within the comparison set they compete in, never across the shoot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from aiculler.gates import GateResults
from aiculler.ingest import Shoot
from aiculler.routing import ComparisonSet, Routing, Unit

if TYPE_CHECKING:
    from aiculler.api import CullSettings, FileDecision

# "relative" normalisation: distance from the best candidate in the set, on an absolute scale, so two
# near-identical frames get near-identical scores (rank normalisation would spread them to 0 and 1).
RELATIVE_SCALE = {
    "sharpness": ("log", math.log(4.0)),  # a quarter of the best frame's Laplacian variance -> 0
    "exposure": ("linear", 0.5),  # half a quality point below the best -> 0
}
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
    captures: list[str], scores: dict[str, dict[str, float]], weights: dict[str, float], method: str
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Combined score per capture within one set, plus each metric's set-normalised value."""
    rel: dict[str, dict[str, float]] = {c: {} for c in captures}
    for metric in weights:
        normed = normalise({c: scores.get(c, {}).get(metric) for c in captures}, metric, method)
        for c in captures:
            rel[c][metric] = normed[c]
    total = sum(weights.values()) or 1.0
    combined = {c: round(sum(w * rel[c][m] for m, w in weights.items()) / total, 4) for c in captures}
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


def _why_better(winner: str, loser: str, rel: dict[str, dict[str, float]]) -> str:
    parts = []
    if rel[winner].get("sharpness", 0) - rel[loser].get("sharpness", 0) >= 0.05:
        parts.append("sharper")
    if rel[winner].get("exposure", 0) - rel[loser].get("exposure", 0) >= 0.05:
        parts.append("better exposed")
    return " and ".join(parts) if parts else "higher combined score"


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

    # 1. Unreadable and passthrough.
    for cid, err in routing.unreadable.items():
        st[cid].set("review", f"unreadable: {err}")
    for cid, reason in routing.passthrough.items():
        st[cid].set("keep", reason)

    # 2. Bursts.
    burst_survivors: dict[str, list[str]] = {}
    for b in routing.bursts:
        _rank_burst(shoot, b, routing.units[b.id], gates, scores, weights, method, st, burst_survivors)

    # 3. Scenes.
    for s in routing.scenes:
        _rank_scene(shoot, s, routing, gates, scores, weights, method, settings, st, burst_survivors)

    # 4. Review overrides.
    for g in shoot.groups.values():
        why = []
        if g.confidence < settings.review_below:
            why.append(f"grouping confidence {g.confidence:.2f} < {settings.review_below:g}")
        why.extend(f"bracketeering warning: {w}" for w in g.warnings)
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
        rel = {f"{k}_rel": v for k, v in s_.rel.items()}
        reasons = list(s_.reasons)
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
        )
    return out


def _rank_burst(shoot, b: ComparisonSet, unit: Unit, gates, scores, weights, method, st, survivors) -> None:
    combined, rel = combine(b.members, scores, weights, method)
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
                f"{_name(shoot, ok[0])} is {_why_better(ok[0], c, rel)}",
            )


def _rank_scene(shoot, s: ComparisonSet, routing, gates, scores, weights, method, settings, st, survivors) -> None:
    units = [routing.units[u] for u in s.members]
    reps = [u.rep for u in units]
    combined, rel = combine(reps, scores, weights, method)
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
                f"{_unit_label(shoot, best)} is {_why_better(best.rep, u.rep, rel)}"
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
