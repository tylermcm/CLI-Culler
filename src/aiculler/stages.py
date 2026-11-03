"""Cheap -> expensive: which frames each stage runs on.

* ``contested``: frames that still compete after the cheap gates, i.e. are in a burst or scene with more
  candidates than it keeps. A frame alone in its set, or in a set that keeps everyone, is decided without
  MANIQA or Q-ReAlign: nothing to rank it against.
* ``appeal_shortlist``: within each contested set, the best few by the technical scores; only these reach
  Q-ReAlign (the runtime driver).
* ``blur_failures``: the severe-blur gate, which needs MANIQA.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

from aiculler.gates import GateFailure, GateResults
from aiculler.ingest import Shoot
from aiculler.routing import Routing

if TYPE_CHECKING:
    from aiculler.api import CullSettings

LATE_METRICS = ("appeal", "preference")


def _ok_members(routing: Routing, gates: GateResults) -> dict[str, list[str]]:
    return {b.id: [c for c in b.members if not gates.failed(c)] for b in routing.bursts}


def contested(routing: Routing, gates: GateResults) -> set[str]:
    """Candidates that still have to beat someone."""
    ok_burst = _ok_members(routing, gates)
    burst_keep = {b.id: b.keep for b in routing.bursts}
    out: set[str] = set()
    for bid, ok in ok_burst.items():
        if burst_keep[bid] is not None and len(ok) > burst_keep[bid]:
            out.update(ok)
    for s in routing.scenes:
        if s.keep is None:
            continue
        units = [routing.units[u] for u in s.members]
        ok_units = [u for u in units if (ok_burst.get(u.id) if u.kind == "burst" else not gates.failed(u.rep))]
        if len(ok_units) > s.keep:
            for u in ok_units:
                out.update(ok_burst[u.id] if u.kind == "burst" else [u.rep])
    return out


def _technical(captures: list[str], scores, settings) -> dict[str, float]:
    from aiculler.decide import combine

    weights = {k: float(v) for k, v in settings.weights.items() if float(v) > 0 and k not in LATE_METRICS}
    combined, _ = combine(captures, scores, weights or {"sharpness": 1.0}, settings.normalise)
    return combined


def appeal_shortlist(routing: Routing, gates: GateResults, scores, settings: "CullSettings", candidates: set[str]) -> set[str]:
    """The top ``appeal_shortlist`` (at least keep + 1) of each contested set, by the technical scores."""
    ok_burst = _ok_members(routing, gates)
    out: set[str] = set()
    best_of_burst: dict[str, list[str]] = {}
    for b in routing.bursts:
        ok = [c for c in ok_burst[b.id] if c in candidates]
        if not ok:
            continue
        tech = _technical(ok, scores, settings)
        ranked = sorted(ok, key=lambda c: (-tech[c], b.members.index(c)))
        k = max(settings.appeal_shortlist, (b.keep or 0) + 1)
        best_of_burst[b.id] = ranked[:k]
        if b.keep is not None and len(ok) > b.keep:
            out.update(ranked[:k])
    for s in routing.scenes:
        if s.keep is None:
            continue
        units = [routing.units[u] for u in s.members]
        reps: dict[str, str] = {}
        for u in units:
            if u.kind == "burst":
                if best_of_burst.get(u.id):
                    reps[u.id] = best_of_burst[u.id][0]
            elif u.rep in candidates:
                reps[u.id] = u.rep
        if len(reps) <= s.keep:
            continue
        tech = _technical(list(reps.values()), scores, settings)
        ranked = sorted(reps, key=lambda uid: (-tech[reps[uid]], s.members.index(uid)))
        for uid in ranked[: max(settings.appeal_shortlist, s.keep + 1)]:
            # A burst's frames that may still become its representative all need appeal to be compared fairly.
            out.update(best_of_burst[uid] if routing.units[uid].kind == "burst" else [reps[uid]])
    return out


def blur_failures(shoot: Shoot, routing: Routing, gates: GateResults, scores, settings: "CullSettings") -> dict[str, GateFailure]:
    """Severe blur: MANIQA in the shoot's bottom ``maniqa_bottom`` AND face (else whole-frame) sharpness below
    ``blur_ratio`` x its comparison set's median. Bursts compare frames of the burst; singles, the candidates of
    their scene. Never bracket frames; a frame alone in its set has no median and is never blur-gated."""
    g = settings.gates
    values = sorted(v["maniqa"] for v in scores.values() if v.get("maniqa") is not None)
    if len(values) < 10:  # too few to call a bottom percentile
        return {}
    cut = values[max(0, math.ceil(len(values) * g.maniqa_bottom) - 1)]  # the worst ceil(2% of n) frames
    sets: dict[str, list[str]] = {}
    for b in routing.bursts:
        if b.label != "burst":  # a moving bracket's frames are bracket frames: never gated
            continue
        for c in b.members:
            sets[c] = b.members
    scene_frames: dict[int, list[str]] = {}
    for s in routing.scenes:
        if s.scene is None:
            continue
        frames = []
        for uid in s.members:
            u = routing.units[uid]
            frames.extend(u.captures if u.kind == "burst" else [u.rep])
        scene_frames[s.scene] = frames
        for uid in s.members:
            u = routing.units[uid]
            if u.kind == "single":
                sets[u.rep] = frames
    out: dict[str, GateFailure] = {}
    for c, members in sets.items():
        mine = scores.get(c, {})
        if mine.get("maniqa") is None or mine["maniqa"] > cut or gates.failed(c):
            continue
        metric = "face_sharpness" if mine.get("face_sharpness") is not None else "sharpness"
        others = [scores.get(m, {}).get(metric) for m in members if m != c]
        others = [v for v in others if v is not None]
        if not others or mine.get(metric) is None:
            continue
        median = statistics.median(others)
        if median > 0 and mine[metric] < g.blur_ratio * median:
            what = "face" if metric == "face_sharpness" else "frame"
            out[c] = GateFailure(
                "blur",
                f"severe blur (MANIQA in the shoot's bottom {g.maniqa_bottom:.0%}; {what} sharpness {mine[metric] / median:.0%} of its set's median)",
            )
    return out
