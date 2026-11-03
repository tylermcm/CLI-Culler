"""Gates: reject only clear failures.

Gates on the preview:

* black frame / misfire: mean luma < ``black_mean`` (always a reject)
* highlights, by connected clipped region (``scorers.clipping``): one region >= ``clip_region_reject`` of the
  frame, or >= ``sky_blown`` of the top third clipped ("sky blown out"), or a washed-out frame (mean luma >=
  ``washed_mean`` and tonal spread <= ``washed_spread``). Smaller regions (``clip_region_ignore`` up to
  ``clip_region_reject``) cost a score penalty in ``decide``; below ``clip_region_ignore``, nothing.
* unusable shadows: ``shadows_crushed`` > threshold
* featureless preview: by default a review flag, not a rejection (a moon in an empty sky is featureless too)

Severe blur needs MANIQA (milestone 2) and is not gated yet.

Gates run only on frames that compete: burst frames and single frames. Bracket frames are never gated
on exposure (they are dark and bright on purpose), and frames left out of culling are not gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiculler.ingest import Shoot
from aiculler.routing import Routing

if TYPE_CHECKING:
    from aiculler.api import CullSettings

BLACK = "black"


@dataclass
class GateFailure:
    gate: str  # "black", "highlights", "shadows", "featureless"
    reason: str


@dataclass
class GateResults:
    failures: dict[str, list[GateFailure]] = field(default_factory=dict)  # capture -> failures (reject)
    flags: dict[str, list[str]] = field(default_factory=dict)  # capture -> review flags (not failures)

    def failed(self, capture: str) -> bool:
        return bool(self.failures.get(capture))

    def only_black(self, capture: str) -> bool:
        fails = self.failures.get(capture) or []
        return bool(fails) and all(f.gate == BLACK for f in fails)


def highlight_failures(values: dict[str, float], settings: "CullSettings") -> list[GateFailure]:
    """Region-aware highlight gates from the clipping scorer's values (nothing when they are missing)."""
    g = settings.gates
    out: list[GateFailure] = []
    region, top = values.get("clip_region"), values.get("top_clip")
    mean, spread = values.get("luma_mean"), values.get("luma_spread")
    if region is not None and region >= g.clip_region_reject:
        out.append(GateFailure("clip_region", f"one clipped region covers {region:.0%} of the frame (>= {g.clip_region_reject:.0%})"))
    if top is not None and top >= g.sky_blown:
        out.append(GateFailure("sky_blown", f"sky blown out ({top:.0%} of the top third clipped, >= {g.sky_blown:.0%})"))
    if mean is not None and spread is not None and mean >= g.washed_mean and spread <= g.washed_spread:
        out.append(GateFailure("washed_out", f"washed out (mean level {mean:.2f} >= {g.washed_mean:g}, tonal spread {spread:.2f} <= {g.washed_spread:g})"))
    return out


def clip_penalty(values: dict[str, float], settings: "CullSettings", *, ungated: bool = False) -> float:
    """Score penalty for a mid-sized clipped region: 0 below clip_region_ignore, rising to clip_penalty at clip_region_reject.

    ``ungated``: for frames the gates never see (a moving bracket's), a region at or past clip_region_reject costs
    the full penalty instead of nothing.
    """
    g = settings.gates
    region = values.get("clip_region")
    if region is None or region < g.clip_region_ignore:
        return 0.0  # small: free
    if region >= g.clip_region_reject:
        return g.clip_penalty if ungated else 0.0  # large: the clip_region gate decides, not a penalty
    span = g.clip_region_reject - g.clip_region_ignore
    return round(g.clip_penalty * min(1.0, (region - g.clip_region_ignore) / span), 4)


def apply_gates(
    shoot: Shoot, routing: Routing, settings: "CullSettings", scores: dict[str, dict[str, float]] | None = None
) -> GateResults:
    g = settings.gates
    out = GateResults()
    candidates: list[str] = []
    for b in routing.bursts:
        if b.label == "burst":  # a moving bracket's frames are bracket frames: never gated
            candidates.extend(b.members)
    for u in routing.units.values():
        if u.kind == "single" and u.rep is not None:
            candidates.append(u.rep)

    for cid in candidates:
        cap = shoot.captures[cid]
        fails: list[GateFailure] = []
        ex = cap.exposure
        if ex is not None:
            if ex.mean < g.black_mean:
                fails.append(GateFailure(BLACK, f"black frame (mean level {ex.mean:.3f} < {g.black_mean})"))
            else:
                fails.extend(highlight_failures((scores or {}).get(cid, {}), settings))
                if ex.shadows_crushed > g.shadows_crushed:
                    fails.append(GateFailure(
                        "shadows",
                        f"unusable shadows ({ex.shadows_crushed:.0%} of pixels crushed > {g.shadows_crushed:.0%})",
                    ))
        if cap.featureless and not any(f.gate == BLACK for f in fails):
            if g.featureless == "reject":
                fails.append(GateFailure("featureless", "featureless frame (misfire or blank)"))
            elif g.featureless == "review":
                out.flags.setdefault(cid, []).append("featureless preview: misfire, blank frame or minimal subject?")
        if fails:
            out.failures[cid] = fails
    return out
