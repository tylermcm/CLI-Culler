"""Gates: reject only clear failures.

Milestone 1 gates use bracketeering's preview exposure stats:

* black frame / misfire: mean luma < ``black_mean``
* unusable clipping: ``highlights_clipped`` > threshold or ``shadows_crushed`` > threshold
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


def apply_gates(shoot: Shoot, routing: Routing, settings: "CullSettings") -> GateResults:
    g = settings.gates
    out = GateResults()
    candidates: list[str] = []
    for b in routing.bursts:
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
                if ex.highlights_clipped > g.highlights_clipped:
                    fails.append(GateFailure(
                        "highlights",
                        f"unusable clipping ({ex.highlights_clipped:.0%} of pixels clipped > {g.highlights_clipped:.0%})",
                    ))
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
