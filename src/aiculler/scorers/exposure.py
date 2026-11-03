"""Exposure quality from bracketeering's preview stats (no pixels decoded here).

Clipped highlights are not scored here: *where* they clip matters more than how much, so the clipping
scorer (connected regions) and its gates and penalty handle them.
"""

from __future__ import annotations


def exposure_quality(shadows_crushed: float, mean: float) -> float:
    """1.0 = well exposed. Penalises crushed shadows and a mean level far from mid-grey."""
    q = 1.0
    q -= 0.6 * min(1.0, shadows_crushed / 0.40)
    q -= 0.4 * min(1.0, abs(mean - 0.45) / 0.45)
    return round(max(0.0, q), 4)


class ExposureScorer:
    name = "exposure"
    revision = "bracketeering-stats-v2-no-highlights"

    def score(self, batch) -> list[dict]:
        out = []
        for cand in batch:
            ex = cand.capture.exposure
            out.append({} if ex is None else {"exposure": exposure_quality(ex.shadows_crushed, ex.mean)})
        return out
