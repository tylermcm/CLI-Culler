"""Exposure quality from bracketeering's preview stats (no pixels decoded here)."""

from __future__ import annotations


def exposure_quality(highlights_clipped: float, shadows_crushed: float, mean: float) -> float:
    """1.0 = well exposed. Penalises clipped highlights most, then crushed shadows, then a far-off mean."""
    q = 1.0
    q -= 0.5 * min(1.0, highlights_clipped / 0.10)
    q -= 0.3 * min(1.0, shadows_crushed / 0.40)
    q -= 0.2 * min(1.0, abs(mean - 0.45) / 0.45)
    return round(max(0.0, q), 4)


class ExposureScorer:
    name = "exposure"
    revision = "bracketeering-stats-v1"

    def score(self, batch) -> list[dict]:
        out = []
        for cand in batch:
            ex = cand.capture.exposure
            out.append({} if ex is None else {"exposure": exposure_quality(ex.highlights_clipped, ex.shadows_crushed, ex.mean)})
        return out
