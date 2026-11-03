"""Whole-frame sharpness: variance of the 4-neighbour Laplacian of the preview's luma.

Same Laplacian as the image-triage app's cascade (``image_triage/cascade/extract.py``,
``_centre_laplacian_variance``), over the whole frame instead of the central half. Its scale depends on
the preview size and mostly measures scene texture, so it is only compared within a comparison set.
Milestone 2 adds face-crop sharpness, which beats it for portraits.
"""

from __future__ import annotations

import numpy as np

from aiculler.ingest import luma


def laplacian_variance(gray: np.ndarray) -> float:
    height, width = gray.shape
    if height < 3 or width < 3:
        return 0.0
    g = gray.astype(np.float32)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var(dtype=np.float64))


class SharpnessScorer:
    name = "sharpness"
    revision = "laplacian-var-wholeframe-1024-v1"

    def score(self, batch) -> list[dict]:
        out = []
        for cand in batch:
            image = cand.image
            out.append({} if image is None else {"sharpness": round(laplacian_variance(luma(image)), 3)})
        return out
