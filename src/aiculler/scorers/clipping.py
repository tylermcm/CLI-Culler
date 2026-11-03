"""Where highlights clip, not just how much: connected clipped regions on the preview.

A sun, a specular glint or a lamp is a small clipped region and costs nothing; a blown sky is one large
region across the top of the frame. Measured on the cached ~1024 px preview:

* ``clip_region``: the largest 8-connected region of clipped pixels (any channel >= 250), as a fraction of the frame
* ``clip_total``: all clipped pixels, as a fraction of the frame
* ``top_clip``: clipped fraction of the top third ("sky blown out")
* ``luma_mean`` (0-1) and ``luma_spread`` (p95 - p5 of luma, 0-1): together they flag a washed-out frame

The embedded preview clips slightly earlier than the RAW, so thresholds that use these are deliberately loose.
"""

from __future__ import annotations

import numpy as np

CLIP_LEVEL = 250  # same level as bracketeering's highlight_level and the app's cascade


def clipping_stats(rgb: np.ndarray) -> dict[str, float]:
    from scipy import ndimage

    mask = (rgb >= CLIP_LEVEL).any(axis=2)
    h, w = mask.shape
    area = float(h * w) or 1.0
    largest = 0.0
    if mask.any():
        labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        if n:
            largest = float(np.bincount(labels.ravel())[1:].max()) / area
    top = mask[: max(1, h // 3)]
    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    p5, p95 = np.percentile(luma, [5, 95])
    return {
        "clip_region": round(largest, 4),
        "clip_total": round(float(mask.mean()), 4),
        "top_clip": round(float(top.mean()), 4),
        "luma_mean": round(float(luma.mean()) / 255.0, 4),
        "luma_spread": round(float(p95 - p5) / 255.0, 4),
    }


class ClippingScorer:
    name = "clipping"
    revision = "regions-8conn-250-v2-upright"
    needs_pixels = True

    def score(self, batch) -> list[dict]:
        out = []
        for cand in batch:
            image = cand.image
            out.append({} if image is None else clipping_stats(np.asarray(image.convert("RGB"))))
        return out
