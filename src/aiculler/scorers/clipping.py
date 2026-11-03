"""Where highlights clip, not just how much: connected clipped regions on the preview.

A sun, a specular glint or a lamp is a small clipped region and costs nothing; a blown sky is one large
region across the top of the frame. Two kinds of clipping are measured, on the cached ~1024 px preview:

* **blown** (to white): every channel at or near the top (max >= 250 and min >= ``WHITE_MIN``). Detail is gone;
  only this can fail a gate.
* **saturated** (any channel >= 250): includes strong colour, such as a yellow light ring or a red neon sign,
  whose detail is still there in the other channels. It costs score (``clip_penalty``) but never fails a gate.

Values:

* ``clip_region``: the largest 8-connected region of blown pixels, as a fraction of the frame (gate)
* ``top_clip``: blown fraction of the top third ("sky blown out", gate)
* ``clip_total``: all blown pixels, as a fraction of the frame
* ``sat_region``: the largest 8-connected region of saturated pixels (any channel), as a fraction of the frame
  (penalty)
* ``luma_mean`` (0-1) and ``luma_spread`` (p95 - p5 of luma, 0-1): together they flag a washed-out frame

The embedded preview clips slightly earlier than the RAW, so thresholds that use these are deliberately loose.
"""

from __future__ import annotations

import numpy as np

CLIP_LEVEL = 250  # same level as bracketeering's highlight_level and the app's cascade
WHITE_MIN = 235  # a blown pixel's lowest channel: near-white, not a saturated colour


def _largest_region(mask: np.ndarray) -> float:
    from scipy import ndimage

    if not mask.any():
        return 0.0
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    return float(np.bincount(labels.ravel())[1:].max()) / float(mask.size) if n else 0.0


def clipping_stats(rgb: np.ndarray) -> dict[str, float]:
    saturated = (rgb >= CLIP_LEVEL).any(axis=2)
    blown = saturated & (rgb.min(axis=2) >= WHITE_MIN)
    h = blown.shape[0]
    top = blown[: max(1, h // 3)]
    luma = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    p5, p95 = np.percentile(luma, [5, 95])
    return {
        "clip_region": round(_largest_region(blown), 4),
        "clip_total": round(float(blown.mean()), 4),
        "top_clip": round(float(top.mean()), 4),
        "sat_region": round(_largest_region(saturated), 4),
        "luma_mean": round(float(luma.mean()) / 255.0, 4),
        "luma_spread": round(float(p95 - p5) / 255.0, 4),
    }


class ClippingScorer:
    name = "clipping"
    revision = "regions-8conn-white235-sat250-v3-upright"
    needs_pixels = True

    def score(self, batch) -> list[dict]:
        out = []
        for cand in batch:
            image = cand.image
            out.append({} if image is None else clipping_stats(np.asarray(image.convert("RGB"))))
        return out
