"""What a frame shows, and how it was shot: semantic categories (TinyCLIP, zero-shot) and technique flags (EXIF).

Categories are multi-label: a frame can be a portrait *and* a night photo. Each category is a prototype, the
normalised mean of a few prompt embeddings; a frame's probabilities are a softmax over its cosine similarity to the
prototypes at CLIP's logit scale (100). The top category always counts; others count at ``LABEL_MIN`` or above (at
most ``MAX_LABELS``).

Technique flags come from EXIF alone, at thresholds where a photograph usually looks different to a quality model:
``high_iso`` (ISO >= 3200), ``slow_shutter`` (>= 1/30 s), ``long_lens`` (>= 200 mm as recorded), ``wide_aperture``
(f/2.8 or wider), ``flash``.

Used by the quality floor to judge a MANIQA score against similar frames (``baseline``), and written to
decisions.json for search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CATEGORIES: dict[str, tuple[str, ...]] = {
    "portrait": ("a portrait photo of a person", "a close-up photo of a person's face", "a headshot of a person"),
    "people": ("a photo of a group of people", "a photo of people at an event", "a candid photo of friends"),
    "landscape": ("a landscape photo", "a photo of mountains", "a photo of a beach", "a photo of a forest", "a scenic nature photo"),
    "architecture": ("a photo of a building", "a photo of architecture", "a photo of a city skyline", "a photo of a temple"),
    "animals": ("a photo of a wild animal", "a wildlife photo", "a photo of a bird", "a photo of a pet"),
    "macro": ("a macro photo", "a close-up photo of a flower", "a close-up photo of an insect", "a close-up photo of a small object"),
    "night": ("a photo taken at night", "a night scene with city lights", "a dark low-light photo"),
    "street": ("a street photo", "a photo of a city street", "a photo of people walking in a city"),
    "food_product": ("a photo of food", "a product photo", "a photo of a dish on a table"),
    "interior": ("a photo of a room interior", "a photo taken indoors in a house", "a photo of a restaurant interior"),
}
VERSION = "cats10-v1"  # bump when the categories or prompts change: baselines built with another version are refused
LOGIT_SCALE = 100.0
LABEL_MIN = 0.20
MAX_LABELS = 3

TECHNIQUES = ("high_iso", "slow_shutter", "long_lens", "wide_aperture", "flash")


@dataclass
class Prototypes:
    names: list[str]
    matrix: np.ndarray  # (categories, dim), unit rows
    model: str  # TinyCLIP revision the prompts were encoded with


def build_prototypes(engine, model: str) -> Prototypes:
    names, rows = [], []
    for name, prompts in CATEGORIES.items():
        mean = np.mean([engine.encode_text(p) for p in prompts], axis=0)
        rows.append(mean / (np.linalg.norm(mean) or 1.0))
        names.append(name)
    return Prototypes(names, np.asarray(rows, dtype=np.float32), model)


def classify(embedding, protos: Prototypes) -> dict:
    """{"labels": [...], "primary": name, "probs": {name: p}} for one unit-length image embedding."""
    v = np.asarray(embedding, dtype=np.float32)
    logits = LOGIT_SCALE * (protos.matrix @ v)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    order = np.argsort(-p)
    labels = [protos.names[order[0]]] + [protos.names[i] for i in order[1:MAX_LABELS] if p[i] >= LABEL_MIN]
    return {"labels": labels, "primary": labels[0], "probs": {n: round(float(x), 4) for n, x in zip(protos.names, p)}}


def techniques(capture) -> list[str]:
    """EXIF technique flags of a capture (``ingest.Capture``)."""
    out = []
    if capture.iso is not None and capture.iso >= 3200:
        out.append("high_iso")
    if capture.exposure_time is not None and capture.exposure_time >= 1 / 30:
        out.append("slow_shutter")
    if capture.focal_length is not None and capture.focal_length >= 200:
        out.append("long_lens")
    if capture.f_number is not None and 0 < capture.f_number <= 2.8:
        out.append("wide_aperture")
    if capture.flash:
        out.append("flash")
    return out
