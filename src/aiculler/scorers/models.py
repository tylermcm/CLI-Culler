"""Model-backed scorers: faces (MediaPipe + AdaFace), MANIQA, Q-ReAlign appeal, TinyCLIP embeddings.

Each takes its engine from the ``ModelRegistry`` and its revision from the model's pinned revision plus the
procedure, so a different model or procedure never reuses cached scores.
"""

from __future__ import annotations

import numpy as np

from aiculler.models.app.blink import BLINK_THRESHOLD, MIN_FACE_AREA
from aiculler.scorers.sharpness import laplacian_variance

FACE_CROP = 160  # face crops are resized to this width before measuring sharpness, so faces of different sizes compare


def face_sharpness(image, box) -> float | None:
    """Laplacian variance of a face crop (box = fractions of the frame), resized to a fixed width."""
    from PIL import Image

    if image is None:
        return None
    w, h = image.size
    left, top, right, bottom = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    if right - left < 8 or bottom - top < 8:
        return None
    crop = image.crop((int(left), int(top), int(round(right)), int(round(bottom)))).convert("L")
    scale = FACE_CROP / crop.width
    crop = crop.resize((FACE_CROP, max(8, round(crop.height * scale))), Image.Resampling.LANCZOS)
    return round(laplacian_variance(np.asarray(crop, dtype=np.float32)), 3)


class FacesScorer:
    """Faces, eyes and face-crop sharpness.

    Scores: ``face_count`` (prominent faces), ``eyes_closed`` (most closed of the faces >= 1% of the frame: one
    person blinking in a group shot counts, the app's ``frame_blink_score``), ``face_sharpness`` (largest face),
    ``face_area`` (largest face). Detail: ``faces`` (boxes, areas, eyes, AdaFace vectors for people coverage).
    """

    name = "faces"
    needs_pixels = True

    def __init__(self, registry):
        self.registry = registry
        adaface = registry.revision("adaface") if registry.available("adaface") else "none"
        search = "tiles" if adaface != "none" else "whole"  # tiled small-face search only for people coverage
        self.revision = f"landmarker:{registry.revision('face_landmarker')}+adaface:{adaface}+{search}+crop{FACE_CROP}-v1"

    def score(self, batch) -> list[dict]:
        engine = self.registry.engine("face_landmarker")
        out = []
        for cand in batch:
            image = cand.model_image(engine.LONG_EDGE)
            if image is None:
                out.append({})
                continue
            faces = engine.detail(image)["faces"]
            values: dict = {"face_count": len(faces), "faces": faces}
            prominent = [f for f in faces if f["area"] >= MIN_FACE_AREA and f.get("eyes_closed") is not None]
            if prominent:
                values["eyes_closed"] = max(f["eyes_closed"] for f in prominent)
            if faces:
                largest = max(faces, key=lambda f: f["area"])
                values["face_area"] = largest["area"]
                sharp = face_sharpness(cand.image, largest["box"])
                if sharp is not None:
                    values["face_sharpness"] = sharp
            out.append(values)
        return out


class ManiqaScorer:
    name = "maniqa"
    needs_pixels = False

    def __init__(self, registry):
        self.registry = registry
        self.revision = f"{registry.revision('maniqa')}+crops20-1280-v1"

    def score(self, batch) -> list[dict]:
        engine = self.registry.engine("maniqa")
        out = []
        for cand in batch:
            image = cand.model_image(engine.LONG_EDGE)
            out.append({} if image is None else {"maniqa": round(engine.score(image), 5)})
        return out


class AppealScorer:
    """Q-ReAlign Mini 0.8B: probability-weighted aesthetic level, 0 (bad) .. 1 (excellent)."""

    name = "appeal"
    needs_pixels = False

    def __init__(self, registry):
        self.registry = registry
        self.revision = f"q_realign:{registry.revision('q_realign')}+1024-v1"

    def score(self, batch) -> list[dict]:
        engine = self.registry.engine("q_realign")
        out = []
        for cand in batch:
            image = cand.model_image(engine.LONG_EDGE)
            out.append({} if image is None else {"appeal": round(engine.score(image), 5)})
        return out


class TinyClipScorer:
    """TinyCLIP image embedding (unit length) of the cached preview. Detail only: ``embedding``."""

    name = "tinyclip"
    needs_pixels = True

    def __init__(self, registry):
        self.registry = registry
        self.revision = f"{registry.revision('tinyclip')}+224-bilinear-v1"

    def score(self, batch) -> list[dict]:
        engine = self.registry.engine("tinyclip")
        return [{} if cand.image is None else {"embedding": engine.embed(cand.image)} for cand in batch]


__all__ = ["AppealScorer", "BLINK_THRESHOLD", "FacesScorer", "ManiqaScorer", "TinyClipScorer", "face_sharpness"]
