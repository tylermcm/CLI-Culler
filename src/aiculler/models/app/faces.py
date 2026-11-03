"""Faces for the People index, found by the same landmarker the blink check uses. Pure logic: geometry, preprocessing and the per-file cache. No models, no Qt.

One detector serves both jobs: the face landmarker finds each face and its 478 landmarks, the five alignment points (eye centres, nose tip, mouth corners)
come from those landmarks by plain geometry, and an AdaFace recognizer turns the aligned 112 x 112 crop into a 512-number identity vector. A face smaller than
``MIN_FACE_AREA`` of the frame is a background person and is skipped, exactly as it is for the blink check. ``rankers/worker.py`` runs the models; the folder
cascade and the People index read and write ``FaceCache`` so a file is analysed once whichever of them reaches it first.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from aiculler.models.app.blink import MIN_FACE_AREA

__all__ = ["ALIGNED_SIZE", "EMBEDDING_DIM", "FACE_MODEL_TAG", "FaceFinding", "MIN_FACE_AREA", "PEOPLE_MIN_FACE_AREA", "align_crop", "five_points", "preprocess", "similarity_transform"]

ALIGNED_SIZE = 112
EMBEDDING_DIM = 512
# Names the recognizer for everything stored with its vectors (``image_faces.identity_model``, the cache). A different recognizer is a different space: change this
# and every stored vector is ignored until the folder is scanned again.
FACE_MODEL_TAG = "adaface:ir50+tiles"   # +tiles: small faces are found by looking at overlapping parts of the frame too (the first version missed them)
# The People index keeps smaller faces than the blink check does: a couple standing a few metres away is the subject of the photo even when each face is half a percent
# of the frame. Only a face under this share of the frame is a bystander.
PEOPLE_MIN_FACE_AREA = 0.003

# The standard 112 x 112 five-point template the recognizer was trained on: eye (image left), eye (image right), nose tip, mouth corner (image left), mouth corner (image right).
TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)

# MediaPipe face-mesh indices. The two corners of each eye average to the eye centre; 1 is the nose tip; 61 and 291 are the mouth corners.
LEFT_EYE_CORNERS = (33, 133)  # the eye that appears on the image's left
RIGHT_EYE_CORNERS = (362, 263)
NOSE_TIP = 1
MOUTH_LEFT, MOUTH_RIGHT = 61, 291


def five_points(landmarks: Sequence[tuple[float, float]], width: int, height: int) -> np.ndarray:
    """The five alignment points in pixels, from a face's normalised (0..1) landmark points."""

    def point(index: int) -> np.ndarray:
        x, y = landmarks[index]
        return np.array([float(x) * width, float(y) * height], dtype=np.float64)

    return np.stack([
        (point(LEFT_EYE_CORNERS[0]) + point(LEFT_EYE_CORNERS[1])) / 2.0,
        (point(RIGHT_EYE_CORNERS[0]) + point(RIGHT_EYE_CORNERS[1])) / 2.0,
        point(NOSE_TIP),
        point(MOUTH_LEFT),
        point(MOUTH_RIGHT),
    ])


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The 2 x 3 rotation + uniform scale + translation that maps ``source`` points onto ``target`` in the least-squares sense (Umeyama)."""
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    variance = (src_c ** 2).sum() / len(src)
    if variance <= 1e-12:
        raise ValueError("degenerate landmarks")
    u, s, vt = np.linalg.svd(dst_c.T @ src_c / len(src))
    flip = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        flip[-1] = -1.0
    rotation = u @ np.diag(flip) @ vt
    scale = (s * flip).sum() / variance
    translation = dst_mean - scale * rotation @ src_mean
    return np.hstack([scale * rotation, translation[:, None]])


def align_crop(image, points: np.ndarray):
    """The 112 x 112 RGB crop (a numpy array) of a PIL image with the face's five points moved onto the template."""
    from PIL import Image

    forward = similarity_transform(points, TEMPLATE)
    full = np.vstack([forward, [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(full)[:2]  # PIL maps each output pixel back to an input pixel
    crop = image.convert("RGB").transform(
        (ALIGNED_SIZE, ALIGNED_SIZE), Image.Transform.AFFINE, tuple(float(v) for v in inverse.reshape(-1)), resample=Image.Resampling.BILINEAR
    )
    return np.asarray(crop, dtype=np.uint8)


def preprocess(rgb_crop: np.ndarray) -> np.ndarray:
    """The recognizer's input for one crop: BGR channel order, scaled to [-1, 1], channels first, with a batch axis (1, 3, 112, 112)."""
    bgr = np.asarray(rgb_crop, dtype=np.float32)[:, :, ::-1]
    return np.ascontiguousarray(((bgr / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None])


@dataclass(frozen=True, slots=True)
class FaceFinding:
    """One prominent face in a frame. ``box`` is (left, top, right, bottom) as fractions of the frame, so it does not depend on the size the frame was read at."""

    box: tuple[float, float, float, float]
    area: float
    embedding: tuple[float, ...] = ()
    eyes_closed: float | None = None

    def to_json(self) -> dict[str, object]:
        return {"box": [round(v, 5) for v in self.box], "area": round(self.area, 5), "embedding": [round(v, 6) for v in self.embedding], "eyes_closed": self.eyes_closed}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "FaceFinding":
        box = tuple(float(v) for v in data["box"])  # type: ignore[union-attr]
        closed = data.get("eyes_closed")
        return cls(
            box=(box[0], box[1], box[2], box[3]),
            area=float(data["area"]),  # type: ignore[arg-type]
            embedding=tuple(float(v) for v in data.get("embedding", ())),  # type: ignore[union-attr]
            eyes_closed=None if closed is None else float(closed),  # type: ignore[arg-type]
        )


def landmark_box(landmarks: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for x, y in landmarks:
        xs.append(float(x))
        ys.append(float(y))
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0.0, 0.0, 0.0, 0.0)
