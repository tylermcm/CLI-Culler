"""Eyes-closed ("blink") decision from a face landmarker's blendshape scores. Pure logic: no models, no Qt.

MediaPipe's Face Landmarker (Apache-2.0) returns, per face, a landmark set and a list of blendshape scores in [0, 1] that include ``eyeBlinkLeft``
and ``eyeBlinkRight`` (1 means the lid is fully closed). A frame's score is the highest eyes-closed value among the faces that are large enough to
matter, so one person blinking in a group shot counts and a tiny background face does not. The worker (``rankers/worker.py``) calls
``frame_blink_score``; the folder runner thresholds it with ``is_blink``.

Thresholds are conservative on purpose (a wrong blink flag rejects a good frame): looking down also lowers the lids, which scores well below 0.6.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

BLINK_LEFT = "eyeBlinkLeft"
BLINK_RIGHT = "eyeBlinkRight"
BLINK_THRESHOLD = 0.6  # mean of both eyes at or above this is a blink
MIN_FACE_AREA = 0.01  # a face must cover at least this fraction of the frame to count


def face_area(landmarks: Iterable[tuple[float, float]]) -> float:
    """Bounding-box area of a face's normalised (0..1) landmark points, as a fraction of the frame."""
    xs: list[float] = []
    ys: list[float] = []
    for x, y in landmarks:
        xs.append(float(x))
        ys.append(float(y))
    if not xs:
        return 0.0
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def eyes_closed(blendshapes: Mapping[str, float]) -> float | None:
    """Mean of the two eyelid-closure scores, or None when the landmarker did not report both."""
    left, right = blendshapes.get(BLINK_LEFT), blendshapes.get(BLINK_RIGHT)
    if left is None or right is None:
        return None
    return (float(left) + float(right)) / 2.0


def frame_blink_score(faces: Sequence[tuple[float, Mapping[str, float]]], *, min_area: float = MIN_FACE_AREA) -> float:
    """Highest eyes-closed score among the prominent faces; 0.0 when no face is large enough or none reports blendshapes.
    ``faces`` is a sequence of ``(area_fraction, blendshape_scores)``."""
    best = 0.0
    for area, shapes in faces:
        if area < min_area:
            continue
        closed = eyes_closed(shapes)
        if closed is not None and closed > best:
            best = closed
    return best


def is_blink(score: float, threshold: float = BLINK_THRESHOLD) -> bool:
    """An unreadable frame scores minus infinity and is never a blink."""
    return score >= threshold
