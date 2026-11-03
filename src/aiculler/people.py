"""People: faces clustered into identities (the ``people`` model feature), and people coverage.

Faces come from the faces scorer (MediaPipe landmarker + AdaFace, as in the app), run on every readable capture
when ``people`` is on. Identities are clustered the way the app's People index clusters AdaFace vectors
(``people_search.cluster_face_identities``: in capture order, each face joins the most similar centroid at cosine
>= ``face_cluster_threshold`` or starts a new person). Every face with an identity vector is clustered, as in the
app, so the index can find a person however small they are in a frame.

People coverage (``keep_people_coverage``) counts only prominent faces (>= 1% of the frame, the blink check's
size), so bystanders do not keep frames. A person is covered when any frame where they are prominent is kept or
sent to review. Otherwise their best acceptable rejected frame (not gate-failed, their eyes open, their face
largest) is promoted to keep with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from aiculler.models.app.blink import MIN_FACE_AREA

if TYPE_CHECKING:
    from aiculler.api import CullSettings, FileDecision


@dataclass
class FaceRef:
    capture: str
    index: int  # position in the capture's face list
    area: float
    eyes_closed: float | None
    vector: np.ndarray
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    person: int = 0  # 1-based person number, set by find_people


def cluster_people(faces: list[FaceRef], threshold: float) -> list[list[FaceRef]]:
    groups: list[list[FaceRef]] = []
    centroids: list[np.ndarray] = []
    for face in faces:
        best, best_score = -1, -1.0
        for i, c in enumerate(centroids):
            score = float(np.dot(face.vector, c))
            if score > best_score:
                best, best_score = i, score
        if best >= 0 and best_score >= threshold:
            groups[best].append(face)
            mean = np.mean([f.vector for f in groups[best]], axis=0)
            norm = float(np.linalg.norm(mean))
            centroids[best] = mean / norm if norm > 0 else mean
        else:
            groups.append([face])
            centroids.append(face.vector)
    return groups


def find_people(shoot, details: dict[str, dict], threshold: float) -> list[list[FaceRef]]:
    """Every face with an identity vector, clustered into people (numbered from 1, in order of first appearance)."""
    faces: list[FaceRef] = []
    for cid in shoot.captures:  # capture order
        for i, f in enumerate((details.get(cid) or {}).get("faces") or []):
            vec = f.get("embedding") or []
            if not vec:
                continue
            v = np.asarray(vec, dtype=np.float32)
            n = float(np.linalg.norm(v))
            box = tuple(float(x) for x in (f.get("box") or (0, 0, 0, 0)))
            faces.append(FaceRef(cid, i, float(f.get("area", 0.0)), f.get("eyes_closed"), v / n if n > 0 else v, box))
    people = cluster_people(faces, threshold)
    for number, person in enumerate(people, 1):
        for face in person:
            face.person = number
    return people


def apply_people_coverage(
    people: list[list[FaceRef]], files: dict[str, "FileDecision"], gates, settings: "CullSettings"
) -> dict[str, int]:
    """Promote one frame per uncovered person (prominent faces only); returns stats."""
    decision = {d.capture: d.decision for d in files.values() if d.primary}
    promoted = 0
    for person in people:
        prominent = [f for f in person if f.area >= MIN_FACE_AREA]
        if not prominent:
            continue
        number = prominent[0].person
        captures = {f.capture for f in prominent}
        if any(decision.get(c) in ("keep", "review") for c in captures):
            continue
        acceptable = [
            f for f in prominent
            if decision.get(f.capture) == "reject" and not gates.failed(f.capture)
            and (f.eyes_closed is None or f.eyes_closed < settings.blink_threshold)
        ]
        if not acceptable:
            continue
        best = max(acceptable, key=lambda f: (f.area, -(f.eyes_closed or 0.0)))
        for d in files.values():
            if d.capture == best.capture:
                d.decision = "keep"
                d.reasons.append(f"kept: only good frame of person #{number} ({len(captures)} frame(s) of them, all otherwise rejected)")
        decision[best.capture] = "keep"
        promoted += 1
    return {"people": len(people), "people_promoted": promoted}


def tag_people(people: list[list[FaceRef]], files: dict[str, "FileDecision"]) -> None:
    """Each file lists the people in its capture (person numbers, ascending)."""
    by_capture: dict[str, set[int]] = {}
    for person in people:
        for f in person:
            by_capture.setdefault(f.capture, set()).add(f.person)
    for d in files.values():
        d.people = sorted(by_capture.get(d.capture, ()))
