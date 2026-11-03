"""People coverage (milestone 5): each recognised person keeps at least one acceptable frame.

Faces come from the faces scorer (MediaPipe landmarker + AdaFace, as in the app). Identities are clustered the
way the app's People index clusters AdaFace vectors (``people_search.cluster_face_identities``: greedy, each face
joins the most similar centroid at cosine >= ``face_cluster_threshold`` (the app's 0.35) or starts a new person).
Only prominent faces (>= 1% of the frame, the blink check's size) count, so bystanders do not.

A person is covered when any frame of theirs is kept or sent to review. Otherwise their best acceptable rejected
frame (not gate-failed, their eyes open, their face largest) is promoted to keep with a reason.
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
    index: int
    area: float
    eyes_closed: float | None
    vector: np.ndarray


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


def apply_people_coverage(
    shoot, files: dict[str, "FileDecision"], details: dict[str, dict], gates, settings: "CullSettings"
) -> dict[str, int]:
    """Promote one frame per uncovered person; returns stats."""
    faces: list[FaceRef] = []
    for cid in shoot.captures:  # capture order
        for i, f in enumerate((details.get(cid) or {}).get("faces") or []):
            vec = f.get("embedding") or []
            if f.get("area", 0) < MIN_FACE_AREA or not vec:
                continue
            v = np.asarray(vec, dtype=np.float32)
            n = float(np.linalg.norm(v))
            faces.append(FaceRef(cid, i, float(f["area"]), f.get("eyes_closed"), v / n if n > 0 else v))
    people = cluster_people(faces, settings.face_cluster_threshold)
    decision = {}
    for path, d in files.items():
        if d.primary:
            decision[d.capture] = d.decision
    promoted = 0
    for number, person in enumerate(people, 1):
        captures = {f.capture for f in person}
        if any(decision.get(c) in ("keep", "review") for c in captures):
            continue
        acceptable = [
            f for f in person
            if decision.get(f.capture) == "reject" and not gates.failed(f.capture)
            and (f.eyes_closed is None or f.eyes_closed < settings.blink_threshold)
        ]
        if not acceptable:
            continue
        best = max(acceptable, key=lambda f: (f.area, -(f.eyes_closed or 0.0)))
        for path, d in files.items():
            if d.capture == best.capture:
                d.decision = "keep"
                d.reasons.append(f"kept: only good frame of person #{number} ({len(captures)} frame(s) of them, all otherwise rejected)")
        decision[best.capture] = "keep"
        promoted += 1
    return {"people": len(people), "people_promoted": promoted}
