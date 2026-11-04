"""People: faces clustered into identities (the ``people`` model feature), and people coverage.

Faces come from the faces scorer (MediaPipe landmarker + AdaFace, as in the app). Which frames get the people search
(the tiled small-face search plus identity vectors) is the app's person pre-filter (``face_index.plan_face_index``):
TinyCLIP's person probability >= ``PERSON_PRESENCE_THRESHOLD`` and not confidently animal-led. The app measured
recall ~0.93 for people while skipping ~74% of frames; a frame the filter misses keeps its whole-frame faces (blinks)
when it is contested, but no identity. Identities are clustered the way the app's People index clusters AdaFace vectors
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

# The app's person pre-filter (image_triage/face_index.py), unchanged: prompts averaged into class centroids, a softmax
# at CLIP's logit scale over person / no-person, and a human / animal margin that drops animal-led frames.
PERSON_PRESENCE_THRESHOLD = 0.15
ANIMAL_REJECTION_MARGIN = -0.05
PERSON_PROMPTS = (
    "a photo of a person",
    "a photo of people",
    "a photo of a man",
    "a photo of a woman",
    "a photo of a child",
    "a photo of a group of people",
    "a portrait of a person",
    "a candid photo of people",
)
NO_PERSON_PROMPTS = (
    "a landscape photo with no people",
    "a photo of scenery",
    "a photo of food",
    "a photo of an animal",
    "a photo of a building",
    "a close-up photo of an object",
    "an empty room with no people",
    "a photo of a plant",
)
HUMAN_PROMPTS = (
    "a photo of a human person",
    "a photograph of a man",
    "a photograph of a woman",
    "a photograph of a child",
    "a close-up portrait of a human face",
    "a group of human people",
)
ANIMAL_PROMPTS = (
    "a photo of an animal",
    "a photo of a monkey",
    "a photo of an ape",
    "a close-up portrait of an animal face",
    "a photo of wildlife",
    "a photo of a dog or cat",
)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def person_filter(engine, embeddings: dict[str, list[float]]) -> dict[str, dict]:
    """{capture: {"person_prob", "animal_margin", "flagged"}} for TinyCLIP image embeddings (``engine.encode_text``
    encodes the prompts)."""

    def proto(prompts):
        return _unit(np.mean([_unit(engine.encode_text(p)) for p in prompts], axis=0))

    person, noperson, human, animal = (proto(p) for p in (PERSON_PROMPTS, NO_PERSON_PROMPTS, HUMAN_PROMPTS, ANIMAL_PROMPTS))
    out = {}
    for cid, emb in embeddings.items():
        v = _unit(emb)
        logits = np.array([float(v @ person), float(v @ noperson)], dtype=np.float64) * 100.0
        logits -= logits.max()
        e = np.exp(logits)
        prob = float(e[0] / e.sum())
        margin = float(v @ human - v @ animal)
        out[cid] = {"person_prob": round(prob, 4), "animal_margin": round(margin, 4),
                    "flagged": prob >= PERSON_PRESENCE_THRESHOLD and margin > ANIMAL_REJECTION_MARGIN}
    return out


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
