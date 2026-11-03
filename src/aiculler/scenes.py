"""Scene refinement with TinyCLIP (milestone 4).

bracketeering's near-duplicate scenes are its weakest output: hazy horizons over-merge and some same-view pairs
split. Per camera body, in capture order, using each group's representative embedding (``best_exposed`` for a
bracket, so its dark and bright frames do not skew it):

* **split** a scene between consecutive groups whose cosine similarity is below ``split_cos``;
* **merge** adjacent scenes (or lone groups) within ``merge_window`` seconds whose boundary groups are at least
  ``merge_cos`` similar, unless bracketeering's registration contradicts it (structure says "different" and
  the tonal layout does not support a match).

Each change is recorded on the group (``scene_note``) and appears in that file's reasons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aiculler.ingest import GroupInfo, Shoot

if TYPE_CHECKING:
    from aiculler.api import CullSettings

LAYOUT_SUPPORT = 0.85  # bracketeering's PixelConfig.layout_similarity


def _vector(shoot: Shoot, g: GroupInfo, embeddings: dict[str, list[float]]) -> np.ndarray | None:
    rep = g.best_exposed or g.representative
    vec = embeddings.get(rep)
    if vec is None:
        vecs = [embeddings[c] for c in g.captures if c in embeddings]
        if not vecs:
            return None
        vec = np.mean(vecs, axis=0)
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _times(shoot: Shoot, g: GroupInfo) -> tuple[float | None, float | None]:
    ts = [shoot.captures[c].time for c in g.captures if shoot.captures[c].time is not None]
    return (min(ts), max(ts)) if ts else (None, None)


def _contradicts(shoot: Shoot, a: GroupInfo, b: GroupInfo) -> bool:
    from bracketeering.pixels import DIFFERENT, layout_similarity, similarity

    ca, cb = shoot.captures[a.best_exposed or a.representative], shoot.captures[b.best_exposed or b.representative]
    if ca.pixels is None or cb.pixels is None:
        return False
    sim = similarity(ca, cb)
    if sim is None or sim >= DIFFERENT:
        return False
    lay = layout_similarity(ca, cb)
    return lay is None or lay < LAYOUT_SUPPORT


def refine_scenes(shoot: Shoot, embeddings: dict[str, list[float]], settings: "CullSettings") -> dict[str, int]:
    """Split and merge scenes in place; returns counts."""
    groups = list(shoot.groups.values())
    next_id = max([g.scene for g in groups if g.scene is not None] + [0]) + 1
    vec = {g.id: _vector(shoot, g, embeddings) for g in groups}
    splits = merges = 0

    by_camera: dict[str, list[GroupInfo]] = {}
    for g in groups:  # bracketeering lists groups in capture order
        by_camera.setdefault(shoot.captures[g.captures[0]].camera, []).append(g)

    for cam_groups in by_camera.values():
        # 1. Split: within each bracketeering scene, cut where consecutive groups do not look alike.
        last_in_scene: dict[int, GroupInfo] = {}
        current: dict[int, int] = {}  # original scene -> scene id now in use
        for g in cam_groups:
            if g.scene is None:
                continue
            original = g.scene
            prev = last_in_scene.get(original)
            if prev is not None and vec[prev.id] is not None and vec[g.id] is not None:
                cos = float(np.dot(vec[prev.id], vec[g.id]))
                if cos < settings.split_cos:
                    current[original] = next_id
                    next_id += 1
                    splits += 1
                    g.scene_note = f"scene split from scene {original} by embedding similarity {cos:.2f} (< {settings.split_cos:g})"
            last_in_scene[original] = g
            g.scene = current.get(original, original)

        # 2. Merge: adjacent segments (a lone group is a segment of its own) that look alike and are close in time.
        segments: list[list[GroupInfo]] = []
        for g in cam_groups:
            if segments and g.scene is not None and segments[-1][-1].scene == g.scene:
                segments[-1].append(g)
            else:
                segments.append([g])
        merged: list[list[GroupInfo]] = [segments[0]] if segments else []
        for seg in segments[1:]:
            a, b = merged[-1][-1], seg[0]
            end_a, start_b = _times(shoot, a)[1], _times(shoot, b)[0]
            close = end_a is not None and start_b is not None and 0 <= start_b - end_a <= settings.merge_window
            va, vb = vec[a.id], vec[b.id]
            cos = float(np.dot(va, vb)) if va is not None and vb is not None else None
            if close and cos is not None and cos >= settings.merge_cos and not _contradicts(shoot, a, b):
                target = next((x.scene for x in merged[-1] if x.scene is not None), None)
                if target is None:
                    target = next_id
                    next_id += 1
                    for x in merged[-1]:
                        x.scene = target
                for x in seg:
                    x.scene = target
                    x.scene_note = f"scene merged with {a.id} by embedding similarity {cos:.2f} (>= {settings.merge_cos:g})"
                merged[-1].extend(seg)
                merges += 1
            else:
                merged.append(seg)
    return {"scene_splits": splits, "scene_merges": merges}
