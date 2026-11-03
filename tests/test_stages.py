"""Milestones 2-5 on hand-built shoots: contested sets, shortlist, blur gate, blinks, scenes, people, preference."""

import numpy as np
import pytest
from shoot_builder import ShootBuilder, run

from aiculler.api import CullSettings
from aiculler.gates import apply_gates
from aiculler.routing import route
from aiculler.stages import appeal_shortlist, blur_failures, contested


def _route(b: ShootBuilder, settings=None):
    settings = settings or CullSettings()
    shoot = b.build()
    routing = route(shoot, settings)
    return shoot, routing, apply_gates(shoot, routing, settings, b.scores), settings


# --------------------------------------------------------------------------- cheap -> expensive


def test_contested_skips_lone_frames_and_sets_that_keep_everyone():
    b = ShootBuilder()
    burst = b.group("burst", [{"name": f"U{i}", "sharpness": 10 + i} for i in range(4)], scene=1)
    lone = b.single("Lone", sharpness=5)  # alone in its scene: nothing to compare against
    pair = b.group("burst", [{"name": "P0", "sharpness": 1}, {"name": "P1", "sharpness": 2}], scene=2)  # keeps 2 of 2
    s1 = b.single("A", scene=3, sharpness=1)
    s2 = b.single("B", scene=3, sharpness=2, mean=0.0)  # black: gated, so scene 3 has one candidate left
    shoot, routing, gates, settings = _route(b)
    live = contested(routing, gates)
    assert live == set(burst)
    assert lone not in live and not live & set(pair) and s1 not in live and s2 not in live


def test_appeal_shortlist_is_the_technical_top_of_each_contested_set():
    b = ShootBuilder()
    ids = b.group("burst", [{"name": f"U{i}", "sharpness": s, "maniqa": m} for i, (s, m) in
                            enumerate([(10, 0.3), (50, 0.6), (40, 0.5), (5, 0.2), (45, 0.55), (8, 0.25)])], scene=1)
    singles = [b.single(f"S{i}", scene=1, sharpness=s, maniqa=m) for i, (s, m) in enumerate([(30, 0.4), (60, 0.7), (1, 0.1), (2, 0.15)])]
    shoot, routing, gates, settings = _route(b, CullSettings(appeal_shortlist=3))
    short = appeal_shortlist(routing, gates, b.scores, settings, contested(routing, gates))
    assert {ids[1], ids[2], ids[4]} <= short  # the burst's top 3 by MANIQA + sharpness
    assert ids[3] not in short and ids[5] not in short
    assert singles[1] in short and singles[0] in short and singles[2] not in short
    assert len(short) < len(contested(routing, gates))


def test_unshortlisted_frames_never_win_on_missing_appeal():
    b = ShootBuilder()
    b.group("burst", [{"name": "Short", "sharpness": 50, "appeal": 0.2}, {"name": "Pruned", "sharpness": 49},
                      {"name": "Other", "sharpness": 10, "appeal": 0.9}], scene=1)
    d, _ = run(b, CullSettings(bursts_keep=1))
    assert d["Other"].decision == "keep"  # the most appealing frame wins the burst
    assert d["Pruned"].decision == "reject" and d["Pruned"].scores["appeal_rel"] == 0.0


def test_blur_gate_needs_both_bottom_maniqa_and_far_below_set_median():
    b = ShootBuilder()
    burst = b.group("burst", [{"name": f"U{i}", "sharpness": s, "maniqa": m} for i, (s, m) in
                              enumerate([(400, 0.6), (410, 0.62), (20, 0.05), (390, 0.04)])], scene=1)
    for i in range(96):  # the rest of the shoot: 100 frames, so the bottom 2% is two frames
        b.single(f"X{i}", scene=100 + i, sharpness=300, maniqa=0.5 + i * 0.001)
    shoot, routing, gates, settings = _route(b)
    fails = blur_failures(shoot, routing, gates, b.scores, settings)
    # U2: bottom MANIQA and 5% of the burst's median sharpness -> gated.
    # U3: bottom MANIQA too but as sharp as its siblings -> not blur (MANIQA alone never gates).
    assert set(fails) == {burst[2]} and "severe blur" in fails[burst[2]].reason


def test_blur_gate_uses_face_sharpness_when_there_is_a_face():
    b = ShootBuilder()
    burst = b.group("burst", [{"name": "Soft", "sharpness": 400, "face_sharpness": 10, "maniqa": 0.01},
                              {"name": "A", "sharpness": 100, "face_sharpness": 300, "maniqa": 0.6},
                              {"name": "B", "sharpness": 100, "face_sharpness": 280, "maniqa": 0.6}], scene=1)
    for i in range(20):
        b.single(f"X{i}", scene=100 + i, sharpness=300, maniqa=0.5)
    shoot, routing, gates, settings = _route(b)
    fails = blur_failures(shoot, routing, gates, b.scores, settings)
    assert set(fails) == {burst[0]} and "face sharpness" in fails[burst[0]].reason


# --------------------------------------------------------------------------- faces and blinks


def test_blink_costs_the_frame_and_the_reason_says_so():
    b = ShootBuilder()
    b.group("burst", [{"name": "Blink", "sharpness": 100, "maniqa": 0.7, "eyes_closed": 0.85, "face_count": 1},
                      {"name": "Open", "sharpness": 90, "maniqa": 0.68, "eyes_closed": 0.1, "face_count": 1}], scene=1)
    d, _ = run(b, CullSettings(bursts_keep=1))
    assert d["Open"].decision == "keep" and d["Blink"].decision == "reject"
    assert "Open.NEF has eyes open" in d["Blink"].reasons[0]
    assert "score -0.30: eyes closed (0.85)" in d["Blink"].reasons
    d, _ = run(b, CullSettings(bursts_keep=1, require_eyes_open=False))
    assert d["Blink"].decision == "keep"


def test_group_shot_blink_names_someone():
    b = ShootBuilder()
    b.single("Group", scene=1, sharpness=100, eyes_closed=0.7, face_count=4)
    d, _ = run(b)
    assert "score -0.30: someone's eyes are closed (0.70)" in d["Group"].reasons


def test_signals_missing_from_a_whole_set_do_not_shrink_its_scores():
    b = ShootBuilder()
    b.single("A", scene=1, sharpness=100)
    b.single("B", scene=1, sharpness=25)
    d, _ = run(b)
    # Only sharpness and exposure take part, renormalised: B is ~2% as sharp-relative, equally exposed.
    assert d["A"].score == 1.0 and 0.40 < d["B"].score < 0.42


# --------------------------------------------------------------------------- scenes


def _emb(*values):
    v = np.asarray(values, dtype=np.float32)
    return list(v / np.linalg.norm(v))


def test_scene_split_and_merge_by_embedding():
    from aiculler.scenes import refine_scenes

    b = ShootBuilder()
    a = b.single("A", scene=1, time=0.0)
    bb = b.single("B", scene=1, time=10.0)  # same bracketeering scene, different view: split
    c = b.single("C", scene=2, time=30.0)  # different bracketeering scene, same view as B: merge
    d = b.single("D", time=500.0)  # same view as C but far later: stays apart
    shoot = b.build()
    emb = {a: _emb(1, 0, 0), bb: _emb(0, 1, 0), c: _emb(0, 1, 0.05), d: _emb(0, 1, 0.05)}
    stats = refine_scenes(shoot, emb, CullSettings())
    g = {x: shoot.group_of(x) for x in (a, bb, c, d)}
    assert stats == {"scene_splits": 1, "scene_merges": 1}
    assert g[a].scene == 1 and g[bb].scene == g[c].scene != 1 and g[d].scene is None
    assert "split from scene 1" in g[bb].scene_note and "merged with" in g[c].scene_note


def test_scene_note_reaches_reasons_and_changes_who_competes():
    from aiculler.decide import decide
    from aiculler.scenes import refine_scenes

    b = ShootBuilder()
    x = b.single("X", scene=1, time=0.0, sharpness=100)
    y = b.single("Y", scene=1, time=5.0, sharpness=10)  # would lose to X, but it is a different view
    settings = CullSettings()
    shoot = b.build()
    refine_scenes(shoot, {x: _emb(1, 0), y: _emb(0, 1)}, settings)
    routing = route(shoot, settings)
    files = decide(shoot, routing, apply_gates(shoot, routing, settings, b.scores), b.scores, settings)
    d = {p.rsplit("/", 1)[-1][:-4]: f for p, f in files.items()}
    assert d["X"].decision == d["Y"].decision == "keep"
    assert any("split from scene 1" in r for r in d["Y"].reasons)


# --------------------------------------------------------------------------- people coverage


def test_people_coverage_promotes_the_only_good_frame_of_a_person():
    from aiculler.decide import decide
    from aiculler.people import apply_people_coverage, find_people, tag_people

    b = ShootBuilder()
    alice, bob = _emb(1, 0, 0, 0), _emb(0, 1, 0, 0)
    ids = b.group("burst", [{"name": "Both", "sharpness": 100}, {"name": "Bob1", "sharpness": 50},
                            {"name": "Bob2", "sharpness": 40}, {"name": "BobBlink", "sharpness": 30}], scene=1)
    settings = CullSettings(bursts_keep=1)
    shoot = b.build()
    face = lambda vec, area=0.05, closed=0.1: {"box": [0.4, 0.3, 0.6, 0.6], "area": area, "eyes_closed": closed, "embedding": vec}  # noqa: E731
    details = {
        ids[0]: {"faces": [face(alice)]},
        ids[1]: {"faces": [face(bob, area=0.04)]},
        ids[2]: {"faces": [face(bob, area=0.06)]},  # Bob's largest face: the one to keep
        ids[3]: {"faces": [face(bob, area=0.2, closed=0.9)]},  # largest, but eyes closed: not acceptable
    }
    routing = route(shoot, settings)
    gates = apply_gates(shoot, routing, settings, b.scores)
    files = decide(shoot, routing, gates, b.scores, settings)
    people = find_people(shoot, details, settings.face_cluster_threshold)
    stats = apply_people_coverage(people, files, gates, settings)
    tag_people(people, files)
    d = {p.rsplit("/", 1)[-1][:-4]: f for p, f in files.items()}
    assert stats == {"people": 2, "people_promoted": 1}
    assert d["Both"].people == [1] and d["BobBlink"].people == [2]
    assert d["Both"].decision == "keep" and d["Bob2"].decision == "keep"
    assert d["Bob1"].decision == "reject" and d["BobBlink"].decision == "reject"
    assert "only good frame of person #2" in d["Bob2"].reasons[-1]


def test_similar_faces_cluster_into_one_person():
    from aiculler.people import FaceRef, cluster_people

    v = np.asarray(_emb(1, 0))
    w = np.asarray(_emb(0.9, 0.44))  # cosine ~0.9: same person
    groups = cluster_people([FaceRef("a", 0, 0.05, None, v), FaceRef("b", 0, 0.05, None, w)], 0.35)
    assert len(groups) == 1


def test_bystanders_are_indexed_but_keep_no_frame():
    from aiculler.decide import decide
    from aiculler.people import apply_people_coverage, find_people

    b = ShootBuilder()
    ids = b.group("burst", [{"name": "A", "sharpness": 100}, {"name": "B", "sharpness": 50}], scene=1)
    settings = CullSettings(bursts_keep=1)
    shoot = b.build()
    tiny = {"box": [0.1, 0.1, 0.12, 0.12], "area": 0.0004, "eyes_closed": 0.1, "embedding": _emb(0, 1, 0, 0)}
    details = {ids[1]: {"faces": [tiny]}}  # a bystander, only in the rejected frame
    routing = route(shoot, settings)
    gates = apply_gates(shoot, routing, settings, b.scores)
    files = decide(shoot, routing, gates, b.scores, settings)
    people = find_people(shoot, details, settings.face_cluster_threshold)
    assert len(people) == 1  # in the index...
    assert apply_people_coverage(people, files, gates, settings)["people_promoted"] == 0  # ...but not kept for


# --------------------------------------------------------------------------- preference


def test_preference_model_learns_and_round_trips(tmp_path):
    from aiculler.preference import PreferenceModel, train

    rng = np.random.default_rng(0)
    kept = rng.normal(1.0, 0.3, size=(15, 16))
    rejected = rng.normal(-1.0, 0.3, size=(60, 16))
    x = np.vstack([kept, rejected])
    y = np.array([1] * 15 + [0] * 60)
    model = train(x, y, embedding_revision="rev")
    assert model.positives == 15 and model.diagnostics["method"] == "leave-one-out"
    assert model.diagnostics["auc"] > 0.95
    path = model.save(tmp_path / "pref.joblib")
    loaded = PreferenceModel.load(path)
    p = loaded.score(np.vstack([kept[:3], rejected[:3]]))
    assert (p[:3] > 0.5).all() and (p[3:] < 0.5).all()
    with pytest.raises(ValueError):
        train(x[:3], y[:3], embedding_revision="rev")
