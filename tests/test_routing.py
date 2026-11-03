import pytest
from shoot_builder import ShootBuilder

from aiculler.api import CullSettings
from aiculler.routing import route


def _bracket(b: ShootBuilder, scene=1):
    return b.group(
        "exposure",
        [{"name": "B1", "mean": 0.45}, {"name": "B2", "mean": 0.1}, {"name": "B3", "mean": 0.8}],
        scene=scene,
        representative=0,
        best_exposed=2,
    )


def test_bracket_keep_set_is_one_unit_represented_by_best_exposed():
    b = ShootBuilder()
    ids = _bracket(b)
    r = route(b.build(), CullSettings(brackets="keep_set"))
    unit = r.units["B00001"]
    assert unit.captures == ids
    assert unit.rep == ids[2]  # best_exposed, not the metered frame
    assert not r.bursts and not r.passthrough
    # Only the representative is scored: the bracket's own frames never compete.
    assert r.scored_captures() == [ids[2]]


def test_bracket_without_best_exposed_uses_representative():
    b = ShootBuilder()
    ids = b.group("exposure", [{"name": "B1"}, {"name": "B2"}, {"name": "B3"}], scene=1, representative=0)
    r = route(b.build(), CullSettings())
    assert r.units["B00001"].rep == ids[0]


def test_bracket_pick_one_and_ignore():
    b = ShootBuilder()
    ids = _bracket(b)
    r = route(b.build(), CullSettings(brackets="pick_one"))
    assert r.pick_one and r.units["B00001"].rep == ids[2]

    r = route(b.build(), CullSettings(brackets="ignore"))
    assert set(r.passthrough) == set(ids)
    assert not r.units and not r.scenes and r.scored_captures() == []


def test_burst_competes_internally_and_enters_its_scene_as_one_unit():
    b = ShootBuilder()
    ids = b.group("burst", [{"name": f"U{i}"} for i in range(4)], scene=7)
    single = b.single("S1", scene=7)
    r = route(b.build(), CullSettings(bursts_keep=2, scene_keep=1))
    (burst,) = r.bursts
    assert burst.members == ids and burst.keep == 2
    (scene,) = r.scenes
    assert scene.id == "scene:7" and scene.members == ["U00001", "S00002"] and scene.keep == 1
    assert r.units["U00001"].rep is None  # chosen by the burst ranking
    assert set(r.scored_captures()) == set(ids) | {single}


def test_bursts_keep_zero_leaves_bursts_out():
    b = ShootBuilder()
    ids = b.group("burst", [{"name": f"U{i}"} for i in range(3)], scene=1)
    r = route(b.build(), CullSettings(bursts_keep=0))
    assert set(r.passthrough) == set(ids) and not r.bursts and not r.units


@pytest.mark.parametrize("kind,setting", [("focus", "focus_stacks"), ("timelapse", "timelapses")])
def test_sequences_pass_through_unless_culled(kind, setting):
    b = ShootBuilder()
    ids = b.group(kind, [{"name": f"F{i}"} for i in range(5)], scene=2)
    r = route(b.build(), CullSettings())
    assert set(r.passthrough) == set(ids)
    r = route(b.build(), CullSettings(**{setting: "cull"}))
    unit = next(iter(r.units.values()))
    assert unit.captures == ids and unit.rep == ids[2] and not r.passthrough


def test_scenes_never_mix_and_unsceneed_groups_are_alone():
    b = ShootBuilder()
    b.single("A", scene=1)
    b.single("B", scene=1)
    b.single("C", scene=2)
    b.single("D")  # no scene id
    r = route(b.build(), CullSettings())
    sets = {s.id: s.members for s in r.scenes}
    assert sets == {"scene:1": ["S00001", "S00002"], "scene:2": ["S00003"], "solo:S00004": ["S00004"]}


def test_unreadable_captures_are_held_out():
    b = ShootBuilder()
    ok, bad = b.group("burst", [{"name": "U1"}, {"name": "U2", "error": "zero-filled file"}], scene=1)
    r = route(b.build(), CullSettings())
    assert r.unreadable == {bad: "zero-filled file"}
    assert r.bursts[0].members == [ok]


def test_settings_validation_and_json_round_trip():
    s = CullSettings.from_dict({"brackets": "pick_one", "bursts_keep": 3, "gates": {"black_mean": 0.03}})
    assert s.gates.black_mean == 0.03
    assert CullSettings.from_dict(s.to_dict()) == s
    with pytest.raises(ValueError):
        CullSettings(brackets="merge")
    with pytest.raises(ValueError):
        CullSettings(bursts_keep=-1)
    with pytest.raises(ValueError):
        CullSettings.from_dict({"burst_keep": 2})  # typo: unknown setting
