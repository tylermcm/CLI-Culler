from shoot_builder import ShootBuilder, run

from aiculler.api import CullSettings, GateSettings
from aiculler.decide import normalise


def _scene_with_bracket(b: ShootBuilder, bracket_sharpness: float, single_sharpness: float):
    # Bracket frames are dark and bright on purpose: B2 would fail any exposure gate on its own.
    b.group(
        "exposure",
        [
            {"name": "B1", "sharpness": 100.0},
            {"name": "B2", "sharpness": 900.0, "mean": 0.01, "sh": 0.9},
            {"name": "B3", "sharpness": bracket_sharpness, "hl": 0.0},
        ],
        scene=1,
        representative=0,
        best_exposed=2,
    )
    b.single("S1", scene=1, sharpness=single_sharpness)


# --------------------------------------------------------------------------- brackets


def test_keep_set_keeps_or_rejects_the_whole_bracket():
    b = ShootBuilder()
    _scene_with_bracket(b, bracket_sharpness=400.0, single_sharpness=100.0)
    d, _ = run(b, CullSettings(brackets="keep_set"))
    assert [d[n].decision for n in ("B1", "B2", "B3")] == ["keep"] * 3
    assert d["S1"].decision == "reject"
    assert "kept as a set (keep_set)" in d["B2"].reasons[0]

    b = ShootBuilder()
    _scene_with_bracket(b, bracket_sharpness=50.0, single_sharpness=400.0)
    d, _ = run(b, CullSettings(brackets="keep_set"))
    assert [d[n].decision for n in ("B1", "B2", "B3")] == ["reject"] * 3
    assert d["S1"].decision == "keep"
    assert "S1.NEF is sharper" in d["B1"].reasons[0]


def test_bracket_is_judged_only_on_its_best_exposed_frame():
    # B2's sharpness (900) would win the scene if bracket frames competed; it must not count.
    b = ShootBuilder()
    _scene_with_bracket(b, bracket_sharpness=50.0, single_sharpness=400.0)
    d, _ = run(b)
    assert d["B2"].decision == "reject"
    assert d["B2"].rank_in_set == 2 and d["B2"].set_size == 2  # two units in the scene, not four frames


def test_bracket_frames_are_never_gated_on_exposure():
    b = ShootBuilder()
    b.group(
        "exposure",
        [{"name": "B1", "mean": 0.01}, {"name": "B2", "hl": 0.6, "clip_region": 0.6, "top_clip": 1.0}, {"name": "B3", "sh": 0.9}],
        representative=0,
        best_exposed=0,
    )
    d, _ = run(b)
    assert {x.decision for x in d.values()} == {"keep"}
    assert not any("black" in r or "clipping" in r for x in d.values() for r in x.reasons)


def test_pick_one_keeps_only_best_exposed():
    b = ShootBuilder()
    _scene_with_bracket(b, bracket_sharpness=400.0, single_sharpness=100.0)
    d, _ = run(b, CullSettings(brackets="pick_one"))
    assert d["B3"].decision == "keep"
    assert d["B1"].decision == d["B2"].decision == "reject"
    assert "not the best-exposed frame" in d["B1"].reasons[0]


def test_ignore_keeps_every_bracket_frame_unranked():
    b = ShootBuilder()
    _scene_with_bracket(b, bracket_sharpness=1.0, single_sharpness=400.0)
    d, _ = run(b, CullSettings(brackets="ignore"))
    assert [d[n].decision for n in ("B1", "B2", "B3")] == ["keep"] * 3
    assert d["B1"].comparison_set is None and "brackets: ignore" in d["B1"].reasons[0]
    # The single is now alone in its scene.
    assert d["S1"].decision == "keep"


# --------------------------------------------------------------------------- bursts


def test_burst_keeps_top_n_and_names_the_winner():
    b = ShootBuilder()
    b.group("burst", [{"name": f"U{i}", "sharpness": s} for i, s in enumerate([50, 400, 100, 380, 20])], scene=1)
    d, _ = run(b, CullSettings(bursts_keep=2))
    assert [d[f"U{i}"].decision for i in range(5)] == ["reject", "keep", "reject", "keep", "reject"]
    assert d["U1"].rank_in_set == 1 and d["U3"].rank_in_set == 2 and d["U1"].comparison_set == "U00001"
    assert d["U2"].reasons[0].startswith("rejected: 2 better frames in burst U00001 (keeping 2); U1.NEF is sharper")


def test_burst_keep_n_larger_than_burst_keeps_all():
    b = ShootBuilder()
    b.group("burst", [{"name": "U0", "sharpness": 10}, {"name": "U1", "sharpness": 20}], scene=1)
    d, _ = run(b, CullSettings(bursts_keep=3))
    assert {x.decision for x in d.values()} == {"keep"}


def test_burst_winner_represents_the_burst_in_its_scene():
    b = ShootBuilder()
    b.group("burst", [{"name": f"U{i}", "sharpness": s} for i, s in enumerate([100, 120, 90])], scene=4)
    b.single("S1", scene=4, sharpness=500)
    d, _ = run(b, CullSettings(bursts_keep=2, scene_keep=1))
    # The burst lost the scene, so even its two burst-winners are rejected.
    assert {d[f"U{i}"].decision for i in range(3)} == {"reject"}
    assert "burst U00001 lost in its scene" in d["U1"].reasons[-1]
    assert d["S1"].decision == "keep"


def test_bursts_never_compared_across_scenes():
    b = ShootBuilder()
    b.group("burst", [{"name": "U0", "sharpness": 10}, {"name": "U1", "sharpness": 12}], scene=1)
    b.single("S1", scene=2, sharpness=10_000)
    d, _ = run(b, CullSettings(bursts_keep=1))
    assert d["U1"].decision == "keep" and d["S1"].decision == "keep"


# --------------------------------------------------------------------------- scenes


def test_scene_keep_n():
    b = ShootBuilder()
    for i, s in enumerate([100, 300, 200, 50]):
        b.single(f"S{i}", scene=9, sharpness=s)
    d, _ = run(b, CullSettings(scene_keep=2))
    assert [d[f"S{i}"].decision for i in range(4)] == ["reject", "keep", "keep", "reject"]
    assert d["S2"].reasons == ["rank 2 of 4 in scene 9 (keeping 2)"]
    assert d["S0"].reasons[0].startswith("rejected: 2 better units in scene 9 (keeping 2)")


def test_lone_frame_is_kept_with_a_reason():
    b = ShootBuilder()
    b.single("S0", sharpness=1.0)
    d, _ = run(b)
    assert d["S0"].decision == "keep"
    assert d["S0"].reasons == ["only unit in its scene: nothing to compare against"]
    assert d["S0"].comparison_set is None


def test_near_identical_frames_get_near_identical_scores():
    rel = normalise({"a": 1000.0, "b": 990.0, "c": 250.0}, "sharpness")
    assert rel["a"] == 1.0 and rel["b"] > 0.99 and rel["c"] < 0.01
    assert normalise({"a": 1.0, "b": 2.0, "c": 3.0}, "sharpness", "rank") == {"a": 0.0, "b": 0.5, "c": 1.0}
    assert normalise({"a": None, "b": 2.0}, "sharpness")["a"] == 0.5


# --------------------------------------------------------------------------- gates and review


def test_gate_rejects_clear_failures_only():
    b = ShootBuilder()
    b.single("Black", scene=3, sharpness=500, mean=0.01)
    b.single("Clipped", scene=3, sharpness=500, clip_region=0.4)
    b.single("Fine", scene=3, sharpness=100)
    d, _ = run(b)
    assert d["Black"].decision == "reject" and "black frame" in d["Black"].reasons[0]
    assert d["Clipped"].decision == "reject" and "one clipped region covers 40% of the frame" in d["Clipped"].reasons[0]
    assert d["Clipped"].gates == ["clip_region"]
    assert d["Fine"].decision == "keep"


def test_whole_scene_failing_gates_keeps_least_bad_for_review():
    b = ShootBuilder()
    b.single("A", scene=3, sharpness=100, clip_region=0.6)
    b.single("B", scene=3, sharpness=300, clip_region=0.4)
    b.single("C", scene=3, sharpness=900, mean=0.005)  # black: never rescued
    d, _ = run(b)
    assert d["B"].decision == "review"
    assert "every candidate in scene 3 failed a gate" in d["B"].reasons[-1]
    assert d["A"].decision == "reject" and d["C"].decision == "reject"


def test_lone_black_frame_is_rejected_not_rescued():
    b = ShootBuilder()
    b.single("Misfire", mean=0.0)
    d, _ = run(b)
    assert d["Misfire"].decision == "reject"


def test_burst_entirely_gated_is_rescued_when_alone_in_scene():
    b = ShootBuilder()
    b.group("burst", [{"name": "U0", "sharpness": 50, "clip_region": 0.5}, {"name": "U1", "sharpness": 80, "top_clip": 0.8}], scene=1)
    d, _ = run(b)
    assert d["U1"].decision == "review" and d["U0"].decision == "reject"
    assert not any(r.startswith("rejected:") for r in d["U1"].reasons)


def test_low_confidence_or_warned_group_goes_to_review():
    b = ShootBuilder()
    b.group("burst", [{"name": f"U{i}", "sharpness": s} for i, s in enumerate([10, 50, 30])], scene=1, confidence=0.55)
    b.group(
        "exposure",
        [{"name": "B1"}, {"name": "B2"}, {"name": "B3"}],
        scene=2,
        warnings=["partial bracket: 3 of 5 slots"],
    )
    d, _ = run(b, CullSettings(bursts_keep=1, review_below=0.7))
    assert {d[f"U{i}"].decision for i in range(3)} == {"review"}
    assert "review instead of auto-reject: grouping confidence 0.55 < 0.7" in d["U0"].reasons[-1]
    assert "review instead of auto-keep" in d["U1"].reasons[-1]
    # A partial bracket is still a bracket: noted, decided as usual.
    assert {d[n].decision for n in ("B1", "B2", "B3")} == {"keep"}
    assert d["B1"].reasons[-1] == "note: bracketeering: partial bracket: 3 of 5 slots"
    d, _ = run(b, CullSettings(bursts_keep=1, review_warnings="all"))
    assert {d[n].decision for n in ("B1", "B2", "B3")} == {"review"}
    assert "partial bracket" in d["B1"].reasons[-1]


def test_warnings_that_doubt_the_grouping_go_to_review():
    b = ShootBuilder()
    b.group("single", [{"name": "W", "sharpness": 50}], scene=1, warnings=["weak bracket candidate (0.41) below threshold"])
    b.group("single", [{"name": "F", "sharpness": 50}], scene=2, warnings=["flagged auto bracket but no partner frames found"])
    b.group("single", [{"name": "P", "sharpness": 50}], scene=3,
            warnings=["auto-bracket frame (slot +1) without partners: rest of its cycle recomposed or deleted"])
    d, _ = run(b)
    assert d["W"].decision == d["F"].decision == "review" and d["P"].decision == "keep"
    d, _ = run(b, CullSettings(review_warnings="none"))
    assert {x.decision for x in d.values()} == {"keep"}


def test_gate_rejection_survives_review_override():
    b = ShootBuilder()
    b.group("burst", [{"name": "U0", "sharpness": 50, "mean": 0.0}, {"name": "U1", "sharpness": 60}], scene=1, confidence=0.4)
    d, _ = run(b)
    assert d["U0"].decision == "reject" and d["U1"].decision == "review"


def test_featureless_goes_to_review_by_default_and_reject_when_asked():
    b = ShootBuilder()
    b.single("Moon", sharpness=5, featureless=True, mean=0.05)
    d, _ = run(b)
    assert d["Moon"].decision == "review"
    d, _ = run(b, CullSettings(gates=GateSettings(featureless="reject")))
    assert d["Moon"].decision == "review"  # alone in its scene and not black: least-bad, kept for review

    b = ShootBuilder()
    b.single("Moon", scene=5, sharpness=50, featureless=True, mean=0.05)
    b.single("Other", scene=5, sharpness=5)
    d, _ = run(b, CullSettings(gates=GateSettings(featureless="reject")))
    assert d["Moon"].decision == "reject" and "featureless" in d["Moon"].reasons[0]
    assert d["Other"].decision == "keep"


# --------------------------------------------------------------------------- files


def test_sidecars_copies_and_exports_follow_their_shot():
    b = ShootBuilder()
    b.group(
        "burst",
        [
            {"name": "U0", "sharpness": 50, "extra_files": ["U0.JPG", "_picks/U0.NEF"]},
            {"name": "U1", "sharpness": 400, "extra_files": ["U1.JPG", "Edits/U1.tif"]},
        ],
        scene=1,
    )
    _, files = run(b, CullSettings(bursts_keep=1))
    assert len(files) == 6
    for path, d in files.items():
        shot = "U0" if "U0" in path else "U1"
        assert d.decision == ("reject" if shot == "U0" else "keep")
        assert d.capture == f"cam|sc:{shot}"
        assert d.primary == path.endswith(f"/shoot/{shot}.NEF")
        if not d.primary:
            assert d.reasons[-1] == f"follows its shot {shot}.NEF"


def test_unreadable_file_goes_to_review():
    b = ShootBuilder()
    b.group("burst", [{"name": "U0", "sharpness": 50}, {"name": "U1", "error": "zero-filled file"}], scene=1)
    d, _ = run(b)
    assert d["U1"].decision == "review" and d["U1"].reasons == ["unreadable: zero-filled file"]
    assert d["U0"].decision == "keep"


def test_passthrough_sequences_are_kept():
    b = ShootBuilder()
    b.group("focus", [{"name": f"F{i}", "sharpness": i} for i in range(4)], scene=1)
    b.group("timelapse", [{"name": f"T{i}", "sharpness": i} for i in range(6)])
    d, _ = run(b)
    assert {x.decision for x in d.values()} == {"keep"}
    assert "focus_stacks: keep_all" in d["F0"].reasons[0]


# --------------------------------------------------------------------------- highlights by region


def test_small_clipped_regions_cost_nothing_however_much_clips_overall():
    b = ShootBuilder()
    # 40% of pixels clipped overall, but in specks (largest region 3%): no gate, no penalty.
    b.single("Specks", scene=1, sharpness=100, hl=0.4, clip_region=0.03, top_clip=0.3)
    b.single("Clean", scene=1, sharpness=100, clip_region=0.0)
    d, _ = run(b)
    assert d["Specks"].gates == [] and "clip_penalty" not in d["Specks"].scores
    assert d["Specks"].score == d["Clean"].score


def test_mid_sized_region_is_a_score_penalty_only():
    b = ShootBuilder()
    b.single("Mid", scene=1, sharpness=100, clip_region=0.20)  # halfway between 5% and 35%
    b.single("Clean", scene=1, sharpness=100, clip_region=0.0)
    d, _ = run(b)
    assert d["Mid"].gates == [] and d["Mid"].scores["clip_penalty"] == 0.075
    assert d["Mid"].decision == "reject" and d["Clean"].decision == "keep"
    assert d["Mid"].reasons[-1] == "score -0.07: one clipped region covers 20% of the frame"
    # A sharper frame still wins despite the penalty: it is a nudge, not a gate.
    b = ShootBuilder()
    b.single("Mid", scene=1, sharpness=400, clip_region=0.20)
    b.single("Clean", scene=1, sharpness=100, clip_region=0.0)
    d, _ = run(b)
    assert d["Mid"].decision == "keep"


def test_region_sky_and_washed_out_gates():
    b = ShootBuilder()
    b.single("Region", scene=1, sharpness=900, clip_region=0.35)
    b.single("Sky", scene=1, sharpness=900, clip_region=0.2, top_clip=0.75)
    b.single("Washed", scene=1, sharpness=900, luma_mean=0.85, luma_spread=0.1)
    b.single("Bright", scene=1, sharpness=900, luma_mean=0.85, luma_spread=0.4)  # high-key but contrasty: fine
    b.single("Fine", scene=1, sharpness=10)
    d, _ = run(b, CullSettings(scene_keep=2))
    assert d["Region"].gates == ["clip_region"]
    assert d["Sky"].gates == ["sky_blown"] and "sky blown out (75% of the top third clipped" in d["Sky"].reasons[0]
    assert d["Washed"].gates == ["washed_out"] and "washed out" in d["Washed"].reasons[0]
    assert d["Bright"].gates == [] and d["Bright"].decision == "keep"
    assert d["Fine"].decision == "keep"  # the only other frame that passed the gates


def test_region_thresholds_are_configurable():
    b = ShootBuilder()
    b.single("Sky", scene=1, sharpness=900, top_clip=0.7, clip_region=0.3)
    b.single("Other", scene=1, sharpness=10)
    d, _ = run(b, CullSettings(gates=GateSettings(clip_region_reject=0.35, sky_blown=0.8)))
    assert d["Sky"].gates == [] and d["Sky"].decision == "keep"


def test_old_highlights_setting_is_refused_with_a_pointer():
    import pytest

    with pytest.raises(ValueError, match="replaced by region rules"):
        CullSettings.from_dict({"gates": {"highlights_clipped": 0.3}})


def test_bracket_frames_never_get_the_clip_penalty():
    b = ShootBuilder()
    b.group("exposure", [{"name": "B1", "sharpness": 100, "clip_region": 0.2}], scene=1, best_exposed=0)
    b.single("S1", scene=1, sharpness=100, clip_region=0.0)
    d, _ = run(b)
    assert "clip_penalty" not in d["B1"].scores and d["B1"].gates == []


# Closest final picks to each gate on China 8-26 (evaluate --truth picks, values as reported, rounded up).
CHINA_8_26_CLOSEST_PICKS = {
    "_DSC1055": {"clip_region": 0.245, "top_clip": 0.595, "luma_mean": 0.6, "luma_spread": 0.5},
    "_DSC1048": {"shadows_crushed": 0.365, "mean": 0.15},
    "_DSC0570": {"luma_mean": 0.465, "luma_spread": 0.69},
}


def test_default_gates_pass_china_picks_with_margin():
    from aiculler.gates import highlight_failures

    s = CullSettings()
    g = s.gates
    for name, v in CHINA_8_26_CLOSEST_PICKS.items():
        assert highlight_failures(v, s) == [], name
    pick = CHINA_8_26_CLOSEST_PICKS["_DSC1055"]
    assert g.clip_region_reject - pick["clip_region"] >= 0.10  # not a one-point pass
    assert g.sky_blown - pick["top_clip"] >= 0.10
    assert g.shadows_crushed - CHINA_8_26_CLOSEST_PICKS["_DSC1048"]["shadows_crushed"] >= 0.10
    assert CHINA_8_26_CLOSEST_PICKS["_DSC1048"]["mean"] - g.black_mean >= 0.10

    b = ShootBuilder()
    b.single("_DSC1055", scene=1, sharpness=100, **CHINA_8_26_CLOSEST_PICKS["_DSC1055"])
    b.single("_DSC1048", scene=2, sharpness=100, sh=0.365, mean=0.15)
    b.single("_DSC0570", scene=3, sharpness=100, **CHINA_8_26_CLOSEST_PICKS["_DSC0570"])
    d, _ = run(b)
    assert all(x.gates == [] and x.decision == "keep" for x in d.values())
