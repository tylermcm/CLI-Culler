"""ai-culler sample / calibrate: a seeded sample across shoots, a blind labelling page, thresholds from labels."""

import csv
import json

import pytest

pytest.importorskip("bracketeering")

from shootgen import Frame, write_shoot  # noqa: E402
from test_models_pipeline import FakeManiqa  # noqa: E402

from aiculler.baseline import calibrate, draw, make_sample, read_labels  # noqa: E402
from aiculler.models.registry import ModelRegistry  # noqa: E402


def test_draw_gives_equal_shares_and_passes_leftovers_on():
    picked = draw({"a": [str(i) for i in range(50)], "b": ["x", "y"], "c": [str(i) for i in range(50)]}, 10, seed=3)
    assert len(picked["b"]) == 2 and len(picked["a"]) + len(picked["c"]) == 8 and abs(len(picked["a"]) - len(picked["c"])) <= 1
    assert picked == draw({"a": [str(i) for i in range(50)], "b": ["x", "y"], "c": [str(i) for i in range(50)]}, 10, seed=3)


def test_sample_then_calibrate(tmp_path):
    shoots = []
    for k in range(2):
        root = tmp_path / f"shoot{k}"
        write_shoot(root, [Frame(f"DSC_{k}{i:03d}", [0, 70, 200, 260, 500, 900][i], i, marker=10 * (i + 1) + k) for i in range(6)])
        shoots.append(root)
    quality = {10 * (i + 1) + k: 0.15 + 0.05 * i + 0.01 * k for i in range(6) for k in range(2)}
    reg = ModelRegistry.fake({"maniqa": FakeManiqa(quality)})
    out = tmp_path / "sample"
    doc = make_sample(shoots, out, n=8, seed=1, models=reg, cache_db=tmp_path / "c.sqlite")
    assert doc["drawn"] == 8 and set(doc["per_folder"].values()) == {4} and doc["warnings"] == []
    page = (out / "label.html").read_text()
    assert all(str(r["scores"]["maniqa"]) not in page for r in doc["rows"])  # blind: no scores on the page
    assert len(list((out / "thumbs").glob("*.jpg"))) == 8

    # Label: the lowest-quality frames rejected, the middle borderline, the rest kept.
    rows = sorted(doc["rows"], key=lambda r: r["scores"]["maniqa"])
    labels = {r["id"]: ("reject" if i < 2 else "borderline" if i < 4 else "keep") for i, r in enumerate(rows)}
    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "label"])
        for i, lab in labels.items():
            w.writerow([i, "x", lab])
    cal = calibrate(doc, read_labels(out / "labels.csv"))["signals"]["maniqa/raw"]
    lowest_protected = rows[2]["scores"]["maniqa"]
    assert cal["suggested"]["quality_reject"] == pytest.approx(lowest_protected - 0.02, abs=1e-3)
    assert cal["caught"]["quality_reject"]["keep"] == cal["caught"]["quality_reject"]["borderline"] == 0
    assert cal["caught"]["quality_reject"]["reject"] >= 1
    assert cal["suggested"]["quality_review"] >= cal["suggested"]["quality_reject"]

    from aiculler.cli import main

    assert main(["--no-log", "calibrate", str(out), "--out", str(tmp_path / "cal.md")]) == 0
    assert "quality_reject" in (tmp_path / "cal.md").read_text()
    json.loads((out / "sample.json").read_text())


def test_bad_labels_are_refused(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("id,name,label\n001,x,maybe\n")
    with pytest.raises(ValueError, match="label must be one of"):
        read_labels(p)


# --------------------------------------------------------------------------- categories and library baselines


class TextClip:
    """Fake TinyCLIP with a text side: an image's marker picks its axis; a prompt's category picks the same axis."""

    AXES = {"animals": 0, "landscape": 1, "night": 2}

    def __init__(self, image_axis):
        self.image_axis = image_axis  # marker -> axis

    def embed(self, image):
        from shootgen import read_marker

        v = [0.05] * 4
        v[self.image_axis.get(read_marker(image), 3)] = 1.0
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v]

    def encode_text(self, prompt):
        from aiculler.categories import CATEGORIES

        cat = next(c for c, ps in CATEGORIES.items() if prompt in ps)
        v = [0.0] * 4
        v[self.AXES.get(cat, 3)] = 1.0
        return v


def test_classify_is_multi_label_and_techniques_read_exif():
    from types import SimpleNamespace

    import numpy as np

    from aiculler.categories import Prototypes, classify, techniques

    protos = Prototypes(["a", "b", "c"], np.eye(3, dtype=np.float32), "m")
    both = classify(np.array([0.7071, 0.7071, 0.0]), protos)
    assert set(both["labels"]) == {"a", "b"} and both["primary"] in ("a", "b")
    assert classify(np.array([1.0, 0.0, 0.0]), protos)["labels"] == ["a"]
    cap = SimpleNamespace(iso=6400, exposure_time=1 / 1000, focal_length=500, f_number=5.6, flash=False)
    assert techniques(cap) == ["high_iso", "long_lens"]


def test_library_percentiles_prefer_the_most_favourable_usable_cell():
    from aiculler.baseline import LIBRARY_SCHEMA, LibraryBaseline
    from aiculler.categories import VERSION

    recs = [{"shoot": "s", "capture": f"a{i}", "maniqa": 0.10 + i / 1000, "labels": ["animals"], "techniques": ["long_lens"]} for i in range(60)]
    recs += [{"shoot": "s", "capture": f"l{i}", "maniqa": 0.40 + i / 1000, "labels": ["landscape"], "techniques": []} for i in range(60)]
    lib = LibraryBaseline({"schema": LIBRARY_SCHEMA, "maniqa": "m", "tinyclip": "t", "categories_version": VERSION, "records": recs})
    # 0.13 is low for the library (bottom quarter) but mid-pack for animals.
    assert lib.percentile(0.13, ["landscape"])[0] == 0.0
    pct, cell = lib.percentile(0.13, ["landscape", "animals"])
    assert abs(pct - 0.5) < 0.02 and "animals" in cell
    assert lib.percentile(0.13, ["macro"])[1].startswith("all 120")  # no cell: the whole library
    assert "animals + long_lens" in lib.percentile(0.13, ["animals"], ["long_lens"])[1] or lib.percentile(0.13, ["animals"], ["long_lens"])[0] >= 0.5
    with pytest.raises(ValueError, match="rebuild it"):
        lib.check("other-maniqa", "t")


def test_category_floor_end_to_end(tmp_path):
    """An animal frame with a low MANIQA score is fine among animals; a landscape frame with the same score is not."""
    from aiculler import CullSettings, cull
    from aiculler.api import GateSettings
    from aiculler.baseline import LibraryBaseline, build_library

    lib_root = tmp_path / "library"
    # Library: animals score low (0.10-0.16), landscapes high (0.40-0.46).
    # (Markers are read back to the nearest 10: 10 marks the animal frames, 20 the landscapes.)
    frames = [Frame(f"DSC_{i:04d}", i * 37.0 + (i % 3), i % 8, marker=10 if i < 60 else 20) for i in range(120)]
    write_shoot(lib_root, frames)
    axis = {10: 0, 20: 1}
    quality = {10: 0.10, 20: 0.40}
    reg = ModelRegistry.fake({"maniqa": FakeManiqa(quality), "tinyclip": TextClip(axis)})
    doc = build_library([lib_root], reg, cache_db=tmp_path / "lib.sqlite")
    lib = LibraryBaseline(doc)
    assert lib.size("animals") >= 50 and lib.size("landscape") >= 50
    path = tmp_path / "library.json"
    path.write_text(json.dumps(doc))

    shoot = tmp_path / "shoot"
    write_shoot(shoot, [Frame("BEAR", 0.0, 0, marker=200), Frame("HILL", 900.0, 1, marker=210)])
    reg = ModelRegistry.fake({"maniqa": FakeManiqa({200: 0.13, 210: 0.13}), "tinyclip": TextClip({200: 0, 210: 1})})
    settings = CullSettings(models=["maniqa"], gates=GateSettings(quality_mode="category", quality_reject=0.05,
                                                                  quality_baseline=str(path), quality_unique="reject"))
    result = cull(shoot, settings, models=reg, cache_db=tmp_path / "c.sqlite")
    d = {p.rsplit("/", 1)[-1]: f for p, f in result.files.items()}
    assert d["BEAR.JPG"].decision == "keep" and d["BEAR.JPG"].categories[0] == "animals"
    assert d["HILL.JPG"].decision == "reject"
    assert "percentile 0.00 of" in d["HILL.JPG"].reasons[0]
    assert "tinyclip" in result.models  # the category floor brings TinyCLIP in without "search"


def test_tinyclip_runs_first_on_every_capture_bracket_frames_included(tmp_path):
    from aiculler import CullSettings, cull

    root = tmp_path / "shoot"
    frames = [Frame(f"B{i}", i * 0.3, 0, bias=b, mode=2, marker=10) for i, b in enumerate((-1.0, 0.0, 1.0))]
    frames.append(Frame("S", 600.0, 1, marker=20))
    write_shoot(root, frames)
    reg = ModelRegistry.fake({"tinyclip": TextClip({10: 0, 20: 1})})
    result = cull(root, CullSettings(models=["search"]), models=reg, cache_db=tmp_path / "c.sqlite")
    assert len(result.index["search"]["embeddings"]) == 4
    assert all(f.categories for f in result.files.values())


def test_calibrate_compares_modes_with_a_library(tmp_path):
    from aiculler.baseline import LIBRARY_SCHEMA, LibraryBaseline
    from aiculler.categories import VERSION

    recs = [{"shoot": "s", "capture": f"a{i}", "maniqa": 0.10 + i / 1000, "labels": ["animals"], "techniques": []} for i in range(60)]
    recs += [{"shoot": "s", "capture": f"l{i}", "maniqa": 0.40 + i / 1000, "labels": ["landscape"], "techniques": []} for i in range(60)]
    lib = LibraryBaseline({"schema": LIBRARY_SCHEMA, "maniqa": "m", "tinyclip": "t", "categories_version": VERSION, "records": recs})
    rows = [  # keeps: a low-scoring animal and a high landscape; rejects: a low landscape, a bottom animal
        {"id": "1", "folder": "s", "scores": {"maniqa": 0.13}, "labels": ["animals"], "techniques": []},
        {"id": "2", "folder": "s", "scores": {"maniqa": 0.45}, "labels": ["landscape"], "techniques": []},
        {"id": "3", "folder": "s", "scores": {"maniqa": 0.20}, "labels": ["landscape"], "techniques": []},
        {"id": "4", "folder": "s", "scores": {"maniqa": 0.101}, "labels": ["animals"], "techniques": []},
    ]
    cal = calibrate({"rows": rows}, {"1": "keep", "2": "keep", "3": "reject", "4": "reject"}, library=lib)
    m = cal["signals"]
    assert m["maniqa/raw"]["all_safe"] == 1  # only the bottom animal is below the low animal keep
    assert m["maniqa/category"]["all_safe"] == 2  # as percentiles of their category, both fall below


def test_shoot_mode_floor_uses_the_shoots_own_percentiles(tmp_path):
    from aiculler import CullSettings, cull
    from aiculler.api import GateSettings

    root = tmp_path / "shoot"
    frames = [Frame(f"S{i:02d}", i * 100.0 + (i * i % 17) * 23, i % 8, marker=10 * (i + 1)) for i in range(25)]
    write_shoot(root, frames)
    quality = {10 * (i + 1): 0.30 + i / 100 for i in range(25)}  # S00 lowest ... S24 highest
    maniqa = FakeManiqa(quality)
    settings = CullSettings(models=["maniqa"], scene_keep=0, gates=GateSettings(quality_mode="shoot", quality_reject=0.10,
                                                                                quality_unique="reject"))
    result = cull(root, settings, models=ModelRegistry.fake({"maniqa": maniqa}), cache_db=tmp_path / "c.sqlite")
    d = {p.rsplit("/", 1)[-1][:-4]: f for p, f in result.files.items()}
    rejected = sorted(n for n, f in d.items() if f.decision == "reject")
    assert rejected == ["S00", "S01", "S02"]  # percentiles 0, 0.04, 0.08 of 25 frames
    assert "percentile 0.00 of this shoot's 25 frames" in d["S00"].reasons[0]
    assert len(set(maniqa.seen)) == 25  # a shoot percentile needs every competing frame scored


def test_fusion_both_catches_what_neither_signal_can_alone():
    rows = [
        {"id": "k1", "folder": "a", "scores": {"maniqa": 0.20, "qra_iqa": 0.90}},  # MANIQA dislikes this keep
        {"id": "k2", "folder": "b", "scores": {"maniqa": 0.90, "qra_iqa": 0.20}},  # Q-ReAlign dislikes this one
        {"id": "r1", "folder": "a", "scores": {"maniqa": 0.30, "qra_iqa": 0.30}},  # both mildly dislike this reject
    ]
    cal = calibrate({"rows": rows}, {"k1": "keep", "k2": "keep", "r1": "reject"})
    assert cal["signals"]["maniqa/raw"]["all_safe"] == 0 and cal["signals"]["qra_iqa/raw"]["all_safe"] == 0
    assert cal["fusion"]["raw"]["either"]["all_safe"] == 0 and cal["fusion"]["raw"]["both"]["all_safe"] == 1


def test_held_out_shoots_report_keeps_lost():
    rows = [{"id": f"a{i}", "folder": "A", "scores": {"maniqa": 0.40 + i / 100}} for i in range(5)]
    rows += [{"id": "b0", "folder": "B", "scores": {"maniqa": 0.30}}, {"id": "b1", "folder": "B", "scores": {"maniqa": 0.10}}]
    labels = {f"a{i}": "keep" for i in range(5)} | {"b0": "keep", "b1": "reject"}
    h = calibrate({"rows": rows}, labels)["signals"]["maniqa/raw"]["held_out"]
    # Shoot B judged with a level fitted on A (0.40 - 0.02): its keep at 0.30 is lost, its reject caught.
    assert h == {"keeps_lost": 1, "borderline_lost": 0, "rejects_caught": 1, "folds": 2}


def test_hero_counts_as_keep(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("id,name,label\n001,x,hero\n002,y,reject\n")
    assert read_labels(p) == {"001": "keep", "002": "reject"}


def test_sample_draws_would_be_keeps_and_runners_up_with_roles(tmp_path):
    from aiculler import CullSettings

    root = tmp_path / "shoot"
    frames = [Frame(f"B{i}", 0.1 * i, 0, marker=10 * (i + 1)) for i in range(5)]  # a burst
    frames += [Frame("L", 700.0, 3, marker=200)]  # a lone frame
    write_shoot(root, frames)
    doc = make_sample([root], tmp_path / "out", n=20, seed=1, cache_db=tmp_path / "c.sqlite", settings=CullSettings(bursts_keep=1))
    roles = sorted(r["role"] for r in doc["rows"])
    assert roles == ["burst_winner", "lone", "runner_up"]  # burst losers beyond the runner-up are not drawn
    assert doc["population"] == "keepers" and doc["per_role"] == {"burst_winner": 1, "lone": 1, "runner_up": 1}
    every = make_sample([root], tmp_path / "out2", n=20, seed=1, cache_db=tmp_path / "c.sqlite", population="all")
    assert every["drawn"] == 6 and all(r["role"] is None for r in every["rows"])
