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
    cal = calibrate(doc, read_labels(out / "labels.csv"))
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
