"""End to end on a synthetic shoot: real JPEGs with EXIF, grouped by bracketeering for real."""

import json
import shutil

import pytest

pytest.importorskip("PIL")
pytest.importorskip("bracketeering")

from shootgen import Frame, bracket, burst, write_shoot  # noqa: E402

from aiculler import CullSettings, cull  # noqa: E402
from aiculler.cli import main  # noqa: E402
from aiculler.outputs import load_decisions  # noqa: E402


@pytest.fixture()
def shoot(tmp_path):
    root = tmp_path / "shoot"
    frames = (
        burst("IMG_", 1, 0.0, 5, scene=0, blurs=[2.5, 0.0, 1.2, 3.0, 0.6])
        + bracket("IMG_", 10, 30.0, scene=1)
        + [Frame("IMG_0020", 60.0, 2), Frame("IMG_0021", 64.0, 2, blur=2.0)]
        + [Frame("IMG_0030", 300.0, 0, gain=0.01)]
    )
    paths = write_shoot(root, frames)
    picks = root / "_picks"
    picks.mkdir()
    shutil.copy2(paths["IMG_0002"], picks / "IMG_0002.JPG")  # a copy: same capture, never scored twice
    return root


def _by_name(result):
    out = {}
    for path, d in result.files.items():
        out.setdefault(path.rsplit("/", 1)[-1].rsplit(".", 1)[0], []).append((path, d))
    return out


def test_cull_end_to_end(shoot, tmp_path):
    result = cull(shoot, CullSettings(), cache_db=tmp_path / "c.sqlite")
    files = _by_name(result)
    decision = {name: [d.decision for _, d in entries] for name, entries in files.items()}

    assert result.summary["files"] == 12  # every input file exactly once, the copy included
    assert decision["IMG_0002"] == ["keep", "keep"]  # sharpest burst frame and its copy
    assert decision["IMG_0005"] == ["keep"]  # second sharpest (bursts_keep 2)
    assert decision["IMG_0001"] == decision["IMG_0003"] == decision["IMG_0004"] == ["reject"]
    assert decision["IMG_0010"] == decision["IMG_0011"] == decision["IMG_0012"] == ["keep"]  # bracket as a set
    assert decision["IMG_0020"] == ["keep"] and decision["IMG_0021"] == ["reject"]
    assert decision["IMG_0030"] == ["reject"]  # black frame

    copy = next(d for p, d in files["IMG_0002"] if "_picks" in p)
    assert not copy.primary and copy.reasons[-1] == "follows its shot IMG_0002.JPG"
    # Captures, not files, are scored: 5 burst frames + bracket rep + 3 singles.
    assert result.stats["sharpness_computed"] == 9 and result.stats["captures"] == 11


def test_new_settings_reuse_cached_scores(shoot, tmp_path):
    db = tmp_path / "c.sqlite"
    cull(shoot, CullSettings(), cache_db=db)
    again = cull(shoot, CullSettings(bursts_keep=1, brackets="pick_one"), cache_db=db)
    assert again.stats["sharpness_computed"] == 0 and again.stats["exposure_computed"] == 0
    decision = {p.rsplit("/", 1)[-1]: d.decision for p, d in again.files.items() if d.primary}
    assert decision["IMG_0005.JPG"] == "reject"
    assert sorted(decision[f"IMG_00{n}.JPG"] for n in (10, 11, 12)) == ["keep", "reject", "reject"]


def test_cli_cull_and_evaluate(shoot, tmp_path):
    (shoot / "IMG_0002.xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="4"/></rdf:RDF></x:xmpmeta>'
    )
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"bursts_keep": 1}))
    out = tmp_path / "decisions.json"
    db = tmp_path / "c.sqlite"
    assert main(["--no-log", "--db", str(db), "cull", str(shoot), "--settings", str(settings), "--out", str(out), "--quiet"]) == 0
    doc = load_decisions(out)
    assert doc["schema"] == "aiculler-decisions/1"
    assert doc["settings"]["bursts_keep"] == 1
    assert doc["summary"]["files"] == len(doc["files"]) == 12
    entry = next(v for k, v in doc["files"].items() if k.endswith("/shoot/IMG_0002.JPG"))
    assert entry["decision"] == "keep" and entry["kind"] == "burst" and entry["primary"] is True
    assert set(entry["scores"]) >= {"sharpness", "exposure", "sharpness_rel", "exposure_rel"}

    report = tmp_path / "report.md"
    metrics = tmp_path / "metrics.json"
    assert main([
        "--no-log", "--db", str(db), "evaluate", str(shoot), "--truth", "xmp",
        "--decisions", str(out), "--out", str(report), "--metrics-out", str(metrics),
    ]) == 0
    text = report.read_text()
    assert "# Cull evaluation" in text and "| burst | 1 | 1 | 1 | 100.0% |" in text
    m = json.loads(metrics.read_text())
    assert m["user_kept"] == 1 and m["recall"] == 1.0


def test_evaluate_picks_end_to_end(tmp_path):
    shoot = tmp_path / "China 8-26"
    frames = burst("DSC_", 1, 0.0, 4, scene=0, blurs=[2.0, 0.0, 1.0, 3.0]) + [Frame("DSC_0010", 60.0, 1), Frame("DSC_0011", 63.0, 1, blur=2.5)]
    paths = write_shoot(shoot, frames)
    winners = tmp_path / "_winners"  # outside the shoot folder: passed as a second folder
    winners.mkdir()
    shutil.copy2(paths["DSC_0002"], winners / "DSC_0002.JPG")  # engine should keep it (sharpest)
    shutil.copy2(paths["DSC_0011"], winners / "DSC_0011.JPG")  # engine will reject it (blurred twin)
    report = tmp_path / "report.md"
    metrics = tmp_path / "metrics.json"
    assert main([
        "--no-log", "--db", str(tmp_path / "c.sqlite"), "evaluate", str(shoot), str(winners), "--quiet",
        "--decisions", str(tmp_path / "decisions.json"), "--out", str(report), "--metrics-out", str(metrics),
    ]) == 0  # --truth picks is the default
    m = json.loads(metrics.read_text())
    assert m["mode"] == "picks" and m["user_kept"] == 2 and m["recall"] == 0.5 and m["precision"] is None
    assert m["sets"]["burst"]["hit_rate"] == 1.0 and m["sets"]["scene"]["hit_rate"] == 0.0
    text = report.read_text()
    assert "_winners/ (2)" in text and "DSC_0011.JPG | reject | picked (copy in _winners/)" in text


def test_cli_expands_quoted_tilde_and_rejects_missing_or_empty_folders(shoot, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(shoot.parent))
    out = tmp_path / "d.json"
    db = str(tmp_path / "c.sqlite")
    assert main(["--no-log", "--db", db, "cull", "~/shoot", "--out", str(out), "--quiet"]) == 0  # as if quoted: "~/shoot"
    assert load_decisions(out)["summary"]["files"] == 12

    with pytest.raises(SystemExit, match="shoot folder not found: .*nope"):
        main(["--no-log", "--db", db, "cull", "~/nope", "--out", str(tmp_path / "x.json"), "--quiet"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no image files found"):
        main(["--no-log", "--db", db, "cull", str(empty), "--out", str(tmp_path / "y.json"), "--quiet"])
    assert not (tmp_path / "x.json").exists() and not (tmp_path / "y.json").exists()
