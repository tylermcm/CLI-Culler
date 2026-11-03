"""cull() end to end with fake model engines (no weights, no torch), and the registry against a fake app."""

import json
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("bracketeering")

from shootgen import Frame, burst, read_marker, write_shoot  # noqa: E402

from aiculler import CullSettings, cull  # noqa: E402
from aiculler.models.registry import ModelRegistry  # noqa: E402


class FakeManiqa:
    LONG_EDGE = 1280

    def __init__(self, table):
        self.table, self.seen = table, []

    def score(self, image):
        m = read_marker(image)
        self.seen.append(m)
        return self.table.get(m, 0.5)


class FakeAppeal(FakeManiqa):
    LONG_EDGE = 1024


class FakeFaces:
    LONG_EDGE = 1280

    def __init__(self, table):
        self.table, self.seen = table, []

    def detail(self, image):
        m = read_marker(image)
        self.seen.append(m)
        return {"faces": self.table.get(m, []), "width": image.width, "height": image.height}


class FakeClip:
    def __init__(self, table):
        self.table = table

    def embed(self, image):
        v = np.asarray(self.table.get(read_marker(image), [1.0, 0.0, 0.0]), dtype=np.float32)
        return list(v / np.linalg.norm(v))


def _face(closed, vec=(1.0, 0.0), area=0.05):
    return {"box": [0.4, 0.2, 0.6, 0.5], "area": area, "eyes_closed": closed, "embedding": list(vec)}


@pytest.fixture()
def shoot(tmp_path):
    root = tmp_path / "shoot"
    # A 5-frame burst, all equally sharp: only the models can tell them apart.
    frames = [Frame(f"DSC_{i:04d}", i * 0.1, 0, marker=10 * (i + 1)) for i in range(1, 6)]
    frames.append(Frame("DSC_0100", 600.0, 1, marker=200))  # alone in its scene
    write_shoot(root, frames)
    return root


def test_models_decide_between_near_identical_frames(shoot, tmp_path):
    maniqa = FakeManiqa({20: 0.6, 30: 0.62, 40: 0.61, 50: 0.3, 60: 0.6})
    appeal = FakeAppeal({20: 0.4, 30: 0.5, 40: 0.9, 50: 0.95, 60: 0.45})
    faces = FakeFaces({20: [_face(0.1)], 30: [_face(0.95)], 40: [_face(0.1)], 50: [_face(0.1)], 60: [_face(0.1)], 200: [_face(0.1, (0, 1))]})
    reg = ModelRegistry.fake({"maniqa": maniqa, "q_realign": appeal, "face_landmarker": faces})
    result = cull(shoot, CullSettings(bursts_keep=1, appeal_shortlist=3), models=reg, cache_db=tmp_path / "c.sqlite")
    d = {p.rsplit("/", 1)[-1]: f for p, f in result.files.items()}

    # DSC_0003 (marker 40): technically in the top 3 and most appealing of those -> kept.
    assert d["DSC_0003.JPG"].decision == "keep"
    # DSC_0004 (50) is the most appealing of all, but its MANIQA put it outside the shortlist: Q-ReAlign never saw it.
    assert 50 not in appeal.seen and d["DSC_0004.JPG"].decision == "reject"
    # DSC_0002 (30) blinked.
    assert any("eyes closed (0.95)" in r for r in d["DSC_0002.JPG"].reasons)
    assert "more appealing" in d["DSC_0001.JPG"].reasons[0]
    # Cheap -> expensive: the lone frame reaches no model (without AdaFace there is no people coverage to need it).
    assert 200 not in maniqa.seen and 200 not in appeal.seen and 200 not in faces.seen
    assert result.stats["contested_frames"] == 5 and result.stats["appeal_frames"] == 3
    assert set(result.models) == {"faces", "maniqa", "appeal"} and result.models["maniqa"].startswith("fake-maniqa")
    assert result.warnings == [f"tinyclip: not used (not provided)"]
    assert set(d["DSC_0003.JPG"].scores) >= {"maniqa", "appeal", "face_sharpness", "eyes_closed", "appeal_rel"}
    assert "faces" not in d["DSC_0003.JPG"].scores  # details stay out of decisions.json scores

    # Re-running with new settings re-runs no model.
    maniqa.seen.clear(), appeal.seen.clear(), faces.seen.clear()
    again = cull(shoot, CullSettings(bursts_keep=2, appeal_shortlist=3), models=reg, cache_db=tmp_path / "c.sqlite")
    assert maniqa.seen == [] and faces.seen == [] and again.stats["maniqa_computed"] == 0
    assert sum(f.decision == "keep" for f in again.files.values()) == 3


def test_missing_or_broken_models_degrade_with_a_warning(shoot, tmp_path):
    class Broken:
        LONG_EDGE = 1280

        def score(self, image):
            raise RuntimeError("CUDA out of memory")

    reg = ModelRegistry.fake({"maniqa": Broken()})
    result = cull(shoot, CullSettings(), models=reg, cache_db=tmp_path / "c.sqlite")
    assert result.summary["files"] == 6 and "maniqa" not in result.models
    assert any(w.startswith("maniqa: not used (RuntimeError: CUDA out of memory)") for w in result.warnings)
    # Without a registry at all: milestone-1 behaviour, no warnings.
    plain = cull(shoot, CullSettings(), cache_db=tmp_path / "c.sqlite")
    assert plain.models == {} and plain.warnings == []


def test_people_coverage_end_to_end(shoot, tmp_path):
    bob = (0.0, 1.0)
    faces = FakeFaces({20: [_face(0.1)], 30: [_face(0.1)], 40: [_face(0.1)], 50: [_face(0.1)], 60: [_face(0.1, bob, area=0.08)],
                       200: [_face(0.1)]})
    reg = ModelRegistry.fake({"face_landmarker": faces, "adaface": object(), "maniqa": FakeManiqa({60: 0.1})})
    result = cull(shoot, CullSettings(bursts_keep=1), models=reg, cache_db=tmp_path / "c.sqlite")
    d = {p.rsplit("/", 1)[-1]: f for p, f in result.files.items()}
    # Bob appears only in DSC_0005, which the burst ranking rejected: it is promoted.
    assert d["DSC_0005.JPG"].decision == "keep" and "only good frame of person #2" in d["DSC_0005.JPG"].reasons[-1]
    assert result.stats["people"] == 2 and result.stats["people_promoted"] == 1
    assert 200 in faces.seen  # with AdaFace, every candidate is searched for people, the lone frame included


def test_tinyclip_scenes_and_preference_end_to_end(tmp_path):
    from aiculler.preference import train

    root = tmp_path / "shoot"
    frames = [Frame("DSC_0001", 0.0, 0, marker=10), Frame("DSC_0002", 4.0, 0, marker=20),  # same bracketeering scene?
              Frame("DSC_0010", 30.0, 2, marker=30)]
    write_shoot(root, frames)
    clip = FakeClip({10: [1, 0, 0], 20: [0, 1, 0], 30: [0, 0, 1]})
    rev = "fake-tinyclip+224-bilinear-v1"
    model = train(np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.1], [0, 0, 1], [0.1, 0, 0.9]]),
                  np.array([1, 1, 0, 0, 0, 0]), embedding_revision=rev, projected_dim=3)
    path = model.save(tmp_path / "pref.joblib")
    reg = ModelRegistry.fake({"tinyclip": clip})
    result = cull(root, CullSettings(preference_model=str(path)), models=reg, cache_db=tmp_path / "c.sqlite")
    d = {p.rsplit("/", 1)[-1]: f for p, f in result.files.items()}
    assert "tinyclip" in result.models and result.models["preference"].startswith(rev)
    assert d["DSC_0001.JPG"].scores["preference"] > d["DSC_0002.JPG"].scores["preference"]
    # Different views by embedding: never one comparison set, whatever bracketeering grouped.
    assert d["DSC_0001.JPG"].decision == d["DSC_0002.JPG"].decision == "keep"


# --------------------------------------------------------------------------- registry


@pytest.fixture()
def fake_app(tmp_path, monkeypatch):
    """A minimal image_triage package with a manifest and the app's path rules, plus installed model files."""
    app = tmp_path / "app"
    pkg = app / "image_triage"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "ai_manifest.py").write_text(textwrap.dedent('''
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class FileDigest:
            size: int
            sha256: str

        @dataclass(frozen=True)
        class ModelBundle:
            key: str
            name: str
            repo_id: str
            revision: str
            filenames: tuple
            install_parts: tuple
            approx_mb: int
            pinned_revision: bool = True
            dir_env: str = ""
            repo_env: str = ""
            revision_env: str = ""

        MODEL_BUNDLES = {
            "maniqa": ModelBundle("maniqa", "MANIQA", "r", "18bbbb8a", ("ckpt_koniq10k.pt",), ("Rankers", "MANIQA"), 1,
                                  dir_env="IMAGE_TRIAGE_MANIQA_MODEL_DIR"),
            "qrealign": ModelBundle("qrealign", "Q", "r", "fe1f45a7", ("model.safetensors",), ("Rankers", "Q"), 1),
        }
        MODEL_FILE_DIGESTS = {"maniqa": {"ckpt_koniq10k.pt": FileDigest(5, "x")}, "qrealign": {"model.safetensors": FileDigest(3, "y")}}
    '''))
    (pkg / "ai_paths.py").write_text(textwrap.dedent(f'''
        from pathlib import Path
        def managed_model_dir(*parts):
            return Path({str(tmp_path / "models")!r}).joinpath(*parts)
    '''))
    maniqa = tmp_path / "custom" / "maniqa"
    maniqa.mkdir(parents=True)
    (maniqa / "ckpt_koniq10k.pt").write_bytes(b"12345")
    q = tmp_path / "models" / "Rankers" / "Q"
    q.mkdir(parents=True)
    (q / "model.safetensors").write_bytes(b"1234")  # wrong size
    monkeypatch.setenv("IMAGE_TRIAGE_MANIQA_MODEL_DIR", str(maniqa))
    for name in [m for m in sys.modules if m == "image_triage" or m.startswith("image_triage.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return app


def test_registry_resolves_the_apps_bundles(fake_app):
    reg = ModelRegistry.from_app_manifest(fake_app / "image_triage" / "ai_manifest.py")
    assert reg.available("maniqa") and reg.revision("maniqa") == "18bbbb8a"
    assert reg.path("maniqa").name == "maniqa"  # the *_MODEL_DIR override, as in the app
    assert not reg.available("q_realign") and "3 bytes" in reg.problems()["q_realign"] or "expected 3" in reg.problems()["q_realign"]
    assert "not in the app's manifest" in reg.problems()["tinyclip"]
    assert reg.describe() == {"maniqa": "18bbbb8a"}


def test_cli_cull_with_models_flag_reports_what_was_used(shoot, tmp_path, fake_app, capsys):
    from aiculler.cli import main

    out = tmp_path / "d.json"
    assert main(["--no-log", "--db", str(tmp_path / "c.sqlite"), "cull", str(shoot), "--quiet", "--out", str(out),
                 "--models", str(fake_app)]) == 0
    doc = json.loads(out.read_text())
    err = capsys.readouterr().err
    assert "q_realign" in err and "not available" in err
    # MANIQA's files are present, but this environment cannot load a real checkpoint from 5 bytes: it degrades.
    assert any(w.startswith("maniqa: not used") for w in doc["warnings"])


def test_learn_trains_on_picks_from_past_shoots(tmp_path):
    import shutil

    from aiculler.preference import PreferenceModel, learn_preference

    shoot = tmp_path / "past"
    frames = [Frame(f"DSC_{i:04d}", i * 30.0, i % 3, marker=10 * i) for i in range(1, 13)]
    paths = write_shoot(shoot, frames)
    (shoot / "_winners").mkdir()
    picked = {"DSC_0002", "DSC_0005", "DSC_0008"}  # markers 20, 50, 80: "warm" embeddings
    for name in picked:
        shutil.copy2(paths[name], shoot / "_winners" / f"{name}.JPG")
    clip = FakeClip({m: ([1.0, 0.1 * (m % 7), 0.0] if m in (20, 50, 80) else [0.0, 0.1 * (m % 7), 1.0]) for m in range(10, 130, 10)})
    model, summary = learn_preference([shoot], "picks", ModelRegistry.fake({"tinyclip": clip}), cache_db=tmp_path / "c.sqlite")
    assert summary[0]["examples"] == 12 and summary[0]["kept"] == 3
    assert model.positives == 3 and model.examples == 12 and model.diagnostics["auc"] > 0.9
    path = model.save(tmp_path / "pref.joblib")
    assert PreferenceModel.load(path).embedding_revision == "fake-tinyclip+224-bilinear-v1"
    with pytest.raises(ValueError, match="TinyCLIP is needed"):
        learn_preference([shoot], "picks", ModelRegistry.none())
