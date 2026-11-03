from pathlib import Path

from aiculler.api import CullResult, CullSettings, FileDecision
from aiculler.outputs import sidecar_path, write_xmp
from aiculler.truth import parse_xmp


def _result(files: dict[str, str]) -> CullResult:
    return CullResult(
        files={p: FileDecision(decision=d, group="S1", kind="single", role="single", scene=None, capture=p, primary=True)
               for p, d in files.items()},
        settings=CullSettings(), bracketeering_version="t", engine_version="t", models={}, timings={},
    )


def _read(path: Path):
    return parse_xmp(path.read_bytes())


def test_sidecar_naming():
    assert sidecar_path("/s/DSC_1.NEF") == Path("/s/DSC_1.xmp")
    assert sidecar_path("/s/DSC_1.JPG") == Path("/s/DSC_1.JPG.xmp")


def test_writes_ratings_and_labels(tmp_path):
    files = {str(tmp_path / f"{n}.NEF"): d for n, d in (("k", "keep"), ("v", "review"), ("r", "reject"))}
    c = write_xmp(_result(files))
    assert c == {"written": 3, "unchanged": 0, "skipped": 0}
    k, v, r = (_read(tmp_path / f"{n}.xmp") for n in "kvr")
    assert (k.rating, k.label) == (3, "")
    assert (v.rating, v.label) == (None, "Yellow")
    assert (r.rating, r.label) == (1, "Red")
    # Same decisions again: nothing to do.
    assert write_xmp(_result(files)) == {"written": 0, "unchanged": 3, "skipped": 0}
    # The culler's own output may be updated by a later run.
    files[str(tmp_path / "r.NEF")] = "keep"
    assert write_xmp(_result(files))["written"] == 1
    assert _read(tmp_path / "r.xmp").rating == 3 and _read(tmp_path / "r.xmp").label == ""


def test_user_ratings_and_imagetriage_decisions_are_left_alone(tmp_path):
    rated = tmp_path / "a.xmp"
    rated.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmp:Rating="5"><dc:subject><rdf:Bag><rdf:li>family</rdf:li></rdf:Bag></dc:subject></rdf:Description></rdf:RDF></x:xmpmeta>'
    )
    triage = tmp_path / "b.xmp"
    triage.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:triage="https://image-triage.app/ns/1.0/" triage:ReviewState="accepted"/>'
        "</rdf:RDF></x:xmpmeta>"
    )
    files = {str(tmp_path / "a.NEF"): "reject", str(tmp_path / "b.NEF"): "reject"}
    before = rated.read_bytes(), triage.read_bytes()
    assert write_xmp(_result(files)) == {"written": 0, "unchanged": 0, "skipped": 2}
    assert (rated.read_bytes(), triage.read_bytes()) == before

    assert write_xmp(_result(files), overwrite_ratings=True)["written"] == 2
    t = _read(rated)
    assert (t.rating, t.label) == (1, "Red") and t.tags == ("family",)  # the rest of the sidecar is kept


def test_raw_and_jpeg_of_one_shot_get_their_own_sidecars(tmp_path):
    files = {str(tmp_path / "s.NEF"): "keep", str(tmp_path / "s.JPG"): "keep"}
    assert write_xmp(_result(files))["written"] == 2
    assert (tmp_path / "s.xmp").is_file() and (tmp_path / "s.JPG.xmp").is_file()
