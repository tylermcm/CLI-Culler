import sqlite3

import json

import pytest

from aiculler.evaluate import Truth, evaluate, load_truth, match_catalog, parse_xmp, read_lrcat, read_xmp_truth, render_report

XMP_ATTR = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="{r}" xmp:Label="{l}"/>'
    "</rdf:RDF></x:xmpmeta>"
)
XMP_ELEM = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/"><xmp:Rating>{r}</xmp:Rating></rdf:Description>'
    "</rdf:RDF></x:xmpmeta>"
)


def test_parse_xmp_attribute_and_element_forms():
    t = parse_xmp(XMP_ATTR.format(r=4, l="Red"))
    assert t.rating == 4 and t.label == "Red" and t.kept(3)
    t = parse_xmp(XMP_ELEM.format(r=2))
    assert t.rating == 2 and not t.kept(3)
    assert parse_xmp(XMP_ATTR.format(r=-1, l="")).rejected_flag
    assert parse_xmp(XMP_ATTR.format(r=0, l="Accepted")).kept(3)  # the app's accept label
    assert parse_xmp("not xml") is None


def test_read_xmp_sidecar_and_embedded(tmp_path):
    raw = tmp_path / "DSC_0001.NEF"
    raw.write_bytes(b"II*\x00")
    (tmp_path / "DSC_0001.xmp").write_text(XMP_ATTR.format(r=5, l=""))
    assert read_xmp_truth(raw).rating == 5

    jpg = tmp_path / "DSC_0002.JPG"
    jpg.write_bytes(b"\xff\xd8\xff\xe1" + XMP_ELEM.format(r=3).encode() + b"\xff\xd9")
    t = read_xmp_truth(jpg)
    assert t.rating == 3 and t.source == "embedded XMP"
    assert read_xmp_truth(tmp_path / "DSC_0003.NEF") is None


def _fake_catalog(path, rows):
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE AgLibraryRootFolder (id_local INTEGER PRIMARY KEY, absolutePath TEXT);
        CREATE TABLE AgLibraryFolder (id_local INTEGER PRIMARY KEY, rootFolder INTEGER, pathFromRoot TEXT);
        CREATE TABLE AgLibraryFile (id_local INTEGER PRIMARY KEY, folder INTEGER, baseName TEXT,
                                    extension TEXT, idx_filename TEXT, sidecarExtensions TEXT);
        CREATE TABLE Adobe_images (id_local INTEGER PRIMARY KEY, rootFile INTEGER, rating REAL, pick REAL,
                                   colorLabels TEXT, masterImage INTEGER);
        INSERT INTO AgLibraryRootFolder VALUES (1, 'D:/Photos/');
        INSERT INTO AgLibraryFolder VALUES (1, 1, '2026/Alaska/');
        """
    )
    for i, (name, rating, pick, sidecars) in enumerate(rows, 1):
        base, ext = name.rsplit(".", 1)
        con.execute("INSERT INTO AgLibraryFile VALUES (?, 1, ?, ?, ?, ?)", (i, base, ext, name, sidecars))
        con.execute("INSERT INTO Adobe_images VALUES (?, ?, ?, ?, '', NULL)", (i, i, rating, pick))
    # A virtual copy of the first image, picked: the capture counts as kept.
    con.execute("INSERT INTO Adobe_images VALUES (99, 1, NULL, 1.0, '', 1)")
    con.commit()
    con.close()


def test_lrcat_read_only_and_matched_by_folder_and_name(tmp_path):
    cat = tmp_path / "copy.lrcat"
    _fake_catalog(cat, [("DSC_0001.NEF", 0, 0, ""), ("DSC_0002.NEF", 4, 0, "JPG"), ("DSC_0003.NEF", None, -1, "")])
    before = cat.read_bytes()
    rows = read_lrcat(cat)
    assert cat.read_bytes() == before  # opened read-only
    shoot = ["/mnt/shoots/Alaska/DSC_0001.NEF", "/mnt/shoots/Alaska/DSC_0002.NEF",
             "/mnt/shoots/Alaska/DSC_0002.JPG", "/mnt/shoots/Alaska/DSC_0003.NEF", "/mnt/shoots/Alaska/DSC_0009.NEF"]
    truth = match_catalog(rows, shoot)
    assert truth[shoot[0]].kept(3)  # via its picked virtual copy
    assert truth[shoot[1]].rating == 4 and truth[shoot[2]].rating == 4  # JPEG sidecar shares the rating
    assert truth[shoot[3]].rejected_flag and not truth[shoot[3]].kept(3)
    assert shoot[4] not in truth
    decisions = {"files": {p: {"capture": p.rsplit(".", 1)[0], "primary": p.endswith(".NEF")} for p in shoot}}
    ts = load_truth(cat, decisions)
    assert len(ts.truth) == 4 and "read-only" in ts.source and ts.mode == "labels"


def _file(capture, group, kind, scene, decision, primary=True, score=0.5):
    return {"capture": capture, "group": group, "kind": kind, "scene": scene, "decision": decision,
            "primary": primary, "score": score, "reasons": [f"{decision} because"]}


def test_metrics_and_per_set_agreement():
    files = {
        # Burst U1: user kept u2, engine kept u1 + u2 -> hit.
        "/s/u1.NEF": _file("c1", "U1", "burst", 1, "keep"),
        "/s/u2.NEF": _file("c2", "U1", "burst", 1, "keep"),
        "/s/u2.JPG": _file("c2", "U1", "burst", 1, "keep", primary=False),
        "/s/u3.NEF": _file("c3", "U1", "burst", 1, "reject"),
        # Burst U2: user kept u5, engine kept u4 -> miss.
        "/s/u4.NEF": _file("c4", "U2", "burst", 2, "keep"),
        "/s/u5.NEF": _file("c5", "U2", "burst", 2, "reject"),
        # Scene 3: two singles, user kept none, engine kept one.
        "/s/s1.NEF": _file("c6", "S1", "single", 3, "keep"),
        "/s/s2.NEF": _file("c7", "S2", "single", 3, "review"),
    }
    truth = {"/s/u2.JPG": Truth(rating=4), "/s/u5.NEF": Truth(pick=1), "/s/s2.NEF": Truth(rating=1)}
    ev = evaluate({"files": files, "timings": {"ingest": 1.5}}, truth, truth_source="test")
    s = ev.summary
    assert s["captures"] == 7 and s["user_kept"] == 2  # truth on a sidecar counts for its capture
    assert s["engine_keep"] == 4 and s["precision"] == 0.25 and s["recall"] == 0.5
    assert s["review_rate"] == round(1 / 7, 4)
    assert s["sets"]["burst"] == {"sets": 2, "user_kept_some": 2, "hit": 1, "hit_rate": 0.5, "user_kept_none": 0, "agree_none": 0}
    assert s["sets"]["scene"]["sets"] == 1 and s["sets"]["scene"]["user_kept_none"] == 1
    report = render_report(ev, shoot="test shoot")
    assert "| Precision of keep | 25.0% |" in report
    assert "## User kept, engine rejected (1)" in report and "u5.NEF | reject | picked" in report
    assert "| ingest | 1.50 |" in report


# --------------------------------------------------------------------------- picks, csv, ImageTriage


def _doc(files):
    return {"files": files, "timings": {}}


def test_picks_truth_from_copies_and_exports_in_other_folders():
    files = {
        # Burst U1: u2 has a copy in _winners/ -> picked; u1's JPEG sidecar sits in its own folder -> not a pick.
        "/s/day1/u1.NEF": _file("c1", "U1", "burst", 1, "keep"),
        "/s/day1/u1.JPG": _file("c1", "U1", "burst", 1, "keep", primary=False),
        "/s/day1/u2.NEF": _file("c2", "U1", "burst", 1, "reject"),
        "/s/_winners/u2.NEF": _file("c2", "U1", "burst", 1, "reject", primary=False),
        "/s/day1/u3.NEF": _file("c3", "U1", "burst", 1, "keep"),
        # Single with an edited export in Bears/ -> picked and kept.
        "/s/day1/s1.NEF": _file("c4", "S1", "single", None, "keep"),
        "/s/Bears/s1-edit.jpg": _file("c4", "S1", "single", None, "keep", primary=False),
        # An export bracketeering could not merge with its original: left out.
        "/s/Bears/crop-77.jpg": _file("c5", "S2", "single", None, "keep"),
    }
    ts = load_truth("picks", _doc(files))
    assert ts.mode == "picks" and set(ts.truth) == {"/s/day1/u2.NEF", "/s/day1/s1.NEF"}
    assert ts.excluded == {"c5": "in pick/export folder Bears/ but not merged with an original"}
    assert "_winners/ (1)" in ts.notes[0] and "Bears/ (1)" in ts.notes[0]

    ev = evaluate(_doc(files), ts)
    s = ev.summary
    assert s["captures"] == 4 and s["excluded"] == 1 and s["user_kept"] == 2
    assert s["recall"] == 0.5 and s["precision"] is None and s["agreement_decided"] is None
    assert s["engine_keep_rate"] == 0.75
    assert s["sets"]["burst"]["hit_rate"] == 0.0 and s["sets"]["burst"]["agree_none"] is None
    report = render_report(ev)
    assert "| Precision of keep | not meaningful for picks |" in report
    assert "## User picked, engine rejected (1)" in report and "picked (copy in _winners/)" in report
    assert "## Sets where the engine kept none of the picked frames (1)" in report
    assert "## Left out of the metrics (1)" in report and "crop-77.jpg" in report
    assert "Engine kept, not picked (not necessarily wrong)" in report


def test_csv_truth_from_review_session_rows(tmp_path):
    files = {
        "/s/a/DSC_0001.NEF": _file("c1", "S1", "single", 1, "keep"),
        "/s/a/DSC_0002.NEF": _file("c2", "S2", "single", 1, "reject"),
        "/s/a/DSC_0003.NEF": _file("c3", "S3", "single", 1, "keep"),
        "/s/a/DSC_0004.NEF": _file("c4", "S4", "single", 2, "keep"),
        "/s/b/DSC_0004.NEF": _file("c9", "S9", "single", 3, "keep"),  # same name, other capture
    }
    feedback = tmp_path / "feedback.csv"
    feedback.write_text(
        "id,filename,source_path,label,reject_tags\n"
        ",DSC_0001.NEF,,keep,\n"
        ",DSC_0002.NEF,,r,blownout\n"
        ",DSC_0003.NEF,/elsewhere/a/DSC_0003.NEF,reject,\n"
        ",DSC_0004.NEF,,keep,\n"  # ambiguous: two captures share the name
        ",DSC_0005.NEF,,s,\n"
    )
    ts = load_truth("csv", _doc(files), truth_file=feedback)
    assert ts.unlabelled == "exclude"
    assert {p: t.state for p, t in ts.truth.items()} == {
        "/s/a/DSC_0001.NEF": "kept", "/s/a/DSC_0002.NEF": "rejected", "/s/a/DSC_0003.NEF": "rejected",
    }
    ev = evaluate(_doc(files), ts)
    assert ev.summary["captures"] == 3  # unlabelled captures left out
    assert ev.summary["precision"] == 0.5 and ev.summary["recall"] == 1.0
    assert load_truth(feedback, _doc(files)).source == "CSV feedback.csv"  # a .csv path works directly
    with pytest.raises(ValueError):
        load_truth("csv", _doc(files))  # needs --truth-file


def _triage_db(path, rows):
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY);
        CREATE TABLE decisions (session_id TEXT NOT NULL, path TEXT NOT NULL, modified_ns INTEGER NOT NULL,
            file_size INTEGER NOT NULL, winner INTEGER NOT NULL DEFAULT 0, reject INTEGER NOT NULL DEFAULT 0,
            photoshop INTEGER NOT NULL DEFAULT 0, rating INTEGER NOT NULL DEFAULT 0,
            tags_json TEXT NOT NULL DEFAULT '[]', review_round TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(session_id, path));
        """
    )
    for session, p, winner, reject, rating, tags in rows:
        con.execute("INSERT INTO decisions VALUES (?, ?, 0, 0, ?, ?, 0, ?, ?, '')", (session, p, winner, reject, rating, json.dumps(tags)))
    con.commit()
    con.close()


def test_triage_truth_from_decision_store_and_sidecars(tmp_path, monkeypatch):
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    names = ["W", "K", "R", "Star", "None", "Side"]
    files = {str(shoot / f"{n}.NEF"): _file(n, f"S{i}", "single", None, "keep") for i, n in enumerate(names)}
    data = tmp_path / "share" / "Image Triage" / "Image Triage"
    data.mkdir(parents=True)
    _triage_db(data / "decisions.sqlite3", [
        ("Default", str(shoot / "W.NEF"), 1, 0, 0, []),
        ("Default", str(shoot / "K.NEF"), 0, 0, 0, ["keep"]),
        ("Default", str(shoot / "R.NEF"), 0, 1, 0, []),
        ("Default", str(shoot / "Star.NEF"), 0, 0, 4, []),
        ("Portraits", str(shoot / "W.NEF"), 0, 1, 0, []),
    ])
    # A file the database has no row for, decided in ImageTriage's own sidecar field.
    (shoot / "Side.xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:triage="https://image-triage.app/ns/1.0/" triage:ReviewState="rejected"/>'
        "</rdf:RDF></x:xmpmeta>"
    )
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setattr("sys.platform", "linux")
    ts = load_truth("triage", _doc(files))
    state = {n: ts.truth[str(shoot / f"{n}.NEF")] for n in names if str(shoot / f"{n}.NEF") in ts.truth}
    assert state["W"].note == "ImageTriage Winner" and state["K"].note == "ImageTriage Keep"
    assert state["R"].rejected_flag and state["Side"].rejected_flag
    assert state["Star"].kept(3) and "None" not in state
    assert "session 'Default'" in ts.notes[0] and "Portraits" in ts.notes[0]
    ev = evaluate(_doc(files), ts)
    assert ev.summary["captures"] == 5 and ev.summary["user_kept"] == 3  # "None" has no label: left out

    other = load_truth("triage", _doc(files), session="Portraits")
    assert other.truth[str(shoot / "W.NEF")].rejected_flag
    with pytest.raises(ValueError):
        load_truth("triage", _doc(files), session="Nope")


def test_xmp_reads_imagetriage_fields():
    t = parse_xmp(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:subject><rdf:Bag>'
        "<rdf:li>keep</rdf:li></rdf:Bag></dc:subject></rdf:Description></rdf:RDF></x:xmpmeta>"
    )
    assert t.kept(3) and t.tags == ("keep",)


def test_report_lists_picks_the_gates_reject_and_their_margins():
    gates = {"clip_region_reject": 0.25, "sky_blown": 0.6, "shadows_crushed": 0.6, "black_mean": 0.02,
             "washed_mean": 0.8, "washed_spread": 0.15}
    files = {
        "/s/a/p1.NEF": {**_file("c1", "S1", "single", 1, "reject"), "gates": ["sky_blown"],
                        "scores": {"clip_region": 0.2, "top_clip": 0.7, "luma_mean": 0.6, "luma_spread": 0.5,
                                   "preview_mean": 0.6, "shadows_crushed": 0.01}},
        "/s/_winners/p1.NEF": _file("c1", "S1", "single", 1, "reject", primary=False),
        "/s/a/p2.NEF": {**_file("c2", "S2", "single", 2, "keep"), "gates": [],
                        "scores": {"clip_region": 0.1, "top_clip": 0.2, "luma_mean": 0.78, "luma_spread": 0.16,
                                   "preview_mean": 0.03, "shadows_crushed": 0.3}},
        "/s/_winners/p2.NEF": _file("c2", "S2", "single", 2, "keep", primary=False),
        "/s/a/x.NEF": {**_file("c3", "S3", "single", 3, "reject"), "gates": ["clip_region"],
                       "scores": {"clip_region": 0.9}},  # not picked: not listed
    }
    doc = {"files": files, "settings": {"gates": gates}}
    ev = evaluate(doc, load_truth("picks", doc))
    g = ev.summary["gates_vs_kept"]
    assert g["kept_checked"] == 2 and [x["file"] for x in g["gated"]] == ["p1.NEF"]
    m = g["metrics"]
    assert m["sky_blown"] == {"setting": "sky_blown", "threshold": 0.6, "worst": 0.7, "file": "p1.NEF", "passes": False}
    assert m["clip_region"]["worst"] == 0.2 and m["clip_region"]["passes"]
    assert m["black"]["file"] == "p2.NEF" and m["black"]["passes"]
    assert m["washed_out"]["file"] == "p2.NEF" and m["washed_out"]["passes"]
    report = render_report(ev)
    assert "## Gates vs your picked frames" in report and "Rejected by a gate: **1**" in report
    assert "| sky_blown | `sky_blown` | 60% | p1.NEF | 70% | **no** |" in report


def test_picks_ignore_card_folders_and_bulk_copies():
    """Alaska Aug '21: originals copied to card folders (104ND800/) and a duplicate dump (Tyler's photos/)
    next to real export folders (Bears/, Eagles/)."""
    files = {}
    for i in range(200):
        cap = f"c{i}"
        files[f"/s/Alaska/DSC_{i:04d}.NEF"] = _file(cap, f"S{i}", "single", None, "keep")
        files[f"/s/Tyler's photos/DSC_{i:04d}.NEF"] = _file(cap, f"S{i}", "single", None, "keep", primary=False)
        if i < 60:
            files[f"/s/104ND800/DSC_{i:04d}.JPG"] = _file(cap, f"S{i}", "single", None, "keep", primary=False)
    files["/s/Bears/DSC_0005-edit.jpg"] = _file("c5", "S5", "single", None, "keep", primary=False)
    files["/s/Eagles/DSC_0150.jpg"] = _file("c150", "S150", "single", None, "keep", primary=False)
    files["/s/Bears/unmatched-crop.jpg"] = _file("x", "SX", "single", None, "keep")  # an export with no original
    ts = load_truth("picks", _doc(files))
    assert set(ts.truth) == {"/s/Alaska/DSC_0005.NEF", "/s/Alaska/DSC_0150.NEF"}
    assert ts.notes[0] == "pick folders: Bears/ (1), Eagles/ (1)"
    assert "104ND800/ (60, camera card folder)" in ts.notes[1] and "Tyler's photos/ (200 = 100% of the shoot" in ts.notes[1]
    assert ts.excluded == {"x": "in pick/export folder Bears/ but not merged with an original"}  # originals stay in
    only = load_truth("picks", _doc(files), pick_folders=["eagles"])
    assert set(only.truth) == {"/s/Alaska/DSC_0150.NEF"}


def test_evaluate_reports_what_the_quality_floor_did_to_your_keeps():
    from aiculler.evaluate import evaluate, render_report
    from aiculler.truth import Truth, TruthSet

    def f(cap, decision, floor, gates=()):
        return {"decision": decision, "group": cap, "kind": "single", "role": "single", "scene": None, "capture": cap,
                "primary": True, "score": None, "scores": {}, "reasons": ["x"], "gates": list(gates), "floor": floor}

    decisions = {"settings": {"gates": {"quality_reject": 0.2, "quality_signal": "maniqa"}}, "files": {
        "/s/A.NEF": f("A", "reject", "reject", ["quality"]),  # a keep the floor lost
        "/s/B.NEF": f("B", "review", "review_unique", ["quality"]),  # a keep it sent to review
        "/s/C.NEF": f("C", "reject", "reject", ["quality"]),  # a reject it caught
        "/s/D.NEF": f("D", "keep", None),
    }}
    truth = TruthSet({"/s/A.NEF": Truth(label="Accepted", state="kept"), "/s/B.NEF": Truth(label="Accepted", state="kept"),
                      "/s/C.NEF": Truth(label="Rejected", state="rejected"), "/s/D.NEF": Truth(label="Accepted", state="kept")},
                     "triage", mode="full")
    ev = evaluate(decisions, truth)
    fl = ev.summary["floor"]
    assert fl["keeps_lost"] == ["A.NEF"] and fl["keeps_reviewed"] == ["B.NEF"] and fl["others_rejected"] == 1
    assert "Your kept frames the floor rejected: 1 of 3" in render_report(ev)
