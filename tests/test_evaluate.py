import sqlite3

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
    t, source = load_truth(cat, shoot)
    assert len(t) == 4 and "read-only" in source


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
