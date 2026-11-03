"""Ground-truth sources for the evaluation harness.

Each loader returns a ``TruthSet``: per-file ``Truth``, the source's *mode* and how captures without a
label count.

| Source | Kept | Mode | Unlabelled captures |
|---|---|---|---|
| ``picks`` | bracketeering merged a copy or edited export of it from another folder (``_winners/``, ``Bears/``) | picks | not kept (not picked) |
| ``csv`` | ``label`` keep in a ``filename,label`` CSV (as written by ``review-session``) | labels | excluded |
| ``triage`` | ImageTriage Winner, Keep, or >= N stars (``decisions.sqlite3``, then its XMP sidecars) | labels | excluded |
| ``xmp`` | rating >= N or an accept label (sidecars or embedded XMP) | labels | not kept |
| ``lrcat`` | picked or rating >= N (read-only Lightroom catalog copy) | labels | not kept |

In ``picks`` mode the truth is a subset of what the user kept (only final picks were exported), so
precision is not meaningful; recall and per-set agreement are.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
DC_NS = "http://purl.org/dc/elements/1.1/"
TRIAGE_NS = "https://image-triage.app/ns/1.0/"
DESCRIPTION = f"{{{RDF_NS}}}Description"
RATING = f"{{{XMP_NS}}}Rating"
LABEL = f"{{{XMP_NS}}}Label"
SUBJECT = f"{{{DC_NS}}}subject"
REVIEW_STATE = f"{{{TRIAGE_NS}}}ReviewState"
LI = f"{{{RDF_NS}}}li"
# Same vocabulary as the image-triage app (image_triage/xmp.py, qml_shell/decision_bridge.py).
ACCEPTED_LABELS = frozenset({"accept", "accepted", "keep", "pick", "picked", "select", "selected", "winner"})
REJECTED_LABELS = frozenset({"reject", "rejected"})
KEEP_TAG = "keep"
CSV_KEEP = frozenset({"keep", "k", "1", "true", "yes", "y"})
CSV_REJECT = frozenset({"reject", "r", "0", "false", "no", "n"})
EMBEDDED_XMP_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".dng", ".heic", ".heif", ".hif"}
_XMP_PACKET = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.S)

SOURCES = ("picks", "csv", "triage", "xmp", "lrcat")


@dataclass
class Truth:
    rating: int | None = None  # 0-5; -1 = rejected (some tools write it)
    pick: int = 0  # 1 picked, -1 rejected, 0 none (Lightroom catalog)
    label: str = ""
    tags: tuple[str, ...] = ()
    state: str | None = None  # explicit "kept" / "rejected" (picks, csv, ImageTriage decisions)
    note: str = ""  # how it was decided, for the report
    source: str = ""

    def kept(self, keep_min_rating: int) -> bool:
        if self.state is not None:
            return self.state == "kept"
        if self.pick == 1 or self.label.strip().casefold() in ACCEPTED_LABELS or KEEP_TAG in {t.casefold() for t in self.tags}:
            return True
        return self.rating is not None and self.rating >= keep_min_rating and self.pick != -1

    @property
    def rejected_flag(self) -> bool:
        if self.state is not None:
            return self.state == "rejected"
        return self.pick == -1 or (self.rating is not None and self.rating < 0) or self.label.strip().casefold() in REJECTED_LABELS

    def labelled(self, keep_min_rating: int) -> bool:
        """Says something either way: kept, rejected, or rated (a low rating counts as not kept)."""
        return self.kept(keep_min_rating) or self.rejected_flag or bool(self.rating and self.rating > 0)

    def describe(self) -> str:
        if self.note:
            return self.note
        parts = []
        if self.state:
            parts.append(self.state)
        if self.pick == 1:
            parts.append("picked")
        elif self.pick == -1:
            parts.append("rejected flag")
        if self.rating:
            parts.append(f"{self.rating}★" if self.rating > 0 else "rated reject")
        if self.label:
            parts.append(f"label {self.label}")
        if KEEP_TAG in {t.casefold() for t in self.tags} and not self.state:
            parts.append("keep tag")
        return ", ".join(parts) or "unrated"


@dataclass
class TruthSet:
    truth: dict[str, Truth]  # file path -> truth (any file of a capture)
    source: str  # description for the report
    mode: str = "labels"  # "labels" or "picks"
    unlabelled: str = "not_kept"  # "not_kept" or "exclude"
    excluded: dict[str, str] = field(default_factory=dict)  # capture -> why it is left out of the metrics
    notes: list[str] = field(default_factory=list)


def merge_truths(truths: list[Truth]) -> Truth:
    """One truth per capture from several files / copies: kept if any is kept."""
    if len(truths) == 1:
        return truths[0]
    out = Truth(source=truths[0].source)
    ratings = [t.rating for t in truths if t.rating is not None]
    out.rating = max(ratings) if ratings else None
    picks = [t.pick for t in truths]
    out.pick = 1 if 1 in picks else (-1 if all(p == -1 for p in picks) else 0)
    out.label = next((t.label for t in truths if t.label.strip().casefold() in ACCEPTED_LABELS), truths[0].label)
    out.tags = tuple(dict.fromkeys(tag for t in truths for tag in t.tags))
    states = [t.state for t in truths if t.state]
    out.state = "kept" if "kept" in states else (states[0] if states else None)
    out.note = next((t.note for t in truths if t.state == out.state and t.note), "")
    return out


# --------------------------------------------------------------------------- path matching


def _norm(path: str) -> str:
    return str(path).replace("\\", "/").rstrip("/").casefold()


def match_paths(entries: list[tuple[str, Truth]], paths: list[str], capture_of: dict[str, str] | None = None) -> dict[str, Truth]:
    """Match truth entries to shoot files: exact path, then parent folder + name, then the name alone.

    The name alone is used only when it is unambiguous on both sides: one entry with that name, and (when
    ``capture_of`` is given) all shoot files with that name belong to one capture.
    """
    by_path: dict[str, list[Truth]] = defaultdict(list)
    by_parent: dict[tuple[str, str], list[Truth]] = defaultdict(list)
    by_name: dict[str, list[Truth]] = defaultdict(list)
    for epath, t in entries:
        n = _norm(epath)
        parts = n.split("/")
        by_path[n].append(t)
        if len(parts) > 1:
            by_parent[(parts[-2], parts[-1])].append(t)
        by_name[parts[-1]].append(t)
    name_captures: dict[str, set[str]] = defaultdict(set)
    if capture_of:
        for p in paths:
            name_captures[_norm(p).rsplit("/", 1)[-1]].add(capture_of.get(p, p))
    out: dict[str, Truth] = {}
    for p in paths:
        n = _norm(p)
        parts = n.split("/")
        hits = by_path.get(n) or (by_parent.get((parts[-2], parts[-1])) if len(parts) > 1 else None)
        name = parts[-1]
        if not hits and len(by_name.get(name, [])) == 1 and len(name_captures.get(name, {None})) <= 1:
            hits = by_name[name]
        if hits:
            out[p] = merge_truths(hits)
    return out


# --------------------------------------------------------------------------- picks


def picks_truth(decisions: dict) -> TruthSet:
    """Kept = bracketeering merged a copy or edited export of the capture from another folder.

    A non-primary file in the shot's own folder is a sidecar (the JPEG of RAW+JPEG) and does not count.
    Captures whose primary file lives in one of those pick/export folders are exports bracketeering could
    not merge with their original: they are not shots of the shoot and are left out of the metrics.
    """
    files_of: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for path, d in decisions["files"].items():
        files_of[d["capture"]].append((path, bool(d["primary"])))
    pick_folders: dict[str, int] = defaultdict(int)
    picked: dict[str, list[str]] = {}
    primary_of: dict[str, str] = {}
    for cap, files in files_of.items():
        primary = next((p for p, prim in files if prim), files[0][0])
        primary_of[cap] = primary
        home = Path(primary).parent
        outside = [p for p, prim in files if not prim and Path(p).parent != home]
        if outside:
            picked[cap] = outside
            for folder in {str(Path(p).parent) for p in outside}:
                pick_folders[folder] += 1
    truth: dict[str, Truth] = {}
    for cap, outside in picked.items():
        where = ", ".join(sorted({Path(p).parent.name + "/" for p in outside}))
        truth[primary_of[cap]] = Truth(state="kept", note=f"picked (copy in {where})", source="picks")
    excluded = {
        cap: f"in pick/export folder {Path(primary).parent.name}/ but not merged with an original"
        for cap, primary in primary_of.items()
        if cap not in picked and str(Path(primary).parent) in pick_folders
    }
    notes = [f"pick folders: " + ", ".join(f"{Path(f).name}/ ({n})" for f, n in sorted(pick_folders.items()))] if pick_folders else [
        "no copies or exports outside the shots' folders were found: nothing is picked"
    ]
    if excluded:
        notes.append(f"{len(excluded)} capture(s) in pick folders could not be matched to an original and are left out")
    return TruthSet(truth, "picks (copies/exports merged by bracketeering)", mode="picks", unlabelled="not_kept", excluded=excluded, notes=notes)


# --------------------------------------------------------------------------- CSV


def csv_truth(path: str | Path, decisions: dict) -> TruthSet:
    """``filename,label`` (or ``source_path,label``) rows; label keep/k/1 or reject/r/0. Others are skipped."""
    entries: list[tuple[str, Truth]] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "label" not in [f.strip() for f in reader.fieldnames]:
            raise ValueError(f"{path}: needs a 'label' column plus 'filename' or 'source_path'")
        for row in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            label = row.get("label", "").casefold()
            state = "kept" if label in CSV_KEEP else "rejected" if label in CSV_REJECT else None
            key = row.get("source_path") or row.get("path") or row.get("filename")
            if state is None or not key:
                skipped += 1
                continue
            entries.append((key, Truth(state=state, source="csv")))
    capture_of = {p: d["capture"] for p, d in decisions["files"].items()}
    truth = match_paths(entries, list(decisions["files"]), capture_of)
    notes = [f"{len(entries)} labelled rows" + (f", {skipped} skipped (no keep/reject label or no file)" if skipped else "")]
    return TruthSet(truth, f"CSV {Path(path).name}", mode="labels", unlabelled="exclude", notes=notes)


# --------------------------------------------------------------------------- XMP (Lightroom, ImageTriage, others)


def _field(description: ET.Element, name: str) -> str:
    value = description.get(name, "")
    if value:
        return value.strip()
    child = description.find(name)
    return (child.text or "").strip() if child is not None and child.text else ""


def parse_xmp(data: bytes | str) -> Truth | None:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    rating, label, state, tags = None, "", "", []
    for desc in root.iter(DESCRIPTION):
        r = _field(desc, RATING)
        if r and rating is None:
            try:
                rating = int(round(float(r)))
            except ValueError:
                pass
        label = label or _field(desc, LABEL)
        state = state or _field(desc, REVIEW_STATE)
        subject = desc.find(SUBJECT)
        if subject is not None:
            tags.extend((li.text or "").strip() for li in subject.iter(LI) if (li.text or "").strip())
    t = Truth(rating=rating, label=label, tags=tuple(tags))
    s = state.casefold()
    if s in ("accepted", "winner"):  # ImageTriage's own field
        t.state, t.note = "kept", "ImageTriage Winner (XMP)"
    elif s in ("rejected", "reject"):
        t.state, t.note = "rejected", "ImageTriage Reject (XMP)"
    if rating is None and not label and not t.state and not tags:
        return None
    return t


def sidecar_candidates(path: str | Path) -> list[Path]:
    """NAME.xmp next to the file (Lightroom, ImageTriage for RAWs), or NAME.EXT.xmp."""
    p = Path(path)
    return [p.with_suffix(".xmp"), p.with_suffix(".XMP"), Path(f"{p}.xmp"), Path(f"{p}.XMP")]


def read_xmp_truth(path: str | Path, embedded_limit: int = 4 << 20, embedded: bool = True) -> Truth | None:
    for cand in sidecar_candidates(path):
        if cand.is_file():
            t = parse_xmp(cand.read_bytes())
            if t is not None:
                t.source = cand.name
                return t
    p = Path(path)
    if embedded and p.suffix.lower() in EMBEDDED_XMP_SUFFIXES and p.is_file():
        with open(p, "rb") as fh:
            head = fh.read(embedded_limit)
        m = _XMP_PACKET.search(head)
        if m:
            t = parse_xmp(m.group(0))
            if t is not None:
                t.source = "embedded XMP"
                return t
    return None


def load_xmp_truth(paths: list[str]) -> dict[str, Truth]:
    out = {}
    for p in paths:
        t = read_xmp_truth(p)
        if t is not None:
            out[p] = t
    return out


def xmp_truth(decisions: dict) -> TruthSet:
    return TruthSet(load_xmp_truth(list(decisions["files"])), "XMP (sidecars and embedded)", mode="labels", unlabelled="not_kept")


# --------------------------------------------------------------------------- ImageTriage


def triage_db_candidates() -> list[Path]:
    """Where ImageTriage's ``DecisionStore`` keeps ``decisions.sqlite3`` (Qt AppDataLocation, org and app
    "Image Triage"; older installs used the organisation "Codex"; fallback ``~/.image-triage``)."""
    home = Path.home()
    bases: list[Path] = []
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        bases.append(Path(appdata) if appdata else home / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        bases.append(home / "Library" / "Application Support")
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        bases.append(Path(xdg) if xdg else home / ".local" / "share")
    out = []
    for base in bases:
        out += [base / "Image Triage" / "Image Triage" / "decisions.sqlite3", base / "Codex" / "Image Triage" / "decisions.sqlite3"]
    out.append(home / ".image-triage" / "decisions.sqlite3")
    return out


def find_triage_db() -> Path | None:
    return next((p for p in triage_db_candidates() if p.is_file()), None)


def _open_ro(path: Path) -> sqlite3.Connection:
    base = path.resolve().as_uri()
    try:
        con = sqlite3.connect(base + "?mode=ro", uri=True)
        con.execute("SELECT 1 FROM decisions LIMIT 1")
        return con
    except sqlite3.OperationalError:
        return sqlite3.connect(base + "?mode=ro&immutable=1", uri=True)


def triage_truth(decisions: dict, db: str | Path | None = None, session: str | None = None) -> TruthSet:
    """Decisions from ImageTriage: Winner or Keep -> kept, Reject -> rejected, else its star rating.

    Reads ``decisions.sqlite3`` read-only (rows of the chosen session; by default the session with the most
    rows for this shoot), then ImageTriage's XMP sidecars for files without a row. Rows are matched by
    path; the app's staleness check (file size and mtime) is not applied, so a file edited since still counts.
    """
    paths = list(decisions["files"])
    capture_of = {p: d["capture"] for p, d in decisions["files"].items()}
    notes: list[str] = []
    db_path = Path(db) if db else find_triage_db()
    truth: dict[str, Truth] = {}
    if db_path is not None and db_path.is_file():
        con = _open_ro(db_path)
        try:
            rows = con.execute("SELECT session_id, path, winner, reject, rating, tags_json FROM decisions").fetchall()
        finally:
            con.close()
        by_session: dict[str, list[tuple[str, Truth]]] = defaultdict(list)
        for sid, path, winner, reject, rating, tags_json in rows:
            try:
                tags = tuple(json.loads(tags_json or "[]"))
            except ValueError:
                tags = ()
            t = Truth(rating=int(rating or 0), tags=tags, source="ImageTriage")
            if reject:
                t.state, t.note = "rejected", "ImageTriage Reject"
            elif winner:
                t.state, t.note = "kept", "ImageTriage Winner"
            elif KEEP_TAG in {x.casefold() for x in tags}:
                t.state, t.note = "kept", "ImageTriage Keep"
            by_session[sid].append((path, t))
        matched = {sid: match_paths(entries, paths, capture_of) for sid, entries in by_session.items()}
        if session is None and matched:
            session = max(matched, key=lambda s: (len(matched[s]), s == "Default"))
        if session is not None:
            if session not in by_session:
                raise ValueError(f"ImageTriage session {session!r} not found; sessions: {', '.join(sorted(by_session)) or 'none'}")
            truth = matched[session]
            others = sorted(s for s in matched if s != session and matched[s])
            notes.append(f"database {db_path} (session {session!r}: {len(truth)} files matched)"
                         + (f"; other sessions with rows for this shoot: {', '.join(others)}" if others else ""))
    else:
        notes.append("no ImageTriage decisions.sqlite3 found" + (f" at {db_path}" if db_path else f" (looked in: {', '.join(str(p) for p in triage_db_candidates())})"))
    sidecar = 0
    for p in paths:
        if p not in truth:
            t = read_xmp_truth(p, embedded=False)
            if t is not None:
                truth[p] = t
                sidecar += 1
    if sidecar:
        notes.append(f"{sidecar} file(s) from XMP sidecars")
    return TruthSet(truth, "ImageTriage decisions", mode="labels", unlabelled="exclude", notes=notes)


# --------------------------------------------------------------------------- Lightroom catalog


def read_lrcat(catalog: str | Path) -> list[tuple[str, Truth]]:
    """(catalog path, truth) per file in the catalog, including JPEG sidecars. Opens the catalog read-only."""
    uri = Path(catalog).resolve().as_uri() + "?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            """
            SELECT i.rating, i.pick, i.colorLabels, rf.absolutePath, fo.pathFromRoot,
                   f.idx_filename, f.baseName, f.sidecarExtensions
            FROM Adobe_images i
            JOIN AgLibraryFile f ON i.rootFile = f.id_local
            JOIN AgLibraryFolder fo ON f.folder = fo.id_local
            JOIN AgLibraryRootFolder rf ON fo.rootFolder = rf.id_local
            """
        ).fetchall()
    finally:
        con.close()
    out = []
    for rating, pick, label, root, sub, filename, base, sidecars in rows:
        folder = (root or "") + (sub or "")
        t = Truth(
            rating=int(rating) if rating is not None else None,
            pick=int(round(float(pick))) if pick is not None else 0,
            label=label or "",
            source="catalog",
        )
        out.append((folder + filename, t))
        for ext in (sidecars or "").split(","):
            ext = ext.strip()
            if ext and ext.casefold() != "xmp":
                out.append((f"{folder}{base}.{ext}", t))
    return out


def match_catalog(catalog_rows: list[tuple[str, Truth]], paths: list[str]) -> dict[str, Truth]:
    return match_paths(catalog_rows, paths)


def lrcat_truth(path: str | Path, decisions: dict) -> TruthSet:
    capture_of = {p: d["capture"] for p, d in decisions["files"].items()}
    truth = match_paths(read_lrcat(path), list(decisions["files"]), capture_of)
    return TruthSet(truth, f"Lightroom catalog {Path(path).name} (read-only)", mode="labels", unlabelled="not_kept")


# --------------------------------------------------------------------------- dispatch


def load_truth(
    source: str | Path,
    decisions: dict,
    *,
    truth_file: str | Path | None = None,
    session: str | None = None,
    unlabelled: str | None = None,
) -> TruthSet:
    """``source`` is one of SOURCES, or a path: ``*.csv``, ``*.lrcat``, ``*.sqlite3`` (ImageTriage)."""
    name = str(source)
    kind = name.casefold()
    if kind not in SOURCES:
        suffix = Path(name).suffix.casefold()
        kind = {".csv": "csv", ".lrcat": "lrcat", ".sqlite3": "triage", ".sqlite": "triage"}.get(suffix)
        if kind is None:
            raise ValueError(f"--truth must be one of {', '.join(SOURCES)} or a .csv/.lrcat/.sqlite3 path; got {name}")
        truth_file = name
    if kind in ("csv", "lrcat"):
        if not truth_file or not Path(truth_file).is_file():
            raise ValueError(f"--truth {kind} needs --truth-file pointing at an existing file")
    if kind == "picks":
        ts = picks_truth(decisions)
    elif kind == "csv":
        ts = csv_truth(truth_file, decisions)
    elif kind == "triage":
        ts = triage_truth(decisions, truth_file, session)
    elif kind == "xmp":
        ts = xmp_truth(decisions)
    else:
        ts = lrcat_truth(truth_file, decisions)
    if unlabelled:
        ts.unlabelled = unlabelled.replace("-", "_")
    return ts
