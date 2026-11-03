"""Outputs: ``decisions.json`` (schema ``aiculler-decisions/1``), the search/people index (``index.npz``, schema
``aiculler-index/1``) and XMP sidecars."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiculler.api import CullResult

SCHEMA = "aiculler-decisions/1"


def decisions_document(result: "CullResult") -> dict:
    files = {path: result.files[path].to_dict() for path in sorted(result.files)}
    return {
        "schema": SCHEMA,
        "engine_version": result.engine_version,
        "bracketeering_version": result.bracketeering_version,
        "models": dict(result.models),
        "settings": result.settings.to_dict(),
        "summary": result.summary,
        "timings": dict(result.timings),
        "stats": dict(result.stats),
        "warnings": list(result.warnings),
        "files": files,
    }


def write_decisions(result: "CullResult", path: str | Path) -> Path:
    """Write decisions.json atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    doc = decisions_document(result)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=target.parent, prefix=target.name, suffix=".tmp") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        tmp = Path(fh.name)
    os.replace(tmp, target)
    return target


def load_decisions(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA}, got {doc.get('schema')!r}")
    return doc


# --------------------------------------------------------------------------- index

INDEX_SCHEMA = "aiculler-index/1"


def write_index(result: "CullResult", path: str | Path) -> Path | None:
    """Write the search and people index (numpy ``.npz``); None when no index model ran.

    Arrays (row i of a per-capture array belongs to ``captures[i]``):

    * ``meta``: JSON (schema, engine version, model revisions, face cluster threshold)
    * ``captures`` (str), ``files`` (str), ``file_capture`` (int, into ``captures``): every file of the shoot
    * search: ``clip`` (float32, unit length, one row per embedded capture), ``clip_capture`` (int),
      ``capture_categories`` (str per capture, comma-separated TinyCLIP categories, most likely first)
    * people: one row per face: ``face_capture`` (int), ``face_index`` (its position in the capture's faces),
      ``face_person`` (1-based person number, as in decisions.json), ``face_box`` (left, top, right, bottom as
      fractions of the upright frame), ``face_area``, ``face_eyes_closed`` (NaN when unknown), ``face_embedding``
      (AdaFace, unit length)
    """
    import numpy as np

    index = result.index or {}
    if not index:
        return None
    captures = list(index["captures"])
    pos = {cid: i for i, cid in enumerate(captures)}
    files = [(f, pos[cid]) for cid in captures for f in index["captures"][cid]]
    meta = {"schema": INDEX_SCHEMA, "engine_version": result.engine_version}
    arrays: dict[str, object] = {
        "captures": np.asarray(captures, dtype=str),
        "files": np.asarray([f for f, _ in files], dtype=str),
        "file_capture": np.asarray([i for _, i in files], dtype=np.int32),
    }
    search = index.get("search")
    if search:
        ids = [cid for cid in captures if cid in search["embeddings"]]
        meta["search_model"] = search["model"]
        arrays["clip"] = np.asarray([search["embeddings"][c] for c in ids], dtype=np.float32)
        arrays["clip_capture"] = np.asarray([pos[c] for c in ids], dtype=np.int32)
        if search.get("categories"):
            meta["categories_version"] = search["categories_version"]
            arrays["capture_categories"] = np.asarray([",".join(search["categories"].get(c, [])) for c in captures], dtype=str)
    people = index.get("people")
    if people:
        faces = people["faces"]
        meta["people_model"] = people["model"]
        meta["face_cluster_threshold"] = people["threshold"]
        meta["people"] = len({f["person"] for f in faces})
        arrays["face_capture"] = np.asarray([pos[f["capture"]] for f in faces], dtype=np.int32)
        arrays["face_index"] = np.asarray([f["index"] for f in faces], dtype=np.int32)
        arrays["face_person"] = np.asarray([f["person"] for f in faces], dtype=np.int32)
        arrays["face_box"] = np.asarray([f["box"] for f in faces], dtype=np.float32).reshape(-1, 4)
        arrays["face_area"] = np.asarray([f["area"] for f in faces], dtype=np.float32)
        arrays["face_eyes_closed"] = np.asarray(
            [np.nan if f["eyes_closed"] is None else f["eyes_closed"] for f in faces], dtype=np.float32)
        arrays["face_embedding"] = np.asarray([f["embedding"] for f in faces], dtype=np.float32)
    arrays["meta"] = np.asarray(json.dumps(meta))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, target)
    return target


def load_index(path: str | Path) -> dict:
    """Read an index written by ``write_index``: {"meta": dict, <array name>: ndarray}."""
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    meta = json.loads(str(out.pop("meta")))
    if meta.get("schema") != INDEX_SCHEMA:
        raise ValueError(f"{path}: expected schema {INDEX_SCHEMA}, got {meta.get('schema')!r}")
    out["meta"] = meta
    return out


# --------------------------------------------------------------------------- XMP

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
X_NS = "adobe:ns:meta/"
XMP_NS = "http://ns.adobe.com/xap/1.0/"
TRIAGE_NS = "https://image-triage.app/ns/1.0/"
CULLER_NS = "https://github.com/tylermcm/cli-culler/ns/1.0/"
RAW_SUFFIXES = {".nef", ".nrw", ".arw", ".cr2", ".cr3", ".crw", ".dng", ".raf", ".rw2", ".orf", ".pef", ".srw", ".x3f", ".3fr", ".iiq", ".raw"}
# keep = 3 stars; review = yellow label; reject = 1 star + red label.
XMP_VALUES = {"keep": ("3", ""), "review": ("", "Yellow"), "reject": ("1", "Red")}


def sidecar_path(path: str | Path) -> Path:
    """NAME.xmp next to a RAW (Lightroom, ImageTriage); NAME.EXT.xmp for other files (ImageTriage's choice)."""
    p = Path(path)
    return p.with_suffix(".xmp") if p.suffix.lower() in RAW_SUFFIXES else Path(f"{p}.xmp")


def write_xmp(result: "CullResult", *, overwrite_ratings: bool = False) -> dict[str, int]:
    """Write each file's decision to its XMP sidecar (xmp:Rating, xmp:Label), keeping everything else in it.

    A sidecar that already holds a rating, label or ImageTriage decision the culler did not write is left alone
    unless ``overwrite_ratings``. What the culler writes is recorded (``aiculler:Rating``/``aiculler:Label``), so
    a later run may update its own output. Returns counts: written, unchanged, skipped (user ratings).
    """
    from xml.etree import ElementTree as ET

    for prefix, ns in (("x", X_NS), ("rdf", RDF_NS), ("xmp", XMP_NS), ("triage", TRIAGE_NS), ("aiculler", CULLER_NS)):
        ET.register_namespace(prefix, ns)
    q = lambda ns, name: f"{{{ns}}}{name}"  # noqa: E731
    rating_k, label_k = q(XMP_NS, "Rating"), q(XMP_NS, "Label")
    ours_rating, ours_label, ours_decision = q(CULLER_NS, "Rating"), q(CULLER_NS, "Label"), q(CULLER_NS, "Decision")
    review_state = q(TRIAGE_NS, "ReviewState")
    desc_tag, about = q(RDF_NS, "Description"), q(RDF_NS, "about")

    def field(desc, key):
        v = desc.get(key)
        if v is not None:
            return v.strip()
        child = desc.find(key)
        return (child.text or "").strip() if child is not None and child.text else ""

    def set_field(desc, key, value):
        child = desc.find(key)
        if child is not None:
            desc.remove(child)
        if value:
            desc.set(key, value)
        else:
            desc.attrib.pop(key, None)

    counts = {"written": 0, "unchanged": 0, "skipped": 0}
    done: set[Path] = set()
    for path, d in sorted(result.files.items()):
        target = sidecar_path(path)
        if target in done:
            continue
        done.add(target)
        rating, label = XMP_VALUES[d.decision]
        tree = None
        if target.is_file():
            try:
                tree = ET.parse(target)
            except (ET.ParseError, OSError):
                counts["skipped"] += 1  # unreadable: never clobber
                continue
        if tree is None:
            root = ET.Element(q(X_NS, "xmpmeta"))
            rdf = ET.SubElement(root, q(RDF_NS, "RDF"))
            ET.SubElement(rdf, desc_tag, {about: ""})
            tree = ET.ElementTree(root)
        descs = list(tree.getroot().iter(desc_tag))
        if not descs:
            rdf = tree.getroot().find(f".//{q(RDF_NS, 'RDF')}")
            if rdf is None:
                counts["skipped"] += 1
                continue
            descs = [ET.SubElement(rdf, desc_tag, {about: ""})]
        desc = descs[0]
        cur_rating = next((field(x, rating_k) for x in descs if field(x, rating_k)), "")
        cur_label = next((field(x, label_k) for x in descs if field(x, label_k)), "")
        user_state = any(field(x, review_state) for x in descs)
        mine = field(desc, ours_rating) == cur_rating and field(desc, ours_label) == cur_label and field(desc, ours_decision)
        users = user_state or ((cur_rating not in ("", "0") or cur_label) and not mine)
        if users and not overwrite_ratings:
            counts["skipped"] += 1
            continue
        if cur_rating == rating and cur_label == label and field(desc, ours_decision) == d.decision:
            counts["unchanged"] += 1
            continue
        for x in descs:  # one value each, on the first description
            if x is not desc:
                set_field(x, rating_k, "")
                set_field(x, label_k, "")
        set_field(desc, rating_k, rating)
        set_field(desc, label_k, label)
        set_field(desc, ours_rating, rating)
        set_field(desc, ours_label, label)
        set_field(desc, ours_decision, d.decision)
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("wb", delete=False, dir=target.parent, prefix=target.name, suffix=".tmp") as fh:
            tree.write(fh, encoding="utf-8", xml_declaration=True)
            tmp = Path(fh.name)
        os.replace(tmp, target)
        counts["written"] += 1
    return counts
