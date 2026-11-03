"""Shoot ingest: bracketeering grouping, capture identity and the preview cache.

Everything downstream works on *captures* (bracketeering's shots), never on paths: a RAW, its JPEG
sidecar, a copy in ``_picks/`` and an edited export are one capture with several files. Only the
capture is scored; every file then gets its capture's decision.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

PREVIEW_SIZE = 1024  # long edge of the cached, decoded preview


@dataclass(slots=True)
class ExposureStats:
    highlights_clipped: float
    shadows_crushed: float
    mean: float
    flag: str

    def to_dict(self) -> dict:
        return {
            "highlights_clipped": self.highlights_clipped,
            "shadows_crushed": self.shadows_crushed,
            "mean": self.mean,
            "flag": self.flag,
        }


@dataclass(slots=True)
class Capture:
    """One physical capture (bracketeering shot)."""

    id: str  # capture identity: body + shutter count, else body + time + frame number, else path
    primary: str  # path of the primary file (the RAW when there is one)
    files: list[str]  # every file of this capture, primary first
    group: str  # bracketeering group id
    position: int  # 1-based capture position in its group
    role: str  # representative / member / single
    exposure: ExposureStats | None = None
    featureless: bool = False  # preview holds no structure (black misfire, moon in an empty sky)
    exposure_time: float | None = None
    focal_length: float | None = None
    time: float | None = None
    error: str | None = None  # primary file could not be read
    pixels: object | None = None  # bracketeering PixelStats (thumb/layout): registration for scene refinement
    camera: str = ""  # body key (make|model|serial)


@dataclass(slots=True)
class GroupInfo:
    id: str
    kind: str  # exposure / burst / focus / timelapse / single
    confidence: float
    warnings: list[str]
    captures: list[str]  # capture ids in capture order
    representative: str
    best_exposed: str | None
    scene: int | None
    camera: str | None = None
    scene_note: str | None = None  # why scene refinement changed this group's scene


@dataclass(slots=True)
class FileInfo:
    path: str
    capture: str
    primary: bool
    error: str | None = None


@dataclass
class Shoot:
    """Bracketeering's view of a shoot, keyed by capture."""

    groups: dict[str, GroupInfo]
    captures: dict[str, Capture]
    files: dict[str, FileInfo]
    bracketeering_version: str
    report: dict = field(default_factory=dict)  # bracketeering/2 JSON, for debugging and fixtures

    def group_of(self, capture_id: str) -> GroupInfo:
        return self.groups[self.captures[capture_id].group]


def capture_identity(meta, fallback_path: Path) -> str:
    """Stable id for one capture.

    Body (make, model, serial) + shutter count when the maker notes give one; otherwise body +
    capture time + file number; otherwise the resolved primary path.
    """
    body = "|".join(x.strip().lower() for x in (meta.make or "", meta.model or "", meta.serial or ""))
    if meta.shutter_count is not None:
        return f"{body}|sc:{meta.shutter_count}"
    if meta.capture_time is not None:
        fn = meta.file_number if meta.file_number is not None else "-"
        return f"{body}|t:{meta.capture_time:.3f}|fn:{fn}"
    return f"path:{fallback_path.resolve()}"


def ingest(
    folder: str | Path | list[str | Path],
    *,
    backend: str = "auto",
    exclude: list[str] | tuple[str, ...] = (),
    workers: int | None = None,
    analyze: Callable | None = None,
) -> Shoot:
    """Run bracketeering on a shoot and index it by capture."""
    import bracketeering
    from bracketeering.pixels import MIN_DETAIL, detail
    from bracketeering.report import to_dict

    run = analyze or bracketeering.analyze
    roots = [Path(p).expanduser().resolve() for p in (folder if isinstance(folder, list) else [folder])]
    missing = [str(r) for r in roots if not r.exists()]
    if missing:
        raise FileNotFoundError(f"shoot folder not found: {', '.join(missing)}")
    groups, unreadable = run(roots, backend=backend, pixels=True, exclude=list(exclude), workers=workers)
    report = to_dict(groups, unreadable)
    if not report["files"]:
        raise ValueError(f"no image files found in {', '.join(str(r) for r in roots)}")
    errors = {e["file"]: e["error"] for e in report["errors"]}

    shoot_groups: dict[str, GroupInfo] = {}
    captures: dict[str, Capture] = {}
    files: dict[str, FileInfo] = {}
    for g, entry in zip(groups, report["groups"]):
        ids: list[str] = []
        for pos, shot in enumerate(g.shots):
            cid = capture_identity(shot.meta, shot.meta.path)
            if cid in captures:  # two shots claim one identity (e.g. no counter, same second): keep them apart
                cid = f"{cid}|path:{shot.meta.path.resolve()}"
            px = shot.pixels
            stats = None
            featureless = False
            if px is not None:
                stats = ExposureStats(round(px.highlights, 4), round(px.shadows, 4), round(px.mean, 3), px.flag)
                featureless = px.thumb is not None and detail(px.thumb) < MIN_DETAIL
            primary = str(shot.meta.path)
            captures[cid] = Capture(
                id=cid,
                primary=primary,
                files=[primary] + [str(f.path) for f in shot.files if f is not shot.meta],
                group=entry["id"],
                position=pos + 1,
                role=report["files"][primary]["role"],
                exposure=stats,
                featureless=featureless,
                exposure_time=shot.meta.exposure_time,
                focal_length=shot.meta.focal_length,
                time=shot.meta.capture_time,
                error=errors.get(primary),
                pixels=px,
                camera="|".join(shot.camera_key),
            )
            for f in shot.files:
                key = str(f.path)
                files[key] = FileInfo(key, cid, f is shot.meta, errors.get(key))
            ids.append(cid)
        rep = ids[g.representative]
        best = ids[g.best_exposed] if g.best_exposed is not None else None
        shoot_groups[entry["id"]] = GroupInfo(
            id=entry["id"],
            kind=g.kind,
            confidence=float(g.confidence),
            warnings=list(g.warnings),
            captures=ids,
            representative=rep,
            best_exposed=best,
            scene=g.scene,
            camera=entry.get("camera"),
        )
    # Every input file exactly once: bracketeering guarantees it; make sure nothing got lost on the way.
    missing = set(report["files"]) - set(files)
    if missing:
        raise RuntimeError(f"bracketeering files not indexed by capture: {sorted(missing)[:5]}")
    return Shoot(shoot_groups, captures, files, bracketeering.__version__, report)


_TIFF_ORIENTATION = 0x0112
_TRANSPOSE = {2: "FLIP_LEFT_RIGHT", 3: "ROTATE_180", 4: "FLIP_TOP_BOTTOM", 5: "TRANSPOSE", 6: "ROTATE_270", 7: "TRANSVERSE", 8: "ROTATE_90"}


def raw_orientation(path: str | Path) -> int | None:
    """EXIF orientation (1-8) from a TIFF-based RAW container (NEF, ARW, CR2, DNG, ORF, ...), else None.

    Same rule as the app (image_triage/raw_embedded_jpeg.py): the first IFD of the *main* chain carrying 0x0112.
    Sub-IFDs and thumbnails are skipped: they hold the sensor's orientation (1) however the camera was held.
    Embedded previews often carry no orientation of their own, so without this a vertical frame is analysed sideways.
    """
    import struct

    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if head[:2] == b"II":
                e = "<"
            elif head[:2] == b"MM":
                e = ">"
            else:
                return None
            if struct.unpack(e + "H", head[2:4])[0] != 42:
                return None
            offset = struct.unpack(e + "I", head[4:8])[0]
            seen: set[int] = set()
            while offset and offset not in seen and len(seen) < 16:
                seen.add(offset)
                fh.seek(offset)
                raw = fh.read(2)
                if len(raw) != 2:
                    return None
                count = struct.unpack(e + "H", raw)[0]
                if not 0 < count <= 4096:
                    return None
                data = fh.read(count * 12 + 4)
                if len(data) < count * 12 + 4:
                    return None
                for i in range(count):
                    tag, typ, n = struct.unpack(e + "HHI", data[i * 12 : i * 12 + 8])
                    if tag == _TIFF_ORIENTATION and n >= 1:
                        value = struct.unpack(e + "H", data[i * 12 + 8 : i * 12 + 10])[0] if typ == 3 else struct.unpack(e + "I", data[i * 12 + 8 : i * 12 + 12])[0]
                        return int(value) if 1 <= value <= 8 else None
                offset = struct.unpack(e + "I", data[count * 12 : count * 12 + 4])[0]
    except (OSError, struct.error):
        return None
    return None


def decode_upright(path: Path, long_edge: int, *, resample: str = "bilinear"):
    """The file's preview as an upright RGB PIL image bounded to ``long_edge`` (never enlarged), or None.

    RAW files are read through bracketeering's embedded-preview extraction (never decoded) and rotated by the
    container's orientation; rendered files and RAWs without one use the JPEG's own EXIF orientation. ``resample``
    "lanczos" matches the app's model inputs (cascade/rankers/worker.load_image); "bilinear" is for the cache.
    """
    from PIL import Image, ImageOps
    from bracketeering.preview import preview_source

    src = preview_source(path)
    if src is None:
        return None
    try:
        with Image.open(io.BytesIO(src) if isinstance(src, bytes) else src) as im:
            im.draft("RGB", (long_edge, long_edge))  # JPEG: decode at a reduced scale directly
            orientation = raw_orientation(path) if isinstance(src, bytes) else None
            if orientation and orientation != 1:
                im = im.convert("RGB").transpose(getattr(Image.Transpose, _TRANSPOSE[orientation]))
            else:
                im = ImageOps.exif_transpose(im).convert("RGB")
            if max(im.size) > long_edge:
                scale = long_edge / max(im.size)
                size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
                im = im.resize(size, Image.Resampling.LANCZOS if resample == "lanczos" else Image.Resampling.BILINEAR)
            return im
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


class PreviewCache:
    """Decoded previews at ~1024 px, cached on disk by capture identity.

    Uses ``bracketeering.preview.preview_source``, which parses the RAW container and returns its
    largest decodable embedded JPEG (or the file itself for rendered formats). Never decodes RAW data.
    Previews are upright (``decode_upright``).
    """

    VERSION = 2  # 2: rotated by the RAW container's orientation

    def __init__(self, cache_dir: str | Path, size: int = PREVIEW_SIZE):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.size = size

    def path_for(self, capture_id: str) -> Path:
        digest = hashlib.sha1(capture_id.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{digest}_{self.size}_v{self.VERSION}.jpg"

    def load(self, capture: Capture):
        """PIL RGB image of the capture's preview, or None if it has none."""
        from PIL import Image

        cached = self.path_for(capture.id)
        if cached.exists():
            try:
                with Image.open(cached) as im:
                    return im.convert("RGB")
            except OSError:
                cached.unlink(missing_ok=True)
        image = None
        for path in capture.files:  # primary first; a sidecar JPEG can stand in for an unreadable RAW
            image = decode_upright(Path(path), self.size)
            if image is not None:
                break
        if image is None:
            return None
        tmp = cached.with_suffix(".tmp")
        image.save(tmp, "JPEG", quality=92)
        tmp.replace(cached)
        return image

    def load_for_model(self, capture: Capture, long_edge: int):
        """The capture's preview at a model's own input size (LANCZOS, like the app); not cached on disk."""
        for path in capture.files:
            image = decode_upright(Path(path), long_edge, resample="lanczos")
            if image is not None:
                return image
        return None


def luma(image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)
