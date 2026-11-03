"""Outputs: ``decisions.json`` (schema ``aiculler-decisions/1``). XMP comes in milestone 5."""

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
