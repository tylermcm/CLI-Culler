"""``ModelRegistry``: where the app's pinned model files are, and loaded engines for the scorers.

Models are never downloaded or re-pinned here. ``ModelRegistry.from_app_manifest()`` imports the app's own
``image_triage.ai_manifest`` (which also registers the ranker bundles from ``cascade/rankers/bundles.py``) and
resolves each bundle's directory the way the app does (``ai_model_store.bundle_install_dir``: the bundle's
``*_MODEL_DIR`` environment override, else ``<managed AI root>/models/<install_parts>``). Install a model
with the app's Set Up AI; the culler only reads it.

Tests build a registry from fake engines with ``ModelRegistry.fake()``.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Spec key -> the app's bundle key (ai_manifest.MODEL_BUNDLES).
MODEL_KEYS: dict[str, str] = {
    "tinyclip": "tinyclip",
    "maniqa": "maniqa",
    "q_realign": "qrealign",
    "face_landmarker": "faces_blink",
    "adaface": "faces_identity",
}


class ModelUnavailable(RuntimeError):
    pass


@dataclass
class ModelEntry:
    key: str  # spec key
    bundle: str  # app bundle key
    path: Path | None  # install directory
    revision: str
    files: tuple[str, ...] = ()
    digests: dict[str, tuple[int, str]] = field(default_factory=dict)  # file -> (size, sha256)
    problem: str | None = None  # why it is unavailable


class ModelRegistry:
    def __init__(self, entries: dict[str, ModelEntry], *, device: str = "auto", engines: dict[str, object] | None = None,
                 factories: dict[str, Callable[["ModelRegistry"], object]] | None = None):
        self.entries = entries
        self.device = device
        self._engines: dict[str, object] = dict(engines or {})
        self._factories = factories
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_app_manifest(cls, app_path: str | Path | None = None, *, device: str = "auto", verify: str = "size") -> "ModelRegistry":
        """Resolve every model from the app's manifest.

        ``app_path``: the app's ``ai_manifest.py`` or its repository/installation root; omitted, the installed
        ``image_triage`` package is used. ``verify``: "size" (default, fast), "sha256" (hashes every file once,
        slow for Q-ReAlign's 2.2 GB) or "none".
        """
        manifest, store = _import_app(app_path)
        entries: dict[str, ModelEntry] = {}
        for key, bundle_key in MODEL_KEYS.items():
            bundle = manifest.MODEL_BUNDLES.get(bundle_key)
            if bundle is None:
                entries[key] = ModelEntry(key, bundle_key, None, "", problem=f"bundle {bundle_key!r} is not in the app's manifest")
                continue
            digests = {name: (d.size, d.sha256) for name, d in manifest.MODEL_FILE_DIGESTS.get(bundle_key, {}).items()}
            path = store(bundle_key, bundle)
            revision = os.environ.get(bundle.revision_env, "").strip() if bundle.revision_env else ""
            entry = ModelEntry(key, bundle_key, path, revision or bundle.revision, tuple(bundle.filenames), digests)
            entry.problem = _check(entry, verify)
            entries[key] = entry
        return cls(entries, device=device)

    @classmethod
    def fake(cls, engines: dict[str, object], revisions: dict[str, str] | None = None) -> "ModelRegistry":
        """A registry whose engines are given objects (tests). Keys not given are unavailable."""
        revisions = revisions or {}
        entries = {
            key: ModelEntry(key, bundle, None, revisions.get(key, f"fake-{key}"),
                            problem=None if key in engines else "not provided")
            for key, bundle in MODEL_KEYS.items()
        }
        return cls(entries, engines=engines, factories={})

    @classmethod
    def none(cls) -> "ModelRegistry":
        """No models: milestone-1 behaviour (exposure, sharpness, clipping only)."""
        return cls.fake({})

    def restrict(self, keys: set[str]) -> "ModelRegistry":
        """A view of this registry with only ``keys`` usable; every other model reads as "not requested".

        Call it before any engine is loaded: the faces engine is built for the view (tiled search and identity
        vectors only when AdaFace is in it).
        """
        entries = {
            key: ModelEntry(e.key, e.bundle, e.path, e.revision, e.files, e.digests,
                            e.problem if key in keys else (e.problem or "not requested"))
            for key, e in self.entries.items()
        }
        engines = {k: v for k, v in self._engines.items() if k in keys}
        return ModelRegistry(entries, device=self.device, engines=engines, factories=self._factories)

    # ------------------------------------------------------------------ queries

    def available(self, key: str) -> bool:
        entry = self.entries.get(key)
        return entry is not None and entry.problem is None

    def revision(self, key: str) -> str:
        return self.entries[key].revision

    def path(self, key: str) -> Path:
        entry = self.entries[key]
        if entry.problem is not None or entry.path is None:
            raise ModelUnavailable(f"{key}: {entry.problem or 'no path'}")
        return entry.path

    def describe(self) -> dict[str, str]:
        """For decisions.json: revision of each model that is available."""
        return {key: e.revision for key, e in self.entries.items() if e.problem is None}

    def problems(self) -> dict[str, str]:
        return {key: e.problem for key, e in self.entries.items() if e.problem is not None}

    def engine(self, key: str):
        """The loaded engine for ``key`` (loaded once, on first use)."""
        with self._lock:
            if key in self._engines:
                return self._engines[key]
            if not self.available(key):
                raise ModelUnavailable(f"{key}: {self.entries[key].problem}")
            factories = self._factories if self._factories is not None else _default_factories()
            if key not in factories:
                raise ModelUnavailable(f"{key}: no engine for this model")
            engine = factories[key](self)
            self._engines[key] = engine
            return engine

    def close(self) -> None:
        for engine in self._engines.values():
            close = getattr(engine, "close", None)
            if close is not None:
                close()
        self._engines.clear()


def _default_factories() -> dict[str, Callable[[ModelRegistry], object]]:
    from aiculler.models import engines

    return engines.FACTORIES


def _import_app(app_path: str | Path | None):
    """(ai_manifest module, function(bundle_key, bundle) -> install dir) from the app."""
    if app_path is not None:
        root = Path(app_path).expanduser().resolve()
        if root.is_file():  # .../image_triage/ai_manifest.py
            root = root.parent.parent
        if (root / "image_triage").is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        manifest = importlib.import_module("image_triage.ai_manifest")
    except ImportError as exc:
        raise ModelUnavailable(
            "the image-triage app's ai_manifest could not be imported; pass its location (--models PATH_TO_APP) "
            f"or install the app in this environment ({exc})"
        ) from exc
    try:
        store = importlib.import_module("image_triage.ai_model_store")

        def install_dir(bundle_key, bundle):
            return Path(store.bundle_install_dir(bundle_key))
    except ImportError:
        paths = importlib.import_module("image_triage.ai_paths")

        def install_dir(bundle_key, bundle):
            override = os.environ.get(bundle.dir_env, "").strip() if bundle.dir_env else ""
            return Path(override).expanduser().resolve() if override else Path(paths.managed_model_dir(*bundle.install_parts))

    return manifest, install_dir


def _check(entry: ModelEntry, verify: str) -> str | None:
    if entry.path is None or not entry.path.is_dir():
        return f"not installed (no {entry.path}); install it with the app's Set Up AI"
    for name in entry.files:
        f = entry.path / name
        if not f.is_file():
            return f"missing {name} in {entry.path}"
        expected = entry.digests.get(name)
        if expected is None or verify == "none":
            continue
        size, sha = expected
        if f.stat().st_size != size:
            return f"{name} is {f.stat().st_size} bytes, expected {size} (re-install with the app)"
        if verify == "sha256":
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != sha:
                return f"{name} does not match its pinned SHA-256 (re-install with the app)"
    return None
