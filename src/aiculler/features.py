from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from aiculler.storage import SQLiteFeatureStore

RAW_EXTENSIONS = {".nef", ".arw", ".cr2", ".cr3", ".crw", ".dng", ".gpr", ".raf", ".rw2"}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class HeadlessFeatureExtractor:
    """Frozen dual-stream ONNX feature extractor for visual and technical signals."""

    def __init__(
        self,
        clip_onnx_path: str | Path,
        topiq_onnx_path: str | Path | None = None,
        *,
        providers: list[str] | None = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNX feature extraction") from exc

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads = max(1, min(8, threading.active_count() or 1))
        providers = providers or self._available_providers(ort)
        self.clip_session = ort.InferenceSession(str(clip_onnx_path), opts, providers=providers)
        self.clip_input_name = self._select_image_input(self.clip_session)
        self.clip_output_name = self._select_embedding_output(self.clip_session)
        self.clip_input_size = self._select_spatial_size(self.clip_session, default=224)

        self.topiq_session = None
        self.topiq_input_name = None
        self.topiq_input_size = 512
        if topiq_onnx_path is not None and str(topiq_onnx_path).lower().endswith(".onnx"):
            self.topiq_session = ort.InferenceSession(str(topiq_onnx_path), opts, providers=providers)
            self.topiq_input_name = self._select_image_input(self.topiq_session)
            self.topiq_input_size = self._select_spatial_size(self.topiq_session, default=512)
        self.technical_scorer = HeuristicTechnicalScorer()

    @staticmethod
    def _available_providers(ort) -> list[str]:
        preferred = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        available = set(ort.get_available_providers())
        return [provider for provider in preferred if provider in available] or ["CPUExecutionProvider"]

    @staticmethod
    def _select_image_input(session) -> str:
        inputs = session.get_inputs()
        image_inputs = [
            input_meta
            for input_meta in inputs
            if "pixel" in input_meta.name.lower()
            or (len(input_meta.shape) == 4 and "float" in input_meta.type)
        ]
        if not image_inputs:
            raise ValueError("ONNX model does not expose a 4D image/pixel input")
        extra_required = [input_meta.name for input_meta in inputs if input_meta.name != image_inputs[0].name]
        if extra_required:
            raise ValueError(
                "Use a vision-only ONNX model for image extraction; "
                f"this model also requires: {', '.join(extra_required)}"
            )
        return image_inputs[0].name

    @staticmethod
    def _select_embedding_output(session) -> str | None:
        outputs = session.get_outputs()
        for output_meta in outputs:
            if "embed" in output_meta.name.lower():
                return output_meta.name
        for output_meta in outputs:
            if len(output_meta.shape) == 2:
                return output_meta.name
        return outputs[0].name if outputs else None

    @staticmethod
    def _select_spatial_size(session, *, default: int) -> int:
        image_input = session.get_inputs()[0]
        if len(image_input.shape) >= 4:
            height, width = image_input.shape[-2:]
            if isinstance(height, int) and isinstance(width, int) and height == width:
                return height
        return default

    def extract_features(self, image_path: str | Path) -> dict[str, np.ndarray | float]:
        with Image.open(image_path) as opened:
            img = opened.convert("RGB")
        clip_input = self._preprocess_for_clip(img)
        clip_outputs = self.clip_session.run([self.clip_output_name], {self.clip_input_name: clip_input})
        clip_embed = clip_outputs[0]
        if self.topiq_session is not None and self.topiq_input_name is not None:
            topiq_input = self._preprocess_for_topiq(img)
            topiq_score = self.topiq_session.run(None, {self.topiq_input_name: topiq_input})[0]
            technical_score = float(np.asarray(topiq_score).reshape(-1)[0])
        else:
            technical_score = self.technical_scorer.score(img)
        return {
            "embedding": np.asarray(clip_embed, dtype=np.float32).reshape(-1),
            "technical_score": technical_score,
        }

    def _preprocess_for_clip(self, img: Image.Image) -> np.ndarray:
        resized = img.resize((self.clip_input_size, self.clip_input_size), Image.Resampling.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        normalized = (arr - mean) / std
        return np.expand_dims(normalized.transpose(2, 0, 1), axis=0).astype(np.float32)

    def _preprocess_for_topiq(self, img: Image.Image) -> np.ndarray:
        resized = img.resize((self.topiq_input_size, self.topiq_input_size), Image.Resampling.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0
        return np.expand_dims(arr.transpose(2, 0, 1), axis=0).astype(np.float32)


class HeuristicTechnicalScorer:
    """Offline focus/exposure scorer used when a TOPIQ ONNX graph is unavailable."""

    def score(self, img: Image.Image) -> float:
        gray = np.asarray(img.convert("L").resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        sharpness = float(np.clip((np.mean(np.abs(gx)) + np.mean(np.abs(gy))) * 12.0, 0.0, 1.0))
        contrast = float(np.clip(np.std(gray) * 4.0, 0.0, 1.0))
        exposure = float(np.clip(1.0 - abs(float(np.mean(gray)) - 0.5) * 2.0, 0.0, 1.0))
        return float(np.clip(0.55 * sharpness + 0.25 * contrast + 0.20 * exposure, 0.0, 1.0))


@dataclass(frozen=True)
class IngestionEvent:
    image_id: int
    source_path: Path
    preview_path: Path | None
    status: str
    message: str = ""
    preview_seconds: float = 0.0
    feature_seconds: float = 0.0
    total_seconds: float = 0.0


class PreviewExtractor:
    """Extract embedded JPEG previews without external tools.

    Many RAW files contain one or more JPEG byte ranges. This extractor scans for
    JPEG SOI/EOI markers, validates candidates with Pillow, and keeps the largest
    readable preview. Regular image files are normalized into the cache.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, source_path: str | Path) -> tuple[Path, tuple[int, int]]:
        source = Path(source_path)
        target = self.cache_dir / f"{self._stable_name(source)}.jpg"
        if target.exists():
            with Image.open(target) as img:
                return target, img.size
        if source.suffix.lower() in RAW_EXTENSIONS:
            return self._extract_embedded_jpeg(source, target)
        return self._normalize_preview(source, target)

    def _normalize_preview(self, source: Path, target: Path) -> tuple[Path, tuple[int, int]]:
        with Image.open(source) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            rgb.save(target, "JPEG", quality=92, optimize=True)
            return target, rgb.size

    def _extract_embedded_jpeg(self, source: Path, target: Path) -> tuple[Path, tuple[int, int]]:
        data = source.read_bytes()
        starts = [idx for idx in self._find_jpeg_starts(data)]
        best: tuple[int, int, bytes] | None = None
        for start in starts:
            end = data.find(b"\xff\xd9", start + 2)
            if end == -1:
                continue
            candidate = data[start : end + 2]
            if len(candidate) < 1024:
                continue
            if self._is_valid_jpeg(candidate):
                size = len(candidate)
                if best is None or size > best[0]:
                    best = (size, start, candidate)
        if best is None:
            raise UnidentifiedImageError(f"No embedded JPEG preview found in {source}")
        target.write_bytes(best[2])
        with Image.open(target) as img:
            img.verify()
        with Image.open(target) as img:
            return target, img.size

    @staticmethod
    def _find_jpeg_starts(data: bytes) -> Iterable[int]:
        idx = data.find(b"\xff\xd8\xff")
        while idx != -1:
            marker_offset = idx + 3
            if marker_offset < len(data):
                marker = data[marker_offset]
                if 0xC0 <= marker <= 0xFE:
                    yield idx
            idx = data.find(b"\xff\xd8\xff", idx + 2)

    @staticmethod
    def _is_valid_jpeg(data: bytes) -> bool:
        try:
            from io import BytesIO

            with Image.open(BytesIO(data)) as img:
                img.verify()
            return True
        except Exception:
            return False

    @staticmethod
    def _stable_name(source: Path) -> str:
        digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
        return f"{source.stem}-{digest}"


class IngestionEngine:
    """Thread-pool ingestion and feature extraction worker."""

    def __init__(
        self,
        store: SQLiteFeatureStore,
        cache_dir: str | Path,
        *,
        extractor: HeadlessFeatureExtractor | None = None,
        max_workers: int = 4,
        on_event: Callable[[IngestionEvent], None] | None = None,
    ):
        self.store = store
        self.preview_extractor = PreviewExtractor(cache_dir)
        self.extractor = extractor
        self.max_workers = max(1, int(max_workers))
        self.on_event = on_event

    def scan(self, folder: str | Path, *, recursive: bool = True) -> list[Path]:
        root = Path(folder)
        iterator = root.rglob("*") if recursive else root.glob("*")
        return sorted(
            path
            for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def ingest(self, folder: str | Path, *, recursive: bool = True) -> list[int]:
        paths = self.scan(folder, recursive=recursive)
        return self.ingest_paths(paths)

    def ingest_paths(self, paths: Iterable[str | Path]) -> list[int]:
        image_ids: list[int] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._ingest_one, Path(path)) for path in paths]
            for future in as_completed(futures):
                image_id = future.result()
                if image_id is not None:
                    image_ids.append(image_id)
        return image_ids

    def _ingest_one(self, source_path: Path) -> int | None:
        started_at = time.perf_counter()
        image_id = self.store.upsert_image(source_path, status="pending")
        try:
            preview_started_at = time.perf_counter()
            preview_path, (width, height) = self.preview_extractor.extract(source_path)
            preview_seconds = time.perf_counter() - preview_started_at
            image_id = self.store.upsert_image(
                source_path,
                preview_path=preview_path,
                status="previewed",
                width=width,
                height=height,
            )
            self._emit(
                IngestionEvent(
                    image_id,
                    source_path,
                    preview_path,
                    "previewed",
                    preview_seconds=preview_seconds,
                    total_seconds=time.perf_counter() - started_at,
                )
            )
            if self.extractor is not None:
                feature_started_at = time.perf_counter()
                features = self.extractor.extract_features(preview_path)
                feature_seconds = time.perf_counter() - feature_started_at
                self.store.save_features(
                    image_id,
                    np.asarray(features["embedding"], dtype=np.float32),
                    technical_score=float(features["technical_score"]),
                    aesthetic_prior=float(features["technical_score"]),
                    status="ready",
                )
                self._emit(
                    IngestionEvent(
                        image_id,
                        source_path,
                        preview_path,
                        "ready",
                        preview_seconds=preview_seconds,
                        feature_seconds=feature_seconds,
                        total_seconds=time.perf_counter() - started_at,
                    )
                )
            return image_id
        except Exception as exc:
            self.store.mark_error(image_id, str(exc))
            self._emit(
                IngestionEvent(
                    image_id,
                    source_path,
                    None,
                    "error",
                    str(exc),
                    total_seconds=time.perf_counter() - started_at,
                )
            )
            return image_id

    def _emit(self, event: IngestionEvent) -> None:
        if self.on_event is not None:
            self.on_event(event)
