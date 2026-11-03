"""Model engines, ported from the app's ranker worker (``image_triage/cascade/rankers/worker.py``).

Same models, pre/post-processing and score scales as the app; the differences are only where images come from
(an upright PIL image from ``ingest.decode_upright`` at the engine's ``LONG_EDGE`` instead of a path) and that
the engines run in-process instead of in the app's worker subprocess. torch, timm, einops, transformers and
mediapipe are imported only when an engine is built (``pip install 'ai-culler[models]'``, or run inside the
app's managed AI runtime). Precision: fp16 on CUDA, fp32 on CPU, as in the app.
"""

from __future__ import annotations

from pathlib import Path

MANIQA_CROPS = 20
MANIQA_CROP_SIZE = 224
MANIQA_LONG_EDGE = 1280
MANIQA_CROP_SEED = 20
FACES_LONG_EDGE = 1280
FACES_MAX = 10
QREALIGN_LONG_EDGE = 1024
QREALIGN_QUESTION = "Can you rate the aesthetics of this picture?"
QREALIGN_ANSWER_STEM = "The aesthetics of the image is"
QREALIGN_LEVELS = ("excellent", "good", "fair", "poor", "bad")
QREALIGN_WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.0)
MANIQA_CHECKPOINT = "ckpt_koniq10k.pt"
LANDMARKER_FILE = "face_landmarker.task"
ADAFACE_FILE = "adaface_ir_50.onnx"


def pick_device(torch, requested: str) -> str:
    value = (requested or "auto").strip().lower() or "auto"
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and torch.cuda.is_available():
        return value
    return "cpu"


class ManiqaEngine:
    """MANIQA (KonIQ-10k): mean over 20 seeded 224 px crops of the frame at a 1280 px long edge."""

    LONG_EDGE = MANIQA_LONG_EDGE

    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        import numpy as np
        import torch

        from aiculler.models.app.maniqa_net import MANIQA

        self.np, self.torch = np, torch
        self.device = pick_device(torch, device)
        self.half = self.device.startswith("cuda")
        checkpoint = Path(model_dir) / MANIQA_CHECKPOINT
        net = MANIQA(embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8, img_size=224, window_size=4, depths=[2, 2],
                     num_heads=[4, 4], num_tab=2, scale=0.8)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        result = net.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.endswith(("relative_position_index", "attn_mask"))]
        if missing:
            raise RuntimeError(f"{MANIQA_CHECKPOINT} does not match the MANIQA network ({len(missing)} missing tensors, e.g. {missing[0]})")
        net = net.to(self.device).eval()
        self.net = net.half() if self.half else net

    def score(self, image) -> float:
        np, torch = self.np, self.torch
        if min(image.size) <= MANIQA_CROP_SIZE:
            from PIL import Image

            scale = (MANIQA_CROP_SIZE + 1) / min(image.size)
            image = image.resize((max(MANIQA_CROP_SIZE + 1, round(image.width * scale)), max(MANIQA_CROP_SIZE + 1, round(image.height * scale))), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        height, width = pixels.shape[:2]
        rng = np.random.default_rng(MANIQA_CROP_SEED)
        crops = []
        for _ in range(MANIQA_CROPS):
            top, left = int(rng.integers(0, height - MANIQA_CROP_SIZE)), int(rng.integers(0, width - MANIQA_CROP_SIZE))
            crops.append(pixels[top:top + MANIQA_CROP_SIZE, left:left + MANIQA_CROP_SIZE])
        batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2)
        batch = ((batch - 0.5) / 0.5).to(self.device, dtype=torch.float16 if self.half else torch.float32)
        with torch.inference_mode():
            return float(self.net(batch).float().mean())


class QRealignEngine:
    """Q-ReAlign Mini 0.8B: probability-weighted aesthetic level (excellent 1.0 .. bad 0.0) at a 1024 px long edge."""

    LONG_EDGE = QREALIGN_LONG_EDGE

    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        import torch
        import transformers

        self.torch = torch
        self.device = pick_device(torch, device)
        self.half = self.device.startswith("cuda")
        dtype = torch.float16 if self.half else torch.float32
        self.processor = transformers.AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        auto = getattr(transformers, "AutoModelForImageTextToText", None)
        model_class = auto or transformers.Qwen3_5ForConditionalGeneration
        model = model_class.from_pretrained(str(model_dir), dtype=dtype, local_files_only=True)
        self.model = model.to(self.device).eval()
        tokenizer = self.processor.tokenizer
        self.level_ids = [tokenizer(" " + word, add_special_tokens=False)["input_ids"][0] for word in QREALIGN_LEVELS]
        if len(set(self.level_ids)) != len(self.level_ids):
            raise RuntimeError("Q-ReAlign level words share a first token; the tokenizer is not the expected one.")
        self.weights = torch.tensor(QREALIGN_WEIGHTS, dtype=torch.float32, device=self.device)

    def score(self, image) -> float:
        torch = self.torch
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QREALIGN_QUESTION}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[prompt + QREALIGN_ANSWER_STEM], images=[image], return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        if self.half:
            inputs = {k: (v.half() if getattr(v, "is_floating_point", lambda: False)() else v) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        probabilities = torch.softmax(logits[self.level_ids], dim=-1)
        return float((probabilities * self.weights).sum())


def _overlap(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / smaller if smaller > 0 else 0.0


class FacesEngine:
    """Prominent faces (MediaPipe Face Landmarker, tiled search for small faces), each with eyes-closed and,
    when the recognizer is available, an AdaFace identity vector. CPU only, no torch.

    ``detail(image)`` -> {"faces": [{"box", "area", "eyes_closed", "embedding"}], "width", "height"}; boxes are
    fractions of the frame."""

    LONG_EDGE = FACES_LONG_EDGE

    def __init__(self, landmarker_dir: Path, recognizer_dir: Path | None = None) -> None:
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.np, self.mp = np, mp
        self.device = "cpu"
        self.half = False
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(Path(landmarker_dir) / LANDMARKER_FILE)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=FACES_MAX,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.recognizer = None
        if recognizer_dir is not None:
            import onnxruntime

            self.recognizer = onnxruntime.InferenceSession(str(Path(recognizer_dir) / ADAFACE_FILE), providers=["CPUExecutionProvider"])
            self.input_name = self.recognizer.get_inputs()[0].name

    def close(self) -> None:
        landmarker, self.landmarker = getattr(self, "landmarker", None), None
        if landmarker is not None:
            landmarker.close()

    def detail(self, image) -> dict:
        from aiculler.models.app.blink import eyes_closed, face_area
        from aiculler.models.app.faces import PEOPLE_MIN_FACE_AREA, align_crop, five_points, landmark_box, preprocess

        width, height = image.size
        windows = [(0, 0, width, height)]
        # The tiled search finds faces too small for the whole-frame pass, for the People index (AdaFace). Without
        # the recognizer there is no people coverage, and blinks only count faces >= 1% of the frame, which the
        # whole-frame pass finds: that is the app's blink check (BlinkEngine), at a fraction of the cost.
        if self.recognizer is not None and min(width, height) >= 480:
            for size, count in ((0.6, 3), (0.4, 4), (0.27, 5)):
                part_w, part_h = int(width * size), int(height * size)
                for row in range(count):
                    for col in range(count):
                        left = int(col * (width - part_w) / (count - 1))
                        top = int(row * (height - part_h) / (count - 1))
                        windows.append((left, top, left + part_w, top + part_h))
        found = []
        for left, top, right, bottom in windows:
            part = image if (right - left, bottom - top) == (width, height) else image.crop((left, top, right, bottom))
            part_width, part_height = part.size
            pixels = self.np.ascontiguousarray(self.np.asarray(part))
            result = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=pixels))
            for index, landmarks in enumerate(result.face_landmarks or []):
                points = [(p.x, p.y) for p in landmarks]
                area = face_area(points) * (part_width * part_height) / (width * height)
                if area < PEOPLE_MIN_FACE_AREA:
                    continue
                shapes = {c.category_name: float(c.score) for c in (result.face_blendshapes[index] if result.face_blendshapes else [])}
                try:
                    crop = align_crop(part, five_points(points, part_width, part_height))
                except ValueError:
                    continue
                x0, y0, x1, y1 = landmark_box(points)
                box = ((left + x0 * part_width) / width, (top + y0 * part_height) / height, (left + x1 * part_width) / width, (top + y1 * part_height) / height)
                found.append((area, box, crop, shapes))
        found.sort(key=lambda item: -item[0])
        kept = []
        for item in found:
            if any(_overlap(item[1], other[1]) > 0.35 for other in kept):
                continue
            kept.append(item)
        faces = []
        for area, box, crop, shapes in kept[:FACES_MAX]:
            embedding = []
            if self.recognizer is not None:
                vector = self.recognizer.run(None, {self.input_name: preprocess(crop)})[0][0]
                norm = float(self.np.linalg.norm(vector))
                vector = vector / norm if norm > 0 else vector
                embedding = [round(float(v), 6) for v in vector]
            closed = eyes_closed(shapes)
            faces.append({
                "box": [round(v, 5) for v in box],
                "area": round(area, 5),
                "embedding": embedding,
                "eyes_closed": None if closed is None else round(closed, 4),
            })
        return {"faces": faces, "width": width, "height": height}


class TinyClipEngine:
    """TinyCLIP image embeddings, as the app computes them (the app's vendored ``aiculler.features``:
    ``SemanticEmbeddingExtractor`` + ``HeadlessFeatureExtractor.preprocess_clip_pixels``): the combined ONNX graph,
    output ``image_embeds``, dummy text inputs, 224 px BILINEAR resize, CLIP mean/std."""

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_dir: Path, device: str = "auto") -> None:
        import numpy as np
        import onnxruntime as ort

        self.np = np
        self.model_dir = Path(model_dir)
        self._tokenizer = None
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = set(ort.get_available_providers())
        preferred = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider") if p in available]
        if str(device).startswith("cpu"):
            preferred = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(Path(model_dir) / "onnx" / "model.onnx"), opts, providers=preferred or ["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        image_inputs = [m for m in inputs if "pixel" in m.name.lower() or (len(m.shape) == 4 and "float" in m.type)]
        if not image_inputs:
            raise RuntimeError("TinyCLIP model has no image input")
        self.input_name = image_inputs[0].name
        shape = image_inputs[0].shape
        self.size = shape[-1] if isinstance(shape[-1], int) and shape[-1] == shape[-2] else 224
        outputs = [m.name for m in self.session.get_outputs()]
        self.output_name = next((n for n in outputs if n.casefold() == "image_embeds"), None) or next(
            (n for n in outputs if "image" in n.casefold() and "embed" in n.casefold()), outputs[0])
        self.extra = {}
        for m in inputs:
            if m.name == self.input_name:
                continue
            length = m.shape[-1] if isinstance(m.shape[-1], int) else 77
            if m.name.casefold() == "input_ids":
                self.extra[m.name] = np.zeros((1, length), dtype=np.int64)
            elif m.name.casefold() == "attention_mask":
                self.extra[m.name] = np.ones((1, length), dtype=np.int64)
            else:
                raise RuntimeError(f"unsupported TinyCLIP input {m.name}")

    def encode_text(self, prompt: str) -> list[float]:
        """Unit-length ``text_embeds`` for a prompt, as the app's ``CLIPTextEncoder`` computes them: tokenizer.json
        padded/truncated to 77 with <|endoftext|>, a blank image on the pixel input of the combined graph."""
        np = self.np
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            pad = tok.token_to_id("<|endoftext|>")
            tok.enable_truncation(max_length=77)
            tok.enable_padding(length=77, pad_id=pad, pad_token="<|endoftext|>")
            self._tokenizer = tok
        enc = self._tokenizer.encode(prompt)
        outputs = [m.name for m in self.session.get_outputs()]
        name = next((n for n in outputs if n.casefold() == "text_embeds"), None)
        if name is None:
            raise RuntimeError("TinyCLIP model has no text_embeds output")
        feeds = {}
        for m in self.session.get_inputs():
            if m.name == self.input_name:
                feeds[m.name] = np.zeros((1, 3, self.size, self.size), dtype=np.float32)
            elif m.name.casefold() == "input_ids":
                feeds[m.name] = np.asarray([enc.ids], dtype=np.int64)
            elif m.name.casefold() == "attention_mask":
                feeds[m.name] = np.asarray([enc.attention_mask], dtype=np.int64)
        vector = np.asarray(self.session.run([name], feeds)[0], dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return [float(v) for v in (vector / norm if norm > 0 else vector)]

    def embed(self, image) -> list[float]:
        from PIL import Image

        np = self.np
        arr = np.asarray(image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        arr = (arr - np.array(self.MEAN, dtype=np.float32)) / np.array(self.STD, dtype=np.float32)
        pixels = np.expand_dims(arr.transpose(2, 0, 1), 0).astype(np.float32)
        vector = np.asarray(self.session.run([self.output_name], {self.input_name: pixels, **self.extra})[0], dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return [round(float(v), 6) for v in (vector / norm if norm > 0 else vector)]


def _faces(registry) -> FacesEngine:
    recognizer = registry.path("adaface") if registry.available("adaface") else None
    return FacesEngine(registry.path("face_landmarker"), recognizer)


FACTORIES = {
    "maniqa": lambda r: ManiqaEngine(r.path("maniqa"), r.device),
    "q_realign": lambda r: QRealignEngine(r.path("q_realign"), r.device),
    "face_landmarker": _faces,
    "tinyclip": lambda r: TinyClipEngine(r.path("tinyclip"), r.device),
}
