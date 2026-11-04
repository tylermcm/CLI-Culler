# Models: where the app's inference lives

**All optional.** The cull runs without any model; each one is asked for with `--with` (`faces`, `maniqa`,
`appeal`, `search`, `people`, `all`) or `"models"` in the settings, and only those are loaded (see the README).

**Using them.** `pip install -e '.[models]'` adds the runtime (torch, timm, einops, transformers, mediapipe; the
app runtime's minimums). Install the model files with the app's Set Up AI, then point the culler at the app:
`--models /path/to/ImageTriage` (its folder or `image_triage/ai_manifest.py`), or `--models auto` when the
`image_triage` package is installed in the same environment. Files are found the way the app finds them: the
bundle's `IMAGE_TRIAGE_*_MODEL_DIR` environment variable if set (e.g. `IMAGE_TRIAGE_MANIQA_MODEL_DIR`,
`IMAGE_TRIAGE_BLINK_MODEL_DIR`, `IMAGE_TRIAGE_FACES_MODEL_DIR`, `IMAGE_TRIAGE_QREALIGN_MODEL_DIR`,
`IMAGE_TRIAGE_AICULLER_CLIP_MODEL_DIR`), else `<IMAGE_TRIAGE_AI_ROOT or ~/.local/share/ImageTriage/AI>/models/...`.
File sizes are checked against the pinned digests on every run.

The ported code lives in `src/aiculler/models/` (`app/` verbatim, `engines.py` adapted to take images).

The engine never downloads or pins models; it loads them through the image-triage app's manifest. The code to
reuse is the app's ranker subprocess, `image_triage/cascade/rankers/` (it holds torch, transformers and mediapipe
so the app process never imports them). `image_triage/quality/face.py` is the legacy AuraFace/InsightFace path:
**not** to be reused.

| Spec key | Bundle key | Inference | Model-free logic around it |
|---|---|---|---|
| `q_realign` | `QREALIGN` (`spec.py:21`) | `QRealignEngine`, `cascade/rankers/worker.py:146`: `transformers.AutoModelForImageTextToText` (fallback `Qwen3_5ForConditionalGeneration`); asks for an aesthetics rating, reads the probabilities of excellent/good/fair/poor/bad and weights them 1.0…0.0; images resized to a 1024 px long edge. Started by `qrealign_ranker` (`client.py:378`), called from `cascade/launch.py:81` | — |
| `maniqa` | `MANIQA` | `ManiqaEngine`, `worker.py:97`; network in `_maniqa_net.py` | — |
| `face_landmarker` (blink) | `BLINK` | `BlinkEngine`, `worker.py:192` | `cascade/blink.py` |
| `face_landmarker` + `adaface` (people) | `faces_identity` | `FacesEngine`, `worker.py:244`: the landmarker finds faces, AdaFace runs in onnxruntime on CPU (`worker.py:274`) | `cascade/faces.py`: five alignment points, similarity transform, 112×112 crop, preprocessing. Which frames: the person pre-filter of `face_index.py` (`plan_face_index`: TinyCLIP person probability ≥ 0.15, human/animal margin > −0.05), ported unchanged to `people.person_filter` |

The four ranker bundles are defined in `cascade/rankers/bundles.py` and added to `ai_manifest.MODEL_BUNDLES` /
`MODEL_FILE_DIGESTS` at import time by `_register_cascade_bundles()` at the end of `ai_manifest.py`.

Pins, checked against `bundles.py` (they match `CULLING_ENGINE_SPEC.md` §2):

| Model | Repo @ revision | File | Size (bytes) | SHA-256 |
|---|---|---|---|---|
| MANIQA KonIQ-10k | `Skulleton12/maniqa-koniq10k` @ `18bbbb8a87d06d7066de3fb0412139520fd6ff91` | `ckpt_koniq10k.pt` | 543,335,435 | `a207f8ab57322e6be38ff5c8d019301dc032b454bef21c3c9f9dbf7974eebff6` |
| MediaPipe Face Landmarker f16 | `Skulleton12/mediapipe-face-landmarker` @ `e8098a34013a130b8d8c88de0772f162b450dfad` | `face_landmarker.task` | 3,758,596 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| AdaFace IR-50 (UniFace ONNX) | `yakhyo/uniface-weights` @ `bfcc7d48ea8412d55de406cecf96b65b3ca720b2` | `adaface_ir_50.onnx` | 174,396,180 | `9c0ae385d7362323c92d218de598d32e52a9130b78faac6a8a924091aca62c0b` |
| Q-ReAlign Mini 0.8B | `q-future/Q-ReAlign-Mini-0.8B` @ `fe1f45a7574c9e9d908875af9f7e90cb946aa19f` | `model.safetensors` (+ 7 config/tokenizer files) | 2,214,590,296 | `bde34df0375fff90d2dee716a127039c57d310c0c868b9f52f4fc2d1ead34aac` |
| TinyCLIP | see `ai_manifest.py` | | | |

## Q-ReAlign and MANIQA: what is kept, and how they are run

- **Q-ReAlign** asks both official tasks (`qalign/prompts.py`, first prompt of each pool, as `qalign/infer.py` does):
  image quality ("Can you rate the quality of this picture?" / stem "The quality of the image is") and aesthetics
  ("How is the aesthetics of this image?" / "The aesthetics of the image is"). As in `qalign/scorer.py`, the stem is
  appended as token ids (with its leading space) after the templated user turn; Qwen3.5's template ends with an
  empty thinking block either way. Scores: `qra_iqa` (quality) and `appeal` (aesthetics); `outputs.qra_probs`
  keeps the five level probabilities per task. The image is given at a 1024 px long edge, as the app does (the
  processor's own resizing applies on top); this differs from the reference, which hands the processor the full
  image. Revision `...+1024+iqa+iaa+official-p0-stemids-v1`.
- **MANIQA** keeps its score (the mean of 20 seeded 224 px crops at 1280 px, unchanged) and now also each crop's
  score and box (`outputs.maniqa_crops`, `maniqa_boxes`, `maniqa_size`) and their spread (`maniqa_spread`).
- **Speed on a V100.** A float16 `Conv3d` (Q-ReAlign's vision patch embedding) falls back to PyTorch's generic
  `slow_conv_dilated3d` on this card (torch 2.9, cuDNN 9.10): 24 s per frame. The engine runs that one layer in
  float32: 0.45 s per frame for both tasks, identical scores, 2 GB of GPU memory. The app's ranker worker loads the
  model the same way and will be just as slow on such a card. `flash-linear-attention` (optional) supplies
  Qwen3.5's linear-attention kernels; without it transformers uses slower reference code.

## Face search speed

The people search looks at each frame whole and then in 50 overlapping tiles (3×3, 4×4, 5×5 grids), so faces a few
percent of the frame wide are found: about 0.45 s per frame on one core. Two things keep it affordable, as in the app:
only frames TinyCLIP's person filter flags get it (the rest of the contested frames get the single whole-frame pass,
which is the blink check), and frames run on 8 threads, each with its own landmarker (MediaPipe releases the GIL
while it detects; results are identical to one thread, 6.2× faster on 64 frames). Before the filter was ported every
readable frame was searched: Alaska (5,846 frames) took 35 minutes to find 100 faces.
