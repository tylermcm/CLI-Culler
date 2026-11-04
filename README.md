# AI Culler

Headless, offline-first photo culling engine for terminal use and for the image-triage app.

Given a shoot, `cull` returns **keep / reject / review for every file, with a reason**. It uses
[bracketeering](https://github.com/tylermcm/bracketeering) to decide *what competes with what* (exposure
brackets, bursts, focus stacks, timelapses, near-duplicate scenes), then compares frames only within those
sets. The full design is in bracketeering's `docs/CULLING_ENGINE_SPEC.md`.

## Culling a shoot

```bash
python -m pip install -e .                 # the cull needs no models
ai-culler --db culler.sqlite cull "/shoots/China 8-26" --out decisions.json [--xmp]
ai-culler --db culler.sqlite cull "/shoots/China 8-26" --with search,people --models ~/src/ImageTriage   # + index
ai-culler --db culler.sqlite evaluate "/shoots/China 8-26" --truth picks --decisions decisions.json --out report.md
ai-culler --db culler.sqlite learn "/shoots/Japan 4-24" "/shoots/Alaska" --truth picks --out preference.joblib
```

**The cull needs no models.** Gates reject the junk (black misfires, blown skies, washed-out frames, crushed
shadows), and bursts, brackets and near-duplicate scenes are culled on exposure, sharpness and clipping: that is
most of the work. Models are optional extras, each asked for by name with `--with` (or `"models"` in the settings):

| `--with` | Model | What it adds | Runs on |
|---|---|---|---|
| `faces` | MediaPipe Face Landmarker | blinks and face-crop sharpness count when ranking | contested frames |
| `maniqa` | MANIQA | technical quality when ranking; the severe-blur gate | contested frames |
| `appeal` | Q-ReAlign | aesthetic appeal when ranking | each contested set's shortlist |
| `search` | TinyCLIP | an embedding per capture, written to the index for semantic search | every readable capture |
| `people` | MediaPipe + AdaFace | faces with identities, clustered into people, written to the index; each person keeps a good frame; implies `faces`, uses TinyCLIP | every frame the app's TinyCLIP person filter flags |

`all` asks for every one. Models come from the image-triage app's install, never downloaded here
(`--models auto`, the installed `image_triage` package, or `--models PATH` to the app; see `docs/MODELS.md`). A
model you asked for that is missing or fails to load is skipped with a warning; `decisions.json` lists the
models actually used (`models`) and what was skipped (`warnings`).

**Quality floor** (`gates.quality_signal`, the levels `qra_reject` / `qra_review` for Q-ReAlign image quality, needs
`--with appeal`; `quality_reject` / `quality_review` for MANIQA, needs `--with maniqa`): every frame about to be kept
is scored, lone frames included. Below the reject level a redundant frame is rejected and its contest moves to the
next-best frame, which is scored in turn; a moment's only or best frame goes to review instead (`quality_unique:
review`); below the review level a frame goes to review. Bracket frames and frames left out of culling are never
floored. Scores can be compared raw or against similar frames (`gates.quality_mode`): `raw`, `shoot` (percentile in
its shoot), `category` (percentile among frames of its TinyCLIP category in a library baseline) or `technique`
(category + EXIF technique: high ISO, slow shutter, long lens, wide aperture, flash). Calibrated on 150 labelled
frames (`docs/RESULTS.md`): Q-ReAlign quality by category, review below the 12th percentile; MANIQA did not work
as a floor. Set it from your own labels:

```bash
M=~/src/ImageTriage
ai-culler --db culler.sqlite baseline /shoots/A /shoots/B ... --qrealign --out library.json --models $M   # no labels
ai-culler --db culler.sqlite sample /shoots/A /shoots/B ... --n 150 --baseline library.json --qrealign --out baseline --models $M
#   open baseline/label.html, mark each frame keep / borderline / reject / hero (keys 1-4), Export labels.csv into baseline/
ai-culler calibrate baseline --baseline library.json --out baseline/calibration.md   # modes compared, suggested levels
```

Step by step, with how to read the comparison and check the result against your past culls: `docs/QUALITY_FLOOR.md`.

**TinyCLIP runs first** whenever anything needs it (`search`, the category floor, `preference_model`,
`refine_scenes`): on every readable capture, bracket and burst frames included, before routing and gates. Every
file then carries its `categories` (portrait, people, landscape, architecture, animals, macro, night, street,
food_product, interior; multi-label) in `decisions.json` and the index; every file carries its EXIF `techniques`.

**The index** (`index.npz` next to `decisions.json`, or `--index PATH`), written when `search` or `people` ran:
TinyCLIP embeddings and faces (box, eyes, AdaFace vector, person number) per capture, with every file of each
capture, for the app's semantic and people search (`aiculler.outputs.load_index`; arrays documented in
`write_index`). Each file in `decisions.json` also lists its `people`.

**Stages**, each on fewer frames than the last: ingest (bracketeering) → [TinyCLIP embeddings; with
`refine_scenes`, scene refinement: split at cosine < `split_cos`, merge adjacent scenes ≥ `merge_cos` within
`merge_window`] → route → cheap scores and gates → [people search on the frames TinyCLIP's person filter flags: tiled face search + AdaFace; faces: the whole-frame pass on the other contested frames] → [MANIQA on *contested* frames only (those in a set
with more candidates than it keeps) → severe-blur gate] → [Q-ReAlign on each contested set's technical shortlist
(`appeal_shortlist`)] → [preference (`preference_model` from `learn`)] → decide → [people: clustering, coverage]
→ `decisions.json` (+ index, XMP). Bracketed stages run only when asked for. `stats` reports how many frames
reached each stage; `timings` the seconds per stage.

**XMP** (`--xmp`): keep = 3★, review = yellow label, reject = 1★ + red label. Ratings, labels or ImageTriage
decisions you set yourself are never overwritten unless `--overwrite-ratings`.

`evaluate --truth` sources (each capture's truth is the union over its files):

| Source | Kept | Metrics |
|---|---|---|
| `picks` (default) | bracketeering merged a copy or edited export of the capture from another folder (`_winners/`, `Bears/`, `Re-edits/`); JPEG sidecars in the shot's own folder don't count | Recall of picks and per-set agreement; precision is not meaningful (frames kept but never exported count as not picked) |
| `csv --truth-file f.csv` | `label` keep in a `filename,label` CSV, as written by `review-session`; unlabelled captures are left out | Full: precision, recall, agreement |
| `triage` | ImageTriage Winner or Keep, Reject = rejected; read-only from its `decisions.sqlite3` (Linux `~/.local/share/Image Triage/Image Triage/`), then its XMP sidecars | Full; unlabelled left out |
| `xmp`, `lrcat --truth-file copy.lrcat` | ratings ≥ 3★, accept labels, Lightroom picks | Full; unrated = not kept |

- Pick/export folders must be scanned to be found: if they sit outside the shoot folder, pass them too
  (`evaluate SHOOT _winners Bears ...`). Exports bracketeering can't match to an original are listed and left out.
- `--unlabelled exclude|not-kept` overrides how unlabelled captures count; `--triage-session` picks an ImageTriage profile.
- Install exiftool for the most accurate grouping (`--backend exiftool`; `auto` uses it when found).
- Scores are cached in `--db` by capture identity and scorer revision; re-running with new settings
  recomputes nothing. Decoded previews are cached next to it (`--cache` to move them).

```python
from aiculler import cull, CullSettings
from aiculler.outputs import write_decisions

result = cull("D:/Shoots/Alaska", CullSettings(bursts_keep=1), cache_db="culler.sqlite",
              on_progress=lambda stage, done, total: print(stage, done, total))
write_decisions(result, "decisions.json")
```

**How it decides**

| Group | Setting | What competes |
|---|---|---|
| exposure bracket | `brackets: keep_set` (default) | The bracket competes as one unit in its scene, judged on its `best_exposed` frame; all frames share the outcome |
| | `pick_one` | Same, then only `best_exposed` is kept |
| | `ignore` | All frames kept, not ranked |
| burst | `bursts_keep: N` (2) | Its frames compete; top N survive and the best represents the burst in its scene. `0` = all kept |
| focus stack, timelapse | `keep_all` (default) / `cull` | Kept; or judged as one unit in the scene |
| single | | Competes in its scene |
| scene | `scene_keep: N` (1) | Top N units kept. A frame alone in its scene has nothing to beat and is kept |

- **Gates** reject only clear failures, never on bracket frames:
  black misfires (always); one connected clipped region ≥ 35% of the frame; ≥ 75% of the top third clipped
  ("sky blown out"); washed out (mean ≥ 0.80 and tonal spread p95−p5 ≤ 0.15); crushed shadows > 60%.
  A clipped region of 5–35% is only a score penalty (up to `clip_penalty`); below 5%, nothing, however much
  clips overall in specks. All thresholds are in `gates`. Featureless previews go to review by default. If every
  unit in a scene fails a gate, the least-bad goes to review (never a black frame). `evaluate` lists the frames
  you kept that a gate rejects, and how close your keeps come to each threshold, for tuning.
- **Review**: groups with confidence below `review_below` or with bracketeering warnings, and unreadable files.
- **Scores**: Q-ReAlign appeal, MANIQA, face-crop sharpness, preference, whole-frame sharpness and exposure,
  each normalised within its comparison set (`normalise`) and combined with `weights` (spec defaults: appeal
  0.4, MANIQA 0.25, face 0.15, preference 0.1). A signal nobody in a set has is left out of that set. Penalties:
  eyes closed on any prominent face (`blink_penalty`), mid-sized clipped regions (`clip_penalty`).
- **Severe blur** (needs MANIQA): bottom 2% MANIQA of the shoot *and* face (else frame) sharpness under 0.35×
  its set's median.
- Sidecars, copies and exports get their shot's decision (`primary: false`).

Output schema `aiculler-decisions/1`: per file `decision`, `group`, `kind`, `role`, `scene`, `capture`,
`primary`, `comparison_set`, `rank_in_set`, `set_size`, `score`, `scores`, `reasons`; plus `summary`,
`settings`, `timings`.

## Python API (for ImageTriage)

```python
from aiculler import CullSettings, ModelRegistry, cull
from aiculler.outputs import write_decisions, write_index

settings = CullSettings.from_json("floor_settings.json")          # or CullSettings(models=["people", "search"])
models = ModelRegistry.from_app_manifest("/path/to/ImageTriage")  # the app's installed, pinned models
result = cull("/shoots/China 8-26", settings, models=models, cache_db="culler.sqlite",
              on_progress=lambda stage, done, total: None)
write_decisions(result, "decisions.json")                         # keep / reject / review + reasons per file
if result.index:
    write_index(result, "index.npz")                              # TinyCLIP embeddings, faces, people
```

No GUI toolkit is imported anywhere in the engine. Install: `pip install -e .` (the cull needs no models), plus
`.[models]` for the model runtime (torch, transformers, mediapipe, timm, einops).
