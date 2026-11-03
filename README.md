# AI Culler

Headless, offline-first photo culling engine for terminal use and for the image-triage app.

Given a shoot, `cull` returns **keep / reject / review for every file, with a reason**. It uses
[bracketeering](https://github.com/tylermcm/bracketeering) to decide *what competes with what* (exposure
brackets, bursts, focus stacks, timelapses, near-duplicate scenes), then compares frames only within those
sets. The full design is in bracketeering's `docs/CULLING_ENGINE_SPEC.md`.

## Culling a shoot

```bash
python -m pip install -e '.[models]'      # or plain -e . to run without models
ai-culler --db culler.sqlite cull "/shoots/China 8-26" --models ~/src/ImageTriage --out decisions.json [--xmp]
ai-culler --db culler.sqlite evaluate "/shoots/China 8-26" --truth picks --decisions decisions.json --out report.md
ai-culler --db culler.sqlite learn "/shoots/Japan 4-24" "/shoots/Alaska" --truth picks --out preference.joblib
```

**Models** come from the image-triage app's install, never downloaded here (see `docs/MODELS.md`):
`--models auto` (the installed `image_triage` package), a path to the app, or `none`. Each model the app has
installed is used; a missing one is skipped with a warning, and `decisions.json` lists the models actually used
(`models`) and what was skipped (`warnings`). Without models the engine runs on exposure, sharpness and clipping.

**Stages**, each on fewer frames than the last: ingest (bracketeering) → TinyCLIP embeddings and scene
refinement (split at cosine < `split_cos`, merge adjacent scenes ≥ `merge_cos` within `merge_window`) → route →
cheap scores and gates → faces (MediaPipe: blinks, face-crop sharpness; AdaFace identities) → MANIQA on
*contested* frames only (those in a set with more candidates than it keeps) → severe-blur gate → Q-ReAlign on
each contested set's technical shortlist only (`appeal_shortlist`) → preference (`preference_model` from
`learn`) → decide → people coverage (each person keeps a frame) → `decisions.json` (+ XMP). `stats` reports how
many frames reached each stage; `timings` the seconds per stage.

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

## Legacy ranking commands

The commands below predate the culling engine (global ranking with CLIP-L/TOPIQ). They still work and will
be retired or moved to `legacy/` as `cull` covers their use.

## Install

```powershell
python -m pip install -e .
```

## Local Assets

Model weights, downloaded RAW samples, SQLite databases, cache folders, logs, and generated ranking CSVs are intentionally not tracked. The CLI defaults expect local models at:

```text
models/Clip/clip-vit-large-patch14/onnx/vision_model_uint8.onnx
models/Clip/clip-vit-large-patch14/onnx/text_model_uint8.onnx
models/Clip/clip-vit-large-patch14/tokenizer.json
models/TOPIQ/topiq_nr.onnx
```

## CLI

```powershell
ai-culler --db .\culler.sqlite ingest C:\photos --cache .\.aiculler_cache --clip .\clip.onnx --topiq .\topiq.onnx
ai-culler --db .\culler.sqlite sort
ai-culler --db .\culler.sqlite export --format csv --out .\ranking.csv
ai-culler --db .\culler.sqlite stage --keep-dir .\kept --reject-dir .\rejected --keep-percent 50
ai-culler --db .\culler.sqlite score-text --prompt "warm candid moments, sharp eyes" --out .\prompt_ranking.csv
ai-culler --db .\culler.sqlite learn-feedback --feedback .\feedback.csv --out .\learned_ranking.csv
ai-culler --db .\culler.sqlite rank --prompt "bright airy street scene" --avoid blownout --out .\ranking.csv
ai-culler review-session --ranking .\ranking.csv --out .\session_feedback.csv --top 50
ai-culler --db .\culler.sqlite rank --feedback .\session_feedback.csv --out .\ranking_after_feedback.csv
ai-culler compare-rankings --before .\ranking.csv --after .\ranking_after_feedback.csv --feedback .\session_feedback.csv --out .\rank_delta.csv
```

Feedback CSVs can use `id`, `filename`, or `source_path` plus a `label` column:

```csv
filename,label
IMG_0001.CR3,keep
IMG_0002.CR3,reject
```

CLI runs write structured logs by default under `logs/`. Use `--run-id` to make a log folder easier to find, or `--no-log` for throwaway runs:

```powershell
ai-culler --run-id portrait_test --db .\culler.sqlite score-profile --profile portrait_bokeh --out .\profile.csv
```

`review-session` walks a ranking CSV in terminal mode and writes a training-ready feedback CSV. Use `k` for keep, `r` for reject, `s` to skip, and `q` to stop. Rejected images can include semicolon-separated reject tags such as `blownout;motionblur`.

For a lightweight dry run that only scans RAW files and extracts embedded JPEG previews:

```powershell
ai-culler --db .\culler.sqlite ingest C:\photos --cache .\.aiculler_cache --no-features
```

## Python API

```python
from aiculler import SQLiteFeatureStore, ActiveQuicksortCuller

store = SQLiteFeatureStore("culler.sqlite")

def on_query(left_id, pivot_id, context):
    return 1  # keep left over pivot

culler = ActiveQuicksortCuller(store, query_callback=on_query)
ranking = culler.sort()
```

No PySide, PyQt, Tkinter, or GUI event loop is imported anywhere in the engine.
