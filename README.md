# AI Culler

Headless, offline-first photo culling engine for terminal use and for the image-triage app.

Given a shoot, `cull` returns **keep / reject / review for every file, with a reason**. It uses
[bracketeering](https://github.com/tylermcm/bracketeering) to decide *what competes with what* (exposure
brackets, bursts, focus stacks, timelapses, near-duplicate scenes), then compares frames only within those
sets. The full design is in bracketeering's `docs/CULLING_ENGINE_SPEC.md`.

## Culling a shoot (milestone 1)

```powershell
python -m pip install -e .
ai-culler --db .\culler.sqlite cull D:\Shoots\Alaska --settings .\cull_settings.example.json --out .\decisions.json
ai-culler --db .\culler.sqlite evaluate D:\Shoots\Alaska --truth .\Alaska-copy.lrcat --decisions .\decisions.json --out .\report.md
```

- `--truth xmp` reads Lightroom ratings and labels from XMP sidecars (or XMP embedded in JPEG/TIFF/DNG).
  Lightroom does not write pick/reject flags to XMP; use a **copy** of the `.lrcat` for flags (opened read-only).
- A capture counts as kept when it is picked, carries an accept label, or is rated ≥ 3★ (`--keep-min-rating`).
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

- **Gates** reject only clear failures: black frames, unusable clipping (`gates`). Bracket frames are never
  gated. Featureless previews go to review by default. If every unit in a scene fails a gate, the least-bad
  goes to review (never a black frame).
- **Review**: groups with confidence below `review_below` or with bracketeering warnings, and unreadable files.
- **Scores** (milestone 1): exposure quality from bracketeering's preview stats and whole-frame Laplacian
  sharpness, each normalised within its comparison set (`normalise`), combined with `weights`.
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
