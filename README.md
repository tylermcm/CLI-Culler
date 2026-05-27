# AI Culler

Headless, offline-first photo culling engine for terminal use and future GUI integration.

The package is intentionally decoupled from UI frameworks. It exposes callback-driven Python classes, uses standard `threading` primitives, stores feature data in SQLite, and can run as a plain CLI with `stdin`/`stdout` prompts.

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
