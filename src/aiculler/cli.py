"""The ai-culler command line: cull, evaluate, and the quality-floor tools (baseline, sample, calibrate), learn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _write_text(path: Path | None, text: str) -> None:
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_cull_settings(args):
    from aiculler.api import CullSettings

    settings = CullSettings.from_json(args.settings) if getattr(args, "settings", None) else CullSettings()
    if getattr(args, "backend", None):
        settings.backend = args.backend
    if getattr(args, "exclude", None):
        settings.exclude = list(settings.exclude) + list(args.exclude)
    if getattr(args, "with_models", None) is not None:
        settings.models = [m.strip() for m in args.with_models.split(",") if m.strip() and m.strip() != "none"]
        if "all" in settings.models:
            from aiculler.api import MODEL_FEATURES

            settings.models = list(MODEL_FEATURES)
    try:
        settings.validate()
    except ValueError as exc:
        raise SystemExit(f"ai-culler: {exc}") from exc
    return settings


def _load_registry(args, keys: set[str] | None = None):
    """The app's models, or None. ``keys``: the models the run will use (problems are reported only for those;
    an empty set loads nothing); None: all of them."""
    from aiculler.models.registry import ModelRegistry, ModelUnavailable

    choice = (args.models or "auto").strip()
    if choice.casefold() == "none" or keys == set():
        return None
    try:
        registry = ModelRegistry.from_app_manifest(None if choice.casefold() == "auto" else choice, device=args.device)
    except ModelUnavailable as exc:
        if choice.casefold() == "auto":
            print(f"models: none ({exc})", file=sys.stderr)
            return None
        raise SystemExit(f"ai-culler: {exc}") from exc
    for key, problem in registry.problems().items():
        if keys is None or key in keys:
            print(f"models: {key} not available: {problem}", file=sys.stderr)
    return registry


def _run_cull(args):
    from aiculler.api import cull

    folders = [Path(f).expanduser() for f in args.folder]
    for folder in folders:
        if not folder.is_dir():
            raise SystemExit(f"ai-culler: shoot folder not found: {folder}")

    settings = _load_cull_settings(args)
    last: dict[str, int] = {}

    def on_progress(stage: str, done: int, total: int) -> None:
        if args.quiet or total <= 0:
            return
        step = max(1, total // 10)
        if done == total or done - last.get(stage, -step) >= step:
            last[stage] = done
            print(f"[{stage}] {done}/{total}", file=sys.stderr)

    registry = _load_registry(args, settings.model_keys())
    try:
        result = cull(
            folders,
            settings,
            models=registry,
            cache_db=args.db,
            cache_dir=args.cache,
            on_progress=on_progress,
            workers=args.workers,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ai-culler: {exc}") from exc
    if not args.quiet:
        if result.models:
            print("models used: " + ", ".join(f"{k} ({v})" for k, v in result.models.items()), file=sys.stderr)
        for w in result.warnings:
            print(f"warning: {w}", file=sys.stderr)
    return result


def command_cull(args) -> int:
    from aiculler.outputs import write_decisions

    result = _run_cull(args)
    path = write_decisions(result, args.out)
    s = result.summary
    print(f"{s['files']} files: {s['keep']} keep, {s['reject']} reject, {s['review']} review -> {path}")
    if result.index:
        from aiculler.outputs import write_index

        index_path = write_index(result, args.index or Path(args.out).with_name("index.npz"))
        parts = []
        if "search" in result.index:
            parts.append(f"{len(result.index['search']['embeddings'])} embeddings")
        if "people" in result.index:
            faces = result.index["people"]["faces"]
            parts.append(f"{len(faces)} faces, {len({f['person'] for f in faces})} people")
        print(f"index: {', '.join(parts)} -> {index_path}")
    if args.xmp:
        from aiculler.outputs import write_xmp

        c = write_xmp(result, overwrite_ratings=args.overwrite_ratings)
        print(f"XMP sidecars: {c['written']} written, {c['unchanged']} unchanged, {c['skipped']} left alone (your own ratings)")
    print("timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in result.timings.items()))
    return 0


def command_evaluate(args) -> int:
    from aiculler.evaluate import evaluate, load_truth, render_report
    from aiculler.outputs import decisions_document, load_decisions, write_decisions

    if args.decisions and args.decisions.exists() and not args.rerun:
        decisions = load_decisions(args.decisions)
    else:
        result = _run_cull(args)
        if args.decisions:
            write_decisions(result, args.decisions)
        decisions = decisions_document(result)
    try:
        truth = load_truth(
            args.truth,
            decisions,
            truth_file=args.truth_file,
            session=args.triage_session,
            unlabelled=None if args.unlabelled == "auto" else args.unlabelled,
            pick_folders=args.pick_folder,
        )
    except ValueError as exc:
        raise SystemExit(f"evaluate: {exc}") from exc
    ev = evaluate(decisions, truth, keep_min_rating=args.keep_min_rating)
    report = render_report(ev, shoot=", ".join(str(f) for f in args.folder), worst=args.worst)
    _write_text(args.out, report)
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(ev.summary, indent=2) + "\n", encoding="utf-8")
    s = ev.summary
    precision = "n/a (picks)" if s["mode"] == "picks" else s["precision"]
    kept_gated = len(s.get("gates_vs_kept", {}).get("gated", []))
    print(
        f"recall {s['recall']}, precision {precision}, keep rate {s['engine_keep_rate']}, review rate {s['review_rate']}, "
        f"burst hit rate {s['sets']['burst']['hit_rate']}, scene hit rate {s['sets']['scene']['hit_rate']}"
        + f", {kept_gated} {'picked' if s['mode'] == 'picks' else 'kept'} frame(s) rejected by a gate"
        + (f"; floor: {len(s['floor']['keeps_lost'])} of your keeps rejected, {len(s['floor']['keeps_reviewed'])} sent to review, "
           f"{s['floor']['others_rejected']} others rejected" if s.get("floor") else "")
        + (f" -> {args.out}" if args.out else ""),
        file=sys.stderr,
    )
    return 0


def command_learn(args) -> int:
    from aiculler.preference import learn_preference

    registry = _load_registry(args)
    if registry is None:
        raise SystemExit("ai-culler learn: TinyCLIP is needed; point --models at the image-triage app")
    try:
        model, summary = learn_preference(
            [Path(f).expanduser() for f in args.folder],
            args.truth,
            registry,
            truth_file=args.truth_file,
            session=args.triage_session,
            cache_db=args.db,
            cache_dir=args.cache,
            backend=args.backend or "auto",
        )
    except ValueError as exc:
        raise SystemExit(f"ai-culler learn: {exc}") from exc
    path = model.save(args.out)
    for s in summary:
        print(f"{s['shoot']}: {s['examples']} labelled captures, {s['kept']} kept ({s['truth']})")
    d = model.diagnostics
    auc = "n/a" if d.get("auc") is None else f"{d['auc']:.3f}"
    print(f"preference model: {model.examples} examples, {model.positives} kept; held-out AUC {auc} ({d.get('method')}) -> {path}")
    print("use it with: \"preference_model\": \"%s\" in the cull settings" % path)
    return 0


def command_sample(args) -> int:
    from aiculler.baseline import make_sample

    folders = [Path(f).expanduser() for f in args.folder]
    for folder in folders:
        if not folder.is_dir():
            raise SystemExit(f"ai-culler: folder not found: {folder}")
    library = None
    if args.baseline:
        from aiculler.baseline import LibraryBaseline

        library = LibraryBaseline.load(args.baseline)
    settings = _load_cull_settings(args)
    registry = _load_registry(args, {"maniqa"} | ({"q_realign"} if args.qrealign else set()) | settings.model_keys())
    try:
        doc = make_sample(folders, args.out, n=args.n, seed=args.seed, models=registry, cache_db=args.db,
                          cache_dir=args.cache, workers=args.workers, library=library, qrealign=args.qrealign,
                          settings=settings, population=args.population)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ai-culler: {exc}") from exc
    for w in doc["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    for folder, k in doc["per_folder"].items():
        pool = "frames the floor would judge" if doc.get("population") == "keepers" else "competing frames"
        print(f"{folder}: {k} of {doc['candidates_per_folder'][folder]} {pool}")
    if doc["per_role"]:
        print("by role: " + ", ".join(f"{r} {k}" for r, k in doc["per_role"].items()))
    if doc["per_category"]:
        print("by category: " + ", ".join(f"{c} {k}" for c, k in sorted(doc["per_category"].items(), key=lambda x: -x[1])))
    print(f"{doc['drawn']} frames -> {args.out / 'label.html'} (open it, label every frame, export labels.csv into {args.out})")
    return 0


def command_calibrate(args) -> int:
    from aiculler.baseline import calibrate, read_labels, render_calibration

    sample = json.loads((args.sample / "sample.json").read_text(encoding="utf-8"))
    labels_path = args.labels or args.sample / "labels.csv"
    try:
        library = None
        if args.baseline:
            from aiculler.baseline import LibraryBaseline

            library = LibraryBaseline.load(args.baseline)
        cal = calibrate(sample, read_labels(labels_path), margin=args.margin, review_share=args.review_share, library=library)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ai-culler calibrate: {exc}") from exc
    _write_text(args.out, render_calibration(cal))
    for name, m in cal["signals"].items():
        h = m["held_out"]
        print(f"{name}: all keeps safe catches {m['all_safe']} of {cal['rejects']} rejects; held out: "
              f"{h['keeps_lost'] + h['borderline_lost']} keeps lost, {h['rejects_caught']} rejects caught", file=sys.stderr)
    for mode, f in cal["fusion"].items():
        print(f"maniqa+qra_iqa/{mode}: either {f['either']['all_safe']}, both {f['both']['all_safe']}, agree {f['agree']['all_safe']} "
              f"rejects with all keeps safe (agree sends {f['agree']['review']['keep']} keeps, {f['agree']['review']['reject']} rejects to review)",
              file=sys.stderr)
    return 0


def command_baseline(args) -> int:
    from aiculler.baseline import LibraryBaseline, build_library

    folders = [Path(f).expanduser() for f in args.folder]
    for folder in folders:
        if not folder.is_dir():
            raise SystemExit(f"ai-culler: folder not found: {folder}")
    registry = _load_registry(args, {"maniqa", "tinyclip"} | ({"q_realign"} if args.qrealign else set()))
    last: dict[str, int] = {}

    def on_progress(stage: str, done: int, total: int) -> None:
        step = max(1, total // 10)
        if total and (done == total or done - last.get(stage, -step) >= step):
            last[stage] = done
            print(f"[{stage}] {done}/{total}", file=sys.stderr)

    try:
        doc = build_library(folders, registry, cache_db=args.db, cache_dir=args.cache, workers=args.workers, progress=on_progress,
                            qrealign=args.qrealign)
    except ValueError as exc:
        raise SystemExit(f"ai-culler baseline: {exc}") from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    lib = LibraryBaseline(doc)
    print(f"{len(doc['records'])} frames from {len(folders)} shoots -> {args.out}")
    print("frames per category: " + ", ".join(f"{k} {v}" for k, v in lib.summary().items() if "+" not in k))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-culler", description="Headless offline photo culling engine")
    parser.add_argument("--db", default="aiculler.sqlite", type=Path, help="Score cache (SQLite)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_cull_args(p) -> None:
        p.add_argument("folder", type=Path, nargs="+", help="Shoot folder(s); include pick/export folders that live outside the shoot")
        p.add_argument("--settings", type=Path, help="CullSettings JSON (defaults if omitted)")
        p.add_argument("--cache", type=Path, help="Preview cache directory (default: next to --db)")
        p.add_argument("--backend", choices=["auto", "exiftool", "native"], help="bracketeering metadata backend")
        p.add_argument("--exclude", action="append", default=[], help="Folder name to skip; repeat for several")
        p.add_argument("--workers", type=int)
        p.add_argument("--quiet", action="store_true", help="No progress output")
        p.add_argument(
            "--models",
            default="auto",
            help="Where the app's models are: 'auto' (the installed image-triage app), a path to the app "
            "(its folder or ai_manifest.py), or 'none'. Only read when --with (or the settings' \"models\") asks for a model",
        )
        p.add_argument(
            "--with",
            dest="with_models",
            metavar="LIST",
            help="Optional models to use, comma-separated (overrides the settings' \"models\"): faces (blinks, face "
            "sharpness), maniqa (quality, blur gate), appeal (Q-ReAlign), search (TinyCLIP embeddings for semantic "
            "search), people (AdaFace identities, people coverage); 'all' or 'none'. Default: none; the cull needs no models",
        )
        p.add_argument("--device", default="auto", help="Model device: auto, cpu or cuda")

    cull_cmd = subparsers.add_parser("cull", help="Cull a shoot: keep / reject / review per file, with reasons")
    add_cull_args(cull_cmd)
    cull_cmd.add_argument("--out", type=Path, default=Path("decisions.json"), help="decisions.json path")
    cull_cmd.add_argument("--index", type=Path, help="Search/people index path (default: index.npz next to --out); written when search or people ran")
    cull_cmd.add_argument("--xmp", action="store_true", help="Also write decisions to XMP sidecars (keep 3*, review yellow, reject 1* red)")
    cull_cmd.add_argument("--overwrite-ratings", action="store_true", help="With --xmp: replace ratings/labels you set yourself")
    cull_cmd.set_defaults(func=command_cull)

    evaluate_cmd = subparsers.add_parser("evaluate", help="Report agreement with your own culls (picks, CSV, ImageTriage, XMP)")
    add_cull_args(evaluate_cmd)
    evaluate_cmd.add_argument(
        "--truth",
        default="picks",
        help="picks (default: copies/exports in other folders are final picks), csv, triage (ImageTriage decisions), "
        "xmp, lrcat; or a .csv / .lrcat / ImageTriage decisions.sqlite3 path",
    )
    evaluate_cmd.add_argument("--truth-file", help="File for --truth csv / lrcat; for triage, a decisions.sqlite3 other than the default")
    evaluate_cmd.add_argument("--pick-folder", action="append", default=[],
                              help="With --truth picks: a folder name holding your final picks (repeat); default: found automatically")
    evaluate_cmd.add_argument("--triage-session", help="ImageTriage session (profile preset); default: the one with most rows for this shoot")
    evaluate_cmd.add_argument(
        "--unlabelled",
        choices=["auto", "not-kept", "exclude"],
        default="auto",
        help="How captures without a label count (auto: excluded for csv/triage, not kept otherwise)",
    )
    evaluate_cmd.add_argument("--decisions", type=Path, help="Existing decisions.json to evaluate (else cull runs, and writes it here if given)")
    evaluate_cmd.add_argument("--rerun", action="store_true", help="Re-run cull even if --decisions exists")
    evaluate_cmd.add_argument("--keep-min-rating", type=int, default=3, help="Stars at or above which a frame counts as kept")
    evaluate_cmd.add_argument("--worst", type=int, default=25, help="Disagreements to list per direction")
    evaluate_cmd.add_argument("--out", type=Path, help="report.md path (default: stdout)")
    evaluate_cmd.add_argument("--metrics-out", type=Path, help="Optional metrics JSON")
    evaluate_cmd.set_defaults(func=command_evaluate)

    sample_cmd = subparsers.add_parser("sample", help="Draw a random sample across shoots for a labelled quality baseline")
    sample_cmd.add_argument("folder", type=Path, nargs="+", help="Shoot folders (each one shoot); equal shares from each")
    sample_cmd.add_argument("--n", type=int, default=100, help="Frames to draw in all (default 100)")
    sample_cmd.add_argument("--seed", type=int, default=1, help="Random seed (the same seed draws the same sample)")
    sample_cmd.add_argument("--out", type=Path, required=True, help="Folder for label.html, thumbs/ and sample.json")
    sample_cmd.add_argument("--cache", type=Path, help="Preview cache directory (default: next to --db)")
    sample_cmd.add_argument("--models", default="auto", help="Where the app's models are (as for cull); MANIQA scores the sample")
    sample_cmd.add_argument("--device", default="auto")
    sample_cmd.add_argument("--workers", type=int)
    sample_cmd.add_argument("--settings", type=Path, help="Your cull settings: the would-be keeps are what this cull keeps (floor off)")
    sample_cmd.add_argument("--population", choices=["keepers", "all"], default="keepers",
                            help="keepers (default): what a quality floor would judge, i.e. would-be keeps and each set's runner-up; "
                            "all: every competing frame")
    sample_cmd.add_argument("--qrealign", action="store_true", help="Also score the sample with Q-ReAlign (quality and aesthetics)")
    sample_cmd.add_argument("--baseline", type=Path, help="Library baseline (ai-culler baseline): spread the draw evenly across categories")
    sample_cmd.set_defaults(func=command_sample)

    calibrate_cmd = subparsers.add_parser("calibrate", help="Suggest quality floor levels from a labelled sample")
    calibrate_cmd.add_argument("sample", type=Path, help="The folder ai-culler sample wrote")
    calibrate_cmd.add_argument("--labels", type=Path, help="labels.csv exported from label.html (default: in the sample folder)")
    calibrate_cmd.add_argument("--margin", type=float, default=0.02, help="quality_reject sits this far below the lowest keep/borderline")
    calibrate_cmd.add_argument("--review-share", type=float, default=0.10, help="At most this share of keeps below quality_review")
    calibrate_cmd.add_argument("--out", type=Path, help="calibration.md path (default: stdout)")
    calibrate_cmd.add_argument("--baseline", type=Path, help="Library baseline: also compare the shoot, category and technique modes")
    calibrate_cmd.set_defaults(func=command_calibrate)

    baseline_cmd = subparsers.add_parser("baseline", help="Build a library baseline: quality scores per category across many shoots (no labels)")
    baseline_cmd.add_argument("folder", type=Path, nargs="+", help="Shoot folders (each one shoot)")
    baseline_cmd.add_argument("--out", type=Path, default=Path("library_baseline.json"))
    baseline_cmd.add_argument("--cache", type=Path, help="Preview cache directory (default: next to --db)")
    baseline_cmd.add_argument("--models", default="auto", help="Where the app's models are (as for cull)")
    baseline_cmd.add_argument("--device", default="auto")
    baseline_cmd.add_argument("--workers", type=int)
    baseline_cmd.add_argument("--qrealign", action="store_true", help="Also score every frame with Q-ReAlign (quality and aesthetics)")
    baseline_cmd.set_defaults(func=command_baseline)

    learn_cmd = subparsers.add_parser("learn", help="Train a preference model (TinyCLIP) from your past culls")
    learn_cmd.add_argument("folder", type=Path, nargs="+", help="Shoot folders with past culls (each one shoot)")
    learn_cmd.add_argument("--truth", default="picks", help="picks, csv, triage, xmp or lrcat (as for evaluate)")
    learn_cmd.add_argument("--truth-file", help="File for --truth csv / lrcat / a triage decisions.sqlite3")
    learn_cmd.add_argument("--triage-session", help="ImageTriage session (profile preset)")
    learn_cmd.add_argument("--out", type=Path, default=Path("preference.joblib"))
    learn_cmd.add_argument("--cache", type=Path, help="Preview cache directory (default: next to --db)")
    learn_cmd.add_argument("--backend", choices=["auto", "exiftool", "native"])
    learn_cmd.add_argument("--models", default="auto", help="Where the app's models are (as for cull)")
    learn_cmd.add_argument("--device", default="auto")
    learn_cmd.set_defaults(func=command_learn)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
