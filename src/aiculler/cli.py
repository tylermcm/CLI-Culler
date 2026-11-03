from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

from aiculler.composite_ranking import CompositeRanker, resolve_active_weights
from aiculler.features import HeadlessFeatureExtractor, IngestionEngine
from aiculler.learning import ThreadSafeLearningEngine
from aiculler.preference_learning import PreferenceLearningScorer
from aiculler.profile_scoring import ProfileScorer, list_profile_names, load_profile_atoms
from aiculler.ranking import ActiveQuicksortCuller
from aiculler.run_logging import RunLogger, normalize_value
from aiculler.session_review import collect_review_feedback, load_csv_rows, write_comparison_report
from aiculler.storage import SQLiteFeatureStore
from aiculler.technical_tags import TechnicalTagScorer, load_tag_penalty_configs
from aiculler.text_scoring import CLIPTextEncoder, TextConditionedScorer


def _event_printer(event) -> None:
    source = event.source_path.name
    if event.status == "error":
        print(f"[error] {source}: {event.message}")
    else:
        print(f"[{event.status}] #{event.image_id} {source}")


def _open_store(args) -> SQLiteFeatureStore:
    return SQLiteFeatureStore(args.db)


def _open_logger(args) -> RunLogger:
    logger = RunLogger(
        args.command,
        log_dir=args.log_dir,
        run_id=args.run_id,
        enabled=not args.no_log,
        metadata={"args": _loggable_args(args)},
    )
    return logger


def _loggable_args(args) -> dict:
    return normalize_value({key: value for key, value in vars(args).items() if key != "func"})


def command_ingest(args) -> int:
    logger = _open_logger(args)
    store = _open_store(args)
    events: list[dict] = []
    try:
        extractor = _build_extractor(args)

        def on_event(event) -> None:
            _event_printer(event)
            record = _ingestion_event_record(event)
            events.append(record)
            logger.event("ingestion_event", record)

        engine = IngestionEngine(
            store,
            args.cache,
            extractor=extractor,
            max_workers=args.workers,
            on_event=on_event,
        )
        logger.event("ingest_start", {"folder": args.folder, "workers": args.workers})
        ids = engine.ingest(args.folder, recursive=not args.no_recursive)
        logger.table("ingestion_events", events)
        logger.summary({"image_count": len(ids), "events": len(events)})
        print(f"Ingested {len(ids)} image(s).")
        return 0
    finally:
        store.close()
        logger.close()


def command_benchmark(args) -> int:
    logger = _open_logger(args)
    store = _open_store(args)
    records: list[dict] = []
    try:
        extractor = _build_extractor(args)

        def on_event(event) -> None:
            record = {
                "image_id": event.image_id,
                "source_path": str(event.source_path),
                "preview_path": "" if event.preview_path is None else str(event.preview_path),
                "status": event.status,
                "preview_seconds": f"{event.preview_seconds:.6f}",
                "feature_seconds": f"{event.feature_seconds:.6f}",
                "total_seconds": f"{event.total_seconds:.6f}",
                "message": event.message,
            }
            records.append(record)
            logger.event("benchmark_event", record)
            if not args.quiet:
                print(
                    f"[{event.status}] #{event.image_id} {event.source_path.name} "
                    f"preview={event.preview_seconds:.3f}s feature={event.feature_seconds:.3f}s "
                    f"total={event.total_seconds:.3f}s",
                    file=sys.stderr,
                )

        engine = IngestionEngine(
            store,
            args.cache,
            extractor=extractor,
            max_workers=args.workers,
            on_event=on_event,
        )
        paths = engine.scan(args.folder, recursive=not args.no_recursive)
        if args.limit is not None:
            paths = paths[: max(0, args.limit)]
        logger.event(
            "benchmark_start",
            {"folder": args.folder, "file_count": len(paths), "workers": args.workers, "features": not args.no_features},
        )
        started_at = time.perf_counter()
        engine.ingest_paths(paths)
        elapsed = time.perf_counter() - started_at

        if args.out is not None:
            _write_csv(args.out, records)
        logger.table("benchmark_events", records)

        terminal_status = "previewed" if args.no_features else "ready"
        completed = sum(1 for record in records if record["status"] == terminal_status)
        errors = sum(1 for record in records if record["status"] == "error")
        rate = completed / elapsed if elapsed > 0 else 0.0
        logger.summary(
            {
                "file_count": len(paths),
                "completed": completed,
                "errors": errors,
                "elapsed_seconds": elapsed,
                "images_per_second": rate,
                "terminal_status": terminal_status,
            }
        )
        print(
            f"Benchmarked {completed}/{len(paths)} image(s) in {elapsed:.2f}s "
            f"({rate:.2f} images/sec, errors={errors})."
        )
        if args.out is not None:
            print(f"Wrote timing CSV to {args.out}.")
        return 0 if errors == 0 else 1
    finally:
        store.close()
        logger.close()


def command_sort(args) -> int:
    logger = _open_logger(args)
    store = _open_store(args)
    try:
        culler = ActiveQuicksortCuller(
            store,
            active_threshold=args.active_threshold,
            technical_threshold=args.technical_threshold,
        )
        ranked = culler.sort()
        rows = {int(row["id"]): row for row in store.list_images(require_embedding=True)}
        ranking_records = []
        for index, image_id in enumerate(ranked, start=1):
            row = rows[image_id]
            ranking_records.append(
                {
                    "rank": index,
                    "id": image_id,
                    "filename": Path(row["source_path"]).name,
                    "source_path": row["source_path"],
                    "technical_score": row["technical_score"],
                    "final_score": row["final_score"],
                }
            )
            print(f"{index:04d} #{image_id} score={row['final_score']:.4f} {row['source_path']}")
        logger.event(
            "sort_equation",
            {
                "active_threshold": args.active_threshold,
                "technical_threshold": args.technical_threshold,
                "equation": "ranked_order_score = len(ranked) - rank_index",
            },
        )
        logger.table("sort_ranking", ranking_records)
        logger.summary({"ranked_count": len(ranking_records)})
        return 0
    finally:
        store.close()
        logger.close()


def command_rank(args) -> int:
    logger = _open_logger(args)
    store = _open_store(args)
    try:
        prompt_active = bool(args.prompt)
        profile_active = bool(args.profile)
        preference_active = bool(args.feedback or args.use_existing_preference)
        weights = resolve_active_weights(
            technical_weight=args.technical_weight,
            prompt_weight=args.prompt_weight,
            profile_weight=args.profile_weight,
            preference_weight=args.preference_weight,
            penalty_weight=args.penalty_weight,
            prompt_active=prompt_active,
            profile_active=profile_active,
            preference_active=preference_active,
        )

        text_encoder = None
        if prompt_active or profile_active:
            text_encoder = CLIPTextEncoder(args.text_model, args.tokenizer)
        profile_atoms = load_profile_atoms(args.profiles) if profile_active else []
        tag_configs = load_tag_penalty_configs(args.tag_config) if args.avoid else []

        ranker = CompositeRanker(store, weights=weights)
        result = ranker.rank(
            text_encoder=text_encoder,
            prompt=args.prompt,
            profile_name=args.profile,
            profile_atoms=profile_atoms,
            feedback_csv=args.feedback,
            tag_configs=tag_configs,
            avoid_tags=args.avoid,
            use_existing_preference=args.use_existing_preference,
            preference_projected_dim=args.projected_dim,
            preference_alpha=args.alpha,
            record_feedback=not args.no_record_feedback,
            diagnostic_top_n=args.diagnostic_top_n,
        )
        output_records = [_composite_record_to_csv(record) for record in result.records]
        if args.out is not None:
            _write_csv(args.out, output_records)
            print(f"Wrote composite ranking CSV to {args.out}.", file=sys.stderr)
        if args.diagnostics_out is not None and result.preference_diagnostics is not None:
            _write_csv(args.diagnostics_out, _diagnostic_records_to_csv(result.preference_diagnostics.records))
            print(f"Wrote preference diagnostics CSV to {args.diagnostics_out}.", file=sys.stderr)

        logger.event(
            "rank_equation",
            {
                "equation": (
                    "final_score = technical_weight*technical + prompt_weight*normalized_prompt "
                    "+ profile_weight*normalized_profile + preference_weight*normalized_preference "
                    "- penalty_weight*tag_penalty"
                ),
                "weights": result.weights.__dict__,
                "prompt": args.prompt or "",
                "profile": args.profile or "",
                "feedback": args.feedback,
                "avoid_tags": args.avoid,
            },
        )
        logger.table("rank_components", output_records)
        if profile_active:
            logger.table("profile_atoms", [atom.__dict__ for atom in profile_atoms if atom.profile == args.profile])
        if args.avoid:
            logger.table("tag_configs", [config.__dict__ for config in tag_configs if config.tag in set(args.avoid)])
        if result.preference_diagnostics is not None:
            logger.table("preference_diagnostics", _diagnostic_records_to_csv(result.preference_diagnostics.records))
        logger.summary(
            {
                "ranked_count": len(output_records),
                "weights": result.weights.__dict__,
                "preference_diagnostics": _diagnostic_summary(result.preference_diagnostics),
            }
        )

        if result.preference_diagnostics is not None:
            _print_preference_summary(result.preference_diagnostics)
        for record in output_records[: args.top]:
            print(
                f"{record['rank']:04d} #{record['id']} final={record['final_score']:.4f} "
                f"pre={record['pre_penalty_score']:.4f} penalty={record['tag_penalty']:.4f} "
                f"{record['source_path']}"
            )
        return 0
    finally:
        store.close()
        logger.close()


def _composite_record_to_csv(record) -> dict:
    return {
        "rank": record.rank,
        "id": record.image_id,
        "filename": record.filename,
        "source_path": record.source_path,
        "technical_score": record.technical_score,
        "prompt_score": record.prompt_score,
        "normalized_prompt_score": record.normalized_prompt_score,
        "profile_score": record.profile_score,
        "normalized_profile_score": record.normalized_profile_score,
        "learned_user_score": record.learned_user_score,
        "normalized_learned_user_score": record.normalized_learned_user_score,
        "pre_penalty_score": record.pre_penalty_score,
        "tag_penalty": record.tag_penalty,
        "triggered_tags": record.triggered_tags,
        "final_score": record.final_score,
    }


def _diagnostic_summary(diagnostics) -> dict | None:
    if diagnostics is None:
        return None
    return {
        "feedback_count": diagnostics.feedback_count,
        "keep_count": diagnostics.keep_count,
        "reject_count": diagnostics.reject_count,
        "train_accuracy": diagnostics.train_accuracy,
        "leave_one_out_accuracy": diagnostics.leave_one_out_accuracy,
        "top_n": diagnostics.top_n,
        "top_n_keep_count": diagnostics.top_n_keep_count,
        "top_n_reject_count": diagnostics.top_n_reject_count,
    }


def command_review_session(args) -> int:
    logger = _open_logger(args)
    try:
        result = collect_review_feedback(
            args.ranking,
            args.out,
            top=args.top,
            start_rank=args.start_rank,
            append=args.append,
            skip_existing=args.skip_existing,
            prompt_text=args.prompt or "",
            profile_name=args.profile or "",
        )
        records = load_csv_rows(args.out) if args.out.exists() else []
        logger.event(
            "review_session",
            {
                "ranking": args.ranking,
                "out": args.out,
                "top": args.top,
                "start_rank": args.start_rank,
                "append": args.append,
                "skip_existing": args.skip_existing,
                "reviewed": result.reviewed_count,
                "kept": result.keep_count,
                "rejected": result.reject_count,
                "skipped": result.skipped_count,
            },
        )
        logger.table("review_feedback", records)
        logger.summary(
            {
                "reviewed": result.reviewed_count,
                "kept": result.keep_count,
                "rejected": result.reject_count,
                "skipped": result.skipped_count,
                "out": result.output_path,
            }
        )
        print(
            f"Recorded {result.reviewed_count} decision(s): "
            f"{result.keep_count} keep, {result.reject_count} reject, {result.skipped_count} skipped."
        )
        print(f"Wrote feedback CSV to {result.output_path}.")
        return 0
    finally:
        logger.close()


def command_compare_rankings(args) -> int:
    logger = _open_logger(args)
    try:
        result = write_comparison_report(
            args.before,
            args.after,
            args.out,
            feedback_csv_path=args.feedback,
        )
        records = load_csv_rows(args.out)
        logger.event(
            "rank_comparison",
            {
                "before": args.before,
                "after": args.after,
                "feedback": args.feedback,
                "out": args.out,
                "compared": result.compared_count,
                "improved": result.improved_count,
                "worsened": result.worsened_count,
                "unchanged": result.unchanged_count,
            },
        )
        logger.table("rank_comparison", records)
        logger.summary(
            {
                "compared": result.compared_count,
                "improved": result.improved_count,
                "worsened": result.worsened_count,
                "unchanged": result.unchanged_count,
                "out": result.output_path,
            }
        )
        print(
            f"Compared {result.compared_count} image(s): "
            f"{result.improved_count} improved, {result.worsened_count} worsened, "
            f"{result.unchanged_count} unchanged."
        )
        print(f"Wrote comparison CSV to {result.output_path}.")
        return 0
    finally:
        logger.close()


def command_feedback(args) -> int:
    logger = _open_logger(args)
    store = _open_store(args)
    try:
        label = _parse_label(args.label)

        def on_scores(scores: dict[int, float]) -> None:
            print(f"Updated {len(scores)} score(s).")

        engine = ThreadSafeLearningEngine(store, on_scores_updated_callback=on_scores, projected_dim=args.projected_dim)
        scores = engine.process_user_feedback(args.image_id, label)
        score_rows = []
        for image_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[: args.top]:
            row = store.get_image(image_id)
            if row is not None:
                score_rows.append({"id": image_id, "score": score, "source_path": row["source_path"]})
                print(f"#{image_id} score={score:.4f} {row['source_path']}")
        logger.event(
            "feedback_update",
            {"image_id": args.image_id, "label": label, "projected_dim": args.projected_dim, "score_count": len(scores)},
        )
        logger.table("feedback_top_scores", score_rows)
        return 0
    finally:
        store.close()
        logger.close()


def command_score_text(args) -> int:
    logger = _open_logger(args)
    store = _open_store(args)
    try:
        encoder = CLIPTextEncoder(args.text_model, args.tokenizer)
        scorer = TextConditionedScorer(
            store,
            encoder,
            technical_weight=args.technical_weight,
            prompt_weight=args.prompt_weight,
            normalize_prompt=args.normalize_prompt,
        )
        records = scorer.score_prompt(args.prompt)
        output_records = [
            {
                "rank": index,
                "id": record.image_id,
                "filename": Path(record.source_path).name,
                "source_path": record.source_path,
                "technical_score": record.technical_score,
                "prompt_score": record.prompt_score,
                "normalized_prompt_score": record.normalized_prompt_score,
                "final_score": record.final_score,
            }
            for index, record in enumerate(records, start=1)
        ]
        if args.out is not None:
            _write_csv(args.out, output_records)
            print(f"Wrote prompt ranking CSV to {args.out}.", file=sys.stderr)
        logger.event(
            "score_text_equation",
            {
                "equation": "final_score = technical_weight * technical_score + prompt_weight * normalized(CLIP_image_text_similarity)",
                "prompt": args.prompt,
                "technical_weight": args.technical_weight,
                "prompt_weight": args.prompt_weight,
                "normalize_prompt": args.normalize_prompt,
                "text_model": args.text_model,
                "tokenizer": args.tokenizer,
            },
        )
        logger.table("score_text_ranking", output_records)
        logger.summary({"ranked_count": len(output_records), "prompt": args.prompt})
        for record in output_records[: args.top]:
            print(
                f"{record['rank']:04d} #{record['id']} "
                f"final={record['final_score']:.4f} prompt={record['prompt_score']:.4f} "
                f"tech={record['technical_score']:.4f} {record['source_path']}"
            )
        return 0
    finally:
        store.close()
        logger.close()


def command_score_profile(args) -> int:
    logger = _open_logger(args)
    atoms = load_profile_atoms(args.profiles)
    if args.list_profiles:
        logger.table("available_profiles", [{"profile": name} for name in list_profile_names(atoms)])
        for name in list_profile_names(atoms):
            print(name)
        logger.close()
        return 0
    if not args.profile:
        logger.close(status="error")
        raise SystemExit("--profile is required unless --list-profiles is set")

    store = _open_store(args)
    try:
        encoder = CLIPTextEncoder(args.text_model, args.tokenizer)
        scorer = ProfileScorer(
            store,
            encoder,
            technical_weight=args.technical_weight,
            profile_weight=args.profile_weight,
        )
        records = scorer.score_profile(args.profile, atoms)
        output_records = [
            {
                "rank": index,
                "id": record.image_id,
                "filename": record.filename,
                "source_path": record.source_path,
                "technical_score": record.technical_score,
                "profile_score": record.profile_score,
                "normalized_profile_score": record.normalized_profile_score,
                "final_score": record.final_score,
            }
            for index, record in enumerate(records, start=1)
        ]
        if args.out is not None:
            _write_csv(args.out, output_records)
            print(f"Wrote profile ranking CSV to {args.out}.", file=sys.stderr)
        selected_atoms = [atom for atom in atoms if atom.profile == args.profile]
        logger.event(
            "score_profile_equation",
            {
                "equation": "final_score = technical_weight * technical_score + profile_weight * normalized(weighted_positive_negative_prompt_similarity)",
                "profile": args.profile,
                "technical_weight": args.technical_weight,
                "profile_weight": args.profile_weight,
                "profiles": args.profiles,
            },
        )
        logger.table("profile_atoms", [atom.__dict__ for atom in selected_atoms])
        logger.table("score_profile_ranking", output_records)
        logger.summary({"ranked_count": len(output_records), "profile": args.profile})
        for record in output_records[: args.top]:
            print(
                f"{record['rank']:04d} #{record['id']} "
                f"final={record['final_score']:.4f} profile={record['profile_score']:.4f} "
                f"tech={record['technical_score']:.4f} {record['source_path']}"
            )
        return 0
    finally:
        store.close()
        logger.close()


def command_score_tags(args) -> int:
    logger = _open_logger(args)
    configs = load_tag_penalty_configs(args.tag_config)
    tags = args.avoid
    if not tags:
        tags = sorted({config.tag for config in configs})
    store = _open_store(args)
    try:
        scorer = TechnicalTagScorer(
            store,
            configs,
            penalty_weight=args.penalty_weight,
            base_column=args.base_column,
        )
        records = scorer.score(tags)
        output_records = [
            {
                "rank": index,
                "id": record.image_id,
                "filename": record.filename,
                "source_path": record.source_path,
                "base_score": record.base_score,
                "tag_penalty": record.tag_penalty,
                "adjusted_score": record.adjusted_score,
                "triggered_tags": record.triggered_tags,
                "highlight_clip_ratio": record.metrics.highlight_clip_ratio,
                "harsh_light_score": record.metrics.harsh_light_score,
                "shadow_clip_ratio": record.metrics.shadow_clip_ratio,
                "contrast_score": record.metrics.contrast_score,
                "focus_score": record.metrics.focus_score,
                "motion_blur_score": record.metrics.motion_blur_score,
                "noise_score": record.metrics.noise_score,
            }
            for index, record in enumerate(records, start=1)
        ]
        if args.out is not None:
            _write_csv(args.out, output_records)
            print(f"Wrote tag-adjusted ranking CSV to {args.out}.", file=sys.stderr)
        selected_configs = [config for config in configs if config.tag in set(tags)]
        logger.event(
            "score_tags_equation",
            {
                "equation": "adjusted_score = base_score - penalty_weight * sum(weight * sigmoid(k * metric_margin))",
                "avoid_tags": tags,
                "penalty_weight": args.penalty_weight,
                "base_column": args.base_column,
                "tag_config": args.tag_config,
            },
        )
        logger.table("tag_configs", [config.__dict__ for config in selected_configs])
        logger.table("score_tags_ranking", output_records)
        logger.summary({"ranked_count": len(output_records), "avoid_tags": tags})
        for record in output_records[: args.top]:
            print(
                f"{record['rank']:04d} #{record['id']} "
                f"adjusted={record['adjusted_score']:.4f} base={record['base_score']:.4f} "
                f"penalty={record['tag_penalty']:.4f} tags={record['triggered_tags'] or '-'} "
                f"{record['source_path']}"
            )
        return 0
    finally:
        store.close()
        logger.close()


def command_learn_feedback(args) -> int:
    logger = _open_logger(args)
    store = _open_store(args)
    try:
        scorer = PreferenceLearningScorer(
            store,
            projected_dim=args.projected_dim,
            technical_weight=args.technical_weight,
            prompt_weight=args.prompt_weight,
            preference_weight=args.preference_weight,
            alpha=args.alpha,
        )
        result = scorer.learn_from_csv(
            args.feedback,
            record_feedback=not args.no_record_feedback,
            top_n=args.diagnostic_top_n,
        )
        output_records = [
            {
                "rank": index,
                "id": record.image_id,
                "filename": record.filename,
                "source_path": record.source_path,
                "technical_score": record.technical_score,
                "prompt_score": record.prompt_score,
                "learned_user_score": record.learned_user_score,
                "normalized_learned_user_score": record.normalized_learned_user_score,
                "final_score": record.final_score,
                "feedback_label": record.feedback_label,
            }
            for index, record in enumerate(result.ranking, start=1)
        ]
        if args.out is not None:
            _write_csv(args.out, output_records)
            print(f"Wrote learned ranking CSV to {args.out}.", file=sys.stderr)
        if args.diagnostics_out is not None:
            _write_csv(args.diagnostics_out, _diagnostic_records_to_csv(result.diagnostics.records))
            print(f"Wrote preference diagnostics CSV to {args.diagnostics_out}.", file=sys.stderr)
        logger.event(
            "learn_feedback_equation",
            {
                "equation": "final_score = technical_weight * technical_score + prompt_weight * normalized_prompt_score + preference_weight * normalized_learned_user_score",
                "model": "SGDClassifier(loss=log_loss, penalty=l2) on StandardScaler(IncrementalPCA(CLIP_embedding))",
                "feedback": args.feedback,
                "technical_weight": args.technical_weight,
                "prompt_weight": args.prompt_weight,
                "preference_weight": args.preference_weight,
                "alpha": args.alpha,
                "projected_dim": args.projected_dim,
            },
        )
        logger.table("learned_ranking", output_records)
        logger.table("preference_diagnostics", _diagnostic_records_to_csv(result.diagnostics.records))
        logger.summary(
            {
                "feedback_count": result.diagnostics.feedback_count,
                "keep_count": result.diagnostics.keep_count,
                "reject_count": result.diagnostics.reject_count,
                "train_accuracy": result.diagnostics.train_accuracy,
                "leave_one_out_accuracy": result.diagnostics.leave_one_out_accuracy,
                "top_n": result.diagnostics.top_n,
                "top_n_keep_count": result.diagnostics.top_n_keep_count,
                "top_n_reject_count": result.diagnostics.top_n_reject_count,
            }
        )
        _print_preference_summary(result.diagnostics)
        for record in output_records[: args.top]:
            print(
                f"{record['rank']:04d} #{record['id']} "
                f"final={record['final_score']:.4f} learned={record['learned_user_score']:.4f} "
                f"label={record['feedback_label'] or '-'} {record['source_path']}"
            )
        return 0
    finally:
        store.close()
        logger.close()


def _diagnostic_records_to_csv(records) -> list[dict]:
    return [
        {
            "id": record.image_id,
            "filename": record.filename,
            "source_path": record.source_path,
            "label": record.label,
            "train_score": record.train_score,
            "predicted_label": record.predicted_label,
            "correct": record.correct,
            "leave_one_out_score": record.leave_one_out_score,
            "leave_one_out_label": record.leave_one_out_label,
            "leave_one_out_correct": record.leave_one_out_correct,
        }
        for record in records
    ]


def _print_preference_summary(diagnostics) -> None:
    loo = "n/a" if diagnostics.leave_one_out_accuracy is None else f"{diagnostics.leave_one_out_accuracy:.3f}"
    print(
        "Preference diagnostics: "
        f"feedback={diagnostics.feedback_count} "
        f"keep={diagnostics.keep_count} "
        f"reject={diagnostics.reject_count} "
        f"train_acc={diagnostics.train_accuracy:.3f} "
        f"loo_acc={loo} "
        f"top_{diagnostics.top_n}_keeps={diagnostics.top_n_keep_count} "
        f"top_{diagnostics.top_n}_rejects={diagnostics.top_n_reject_count}"
    )


def _build_extractor(args):
    if args.no_features:
        return None
    if not args.clip:
        raise SystemExit("--clip is required unless --no-features is set")
    if args.topiq is None:
        print("[info] No TOPIQ ONNX supplied; using offline heuristic technical scoring.")
    elif args.topiq.suffix.lower() != ".onnx":
        print(f"[info] {args.topiq.name} is not ONNX; using offline heuristic technical scoring.")
    return HeadlessFeatureExtractor(args.clip, args.topiq)


def command_list(args) -> int:
    logger = _open_logger(args)
    store = _open_store(args)
    try:
        rows = store.list_images(require_embedding=args.ready_only)
        for row in rows:
            print(
                f"#{row['id']} status={row['status']} "
                f"tech={_fmt(row['technical_score'])} final={_fmt(row['final_score'])} "
                f"{row['source_path']}"
            )
        logger.summary({"listed_count": len(rows), "ready_only": args.ready_only})
        return 0
    finally:
        store.close()
        logger.close()


def command_export(args) -> int:
    logger = _open_logger(args)
    store = _open_store(args)
    try:
        rows = _ranked_rows(store, require_scored=args.scored_only)
        records = [_row_to_record(index, row) for index, row in enumerate(rows, start=1)]
        if args.format == "json":
            payload = json.dumps(records, indent=2)
            _write_text(args.out, payload + "\n")
        else:
            _write_csv(args.out, records)
        target = str(args.out) if args.out else "stdout"
        print(f"Exported {len(records)} row(s) to {target}.", file=sys.stderr)
        logger.event("export", {"format": args.format, "out": args.out, "rows": len(records), "scored_only": args.scored_only})
        logger.summary({"exported_rows": len(records), "target": target})
        return 0
    finally:
        store.close()
        logger.close()


def command_stage(args) -> int:
    logger = _open_logger(args)
    store = _open_store(args)
    stage_records: list[dict] = []
    try:
        rows = _ranked_rows(store, require_scored=not args.include_unscored)
        if not rows:
            print("No ranked images found. Run sort first, or pass --include-unscored.", file=sys.stderr)
            return 1
        keep_count = _resolve_keep_count(rows, args)
        keep_dir = Path(args.keep_dir)
        reject_dir = Path(args.reject_dir)
        keep_dir.mkdir(parents=True, exist_ok=True)
        reject_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        skipped = 0
        for index, row in enumerate(rows, start=1):
            source = Path(row["source_path"])
            if not source.exists():
                skipped += 1
                print(f"[missing] #{row['id']} {source}", file=sys.stderr)
                continue
            target_dir = keep_dir if index <= keep_count else reject_dir
            target = target_dir / _stage_name(index, row, source)
            shutil.copy2(source, target)
            copied += 1
            stage_records.append(
                {
                    "rank": index,
                    "id": int(row["id"]),
                    "action": "keep" if index <= keep_count else "reject",
                    "source": source,
                    "target": target,
                    "final_score": row["final_score"],
                }
            )
            print(f"[{'keep' if index <= keep_count else 'reject'}] {source} -> {target}")

        print(f"Staged {copied} file(s): {keep_count} keep target(s), {max(0, len(rows) - keep_count)} reject target(s).")
        if skipped:
            print(f"Skipped {skipped} missing source file(s).", file=sys.stderr)
        logger.table("stage_records", stage_records)
        logger.summary({"copied": copied, "skipped": skipped, "keep_count": keep_count, "total_rows": len(rows)})
        return 0 if copied else 1
    finally:
        store.close()
        logger.close()


def _ingestion_event_record(event) -> dict:
    return {
        "image_id": event.image_id,
        "source_path": event.source_path,
        "preview_path": event.preview_path,
        "status": event.status,
        "message": event.message,
        "preview_seconds": event.preview_seconds,
        "feature_seconds": event.feature_seconds,
        "total_seconds": event.total_seconds,
    }


def _ranked_rows(store: SQLiteFeatureStore, *, require_scored: bool) -> list:
    rows = store.list_images(require_embedding=True)
    if require_scored:
        rows = [row for row in rows if row["final_score"] is not None]
    return rows


def _row_to_record(rank: int, row) -> dict:
    return {
        "rank": rank,
        "id": int(row["id"]),
        "filename": Path(row["source_path"]).name,
        "source_path": row["source_path"],
        "preview_path": row["preview_path"],
        "status": row["status"],
        "width": row["width"],
        "height": row["height"],
        "technical_score": row["technical_score"],
        "aesthetic_prior": row["aesthetic_prior"],
        "pointwise_score": row["pointwise_score"],
        "prompt_score": row["prompt_score"],
        "prompt_text": row["prompt_text"],
        "learned_user_score": row["learned_user_score"],
        "profile_score": row["profile_score"],
        "profile_name": row["profile_name"],
        "tag_base_score": row["tag_base_score"],
        "tag_penalty": row["tag_penalty"],
        "tag_flags": row["tag_flags"],
        "final_score": row["final_score"],
        "error": row["error"],
    }


def _write_text(path: Path | None, text: str) -> None:
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path | None, records: list[dict]) -> None:
    if not records:
        text = ""
        if path is None:
            print(text, end="")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="")
        return

    fieldnames = list(records[0].keys())
    if path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _resolve_keep_count(rows: list, args) -> int:
    total = len(rows)
    if args.score_threshold is not None:
        return sum(1 for row in rows if row["final_score"] is not None and float(row["final_score"]) >= args.score_threshold)
    if args.keep_count is not None:
        return max(0, min(total, args.keep_count))
    keep_percent = args.keep_percent if args.keep_percent is not None else 50.0
    return max(0, min(total, round(total * (keep_percent / 100.0))))


def _stage_name(rank: int, row, source: Path) -> str:
    return f"{rank:04d}_id{int(row['id']):04d}_{source.name}"


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _parse_label(label: str) -> int:
    normalized = label.strip().lower()
    if normalized in {"k", "keep", "1", "true", "yes", "y"}:
        return 1
    if normalized in {"r", "reject", "0", "false", "no", "n"}:
        return 0
    raise SystemExit("label must be keep/k/1 or reject/r/0")


def _load_cull_settings(args):
    from aiculler.api import CullSettings

    settings = CullSettings.from_json(args.settings) if getattr(args, "settings", None) else CullSettings()
    if getattr(args, "backend", None):
        settings.backend = args.backend
    if getattr(args, "exclude", None):
        settings.exclude = list(settings.exclude) + list(args.exclude)
    settings.validate()
    return settings


def _run_cull(args, logger: RunLogger | None = None):
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

    try:
        result = cull(
            folders,
            settings,
            cache_db=args.db,
            cache_dir=args.cache,
            on_progress=on_progress,
            workers=args.workers,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ai-culler: {exc}") from exc
    if logger is not None:
        logger.event("cull_summary", {"summary": result.summary, "timings": result.timings, "stats": result.stats})
    return result


def command_cull(args) -> int:
    from aiculler.outputs import write_decisions

    logger = _open_logger(args)
    result = _run_cull(args, logger)
    path = write_decisions(result, args.out)
    s = result.summary
    print(f"{s['files']} files: {s['keep']} keep, {s['reject']} reject, {s['review']} review -> {path}")
    print("timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in result.timings.items()))
    logger.summary({"out": str(path), "summary": s})
    logger.close()
    return 0


def command_evaluate(args) -> int:
    from aiculler.evaluate import evaluate, load_truth, render_report
    from aiculler.outputs import decisions_document, load_decisions, write_decisions

    logger = _open_logger(args)
    if args.decisions and args.decisions.exists() and not args.rerun:
        decisions = load_decisions(args.decisions)
    else:
        result = _run_cull(args, logger)
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
        + (f" -> {args.out}" if args.out else ""),
        file=sys.stderr,
    )
    logger.summary({"summary": s})
    logger.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-culler", description="Headless offline photo culling engine")
    parser.add_argument("--db", default="aiculler.sqlite", type=Path, help="SQLite database path")
    parser.add_argument("--log-dir", default=Path("logs"), type=Path, help="Structured run log directory")
    parser.add_argument("--run-id", help="Optional stable run id for the log folder")
    parser.add_argument("--no-log", action="store_true", help="Disable structured run logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Scan a folder and extract previews/features")
    ingest.add_argument("folder", type=Path)
    ingest.add_argument("--cache", default=".aiculler_cache", type=Path)
    ingest.add_argument("--clip", type=Path, help="CLIP ONNX model path")
    ingest.add_argument("--topiq", type=Path, help="TOPIQ ONNX model path; non-ONNX files fall back to heuristic scoring")
    ingest.add_argument("--workers", type=int, default=4)
    ingest.add_argument("--no-recursive", action="store_true")
    ingest.add_argument("--no-features", action="store_true", help="Only extract/cache previews")
    ingest.set_defaults(func=command_ingest)

    benchmark = subparsers.add_parser("benchmark", help="Measure ingest/extraction throughput and write timing CSV")
    benchmark.add_argument("folder", type=Path)
    benchmark.add_argument("--cache", default=".aiculler_cache", type=Path)
    benchmark.add_argument("--clip", type=Path, help="CLIP ONNX model path")
    benchmark.add_argument("--topiq", type=Path, help="TOPIQ ONNX model path; non-ONNX files fall back to heuristic scoring")
    benchmark.add_argument("--workers", type=int, default=4)
    benchmark.add_argument("--limit", type=int, help="Maximum number of discovered files to benchmark")
    benchmark.add_argument("--out", type=Path, help="Timing CSV path")
    benchmark.add_argument("--quiet", action="store_true", help="Only print final summary")
    benchmark.add_argument("--no-recursive", action="store_true")
    benchmark.add_argument("--no-features", action="store_true", help="Only benchmark preview extraction")
    benchmark.set_defaults(func=command_benchmark)

    sort = subparsers.add_parser("sort", help="Run active culling sort")
    sort.add_argument("--active-threshold", type=float, default=0.10)
    sort.add_argument("--technical-threshold", type=float, default=0.25)
    sort.set_defaults(func=command_sort)

    rank = subparsers.add_parser("rank", help="Run the full composable ranking stack in one audited pass")
    rank.add_argument("--prompt", help="Optional freeform CLIP text prompt")
    rank.add_argument("--profile", help="Optional named profile from profiles CSV")
    rank.add_argument("--profiles", default=Path("profiles.csv"), type=Path)
    rank.add_argument("--feedback", type=Path, help="Optional keep/reject feedback CSV")
    rank.add_argument("--avoid", action="append", default=[], help="Technical tag to penalize; repeat for multiple")
    rank.add_argument("--tag-config", default=Path("tag_penalties.csv"), type=Path)
    rank.add_argument(
        "--text-model",
        default=Path("models/Clip/clip-vit-large-patch14/onnx/text_model_uint8.onnx"),
        type=Path,
    )
    rank.add_argument(
        "--tokenizer",
        default=Path("models/Clip/clip-vit-large-patch14/tokenizer.json"),
        type=Path,
    )
    rank.add_argument("--technical-weight", type=float)
    rank.add_argument("--prompt-weight", type=float)
    rank.add_argument("--profile-weight", type=float)
    rank.add_argument("--preference-weight", type=float)
    rank.add_argument("--penalty-weight", type=float, default=0.50)
    rank.add_argument("--projected-dim", type=int, default=64)
    rank.add_argument("--alpha", type=float, default=0.0001)
    rank.add_argument("--use-existing-preference", action="store_true")
    rank.add_argument("--diagnostics-out", type=Path)
    rank.add_argument("--diagnostic-top-n", type=int, default=10)
    rank.add_argument("--no-record-feedback", action="store_true")
    rank.add_argument("--top", type=int, default=20)
    rank.add_argument("--out", type=Path, help="Composite ranking CSV output")
    rank.set_defaults(func=command_rank)

    review_session = subparsers.add_parser("review-session", help="Interactively record keep/reject feedback from a ranking CSV")
    review_session.add_argument("--ranking", required=True, type=Path, help="Ranking CSV to review")
    review_session.add_argument("--out", required=True, type=Path, help="Feedback CSV to write")
    review_session.add_argument("--top", type=int, help="Only review this many rows")
    review_session.add_argument("--start-rank", type=int, default=1, help="First rank to show")
    review_session.add_argument("--append", action="store_true", help="Append decisions to an existing feedback CSV")
    review_session.add_argument("--skip-existing", action="store_true", help="Skip rows already present in the output CSV; implies append")
    review_session.add_argument("--prompt", help="Prompt text to stamp into the feedback audit rows")
    review_session.add_argument("--profile", help="Profile name to stamp into the feedback audit rows")
    review_session.set_defaults(func=command_review_session)

    compare_rankings = subparsers.add_parser("compare-rankings", help="Compare before/after ranking CSVs")
    compare_rankings.add_argument("--before", required=True, type=Path, help="Original ranking CSV")
    compare_rankings.add_argument("--after", required=True, type=Path, help="New ranking CSV")
    compare_rankings.add_argument("--out", required=True, type=Path, help="Comparison CSV to write")
    compare_rankings.add_argument("--feedback", type=Path, help="Optional feedback CSV to include labels and reject tags")
    compare_rankings.set_defaults(func=command_compare_rankings)

    score_text = subparsers.add_parser("score-text", help="Rank images using a CLIP text prompt plus technical score")
    score_text.add_argument("--prompt", required=True, help="Prompt describing what should be prioritized")
    score_text.add_argument(
        "--text-model",
        default=Path("models/Clip/clip-vit-large-patch14/onnx/text_model_uint8.onnx"),
        type=Path,
        help="CLIP text ONNX model path",
    )
    score_text.add_argument(
        "--tokenizer",
        default=Path("models/Clip/clip-vit-large-patch14/tokenizer.json"),
        type=Path,
        help="CLIP tokenizer.json path",
    )
    score_text.add_argument("--technical-weight", type=float, default=0.45)
    score_text.add_argument("--prompt-weight", type=float, default=0.55)
    score_text.add_argument("--normalize-prompt", choices=["minmax", "none"], default="minmax")
    score_text.add_argument("--top", type=int, default=20)
    score_text.add_argument("--out", type=Path, help="Optional prompt ranking CSV output")
    score_text.set_defaults(func=command_score_text)

    score_profile = subparsers.add_parser("score-profile", help="Rank images using a named weighted prompt profile")
    score_profile.add_argument("--profile", help="Profile name from profiles CSV")
    score_profile.add_argument("--profiles", default=Path("profiles.csv"), type=Path, help="Profiles CSV path")
    score_profile.add_argument("--list-profiles", action="store_true")
    score_profile.add_argument(
        "--text-model",
        default=Path("models/Clip/clip-vit-large-patch14/onnx/text_model_uint8.onnx"),
        type=Path,
        help="CLIP text ONNX model path",
    )
    score_profile.add_argument(
        "--tokenizer",
        default=Path("models/Clip/clip-vit-large-patch14/tokenizer.json"),
        type=Path,
        help="CLIP tokenizer.json path",
    )
    score_profile.add_argument("--technical-weight", type=float, default=0.35)
    score_profile.add_argument("--profile-weight", type=float, default=0.65)
    score_profile.add_argument("--top", type=int, default=20)
    score_profile.add_argument("--out", type=Path, help="Optional profile ranking CSV output")
    score_profile.set_defaults(func=command_score_profile)

    score_tags = subparsers.add_parser("score-tags", help="Apply measurable technical reject-tag penalties")
    score_tags.add_argument("--avoid", action="append", default=[], help="Tag to penalize; repeat for multiple tags")
    score_tags.add_argument("--tag-config", default=Path("tag_penalties.csv"), type=Path)
    score_tags.add_argument("--penalty-weight", type=float, default=0.50)
    score_tags.add_argument("--base-column", choices=["final_score", "technical_score"], default="final_score")
    score_tags.add_argument("--top", type=int, default=20)
    score_tags.add_argument("--out", type=Path, help="Optional tag-adjusted ranking CSV output")
    score_tags.set_defaults(func=command_score_tags)

    learn_feedback = subparsers.add_parser("learn-feedback", help="Train user preference score from keep/reject CSV")
    learn_feedback.add_argument("--feedback", required=True, type=Path, help="CSV with label plus id, filename, or source_path")
    learn_feedback.add_argument("--projected-dim", type=int, default=64)
    learn_feedback.add_argument("--technical-weight", type=float, default=0.25)
    learn_feedback.add_argument("--prompt-weight", type=float, default=0.25)
    learn_feedback.add_argument("--preference-weight", type=float, default=0.50)
    learn_feedback.add_argument("--alpha", type=float, default=0.0001)
    learn_feedback.add_argument("--top", type=int, default=20)
    learn_feedback.add_argument("--out", type=Path, help="Optional learned ranking CSV output")
    learn_feedback.add_argument("--diagnostics-out", type=Path, help="Optional per-feedback diagnostics CSV output")
    learn_feedback.add_argument("--diagnostic-top-n", type=int, default=10)
    learn_feedback.add_argument("--no-record-feedback", action="store_true", help="Do not append labels to the feedback audit table")
    learn_feedback.set_defaults(func=command_learn_feedback)

    feedback = subparsers.add_parser("feedback", help="Apply one keep/reject feedback label")
    feedback.add_argument("image_id", type=int)
    feedback.add_argument("label")
    feedback.add_argument("--projected-dim", type=int, default=64)
    feedback.add_argument("--top", type=int, default=20)
    feedback.set_defaults(func=command_feedback)

    list_cmd = subparsers.add_parser("list", help="List database entries")
    list_cmd.add_argument("--ready-only", action="store_true")
    list_cmd.set_defaults(func=command_list)

    export = subparsers.add_parser("export", help="Export ranked results to CSV or JSON")
    export.add_argument("--format", choices=["csv", "json"], default="csv")
    export.add_argument("--out", type=Path, help="Output path; defaults to stdout")
    export.add_argument("--scored-only", action="store_true", help="Only export rows with a final score")
    export.set_defaults(func=command_export)

    stage = subparsers.add_parser("stage", help="Copy ranked originals into keep/reject folders")
    stage.add_argument("--keep-dir", required=True, type=Path)
    stage.add_argument("--reject-dir", required=True, type=Path)
    policy = stage.add_mutually_exclusive_group()
    policy.add_argument("--keep-count", type=int, help="Number of top-ranked files to copy to keep-dir")
    policy.add_argument("--keep-percent", type=float, help="Percent of top-ranked files to copy to keep-dir")
    policy.add_argument("--score-threshold", type=float, help="Keep files with final_score >= threshold")
    stage.add_argument("--include-unscored", action="store_true", help="Allow staging before sort has set final scores")
    stage.set_defaults(func=command_stage)

    def add_cull_args(p) -> None:
        p.add_argument("folder", type=Path, nargs="+", help="Shoot folder(s); include pick/export folders that live outside the shoot")
        p.add_argument("--settings", type=Path, help="CullSettings JSON (defaults if omitted)")
        p.add_argument("--cache", type=Path, help="Preview cache directory (default: next to --db)")
        p.add_argument("--backend", choices=["auto", "exiftool", "native"], help="bracketeering metadata backend")
        p.add_argument("--exclude", action="append", default=[], help="Folder name to skip; repeat for several")
        p.add_argument("--workers", type=int)
        p.add_argument("--quiet", action="store_true", help="No progress output")

    cull_cmd = subparsers.add_parser("cull", help="Cull a shoot: keep / reject / review per file, with reasons")
    add_cull_args(cull_cmd)
    cull_cmd.add_argument("--out", type=Path, default=Path("decisions.json"), help="decisions.json path")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
