from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


REVIEW_FIELDNAMES = [
    "id",
    "image_id",
    "filename",
    "source_path",
    "shown_rank",
    "label",
    "reject_tags",
    "review_note",
    "prompt",
    "profile",
    "technical_score",
    "prompt_score",
    "normalized_prompt_score",
    "profile_score",
    "normalized_profile_score",
    "learned_user_score",
    "normalized_learned_user_score",
    "pre_penalty_score",
    "tag_penalty",
    "triggered_tags",
    "final_score",
]


COMPARISON_FIELDNAMES = [
    "id",
    "image_id",
    "filename",
    "source_path",
    "label",
    "reject_tags",
    "before_rank",
    "after_rank",
    "rank_delta",
    "before_final_score",
    "after_final_score",
    "score_delta",
    "before_tag_penalty",
    "after_tag_penalty",
    "after_triggered_tags",
]


@dataclass(frozen=True)
class ReviewSessionResult:
    reviewed_count: int
    keep_count: int
    reject_count: int
    skipped_count: int
    output_path: Path


@dataclass(frozen=True)
class ComparisonResult:
    compared_count: int
    improved_count: int
    worsened_count: int
    unchanged_count: int
    output_path: Path


def collect_review_feedback(
    ranking_csv_path: str | Path,
    output_csv_path: str | Path,
    *,
    top: int | None = None,
    start_rank: int = 1,
    append: bool = False,
    skip_existing: bool = False,
    prompt_text: str = "",
    profile_name: str = "",
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> ReviewSessionResult:
    ranking_rows = select_review_rows(load_csv_rows(ranking_csv_path), start_rank=start_rank, top=top)
    output_path = Path(output_csv_path)
    append = append or skip_existing
    existing_keys = load_existing_review_keys(output_path) if skip_existing else set()
    mode = "a" if append and output_path.exists() else "w"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reviewed = keeps = rejects = skipped = 0
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDNAMES)
        if mode == "w" or handle.tell() == 0:
            writer.writeheader()

        for row in ranking_rows:
            key = review_key(row)
            if skip_existing and key in existing_keys:
                skipped += 1
                continue

            label = ask_for_label(row, input_func=input_func, print_func=print_func)
            if label == "quit":
                break
            if label == "skip":
                skipped += 1
                continue

            reject_tags = ""
            if label == "reject":
                reject_tags = normalize_tags(input_func("Reject tags, comma-separated (optional): "))
            note = input_func("Note (optional): ").strip()
            writer.writerow(
                build_review_record(
                    row,
                    label=label,
                    reject_tags=reject_tags,
                    note=note,
                    prompt_text=prompt_text,
                    profile_name=profile_name,
                )
            )
            reviewed += 1
            keeps += int(label == "keep")
            rejects += int(label == "reject")

    return ReviewSessionResult(
        reviewed_count=reviewed,
        keep_count=keeps,
        reject_count=rejects,
        skipped_count=skipped,
        output_path=output_path,
    )


def ask_for_label(row: dict[str, str], *, input_func: Callable[[str], str], print_func: Callable[[str], None]) -> str:
    rank = row.get("rank", "")
    score = row.get("final_score", "")
    filename = row.get("filename") or Path(row.get("source_path", "")).name
    print_func(f"#{rank} final={score} {filename}")
    while True:
        response = input_func("[K]eep / [R]eject / [S]kip / [Q]uit: ").strip().lower()
        if response in {"k", "keep", "1", "y", "yes"}:
            return "keep"
        if response in {"r", "reject", "0", "n", "no", "remove"}:
            return "reject"
        if response in {"s", "skip", ""}:
            return "skip"
        if response in {"q", "quit", "exit"}:
            return "quit"
        print_func("Enter k, r, s, or q.")


def build_review_record(
    row: dict[str, str],
    *,
    label: str,
    reject_tags: str,
    note: str,
    prompt_text: str,
    profile_name: str,
) -> dict[str, str]:
    image_id = row.get("id") or row.get("image_id") or ""
    return {
        "id": image_id,
        "image_id": image_id,
        "filename": row.get("filename") or Path(row.get("source_path", "")).name,
        "source_path": row.get("source_path", ""),
        "shown_rank": row.get("rank", ""),
        "label": label,
        "reject_tags": reject_tags,
        "review_note": note,
        "prompt": prompt_text or row.get("prompt") or row.get("prompt_text") or "",
        "profile": profile_name or row.get("profile") or row.get("profile_name") or "",
        "technical_score": row.get("technical_score", ""),
        "prompt_score": row.get("prompt_score", ""),
        "normalized_prompt_score": row.get("normalized_prompt_score", ""),
        "profile_score": row.get("profile_score", ""),
        "normalized_profile_score": row.get("normalized_profile_score", ""),
        "learned_user_score": row.get("learned_user_score", ""),
        "normalized_learned_user_score": row.get("normalized_learned_user_score", ""),
        "pre_penalty_score": row.get("pre_penalty_score", ""),
        "tag_penalty": row.get("tag_penalty", ""),
        "triggered_tags": row.get("triggered_tags", ""),
        "final_score": row.get("final_score", ""),
    }


def write_comparison_report(
    before_csv_path: str | Path,
    after_csv_path: str | Path,
    output_csv_path: str | Path,
    *,
    feedback_csv_path: str | Path | None = None,
) -> ComparisonResult:
    before_rows = {match_key(row): row for row in load_csv_rows(before_csv_path)}
    after_rows = load_csv_rows(after_csv_path)
    feedback_rows = load_feedback_lookup(feedback_csv_path) if feedback_csv_path else {}
    records: list[dict[str, str | int | float]] = []

    improved = worsened = unchanged = 0
    for after in after_rows:
        key = match_key(after)
        before = before_rows.get(key)
        if before is None:
            continue
        before_rank = parse_int(before.get("rank"))
        after_rank = parse_int(after.get("rank"))
        rank_delta = before_rank - after_rank if before_rank is not None and after_rank is not None else ""
        if isinstance(rank_delta, int) and rank_delta > 0:
            improved += 1
        elif isinstance(rank_delta, int) and rank_delta < 0:
            worsened += 1
        else:
            unchanged += 1
        before_score = parse_float(before.get("final_score"))
        after_score = parse_float(after.get("final_score"))
        feedback = feedback_rows.get(key, {})
        image_id = after.get("id") or after.get("image_id") or before.get("id") or before.get("image_id") or ""
        records.append(
            {
                "id": image_id,
                "image_id": image_id,
                "filename": after.get("filename") or before.get("filename") or Path(after.get("source_path", "")).name,
                "source_path": after.get("source_path") or before.get("source_path") or "",
                "label": feedback.get("label", ""),
                "reject_tags": feedback.get("reject_tags", ""),
                "before_rank": before_rank if before_rank is not None else "",
                "after_rank": after_rank if after_rank is not None else "",
                "rank_delta": rank_delta,
                "before_final_score": before_score if before_score is not None else "",
                "after_final_score": after_score if after_score is not None else "",
                "score_delta": (
                    after_score - before_score
                    if before_score is not None and after_score is not None
                    else ""
                ),
                "before_tag_penalty": before.get("tag_penalty", ""),
                "after_tag_penalty": after.get("tag_penalty", ""),
                "after_triggered_tags": after.get("triggered_tags", ""),
            }
        )

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_path, records, fieldnames=COMPARISON_FIELDNAMES)
    return ComparisonResult(
        compared_count=len(records),
        improved_count=improved,
        worsened_count=worsened,
        unchanged_count=unchanged,
        output_path=output_path,
    )


def load_feedback_lookup(feedback_csv_path: str | Path) -> dict[str, dict[str, str]]:
    return {match_key(row): row for row in load_csv_rows(feedback_csv_path)}


def load_existing_review_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {match_key(row) for row in load_csv_rows(path)}


def select_review_rows(rows: list[dict[str, str]], *, start_rank: int, top: int | None) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        rank = parse_int(row.get("rank"))
        if rank is not None and rank < start_rank:
            continue
        selected.append(row)
        if top is not None and len(selected) >= max(0, top):
            break
    return selected


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} must include CSV headers")
        return [{key: value or "" for key, value in row.items()} for row in reader]


def write_csv_rows(path: Path, rows: Iterable[dict], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def review_key(row: dict[str, str]) -> str:
    return match_key(row)


def match_key(row: dict[str, str]) -> str:
    image_id = (row.get("id") or row.get("image_id") or "").strip()
    if image_id:
        return f"id:{image_id}"
    source_path = (row.get("source_path") or row.get("path") or "").strip()
    if source_path:
        normalized = str(Path(source_path)).replace("/", "\\").lower()
        return f"path:{normalized}"
    return f"filename:{(row.get('filename') or '').strip().lower()}"


def normalize_tags(value: str) -> str:
    tags = []
    for part in value.replace(";", ",").split(","):
        tag = part.strip().lower().lstrip("#")
        if tag and tag not in tags:
            tags.append(tag)
    return ";".join(tags)


def parse_int(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_float(value: str | None) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
