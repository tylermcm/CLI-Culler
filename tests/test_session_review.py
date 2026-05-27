import csv
import tempfile
import unittest
from pathlib import Path

from aiculler.session_review import collect_review_feedback, load_csv_rows, normalize_tags, write_comparison_report


class SessionReviewTests(unittest.TestCase):
    def test_collect_review_feedback_writes_training_ready_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ranking_path = tmp_path / "ranking.csv"
            feedback_path = tmp_path / "session_feedback.csv"
            self._write_rows(
                ranking_path,
                [
                    {
                        "rank": "1",
                        "id": "10",
                        "filename": "keep.jpg",
                        "source_path": str(tmp_path / "keep.jpg"),
                        "technical_score": "0.8",
                        "prompt_score": "0.7",
                        "final_score": "0.9",
                    },
                    {
                        "rank": "2",
                        "id": "11",
                        "filename": "reject.jpg",
                        "source_path": str(tmp_path / "reject.jpg"),
                        "technical_score": "0.3",
                        "prompt_score": "0.2",
                        "final_score": "0.1",
                    },
                ],
            )
            responses = iter(["k", "good match", "r", "#blownout, harshlight", "too harsh"])

            result = collect_review_feedback(
                ranking_path,
                feedback_path,
                input_func=lambda prompt: next(responses),
                print_func=lambda text: None,
                prompt_text="sun prompt",
                profile_name="travel",
            )

            rows = load_csv_rows(feedback_path)
            self.assertEqual(result.reviewed_count, 2)
            self.assertEqual(result.keep_count, 1)
            self.assertEqual(result.reject_count, 1)
            self.assertEqual(rows[0]["id"], "10")
            self.assertEqual(rows[0]["label"], "keep")
            self.assertEqual(rows[0]["prompt"], "sun prompt")
            self.assertEqual(rows[1]["reject_tags"], "blownout;harshlight")

    def test_write_comparison_report_tracks_rank_movement(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            before = tmp_path / "before.csv"
            after = tmp_path / "after.csv"
            feedback = tmp_path / "feedback.csv"
            report = tmp_path / "comparison.csv"
            self._write_rows(
                before,
                [
                    {"rank": "1", "id": "10", "filename": "a.jpg", "source_path": "a.jpg", "final_score": "0.9"},
                    {"rank": "2", "id": "11", "filename": "b.jpg", "source_path": "b.jpg", "final_score": "0.1"},
                ],
            )
            self._write_rows(
                after,
                [
                    {"rank": "1", "id": "11", "filename": "b.jpg", "source_path": "b.jpg", "final_score": "0.8"},
                    {"rank": "2", "id": "10", "filename": "a.jpg", "source_path": "a.jpg", "final_score": "0.2"},
                ],
            )
            self._write_rows(
                feedback,
                [
                    {"id": "11", "filename": "b.jpg", "source_path": "b.jpg", "label": "keep", "reject_tags": ""},
                    {"id": "10", "filename": "a.jpg", "source_path": "a.jpg", "label": "reject", "reject_tags": "blownout"},
                ],
            )

            result = write_comparison_report(before, after, report, feedback_csv_path=feedback)

            rows = load_csv_rows(report)
            self.assertEqual(result.compared_count, 2)
            self.assertEqual(result.improved_count, 1)
            self.assertEqual(result.worsened_count, 1)
            self.assertEqual(rows[0]["id"], "11")
            self.assertEqual(rows[0]["rank_delta"], "1")
            self.assertEqual(rows[0]["label"], "keep")

    def test_normalize_tags(self):
        self.assertEqual(normalize_tags("#BlownOut, harshlight; blownout"), "blownout;harshlight")

    @staticmethod
    def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
