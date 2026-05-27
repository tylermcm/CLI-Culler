import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiculler.preference_learning import PreferenceLearningScorer, parse_feedback_label, resolve_feedback_examples
from aiculler.storage import SQLiteFeatureStore


class PreferenceLearningTests(unittest.TestCase):
    def test_parse_feedback_label(self):
        self.assertEqual(parse_feedback_label("keep"), 1)
        self.assertEqual(parse_feedback_label("R"), 0)
        with self.assertRaises(ValueError):
            parse_feedback_label("maybe")

    def test_resolves_feedback_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = self._seed_store(tmp_path)
            feedback_path = tmp_path / "feedback.csv"
            self._write_feedback(feedback_path, [("good.jpg", "keep"), ("bad.jpg", "reject")])

            examples = resolve_feedback_examples(store, feedback_path)

            self.assertEqual([example.filename for example in examples], ["good.jpg", "bad.jpg"])
            self.assertEqual([example.label for example in examples], [1, 0])
            store.close()

    def test_learned_scores_promote_keep_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = self._seed_store(tmp_path)
            feedback_path = tmp_path / "feedback.csv"
            self._write_feedback(feedback_path, [("good.jpg", "keep"), ("bad.jpg", "reject")])

            scorer = PreferenceLearningScorer(
                store,
                projected_dim=2,
                technical_weight=0.0,
                prompt_weight=0.0,
                preference_weight=1.0,
            )
            result = scorer.learn_from_csv(feedback_path, record_feedback=False)

            self.assertEqual(result.ranking[0].filename, "good.jpg")
            self.assertGreater(store.get_image(result.ranking[0].image_id)["learned_user_score"], 0.0)
            self.assertEqual(result.diagnostics.feedback_count, 2)
            self.assertEqual(result.diagnostics.keep_count, 1)
            self.assertEqual(result.diagnostics.reject_count, 1)
            self.assertEqual(result.diagnostics.train_accuracy, 1.0)
            store.close()

    def test_leave_one_out_diagnostics_are_reported_when_possible(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = self._seed_store(tmp_path)
            extra_id = store.upsert_image(tmp_path / "bad2.jpg", status="ready")
            store.save_features(extra_id, np.array([-1.2, 0.0, 0.0], dtype=np.float32), technical_score=0.5, prompt_score=0.0)
            feedback_path = tmp_path / "feedback.csv"
            self._write_feedback(
                feedback_path,
                [("good.jpg", "keep"), ("bad.jpg", "reject"), ("bad2.jpg", "reject")],
            )

            scorer = PreferenceLearningScorer(store, projected_dim=2)
            result = scorer.learn_from_csv(feedback_path, record_feedback=False)

            self.assertEqual(len(result.diagnostics.records), 3)
            self.assertIsNotNone(result.diagnostics.leave_one_out_accuracy)
            store.close()

    @staticmethod
    def _seed_store(tmp_path: Path) -> SQLiteFeatureStore:
        store = SQLiteFeatureStore(tmp_path / "test.sqlite")
        samples = {
            "good.jpg": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "bad.jpg": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            "neutral.jpg": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        for filename, embedding in samples.items():
            image_id = store.upsert_image(tmp_path / filename, status="ready")
            store.save_features(image_id, embedding, technical_score=0.5, prompt_score=0.0)
        return store

    @staticmethod
    def _write_feedback(path: Path, rows: list[tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "label"])
            writer.writeheader()
            for filename, label in rows:
                writer.writerow({"filename": filename, "label": label})


if __name__ == "__main__":
    unittest.main()
