import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiculler.storage import SQLiteFeatureStore
from aiculler.text_scoring import TextConditionedScorer, cosine_similarity, normalize_scores


class FakeTextEncoder:
    def encode(self, prompt: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class TextScoringTests(unittest.TestCase):
    def test_cosine_similarity(self):
        self.assertAlmostEqual(
            cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])),
            1.0,
        )
        self.assertAlmostEqual(
            cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            0.0,
        )

    def test_minmax_normalization(self):
        self.assertEqual(normalize_scores({1: 2.0, 2: 4.0}, mode="minmax"), {1: 0.0, 2: 1.0})
        self.assertEqual(normalize_scores({1: 2.0, 2: 2.0}, mode="minmax"), {1: 0.5, 2: 0.5})

    def test_text_conditioned_scorer_updates_prompt_and_final_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteFeatureStore(Path(tmp) / "test.sqlite")
            try:
                left = store.upsert_image(Path(tmp) / "left.jpg", status="ready")
                right = store.upsert_image(Path(tmp) / "right.jpg", status="ready")
                store.save_features(left, np.array([1.0, 0.0, 0.0], dtype=np.float32), technical_score=0.5)
                store.save_features(right, np.array([0.0, 1.0, 0.0], dtype=np.float32), technical_score=0.9)

                scorer = TextConditionedScorer(
                    store,
                    FakeTextEncoder(),
                    technical_weight=0.0,
                    prompt_weight=1.0,
                )
                records = scorer.score_prompt("match left")

                self.assertEqual([record.image_id for record in records], [left, right])
                self.assertEqual(store.get_image(left)["prompt_text"], "match left")
                self.assertGreater(store.get_image(left)["final_score"], store.get_image(right)["final_score"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()

