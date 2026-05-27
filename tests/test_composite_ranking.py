import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiculler.composite_ranking import CompositeRanker, CompositeWeights, resolve_active_weights
from aiculler.storage import SQLiteFeatureStore


class FakeTextEncoder:
    def encode(self, prompt: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class CompositeRankingTests(unittest.TestCase):
    def test_resolve_active_weights_splits_remaining_weight(self):
        weights = resolve_active_weights(
            technical_weight=None,
            prompt_weight=None,
            profile_weight=None,
            preference_weight=None,
            penalty_weight=0.7,
            prompt_active=True,
            profile_active=True,
            preference_active=False,
        )

        self.assertAlmostEqual(weights.technical, 0.35)
        self.assertAlmostEqual(weights.prompt, 0.325)
        self.assertAlmostEqual(weights.profile, 0.325)
        self.assertAlmostEqual(weights.preference, 0.0)
        self.assertAlmostEqual(weights.penalty, 0.7)

    def test_resolve_active_weights_ignores_inactive_explicit_weight(self):
        weights = resolve_active_weights(
            technical_weight=0.5,
            prompt_weight=None,
            profile_weight=None,
            preference_weight=0.5,
            penalty_weight=0.5,
            prompt_active=True,
            profile_active=False,
            preference_active=False,
        )

        self.assertAlmostEqual(weights.prompt, 0.5)
        self.assertAlmostEqual(weights.preference, 0.0)

    def test_composite_ranker_blends_prompt_and_technical_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = SQLiteFeatureStore(tmp_path / "test.sqlite")
            try:
                prompt_match = store.upsert_image(tmp_path / "prompt_match.jpg", status="ready")
                technical_match = store.upsert_image(tmp_path / "technical_match.jpg", status="ready")
                store.save_features(
                    prompt_match,
                    np.array([1.0, 0.0], dtype=np.float32),
                    technical_score=0.2,
                    learned_user_score=99.0,
                )
                store.save_features(
                    technical_match,
                    np.array([0.0, 1.0], dtype=np.float32),
                    technical_score=0.9,
                )

                ranker = CompositeRanker(
                    store,
                    weights=CompositeWeights(
                        technical=0.2,
                        prompt=0.8,
                        profile=0.0,
                        preference=0.0,
                        penalty=0.5,
                    ),
                )
                result = ranker.rank(text_encoder=FakeTextEncoder(), prompt="match prompt")

                self.assertEqual([record.image_id for record in result.records], [prompt_match, technical_match])
                self.assertAlmostEqual(result.records[0].final_score, 0.84)
                self.assertAlmostEqual(result.records[0].learned_user_score, 0.0)
                self.assertEqual(store.get_image(prompt_match)["prompt_text"], "match prompt")
                self.assertAlmostEqual(store.get_image(prompt_match)["tag_base_score"], 0.84)
                self.assertAlmostEqual(store.get_image(prompt_match)["final_score"], 0.84)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
