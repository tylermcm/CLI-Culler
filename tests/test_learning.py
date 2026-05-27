import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiculler.learning import ThreadSafeLearningEngine
from aiculler.storage import SQLiteFeatureStore


class ThreadSafeLearningEngineTests(unittest.TestCase):
    def test_feedback_updates_scores_and_fires_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteFeatureStore(Path(tmp) / "test.sqlite")
            image_ids = []
            for idx in range(3):
                image_id = store.upsert_image(Path(tmp) / f"{idx}.jpg", status="ready")
                store.save_features(image_id, np.array([idx, idx + 1, idx + 2], dtype=np.float32), technical_score=0.9)
                image_ids.append(image_id)

            seen = []
            engine = ThreadSafeLearningEngine(store, on_scores_updated_callback=seen.append, projected_dim=2)
            scores = engine.process_user_feedback(image_ids[0], 1)

            self.assertEqual(set(scores), set(image_ids))
            self.assertEqual(len(seen), 1)
            self.assertEqual(set(seen[0]), set(image_ids))
            store.close()


if __name__ == "__main__":
    unittest.main()

