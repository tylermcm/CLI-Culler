import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiculler.storage import SQLiteFeatureStore


class SQLiteFeatureStoreTests(unittest.TestCase):
    def test_round_trips_embedding_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteFeatureStore(Path(tmp) / "test.sqlite")
            image_id = store.upsert_image(Path(tmp) / "image.jpg", status="ready")
            embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            store.save_features(image_id, embedding, technical_score=0.75, final_score=1.25)

            np.testing.assert_allclose(store.get_embedding(image_id), embedding)
            row = store.get_image(image_id)
            self.assertEqual(row["technical_score"], 0.75)
            self.assertEqual(row["final_score"], 1.25)
            store.close()


if __name__ == "__main__":
    unittest.main()

