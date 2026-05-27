import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiculler.ranking import ActiveQuicksortCuller
from aiculler.storage import SQLiteFeatureStore


class ActiveQuicksortCullerTests(unittest.TestCase):
    def test_low_technical_score_routes_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteFeatureStore(Path(tmp) / "test.sqlite")
            high = store.upsert_image(Path(tmp) / "high.jpg", status="ready")
            mid = store.upsert_image(Path(tmp) / "mid.jpg", status="ready")
            low = store.upsert_image(Path(tmp) / "low.jpg", status="ready")
            store.save_features(high, np.array([3.0, 0.0, 0.0], dtype=np.float32), technical_score=0.9, aesthetic_prior=0.9)
            store.save_features(mid, np.array([2.0, 0.0, 0.0], dtype=np.float32), technical_score=0.8, aesthetic_prior=0.8)
            store.save_features(low, np.array([9.0, 0.0, 0.0], dtype=np.float32), technical_score=0.1, aesthetic_prior=0.1)

            culler = ActiveQuicksortCuller(
                store,
                query_callback=lambda *_: True,
                active_threshold=0.0,
                technical_threshold=0.25,
                rng=random.Random(1),
            )
            ranked = culler.sort()

            self.assertEqual(ranked[-1], low)
            self.assertEqual(set(ranked), {high, mid, low})
            store.close()


if __name__ == "__main__":
    unittest.main()

