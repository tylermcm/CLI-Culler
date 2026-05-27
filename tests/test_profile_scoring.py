import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aiculler.profile_scoring import ProfileScorer, list_profile_names, load_profile_atoms
from aiculler.storage import SQLiteFeatureStore


class FakeTextEncoder:
    def encode(self, prompt: str) -> np.ndarray:
        if "positive" in prompt:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


class ProfileScoringTests(unittest.TestCase):
    def test_load_profile_atoms_and_list_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.csv"
            self._write_profiles(path)

            atoms = load_profile_atoms(path)

            self.assertEqual(list_profile_names(atoms), ["test_profile"])
            self.assertEqual(len(atoms), 2)

    def test_profile_scoring_uses_positive_and_negative_atoms(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            profiles_path = tmp_path / "profiles.csv"
            self._write_profiles(profiles_path)
            store = SQLiteFeatureStore(tmp_path / "test.sqlite")
            try:
                good = store.upsert_image(tmp_path / "good.jpg", status="ready")
                bad = store.upsert_image(tmp_path / "bad.jpg", status="ready")
                store.save_features(good, np.array([1.0, 0.0], dtype=np.float32), technical_score=0.5)
                store.save_features(bad, np.array([0.0, 1.0], dtype=np.float32), technical_score=0.5)

                scorer = ProfileScorer(store, FakeTextEncoder(), technical_weight=0.0, profile_weight=1.0)
                records = scorer.score_profile("test_profile", load_profile_atoms(profiles_path))

                self.assertEqual(records[0].filename, "good.jpg")
                self.assertEqual(store.get_image(records[0].image_id)["profile_name"], "test_profile")
            finally:
                store.close()

    @staticmethod
    def _write_profiles(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["profile", "type", "weight", "prompt"])
            writer.writeheader()
            writer.writerow({"profile": "test_profile", "type": "positive", "weight": "1.0", "prompt": "positive target"})
            writer.writerow({"profile": "test_profile", "type": "negative", "weight": "1.0", "prompt": "negative target"})


if __name__ == "__main__":
    unittest.main()

