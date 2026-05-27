import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import numpy as np

from aiculler.storage import SQLiteFeatureStore
from aiculler.technical_tags import (
    TagPenaltyConfig,
    TechnicalTagScorer,
    compute_technical_metrics,
    load_tag_penalty_configs,
    severity_from_metric,
)


class TechnicalTagTests(unittest.TestCase):
    def test_severity_direction(self):
        high_config = TagPenaltyConfig("blownout", "highlight_clip_ratio", "higher_is_worse", 0.1, 1.0, 20.0)
        low_config = TagPenaltyConfig("outoffocus", "focus_score", "lower_is_worse", 0.5, 1.0, 20.0)

        self.assertGreater(severity_from_metric(0.2, high_config), severity_from_metric(0.01, high_config))
        self.assertGreater(severity_from_metric(0.1, low_config), severity_from_metric(0.9, low_config))

    def test_metrics_detect_bright_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bright.jpg"
            Image.new("RGB", (64, 64), "white").save(path)

            metrics = compute_technical_metrics(path)

            self.assertGreater(metrics.highlight_clip_ratio, 0.9)
            self.assertGreater(metrics.harsh_light_score, 0.5)

    def test_load_tag_penalty_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tags.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["tag", "metric", "direction", "threshold", "weight", "k"])
                writer.writeheader()
                writer.writerow(
                    {
                        "tag": "blownout",
                        "metric": "highlight_clip_ratio",
                        "direction": "higher_is_worse",
                        "threshold": "0.02",
                        "weight": "1.0",
                        "k": "40",
                    }
                )

            configs = load_tag_penalty_configs(path)

            self.assertEqual(configs[0].tag, "blownout")
            self.assertEqual(configs[0].metric, "highlight_clip_ratio")

    def test_score_tags_does_not_compound_on_repeat_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "bright.jpg"
            Image.new("RGB", (64, 64), "white").save(image_path)
            store = SQLiteFeatureStore(tmp_path / "test.sqlite")
            try:
                image_id = store.upsert_image(image_path, preview_path=image_path, status="ready")
                store.save_features(
                    image_id,
                    np.array([1.0, 0.0], dtype=np.float32),
                    technical_score=0.5,
                    final_score=1.0,
                )
                configs = [
                    TagPenaltyConfig("blownout", "highlight_clip_ratio", "higher_is_worse", 0.01, 1.0, 80.0)
                ]
                scorer = TechnicalTagScorer(store, configs, penalty_weight=0.5)

                first = scorer.score(["blownout"])[0]
                second = scorer.score(["blownout"])[0]

                self.assertEqual(first.base_score, 1.0)
                self.assertEqual(second.base_score, 1.0)
                self.assertAlmostEqual(first.adjusted_score, second.adjusted_score)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
