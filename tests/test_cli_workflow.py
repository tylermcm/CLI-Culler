import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from aiculler.cli import main
from aiculler.storage import SQLiteFeatureStore


class CliWorkflowTests(unittest.TestCase):
    def test_export_writes_ranked_csv_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "test.sqlite"
            self._seed_store(db_path, tmp_path)
            csv_path = tmp_path / "ranking.csv"
            json_path = tmp_path / "ranking.json"

            with self._quiet():
                self.assertEqual(main(["--db", str(db_path), "export", "--out", str(csv_path), "--scored-only"]), 0)
                self.assertEqual(
                    main(
                        [
                            "--db",
                            str(db_path),
                            "export",
                            "--format",
                            "json",
                            "--out",
                            str(json_path),
                            "--scored-only",
                        ]
                    ),
                    0,
                )

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with json_path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)

            self.assertEqual([row["rank"] for row in rows], ["1", "2", "3"])
            self.assertEqual(rows[0]["final_score"], "30.0")
            self.assertEqual(records[0]["final_score"], 30.0)

    def test_stage_copies_ranked_originals_into_keep_and_reject_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "test.sqlite"
            self._seed_store(db_path, tmp_path)
            keep_dir = tmp_path / "kept"
            reject_dir = tmp_path / "rejected"

            with self._quiet():
                result = main(
                    [
                        "--db",
                        str(db_path),
                        "stage",
                        "--keep-dir",
                        str(keep_dir),
                        "--reject-dir",
                        str(reject_dir),
                        "--keep-count",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            kept = sorted(path.name for path in keep_dir.iterdir())
            rejected = sorted(path.name for path in reject_dir.iterdir())
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(rejected), 2)
            self.assertIn("photo_2.nef", kept[0])

    def test_benchmark_writes_timing_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            folder = tmp_path / "photos"
            folder.mkdir()
            for idx in range(2):
                Image.new("RGB", (64, 64), (idx * 40, 100, 140)).save(folder / f"photo_{idx}.jpg")
            csv_path = tmp_path / "timings.csv"
            db_path = tmp_path / "bench.sqlite"

            with self._quiet():
                result = main(
                    [
                        "--db",
                        str(db_path),
                        "benchmark",
                        str(folder),
                        "--cache",
                        str(tmp_path / "cache"),
                        "--no-features",
                        "--out",
                        str(csv_path),
                    ]
                )

            self.assertEqual(result, 0)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["status"] for row in rows}, {"previewed"})

    @staticmethod
    def _seed_store(db_path: Path, tmp_path: Path) -> None:
        store = SQLiteFeatureStore(db_path)
        try:
            for idx, score in enumerate([10.0, 20.0, 30.0]):
                source = tmp_path / f"photo_{idx}.nef"
                source.write_bytes(f"raw-{idx}".encode("ascii"))
                image_id = store.upsert_image(source, status="ready")
                store.save_features(
                    image_id,
                    np.array([idx, idx + 1, idx + 2], dtype=np.float32),
                    technical_score=0.9,
                    final_score=score,
                )
        finally:
            store.close()

    @staticmethod
    @contextlib.contextmanager
    def _quiet():
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield


if __name__ == "__main__":
    unittest.main()
