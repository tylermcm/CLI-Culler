import csv
import json
import tempfile
import unittest
from pathlib import Path

from aiculler.run_logging import RunLogger


class RunLoggerTests(unittest.TestCase):
    def test_writes_events_table_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger("score-text", log_dir=tmp, run_id="unit-test", metadata={"x": 1})
            logger.event("weights", {"prompt_weight": 0.9})
            table_path = logger.table("scores", [{"filename": "a.jpg", "score": 1.0}])
            summary_path = logger.summary({"rows": 1})
            logger.close()

            run_dir = Path(tmp) / "unit-test_score-text"
            events_path = run_dir / "events.jsonl"
            self.assertTrue(events_path.exists())
            self.assertEqual(table_path, run_dir / "scores.csv")
            self.assertEqual(summary_path, run_dir / "summary.json")

            events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["type"], "run_start")
            self.assertEqual(events[-1]["type"], "run_end")

            with table_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["filename"], "a.jpg")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["rows"], 1)


if __name__ == "__main__":
    unittest.main()

