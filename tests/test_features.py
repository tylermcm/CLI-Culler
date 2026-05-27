import unittest

from PIL import Image

from aiculler.features import HeuristicTechnicalScorer


class HeuristicTechnicalScorerTests(unittest.TestCase):
    def test_score_is_normalized(self):
        scorer = HeuristicTechnicalScorer()
        score = scorer.score(Image.new("RGB", (64, 64), "white"))

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
