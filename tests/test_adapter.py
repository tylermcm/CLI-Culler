import unittest

import numpy as np

from aiculler.adapter import RankingAwareAdapter


class RankingAwareAdapterTests(unittest.TestCase):
    def test_adapter_projects_scores_and_compares_pairs(self):
        adapter = RankingAwareAdapter(input_dim=8, projected_dim=16, text_dim=12, hidden_dim=8, seed=7)
        embedding = np.arange(8, dtype=np.float32)

        output = adapter.adapt(embedding, text_query="sharp natural light")
        other = adapter.adapt(embedding[::-1], text_query="sharp natural light")

        self.assertEqual(output.embedding.shape, (16,))
        self.assertIsInstance(output.pointwise_score, float)
        self.assertIsInstance(adapter.pairwise_distance(output.embedding, other.embedding), float)


if __name__ == "__main__":
    unittest.main()

