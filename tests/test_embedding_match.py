"""Tests for neural embedding decision gate and duplicate skipping."""
import unittest

from core.embedding_match import decide, MIN_SIMILARITY, MIN_MARGIN


class EmbeddingDecisionGateTests(unittest.TestCase):

    def test_clear_winner_accepted(self):
        ranked = [(0.95, "track-1"), (0.40, "track-2")]
        accepted, tid, sim, margin = decide(ranked)
        self.assertTrue(accepted)
        self.assertEqual("track-1", tid)
        self.assertAlmostEqual(0.95, sim)
        self.assertAlmostEqual(0.55, margin)

    def test_below_min_similarity_rejected(self):
        ranked = [(0.50, "track-1"), (0.20, "track-2")]
        accepted, tid, sim, margin = decide(ranked)
        self.assertFalse(accepted)
        self.assertEqual("track-1", tid)

    def test_close_competitor_rejected(self):
        ranked = [(0.95, "track-1"), (0.93, "track-2")]
        accepted, tid, sim, margin = decide(ranked)
        self.assertFalse(accepted)
        self.assertAlmostEqual(0.02, margin)

    def test_duplicate_runner_up_skipped_for_distinct_competitor(self):
        ranked = [(0.998, "navidrome:1"), (0.998, "ytm:1"), (0.400, "navidrome:2")]

        def is_duplicate(a, b):
            return {"navidrome:1", "ytm:1"} == {a, b}

        accepted, tid, sim, margin = decide(ranked, is_duplicate=is_duplicate)
        self.assertTrue(accepted)
        self.assertEqual("navidrome:1", tid)
        # Margin should be computed against the first non-duplicate competitor (navidrome:2)
        self.assertAlmostEqual(0.998 - 0.400, margin, places=3)

    def test_all_candidates_duplicates(self):
        ranked = [(0.998, "copy-1"), (0.998, "copy-2")]

        def is_duplicate(a, b):
            return True

        accepted, tid, sim, margin = decide(ranked, is_duplicate=is_duplicate)
        self.assertTrue(accepted)
        # Runner up defaults to 0.0 when no non-duplicate competitor exists
        self.assertAlmostEqual(0.998, margin, places=3)


if __name__ == "__main__":
    unittest.main()
