"""The embedder's contract with the Android app.

Everything here is a parity assertion in disguise. A vector this module computes
is compared, on a phone, against one the app computed from a microphone; the two
have to be the same alphabet or the comparison is noise. None of these failures
would announce themselves at runtime -- a wrongly segmented or wrongly quantised
vector is a perfectly well-formed vector that simply never matches anything.

The cross-engine check that needs a real device is `tools/check_parity.py`.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import embedder


class SegmentationTest(unittest.TestCase):
    """Ported from `AudioEmbedder.segment`; the cases are its two padding branches."""

    def test_window_counts_match_the_app(self):
        # Exactly one window.
        self.assertEqual(embedder.segment(np.zeros(8000, np.float32)).shape, (1, 8000))
        # Shorter than a window is zero-padded up to one.
        self.assertEqual(embedder.segment(np.zeros(4000, np.float32)).shape, (1, 8000))
        # Exactly two windows at a 0.5 s hop.
        self.assertEqual(embedder.segment(np.zeros(12000, np.float32)).shape, (2, 8000))
        # A remainder pads out to a further window rather than being dropped.
        self.assertEqual(embedder.segment(np.zeros(10000, np.float32)).shape, (2, 8000))
        # Sixty seconds: the length every pre-existing row in the database has.
        self.assertEqual(embedder.segment(np.zeros(480000, np.float32)).shape, (119, 8000))
        self.assertEqual(embedder.segment(np.zeros(480001, np.float32)).shape, (120, 8000))

    def test_windows_are_contiguous_at_the_hop(self):
        pcm = np.arange(20000, dtype=np.float32)
        w = embedder.segment(pcm)
        self.assertEqual(w[0][0], 0)
        self.assertEqual(w[1][0], embedder.HOP_SAMPLES)
        self.assertEqual(w[2][0], 2 * embedder.HOP_SAMPLES)

    def test_the_tail_is_zero_padded_not_wrapped(self):
        pcm = np.ones(10000, dtype=np.float32)
        w = embedder.segment(pcm)
        # The last window runs past the end of the signal; what is past it is silence.
        self.assertEqual(w[1][-1], 0.0)


class PackingTest(unittest.TestCase):
    """One int8 per component, segment-major -- `AudioEmbedder.pack`."""

    def unit(self, n, seed=0):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, embedder.EMBED_DIM)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def test_blob_is_one_byte_per_component(self):
        v = self.unit(7)
        self.assertEqual(len(embedder.pack(v)), 7 * embedder.EMBED_DIM)

    def test_round_trip_stays_within_a_quantisation_step(self):
        v = self.unit(20)
        back = embedder.unpack(embedder.pack(v))
        self.assertLessEqual(float(np.abs(back - v).max()), 0.5 / embedder.QUANT_SCALE)

    def test_out_of_range_saturates_rather_than_wrapping(self):
        # 0.6 * 255 is 153, which as a signed byte is -103: a large positive
        # component stored as a large negative one. It has to clip instead.
        blob = embedder.pack(np.full((1, embedder.EMBED_DIM), 0.9, np.float32))
        self.assertEqual(np.frombuffer(blob, np.int8).min(), 127)
        blob = embedder.pack(np.full((1, embedder.EMBED_DIM), -0.9, np.float32))
        self.assertEqual(np.frombuffer(blob, np.int8).max(), -127)


class SummaryTest(unittest.TestCase):
    """One mean per 30 s, matching `EmbeddingRepository.summaryOf`."""

    def vectors(self, n):
        rng = np.random.default_rng(3)
        v = rng.standard_normal((n, embedder.EMBED_DIM)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def test_one_chunk_per_thirty_seconds(self):
        self.assertEqual(len(embedder.summary(self.vectors(60))), 1)
        # 119 segments is a 60 s track -- every row written before whole-track
        # indexing -- and must still summarise to exactly one chunk, or those
        # rows would have to be rewritten to stay comparable.
        self.assertEqual(len(embedder.summary(self.vectors(119))), 1)
        self.assertEqual(len(embedder.summary(self.vectors(360))), 6)
        # A short remainder joins the last chunk rather than becoming its own.
        self.assertEqual(len(embedder.summary(self.vectors(370))), 6)
        self.assertEqual(len(embedder.summary(self.vectors(11))), 1)

    def test_every_chunk_is_normalised(self):
        for chunk in embedder.summary(self.vectors(300)):
            self.assertAlmostEqual(float(np.linalg.norm(chunk)), 1.0, places=4)


class ModelTest(unittest.TestCase):
    """Skipped where no TFLite runtime or no network -- the rest still runs."""

    @classmethod
    def setUpClass(cls):
        if not embedder.is_available():
            raise unittest.SkipTest("no TFLite runtime or model available")

    def test_a_minute_of_audio_embeds_to_unit_vectors(self):
        rng = np.random.default_rng(0)
        pcm = (rng.standard_normal(480000) * 0.05).astype(np.float32)
        v = embedder.embed(pcm)
        self.assertEqual(v.shape, (119, embedder.EMBED_DIM))
        norms = np.linalg.norm(v, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-4), f"norms {norms.min()}..{norms.max()}")

    def test_the_same_audio_embeds_the_same_way_twice(self):
        rng = np.random.default_rng(1)
        pcm = (rng.standard_normal(48000) * 0.05).astype(np.float32)
        np.testing.assert_array_equal(embedder.embed(pcm), embedder.embed(pcm))


if __name__ == "__main__":
    unittest.main()
