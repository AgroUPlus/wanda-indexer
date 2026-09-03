"""Smoke test for the three DSP stages on synthetic audio.

Fast, offline, no database. For strict bit-compatibility use test_golden_dsp.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.acoustic_features import extract_features
from core.fingerprinter import compute_landmarks, compute_recording_fingerprint
from core.spectrogram import compute_stft, frame_count_for


def make_samples(seconds: int = 10, sample_rate: int = 8000) -> np.ndarray:
    t = np.linspace(0, seconds, sample_rate * seconds, endpoint=False, dtype=np.float32)
    beat = 0.5 * (1.0 + np.sin(2 * np.pi * 2.0 * t))
    return (np.sin(2 * np.pi * 440.0 * t) * beat).astype(np.float32)


def test_landmarks_are_generated():
    landmarks = compute_landmarks(make_samples())
    assert landmarks, "no landmarks generated"
    for packed_hash, anchor_frame in landmarks[:100]:
        assert -(2 ** 31) <= packed_hash < 2 ** 31, "hash outside signed 32-bit range"
        assert anchor_frame >= 0


def test_recording_fingerprint_is_generated():
    blob = compute_recording_fingerprint(make_samples())
    assert blob, "no recording fingerprint generated"
    assert len(blob) % 4 == 0, "sub-hash blob is not a whole number of 32-bit ints"


def test_features_are_in_range():
    features = extract_features(make_samples())
    assert features is not None, "feature extraction returned nothing"
    for name in ("tempo", "energy", "brightness", "danceability"):
        assert 0.0 <= features[name] <= 1.0, f"{name} out of range: {features[name]}"
    radius = (features["keyX"] ** 2 + features["keyY"] ** 2) ** 0.5
    assert radius <= 1.0 + 1e-6, "key vector outside the unit circle"


def test_shared_stft_matches_independent_computation():
    """Passing a shared STFT must not change a single output value."""
    samples = make_samples()
    stft = compute_stft(samples, frame_count_for(len(samples)))

    assert compute_landmarks(samples) == compute_landmarks(samples, stft)
    assert compute_recording_fingerprint(samples) == compute_recording_fingerprint(samples, stft)
    assert extract_features(samples) == extract_features(samples, stft)


def test_short_input_is_handled():
    """Too-short audio must return empty results, not raise."""
    tiny = np.zeros(500, dtype=np.float32)
    assert compute_landmarks(tiny) == []
    assert compute_recording_fingerprint(tiny) == b""
    assert extract_features(tiny) is None


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  ok  {name}")
    print("All pipeline tests passed.")
