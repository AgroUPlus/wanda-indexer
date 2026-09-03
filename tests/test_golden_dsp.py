"""Golden-vector tests: the Python DSP must stay bit-compatible with the Kotlin
implementations on the phone. Any refactor that changes a single hash is a bug.

Regenerate deliberately (only when the Kotlin side changes) with:
    python3 tests/test_golden_dsp.py --regenerate
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.acoustic_features import extract_features
from core.fingerprinter import compute_landmarks, compute_recording_fingerprint

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_dsp.json")

# (kind, seed, sample_count) -- covers noise, pure tones, a beat pattern,
# digital silence, and a clip too short for feature extraction.
CASES = [
    ("noise", 7, 8000 * 30),
    ("tone", 1, 8000 * 20),
    ("beat", 3, 8000 * 45),
    ("silence", 0, 8000 * 10),
    ("noise", 11, 8000 * 3),
]


def make_signal(kind: str, seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 8000.0
    if kind == "noise":
        x = rng.standard_normal(n) * 0.1
    elif kind == "tone":
        x = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)
    elif kind == "beat":
        env = (np.sin(2 * np.pi * 2.0 * t) > 0.7).astype(np.float64)
        x = env * np.sin(2 * np.pi * 220 * t) * 0.5 + rng.standard_normal(n) * 0.02
    elif kind == "silence":
        x = np.zeros(n)
    else:
        raise ValueError(kind)
    return x.astype(np.float32)


def measure(samples: np.ndarray) -> dict:
    landmarks = compute_landmarks(samples)
    recording = compute_recording_fingerprint(samples)
    return {
        "landmarks_count": len(landmarks),
        "landmarks_sha": hashlib.sha256(np.array(landmarks, dtype=np.int64).tobytes()).hexdigest(),
        "features": extract_features(samples),
        "rec_len": len(recording),
        "rec_sha": hashlib.sha256(recording).hexdigest(),
    }


def load_golden() -> dict:
    with open(GOLDEN_PATH) as handle:
        return json.load(handle)


def compare(key: str, actual: dict, expected: dict) -> list:
    """Returns a list of human-readable mismatch descriptions."""
    problems = []
    for field in ("landmarks_count", "landmarks_sha", "rec_len", "rec_sha"):
        if actual[field] != expected[field]:
            problems.append(f"{key}.{field}: got {actual[field]!r}, want {expected[field]!r}")

    got_feats, want_feats = actual["features"], expected["features"]
    if (got_feats is None) != (want_feats is None):
        problems.append(f"{key}.features: got {got_feats!r}, want {want_feats!r}")
    elif got_feats is not None:
        for name, want in want_feats.items():
            got = got_feats[name]
            # Reassociating float sums shifts the last ulp or two; the phone
            # tolerates that, a changed algorithm would move far more.
            if abs(got - want) > 1e-9:
                problems.append(f"{key}.features.{name}: got {got!r}, want {want!r}")
    return problems


def test_dsp_matches_golden_vectors():
    golden = load_golden()
    problems = []
    for kind, seed, n in CASES:
        key = f"{kind}_{seed}_{n}"
        assert key in golden, f"missing golden entry {key}; regenerate deliberately"
        problems.extend(compare(key, measure(make_signal(kind, seed, n)), golden[key]))
    assert not problems, "DSP output changed:\n  " + "\n  ".join(problems)


def _regenerate():
    out = {}
    for kind, seed, n in CASES:
        out[f"{kind}_{seed}_{n}"] = measure(make_signal(kind, seed, n))
    with open(GOLDEN_PATH, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(f"regenerated {GOLDEN_PATH}")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        test_dsp_matches_golden_vectors()
        print("OK: DSP output matches golden vectors")
