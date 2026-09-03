"""The sub-hash index must match what the Android app writes, exactly.

`RecordingIdentityRepository.matchesForFingerprint` proposes candidates from
`recording_sub_hashes` and only then compares full sequences. If the desktop
computes these halves differently from the phone, the two disagree about which
tracks are even worth comparing, and cross-device recognition quietly stops
working. Hence a golden fixture captured from rows the phone itself wrote.
"""
import base64
import hashlib
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fingerprinter import HIGH_HALF_TAG, sub_hash_halves

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_sub_hashes.json")


def load_golden() -> dict:
    with open(GOLDEN_PATH) as handle:
        return json.load(handle)


def test_halves_match_rows_the_phone_wrote():
    """Recompute halves from a real fingerprint; compare to the phone's own rows.

    The blobs and the expected halves both came out of a library indexed by the
    Android app, so this is a genuine cross-implementation check rather than
    this module agreeing with itself.
    """
    golden = load_golden()
    assert golden, "golden fixture is empty"

    for track_id, expected in golden.items():
        blob = zlib.decompress(base64.b64decode(expected["blob_deflate_b64"]))
        assert len(blob) == expected["blob_len"], f"{track_id}: fixture blob is damaged"

        halves = sorted(sub_hash_halves(blob))
        assert len(halves) == expected["half_count"], (
            f"{track_id}: produced {len(halves)} halves, the phone stored "
            f"{expected['half_count']}"
        )
        digest = hashlib.sha256(",".join(map(str, halves)).encode()).hexdigest()
        assert digest == expected["halves_sha256"], (
            f"{track_id}: halves differ from the rows the phone wrote. "
            "The desktop and the app now disagree about candidate lookup."
        )


def test_halves_are_exactly_two_per_distinct_hash():
    hashes = np.array([0x0001_0002, 0xABCD_1234, 0x0000_0000], dtype=">u4")
    halves = sub_hash_halves(hashes.tobytes())
    expected = {
        0x0002, 0x0001 | HIGH_HALF_TAG,
        0x1234, 0xABCD | HIGH_HALF_TAG,
        0x0000, 0x0000 | HIGH_HALF_TAG,
    }
    assert halves == expected, f"got {sorted(halves)}, want {sorted(expected)}"


def test_halves_are_deduplicated():
    """A repeated hash indexes once: a repeated passage must not weigh more."""
    single = sub_hash_halves(np.array([0x1111_2222], dtype=">u4").tobytes())
    repeated = sub_hash_halves(np.array([0x1111_2222] * 50, dtype=">u4").tobytes())
    assert single == repeated
    assert len(single) == 2


def test_high_and_low_halves_do_not_collide():
    """Without the tag, one hash's low half would collide with another's high.

    0x000000FF carries 0x00FF in its low half; 0x00FF0000 carries the same value
    in its high half. Untagged they would be one index entry and inflate each
    other's candidate score.
    """
    low_only = sub_hash_halves(np.array([0x0000_00FF], dtype=">u4").tobytes())
    high_only = sub_hash_halves(np.array([0x00FF_0000], dtype=">u4").tobytes())

    assert low_only == {0x00FF, 0x0000 | HIGH_HALF_TAG}, sorted(low_only)
    assert high_only == {0x0000, 0x00FF | HIGH_HALF_TAG}, sorted(high_only)
    # The shared raw value 0x00FF appears in both, but never as the same entry.
    assert 0x00FF in low_only and 0x00FF not in high_only
    assert (0x00FF | HIGH_HALF_TAG) in high_only


def test_accepts_blob_or_array():
    hashes = np.array([0xDEAD_BEEF, 0x1234_5678], dtype=">u4")
    assert sub_hash_halves(hashes.tobytes()) == sub_hash_halves(hashes)


def test_degenerate_input_returns_empty():
    assert sub_hash_halves(b"") == set()
    assert sub_hash_halves(b"\x00\x01\x02") == set()   # not a whole uint32
    assert sub_hash_halves(np.array([], dtype=">u4")) == set()


def test_byte_order_is_big_endian():
    """Kotlin's ByteBuffer defaults to big-endian; a mismatch here is silent."""
    blob = bytes([0x12, 0x34, 0x56, 0x78])
    halves = sub_hash_halves(blob)
    assert halves == {0x5678, 0x1234 | HIGH_HALF_TAG}, sorted(halves)


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  ok  {name}")
    print("All sub-hash tests passed.")
