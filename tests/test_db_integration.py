"""Round-trip test against a throwaway database with the real Room schema."""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db_sync import (
    EXTRACTOR_VERSION,
    backfill_sub_hashes,
    batch_insert_index_data,
    check_integrity,
    count_index_rows,
    get_pending_tracks,
    repair_db,
    sub_hash_gap,
)
from core.fingerprinter import sub_hash_halves

SCHEMA = """
CREATE TABLE tracks (
    id TEXT PRIMARY KEY, sourceTrackId TEXT NOT NULL, source TEXT NOT NULL,
    title TEXT NOT NULL, artist TEXT NOT NULL, album TEXT,
    durationMs INTEGER NOT NULL, streamUri TEXT, localFilePath TEXT
);
CREATE TABLE fingerprints (
    hash INTEGER NOT NULL, trackId TEXT NOT NULL, anchorFrame INTEGER NOT NULL,
    PRIMARY KEY (hash, trackId, anchorFrame)
);
CREATE INDEX index_fingerprints_trackId ON fingerprints (trackId);
CREATE TABLE track_features (
    trackId TEXT NOT NULL PRIMARY KEY, tempo REAL NOT NULL, energy REAL NOT NULL,
    brightness REAL NOT NULL, danceability REAL NOT NULL, keyX REAL NOT NULL,
    keyY REAL NOT NULL, version INTEGER NOT NULL, measuredAt INTEGER NOT NULL
);
CREATE TABLE recording_fingerprints (
    trackId TEXT NOT NULL PRIMARY KEY, subHashes BLOB NOT NULL,
    durationMs INTEGER NOT NULL, computedAt INTEGER NOT NULL
);
CREATE TABLE recording_sub_hashes (
    half INTEGER NOT NULL, trackId TEXT NOT NULL,
    PRIMARY KEY (half, trackId)
);
CREATE INDEX index_recording_sub_hashes_trackId ON recording_sub_hashes (trackId);
"""

import numpy as np


def blob_of(values):
    """A recording fingerprint blob, big-endian as Room expects."""
    return np.array(values, dtype=">u4").tobytes()

FEATURES = {"tempo": 0.5, "energy": 0.8, "brightness": 0.4,
            "danceability": 0.6, "keyX": 0.7, "keyY": -0.2}


def build_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO tracks (id, sourceTrackId, source, title, artist, durationMs) "
        "VALUES (?, ?, ?, ?, ?, ?);",
        [
            ("t1", "s1", "NAVIDROME", "First", "Artist A", 180000),
            ("t2", "s2", "YTMUSIC", "Second", "Artist B", 200000),
            ("t3", "s3", "YTMUSIC", "Third", "Artist C", 240000),
        ],
    )
    conn.commit()
    conn.close()


def test_pending_then_insert_then_empty():
    with tempfile.TemporaryDirectory() as workdir:
        db = os.path.join(workdir, "test.db")
        build_db(db)

        pending = get_pending_tracks(db)
        assert len(pending) == 3, f"expected 3 pending, got {len(pending)}"
        assert all(t["needs_landmarks"] and t["needs_features"] for t in pending)

        counts = batch_insert_index_data(
            db,
            {"t1": [(123, 0), (456, 4)], "t2": [(789, 1)]},
            {"t1": dict(FEATURES), "t2": dict(FEATURES)},
            {"t1": (b"\x00\x01\x02\x03", 180000)},
        )
        assert counts["landmarks"] == 3, counts
        assert counts["features"] == 2, counts

        remaining = {t["id"] for t in get_pending_tracks(db)}
        # t1 is fully indexed; t2 still lacks a recording fingerprint; t3 has nothing.
        assert "t1" not in remaining, "fully indexed track still reported as pending"
        assert remaining == {"t2", "t3"}, remaining

        flags = {t["id"]: t for t in get_pending_tracks(db)}
        assert not flags["t2"]["needs_landmarks"], "t2 should not redo landmarks"
        assert flags["t2"]["needs_recording_fp"], "t2 should still need a recording fp"

        rows = count_index_rows(db)
        assert rows["fingerprints"] == 3, rows
        ok, problems = check_integrity(db, thorough=True)
        assert ok, problems


def test_null_track_ids_do_not_hide_pending_work():
    """The old NOT IN query returned zero rows if any trackId was NULL."""
    with tempfile.TemporaryDirectory() as workdir:
        db = os.path.join(workdir, "nulls.db")
        build_db(db)
        conn = sqlite3.connect(db)
        # A stray feature row with no matching track, as a rogue writer might leave.
        conn.execute(
            "INSERT INTO track_features VALUES ('ghost', 0,0,0,0,0,0, ?, 0);",
            (EXTRACTOR_VERSION,),
        )
        conn.commit()
        conn.close()

        assert len(get_pending_tracks(db)) == 3, "pending work disappeared"


def test_repair_rebuilds_a_damaged_index():
    with tempfile.TemporaryDirectory() as workdir:
        db = os.path.join(workdir, "damaged.db")
        build_db(db)
        batch_insert_index_data(db, {"t1": [(i, i) for i in range(500)]}, {}, {})

        ok, _ = check_integrity(db, thorough=True)
        assert ok, "fixture should start healthy"

        assert repair_db(db, log=lambda *a: None), "repair reported failure"
        ok, problems = check_integrity(db, thorough=True)
        assert ok, problems
        assert count_index_rows(db)["fingerprints"] == 500, "rows lost during repair"


def stored_halves(db: str, track_id: str) -> set:
    conn = sqlite3.connect(db)
    try:
        return {r[0] for r in conn.execute(
            "SELECT half FROM recording_sub_hashes WHERE trackId = ?;", (track_id,))}
    finally:
        conn.close()


def test_sub_hashes_written_with_fingerprint():
    """A fingerprint stored without its index can never be proposed as a candidate."""
    with tempfile.TemporaryDirectory() as workdir:
        db = os.path.join(workdir, "sh.db")
        build_db(db)
        blob = blob_of([0x1111_2222, 0x3333_4444, 0xAAAA_BBBB])

        counts = batch_insert_index_data(db, {}, {}, {"t1": (blob, 180000)})
        assert counts["sub_hashes"] == 6, counts
        assert stored_halves(db, "t1") == sub_hash_halves(blob)
        assert sub_hash_gap(db) == (1, 0), sub_hash_gap(db)


def test_replacing_a_fingerprint_clears_stale_halves():
    """Stale index rows propose candidates whose sequences no longer match."""
    with tempfile.TemporaryDirectory() as workdir:
        db = os.path.join(workdir, "replace.db")
        build_db(db)

        first = blob_of([0x0001_0002])
        batch_insert_index_data(db, {}, {}, {"t1": (first, 1000)})
        assert stored_halves(db, "t1") == sub_hash_halves(first)

        second = blob_of([0xF00D_BEEF])
        batch_insert_index_data(db, {}, {}, {"t1": (second, 1000)})
        after = stored_halves(db, "t1")
        assert after == sub_hash_halves(second), "halves do not match the new fingerprint"
        assert not (after & sub_hash_halves(first)), "stale halves from the old fingerprint survived"


def test_backfill_fills_only_missing_tracks():
    with tempfile.TemporaryDirectory() as workdir:
        db = os.path.join(workdir, "backfill.db")
        build_db(db)

        indexed = blob_of([0x1234_5678, 0x8765_4321])
        orphan = blob_of([0x0BAD_F00D, 0x1111_1111, 0x2222_2222])
        batch_insert_index_data(db, {}, {}, {"t1": (indexed, 1000)})

        # A fingerprint written the old way: no sub-hash rows at all.
        conn = sqlite3.connect(db)
        conn.execute("INSERT INTO recording_fingerprints VALUES ('t2', ?, 2000, 0);", (orphan,))
        conn.commit()
        conn.close()

        assert sub_hash_gap(db) == (2, 1), sub_hash_gap(db)
        before = stored_halves(db, "t1")

        written = backfill_sub_hashes(db, log=lambda *a: None)
        assert written == len(sub_hash_halves(orphan)), written
        assert stored_halves(db, "t2") == sub_hash_halves(orphan)
        assert stored_halves(db, "t1") == before, "an already-indexed track was touched"
        assert sub_hash_gap(db) == (2, 0)

        # Idempotent: a second run has nothing to do.
        assert backfill_sub_hashes(db, log=lambda *a: None) == 0


def test_backfill_is_chunked_and_resumable():
    """Each chunk commits, so an interrupt costs at most one chunk."""
    with tempfile.TemporaryDirectory() as workdir:
        db = os.path.join(workdir, "chunked.db")
        build_db(db)
        conn = sqlite3.connect(db)
        for index in range(10):
            conn.execute(
                "INSERT INTO recording_fingerprints VALUES (?, ?, 1000, 0);",
                (f"x{index}", blob_of([0x1000_0000 + index, 0x2000_0000 + index])),
            )
        conn.commit()
        conn.close()

        assert sub_hash_gap(db)[1] == 10
        backfill_sub_hashes(db, log=lambda *a: None, chunk_size=3)
        assert sub_hash_gap(db)[1] == 0
        for index in range(10):
            assert stored_halves(db, f"x{index}"), f"x{index} was skipped"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  ok  {name}")
    print("All database tests passed.")