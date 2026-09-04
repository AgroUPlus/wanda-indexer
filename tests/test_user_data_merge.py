"""`merge_user_data` must carry the phone's own data across a push, and nothing else.

The bug being guarded against: `isLiked`, `playCount` and friends are columns on `tracks`, so
pushing a freshly indexed database to deliver new fingerprints also reverts every like to
whatever the desktop copy last captured. These tests assert the merge fixes that without
touching the index it was built to deliver.
"""
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import db_sync

SCHEMA = """
CREATE TABLE tracks (
    id TEXT PRIMARY KEY, title TEXT, isLiked INTEGER DEFAULT 0,
    isDownloaded INTEGER DEFAULT 0, isCached INTEGER DEFAULT 0, localFilePath TEXT,
    playCount INTEGER DEFAULT 0, lastPlayedTimestamp INTEGER DEFAULT 0,
    addedTimestamp INTEGER DEFAULT 0, isLibrary INTEGER DEFAULT 0, contentHash TEXT
);
CREATE TABLE history (id INTEGER PRIMARY KEY, trackId TEXT, playedAt INTEGER);
CREATE TABLE local_playlists (id INTEGER PRIMARY KEY, name TEXT, trackIds TEXT);
CREATE TABLE recording_splits (idA TEXT, idB TEXT, PRIMARY KEY (idA, idB));
CREATE TABLE canonical_metadata (trackId TEXT PRIMARY KEY, title TEXT);
CREATE TABLE drops (id INTEGER PRIMARY KEY, trackTitle TEXT);
CREATE TABLE track_embeddings (
    trackId TEXT PRIMARY KEY, vector BLOB, dim INTEGER, model TEXT,
    version INTEGER, computedAt INTEGER
);
"""


def build(path, tracks, likes=(), history=(), embeddings=()):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    for tid in tracks:
        conn.execute("INSERT INTO tracks (id, title) VALUES (?, ?)", (tid, f"title {tid}"))
    for tid, count in likes:
        conn.execute("UPDATE tracks SET isLiked = 1, playCount = ? WHERE id = ?", (count, tid))
    for row_id, tid in history:
        conn.execute("INSERT INTO history (id, trackId, playedAt) VALUES (?, ?, 1)",
                     (row_id, tid))
    for tid in embeddings:
        conn.execute(
            "INSERT INTO track_embeddings VALUES (?, ?, 128, 'nmfp-triplet', 1, 0)",
            (tid, b"\x00" * 512),
        )
    conn.commit()
    conn.close()


class MergeUserDataTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.phone = os.path.join(self.dir, "phone.db")
        self.desktop = os.path.join(self.dir, "desktop.db")

    def test_likes_and_play_counts_come_from_the_phone(self):
        build(self.phone, ["a", "b", "c"], likes=[("a", 7), ("b", 3)])
        # The desktop copy is an older snapshot: it thinks only "c" was liked.
        build(self.desktop, ["a", "b", "c"], likes=[("c", 1)], embeddings=["a", "b", "c"])

        db_sync.merge_user_data(self.phone, self.desktop, log=lambda *_: None)

        conn = sqlite3.connect(self.desktop)
        liked = dict(conn.execute("SELECT id, isLiked FROM tracks").fetchall())
        counts = dict(conn.execute("SELECT id, playCount FROM tracks").fetchall())
        embeddings = conn.execute("SELECT count(*) FROM track_embeddings").fetchone()[0]
        conn.close()

        self.assertEqual(liked, {"a": 1, "b": 1, "c": 0}, "phone's likes must win")
        self.assertEqual(counts["a"], 7)
        self.assertEqual(embeddings, 3, "the index being delivered must survive the merge")

    def test_user_tables_are_replaced_not_appended(self):
        build(self.phone, ["a"], history=[(1, "a"), (2, "a")])
        build(self.desktop, ["a"], history=[(9, "a")])

        db_sync.merge_user_data(self.phone, self.desktop, log=lambda *_: None)

        conn = sqlite3.connect(self.desktop)
        rows = sorted(r[0] for r in conn.execute("SELECT id FROM history"))
        conn.close()
        self.assertEqual(rows, [1, 2], "a stale desktop row must not linger alongside")

    def test_tracks_only_on_the_phone_are_reported(self):
        build(self.phone, ["a", "phone-only"], likes=[("phone-only", 4)])
        build(self.desktop, ["a"])

        counts = db_sync.merge_user_data(self.phone, self.desktop, log=lambda *_: None)

        self.assertEqual(counts.get("tracks_only_on_phone"), 1,
                         "divergence must be counted, not silently dropped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
