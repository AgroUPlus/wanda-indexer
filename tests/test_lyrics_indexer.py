import os
import sqlite3
import tempfile
import unittest

from core.db_sync import batch_insert_lyrics, search_lyrics_fts, ensure_lyrics_tables
from core.lyrics_indexer import (
    find_matching_lyric_line,
    parse_lrc,
    plain_text_from_lrc,
)

SAMPLE_LRC = """
[00:00.00]
[00:03.14]Hello darkness my old friend
[00:07.82]I've come to talk with you again
[00:12.45]Because a vision softly creeping
[00:16.89]Left its seeds while I was sleeping
[00:21.10]And the vision that was planted in my brain
[00:27.40]Still remains
[00:31.05]Within the sound of silence
"""


class TestLyricsIndexer(unittest.TestCase):
    def test_parse_lrc(self):
        lines = parse_lrc(SAMPLE_LRC)
        self.assertEqual(len(lines), 7)
        self.assertEqual(lines[0][0], 3140)
        self.assertEqual(lines[0][1], "Hello darkness my old friend")
        self.assertEqual(lines[1][0], 7820)
        self.assertEqual(lines[1][1], "I've come to talk with you again")

    def test_plain_text_from_lrc(self):
        plain = plain_text_from_lrc(SAMPLE_LRC)
        self.assertIn("Hello darkness my old friend", plain)
        self.assertNotIn("[00:03.14]", plain)

    def test_find_matching_lyric_line_synced(self):
        ts, line = find_matching_lyric_line(None, SAMPLE_LRC, "darkness my old friend")
        self.assertEqual(ts, 3140)
        self.assertEqual(line, "Hello darkness my old friend")

        ts2, line2 = find_matching_lyric_line(None, SAMPLE_LRC, "planted in my brain")
        self.assertEqual(ts2, 21100)
        self.assertEqual(line2, "And the vision that was planted in my brain")

    def test_find_matching_lyric_line_plain(self):
        plain = plain_text_from_lrc(SAMPLE_LRC)
        ts, line = find_matching_lyric_line(plain, None, "seeds while I was sleeping")
        self.assertIsNone(ts)
        self.assertEqual(line, "Left its seeds while I was sleeping")

    def test_sqlite_fts_search(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE tracks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    artist TEXT NOT NULL,
                    album TEXT,
                    durationMs INTEGER NOT NULL
                );
            """)
            conn.execute("""
                INSERT INTO tracks VALUES
                ('simon_01', 'The Sound of Silence', 'Simon & Garfunkel', 'Wednesday Morning, 3 A.M.', 185000),
                ('queen_01', 'Bohemian Rhapsody', 'Queen', 'A Night at the Opera', 354000);
            """)
            conn.commit()
            conn.close()

            lyrics_data = {
                'simon_01': {
                    'plainLyrics': plain_text_from_lrc(SAMPLE_LRC),
                    'syncedLyrics': SAMPLE_LRC,
                    'source': 'LRCLIB'
                },
                'queen_01': {
                    'plainLyrics': 'Is this the real life? Is this just fantasy? Caught in a landslide, no escape from reality',
                    'syncedLyrics': None,
                    'source': 'LRCLIB'
                }
            }

            inserted = batch_insert_lyrics(db_path, lyrics_data)
            self.assertEqual(inserted, 2)

            # Query 1: Search for words in Simon & Garfunkel lyrics
            results1 = search_lyrics_fts(db_path, "darkness friend")
            self.assertEqual(len(results1), 1)
            self.assertEqual(results1[0]['trackId'], 'simon_01')
            self.assertEqual(results1[0]['title'], 'The Sound of Silence')
            self.assertIn('<b>darkness</b>', results1[0]['snippet'])

            # Query 2: Search for words in Queen lyrics
            results2 = search_lyrics_fts(db_path, "landslide escape")
            self.assertEqual(len(results2), 1)
            self.assertEqual(results2[0]['trackId'], 'queen_01')
            self.assertEqual(results2[0]['title'], 'Bohemian Rhapsody')
            self.assertIn('<b>landslide</b>', results2[0]['snippet'])

            # Query 3: Non-existent lyric
            results3 = search_lyrics_fts(db_path, "supercalifragilistic")
            self.assertEqual(len(results3), 0)

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


if __name__ == '__main__':
    unittest.main()
