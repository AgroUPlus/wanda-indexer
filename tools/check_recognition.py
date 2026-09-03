"""Check that recording recognition actually works, on your real library.

Answers three questions, in order:

  1. Is the sub-hash lookup index complete?  (it was missing for 91% of tracks)
  2. Which duplicate recordings does the library contain?
  3. How many of them would the app find -- before and after the alignment fix?

It reproduces what the phone does: candidates from `recording_sub_hashes`, then
a full sequence comparison. The offset-alignment logic mirrors
RecordingFingerprinter.aligned in the Android app.

    python3 tools/check_recognition.py              # summary
    python3 tools/check_recognition.py --pairs      # list every duplicate found
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db_sync import connect, sub_hash_gap
from core.fingerprinter import sub_hash_halves

# --- copied from RecordingIdentityRepository.kt / RecordingFingerprinter.kt ---
MATCH_THRESHOLD = 0.72
MIN_CANDIDATE_HITS = 4
CANDIDATE_LIMIT = 20
MAX_OFFSET_FRAMES = 400
REFINE_FRAMES = 3
MIN_OVERLAP_FRAMES = 100
MAX_VOTES_PER_VALUE = 32
BITS = 32
FRAMES_PER_SECOND = 8000 / 256


def similarity_at(a, b, offset):
    """Bit agreement with `b` shifted `offset` frames against `a`."""
    a_start = offset if offset >= 0 else 0
    b_start = 0 if offset >= 0 else -offset
    length = min(len(a) - a_start, len(b) - b_start)
    if length < MIN_OVERLAP_FRAMES:
        return 0.0
    differing = np.bitwise_xor(a[a_start:a_start + length], b[b_start:b_start + length])
    return 1.0 - np.unpackbits(differing.view(np.uint8)).sum() / (length * BITS)


def aligned(a, b):
    """(similarity, offset) at the best alignment, found by voting not scanning."""
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0

    positions = defaultdict(list)
    for index, value in enumerate(a.tolist()):
        positions[value].append(index)

    bins = Counter()
    for j, value in enumerate(b.tolist()):
        hits = positions.get(value)
        # A value occurring everywhere says nothing about position.
        if not hits or len(hits) > MAX_VOTES_PER_VALUE:
            continue
        for i in hits:
            offset = i - j
            if -MAX_OFFSET_FRAMES <= offset <= MAX_OFFSET_FRAMES:
                bins[offset] += 1
    if not bins:
        return 0.0, 0

    # OffsetAlignment.best: each bin summed with its two neighbours.
    best_offset, best_votes = 0, 0
    for offset in bins:
        votes = bins.get(offset - 1, 0) + bins[offset] + bins.get(offset + 1, 0)
        if votes > best_votes:
            best_offset, best_votes = offset, votes

    return max(
        ((similarity_at(a, b, o), o)
         for o in range(best_offset - REFINE_FRAMES, best_offset + REFINE_FRAMES + 1)),
        key=lambda pair: pair[0],
    )


def load(conn):
    rows = conn.execute(
        """
        SELECT f.trackId, f.subHashes, t.artist, t.title, t.source
        FROM recording_fingerprints f JOIN tracks t ON t.id = f.trackId;
        """
    ).fetchall()
    return [
        {"id": r[0], "hashes": np.frombuffer(r[1], dtype=">u4"),
         "artist": r[2], "title": r[3], "source": r[4]}
        for r in rows if r[1]
    ]


def candidates_for(conn, track, halves):
    """The phone's candidate query: shared halves, ranked, capped."""
    tally = Counter()
    ordered = sorted(halves)
    for start in range(0, len(ordered), 900):   # SQLite variable limit
        chunk = ordered[start:start + 900]
        query = (
            f"SELECT trackId, COUNT(*) FROM recording_sub_hashes "
            f"WHERE half IN ({','.join('?' * len(chunk))}) AND trackId != ? GROUP BY trackId"
        )
        for other, hits in conn.execute(query, chunk + [track]):
            tally[other] += hits
    return [t for t, hits in tally.most_common(CANDIDATE_LIMIT) if hits >= MIN_CANDIDATE_HITS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="wanda_music.db")
    parser.add_argument("--pairs", action="store_true", help="List every duplicate found")
    parser.add_argument("--sample", type=int, default=0,
                        help="Only test N tracks (the full sweep is slow on a network mount)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"[ERROR] Database not found: {args.db}")
        return 1

    print("1. Sub-hash lookup index")
    total, missing = sub_hash_gap(args.db)
    if missing:
        print(f"   BROKEN: {missing:,} of {total:,} fingerprints have no index entry.")
        print("   Those tracks can never be recognised. Fix: main.py --backfill-subhashes")
        return 1
    print(f"   OK: all {total:,} fingerprints are in the lookup index.\n")

    conn = connect(args.db, readonly=True)
    try:
        tracks = load(conn)
        by_id = {t["id"]: t for t in tracks}
        print(f"2. Searching {len(tracks):,} fingerprints for duplicate recordings")

        # Same artist+title is a cheap way to find pairs worth checking without
        # comparing all ~900k combinations. Fingerprints decide, not the titles.
        groups = defaultdict(list)
        for track in tracks:
            groups[(track["artist"].lower(), track["title"].lower())].append(track)
        pairs = [(g[i], g[j]) for g in groups.values() if len(g) > 1
                 for i in range(len(g)) for j in range(i + 1, len(g))]
        if args.sample:
            pairs = pairs[: args.sample]
        print(f"   {len(pairs)} candidate pair(s) to verify\n")

        print("3. What the app would find")
        before = after = 0
        rescued, missed = [], []
        for one, two in pairs:
            old = similarity_at(one["hashes"], two["hashes"], 0)
            new, offset = aligned(one["hashes"], two["hashes"])
            if old >= MATCH_THRESHOLD:
                before += 1
            if new >= MATCH_THRESHOLD:
                after += 1
                if old < MATCH_THRESHOLD:
                    rescued.append((one, two, old, new, offset))
            else:
                missed.append((one, two, new))

        print(f"   matched before the alignment fix : {before}/{len(pairs)}")
        print(f"   matched after  the alignment fix : {after}/{len(pairs)}")
        if rescued:
            print(f"\n   {len(rescued)} pair(s) the fix rescued:")
            for one, two, old, new, offset in rescued:
                print(f"     {old:.3f} -> {new:.3f}  shifted {offset / FRAMES_PER_SECOND:+.1f}s   "
                      f"{one['artist']} - {one['title']}"[:96])
        if missed:
            print(f"\n   {len(missed)} pair(s) still unmatched "
                  "(same title, but different recordings):")
            for one, two, score in missed:
                print(f"     {score:.3f}  {one['artist']} - {one['title']}"[:96])

        if args.pairs:
            print("\n4. Live lookup through the app's own candidate query")
            for one, two in pairs[:10]:
                found = candidates_for(conn, one["id"], sub_hash_halves(one["hashes"]))
                hit = two["id"] in found
                print(f"   {'FOUND' if hit else 'miss ':<6} {one['artist']} - {one['title']}"[:80])
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
