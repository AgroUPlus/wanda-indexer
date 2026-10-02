"""Rebuild the index from scratch without losing anything the user made.

"Start clean" is tempting to implement as "delete the tracks and resync", and that would be
wrong here: `isLiked`, `playCount`, `lastPlayedTimestamp`, `isLibrary` and `localFilePath` are
**columns on `tracks`**, not a separate table, and nothing re-derives them. A resync restores the
titles and the artwork; it does not restore what the user liked.

So this deletes only what the indexer computed and can compute again, and never touches `tracks`
itself. That is safe precisely because the app's own resync is already non-destructive:
`TrackDao.upsertTracks` does an `@Insert(IGNORE)` followed by a partial `@Update` limited to the
columns a backend owns, so re-scanning Navidrome or YouTube Music refreshes metadata and leaves
user state alone.

    python3 tools/reset_derived.py --db wanda_music.db
    python3 tools/reset_derived.py --db wanda_music.db --keep-embeddings

Afterwards, re-run the indexer to refill it:

    python3 main.py --db wanda_music.db --no-landmarks
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import db_sync

# Everything the indexer or the server can rebuild.
#
# `shelves` and `friends` are here because they are a cached copy of a server response, not
# something the user authored -- they refill on the next sync.
DERIVED_TABLES = (
    "fingerprints",
    "track_embeddings",
    "track_features",
    "melody_contours",
    "recording_fingerprints",
    "recording_sub_hashes",
    "albums",
    "artists",
    "shelves",
)

# Named so the safety of the operation can be read off the file rather than inferred.
PRESERVED = (
    "tracks (with isLiked, playCount, lastPlayedTimestamp, isLibrary, localFilePath)",
    "history",
    "local_playlists",
    "recording_splits",
    "canonical_metadata",
    "drops",
    "friends",
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="wanda_music.db")
    ap.add_argument("--keep-embeddings", action="store_true",
                    help="Leave track_embeddings alone; rebuild everything else")
    ap.add_argument("--vacuum-into", default="",
                    help="Write the compacted result here instead of vacuuming in place. Use a "
                         "local disk when the database sits on a Windows mount")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"[ERROR] No such database: {args.db}")
        return 2

    tables = [t for t in DERIVED_TABLES
              if not (args.keep_embeddings and t == "track_embeddings")]

    conn = db_sync.connect(args.db, readonly=True)
    present = {
        row[0] for row in
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    counts = {}
    for table in tables:
        if table in present:
            counts[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    likes = conn.execute("SELECT count(*) FROM tracks WHERE isLiked = 1").fetchone()[0]
    played = conn.execute("SELECT count(*) FROM tracks WHERE playCount > 0").fetchone()[0]
    conn.close()

    print(f"[DB]    {args.db}")
    print("\nWill be emptied and rebuilt:")
    for table, count in counts.items():
        print(f"  {table:<26} {count:>12,} rows")
    print("\nWill be kept:")
    for name in PRESERVED:
        print(f"  {name}")
    print(f"\n  {likes} liked track(s), {played} with a play count -- these are the ones that "
          "cannot be rebuilt.")

    if not args.yes:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Nothing was changed.")
            return 1

    started = time.time()
    conn = db_sync.connect(args.db)
    try:
        for table in counts:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()
    finally:
        conn.close()
    print(f"\n[RESET] Emptied {len(counts)} table(s) in {time.time() - started:.1f}s")

    if args.vacuum_into:
        if os.path.exists(args.vacuum_into):
            os.remove(args.vacuum_into)
        conn = db_sync.connect(args.db)
        try:
            conn.execute("VACUUM INTO ?", (args.vacuum_into,))
        finally:
            conn.close()
        size = os.path.getsize(args.vacuum_into) / 1e6
        print(f"[RESET] Compacted into {args.vacuum_into} ({size:.0f} MB)")
    else:
        conn = db_sync.connect(args.db)
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()
        print(f"[RESET] Compacted to {os.path.getsize(args.db) / 1e6:.0f} MB")

    print(f"\nRefill it with:\n  python3 main.py --db {args.db} --no-landmarks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
