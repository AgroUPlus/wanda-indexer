"""Check the neural embedding fingerprint on your real library -- no audio re-decode.

Works entirely from the `track_embeddings` rows the indexer wrote, so run
`python main.py --db wanda_music.db` (with the embedder model present) first.

Answers:

  1. Coverage: how many tracks have an embedding.
  2. Self-consistency: does the first half of a track's segment sequence match its
     second half far better than it matches a random other track? (a sanity check
     that the vectors carry track identity, needing no second recording)
  3. Rank-1 retrieval: take a short window from the middle of each embedded
     track as a "mic clip", match it against every track's full sequence the way
     the phone does, and check the track ranks first -- with what margin over the
     runner-up. This is the recognition test.

    python3 tools/check_embedding_recognition.py
    python3 tools/check_embedding_recognition.py --threshold 0.55 --pairs
"""
import argparse
import os
import random
import sys

import numpy as np

# Runnable from anywhere, like the other tools in this directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db_sync import connect
from core.embedder import EMBEDDER_VERSION, MODEL_NAME
from core.embedding_match import best_cosine, load_embeddings, score


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="wanda_music.db")
    ap.add_argument("--window", type=int, default=12,
                    help="query length in 0.5 s segments (12 = 6 s, the phone's LISTEN_SECONDS)")
    ap.add_argument("--pairs", action="store_true", help="list the retrieval misses")
    args = ap.parse_args(argv)

    conn = connect(args.db, readonly=True)
    emb = load_embeddings(conn)
    total_tracks = conn.execute("SELECT count(*) FROM tracks").fetchone()[0]

    print(f"[1] Coverage: {len(emb):,} / {total_tracks:,} tracks have an embedding "
          f"(model {MODEL_NAME} v{EMBEDDER_VERSION})")
    if len(emb) < 5:
        print("    Not enough to test. Run: python main.py --db wanda_music.db --limit 200")
        return 1

    # 2. Self-consistency: split each track's sequence, match halves.
    ids = list(emb)
    rng = random.Random(0)
    same, diff = [], []
    for tid in ids:
        v = emb[tid]
        if len(v) < 4:
            continue
        mid = len(v) // 2
        same.append(best_cosine(v[:mid], v[mid:]))
        other = emb[rng.choice([x for x in ids if x != tid])]
        diff.append(best_cosine(v[:mid], other))
    same, diff = np.array(same), np.array(diff)
    print(f"\n[2] Self-consistency (n={len(same)}):")
    print(f"    same track  halves : mean {same.mean():.3f}  p10 {np.percentile(same,10):.3f}")
    print(f"    vs random track     : mean {diff.mean():.3f}  p90 {np.percentile(diff,90):.3f}")
    # a threshold that separates them, if one exists
    order = np.argsort(np.concatenate([same, diff]))
    labels = np.concatenate([np.ones_like(same), np.zeros_like(diff)])[order]
    best_acc = max(
        (labels[:k].tolist().count(0) + labels[k:].tolist().count(1)) / len(labels)
        for k in range(len(labels) + 1)
    )
    print(f"    best separating accuracy: {best_acc:.1%}")

    # 3. Rank-1 retrieval: a window from the middle of each track, matched against
    #    every track the way RecognitionRepository.match does on the phone.
    titles = dict(conn.execute("SELECT id, artist || ' - ' || title FROM tracks").fetchall())
    win = args.window          # query segments, 0.5 s each -> default 12 == 6 s
    catalogue = [(tid, emb[tid]) for tid in ids if len(emb[tid]) >= win + 4]

    top1 = 0
    margins = []
    misses = []
    for tid, v in catalogue:
        start = (len(v) - win) // 2
        query = v[start:start + win]
        ranked = sorted(
            ((score(query, tv), other) for other, tv in catalogue),
            reverse=True,
        )
        best_score, best_id = ranked[0]
        runner = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_id == tid:
            top1 += 1
            margins.append(best_score - runner)
        else:
            misses.append((titles.get(tid, tid), best_score, titles.get(best_id, best_id)))

    n = len(catalogue)
    print(f"\n[3] Rank-1 retrieval (query = {win/2:.0f}s from mid-track, {n} tracks):")
    print(f"    correct #1: {top1}/{n}  ({top1 / n:.1%})")
    if margins:
        m = np.array(margins)
        print(f"    winner's lead over #2 (hits): mean {m.mean():.3f}  p10 {np.percentile(m, 10):.3f}")
    if args.pairs and misses:
        print("\n    misses (query track -> what it matched instead):")
        for got, s, wrong in misses[:20]:
            print(f"      {s:.3f}  {got}   ->   {wrong}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
