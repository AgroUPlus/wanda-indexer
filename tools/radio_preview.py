"""Preview Smart Radio against the indexed features, without a phone.

Mirrors the app's ranking exactly: the weights and MAX_STEP below are copied
from AcousticFeatures.kt / SmartRadioBuilder.kt. If those change in the app,
change them here too, or this stops predicting what the phone will do.

    python3 tools/radio_preview.py --diagnose
    python3 tools/radio_preview.py --seed "Konbini" --queue 12
    python3 tools/radio_preview.py --seed "someday" --neighbours 8
    python3 tools/radio_preview.py --duplicates
"""
import argparse
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db_sync import connect

# --- copied from AcousticFeatures.kt -------------------------------------
TEMPO_WEIGHT = 1.6
ENERGY_WEIGHT = 1.3
BRIGHTNESS_WEIGHT = 0.8
DANCE_WEIGHT = 1.0
KEY_WEIGHT = 0.6
MIN_BPM = 60.0
MAX_BPM = 180.0
# --- copied from SmartRadioBuilder.kt ------------------------------------
MAX_STEP = 0.55
EXPLORATION_SHARE = 0.35


class Track:
    __slots__ = ("id", "artist", "title", "source", "album", "genre", "vector")

    def __init__(self, row):
        (self.id, self.artist, self.title, self.source, self.album, self.genre,
         tempo, energy, brightness, dance, key_x, key_y) = row
        self.vector = (tempo, energy, brightness, dance, key_x, key_y)

    @property
    def bpm(self) -> int:
        return int(round(MIN_BPM + self.vector[0] * (MAX_BPM - MIN_BPM)))

    @property
    def energy_pct(self) -> int:
        return int(round(self.vector[1] * 100))

    def label(self, width: int = 52) -> str:
        text = f"{self.artist} - {self.title}"
        return text if len(text) <= width else text[: width - 1] + "…"


def distance(a, b) -> float:
    """Weighted Euclidean, matching AcousticFeatures.distanceTo."""
    total = TEMPO_WEIGHT * (a[0] - b[0]) ** 2
    total += ENERGY_WEIGHT * (a[1] - b[1]) ** 2
    total += BRIGHTNESS_WEIGHT * (a[2] - b[2]) ** 2
    total += DANCE_WEIGHT * (a[3] - b[3]) ** 2
    # The key pair is one axis, not two.
    total += KEY_WEIGHT * ((a[4] - b[4]) ** 2 + (a[5] - b[5]) ** 2)
    return math.sqrt(total)


def load_tracks(db_path: str):
    conn = connect(db_path, readonly=True)
    try:
        rows = conn.execute(
            """
            SELECT f.trackId, t.artist, t.title, t.source, t.album, t.genre,
                   f.tempo, f.energy, f.brightness, f.danceability, f.keyX, f.keyY
            FROM track_features f
            JOIN tracks t ON t.id = f.trackId
            ORDER BY t.artist, t.title;
            """
        ).fetchall()
    finally:
        conn.close()
    return [Track(row) for row in rows]


def find_seed(tracks, query: str):
    needle = query.lower()
    matches = [t for t in tracks if needle in f"{t.artist} {t.title}".lower()]
    if not matches:
        return None, []
    return matches[0], matches


def cmd_neighbours(tracks, seed, count: int) -> None:
    print(f"SEED  {seed.label(60)}   {seed.bpm} BPM · energy {seed.energy_pct}%")
    print()
    ranked = sorted(
        ((distance(seed.vector, other.vector), other) for other in tracks if other.id != seed.id),
        key=lambda pair: pair[0],
    )
    for dist, other in ranked[:count]:
        flag = "" if dist <= MAX_STEP else "   beyond MAX_STEP"
        print(f"  {dist:.3f}  {other.label():<52} {other.bpm:>3} BPM  e{other.energy_pct:>3}%{flag}")


def cmd_queue(tracks, seed, count: int) -> None:
    """The greedy walk from SmartRadioBuilder.walkFrom."""
    print(f"SEED  {seed.label(60)}   {seed.bpm} BPM · energy {seed.energy_pct}%")
    print()
    remaining = [t for t in tracks if t.id != seed.id]
    position = seed.vector
    artists = Counter([seed.artist])

    for index in range(count):
        if not remaining:
            break
        best = min(remaining, key=lambda t: distance(position, t.vector))
        step = distance(position, best.vector)
        if step > MAX_STEP:
            print(f"  -- walk stops: nearest remaining is {step:.3f} > MAX_STEP ({MAX_STEP})")
            print("     the app fills the rest from the exploration share "
                  f"({EXPLORATION_SHARE:.0%} of the queue)")
            break
        remaining.remove(best)
        position = best.vector
        artists[best.artist] += 1
        print(f"  {index + 1:>2}. +{step:.3f}  {best.label(54):<54} {best.bpm:>3} BPM  e{best.energy_pct:>3}%")

    repeated = [(name, n) for name, n in artists.most_common() if n > 1]
    if repeated:
        summary = ", ".join(f"{name} x{n}" for name, n in repeated[:4])
        print(f"\n  artist repeats in queue: {summary}")


def cmd_duplicates(tracks) -> None:
    """Near-identical vectors: usually two uploads of one recording."""
    def normalise(title: str) -> str:
        stripped = re.sub(
            r"\((?:audio|visualiz\w+|lyrics?|official[^)]*|live[^)]*|remaster[^)]*)\)",
            "", title, flags=re.I,
        )
        return re.sub(r"\W+", "", stripped.lower())

    groups = {}
    for track in tracks:
        groups.setdefault((track.artist.lower(), normalise(track.title)), []).append(track)

    found = False
    print("Same recording, different uploads (distance should be near zero):\n")
    for group in groups.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                found = True
                dist = distance(group[i].vector, group[j].vector)
                print(f"  {dist:.4f}  {group[i].title[:34]:<34} vs {group[j].title[:34]}")
    if not found:
        print("  none found by title matching")


def cmd_diagnose(tracks) -> None:
    """Whether each axis is actually carrying information."""
    count = len(tracks)
    print(f"{count} tracks have feature vectors.\n")
    if count < 2:
        print("Not enough data to analyse. Index more tracks first.")
        return

    axes = ("tempo", "energy", "brightness", "danceability", "keyX", "keyY")
    weights = (TEMPO_WEIGHT, ENERGY_WEIGHT, BRIGHTNESS_WEIGHT, DANCE_WEIGHT,
               KEY_WEIGHT, KEY_WEIGHT)

    print("Axis utilisation — an axis with little spread cannot rank anything,")
    print("whatever weight it carries:\n")
    print(f"  {'axis':<13}{'weight':>7}{'min':>8}{'max':>8}{'stdev':>8}   effective")
    for index, (name, weight) in enumerate(zip(axes, weights)):
        values = [t.vector[index] for t in tracks]
        low, high = min(values), max(values)
        mean = sum(values) / count
        stdev = math.sqrt(sum((v - mean) ** 2 for v in values) / count)
        # What the axis contributes to a typical distance: weight x spread.
        effective = weight * stdev
        bar = "#" * min(30, int(effective * 100))
        print(f"  {name:<13}{weight:>7.1f}{low:>8.3f}{high:>8.3f}{stdev:>8.3f}   {bar} {effective:.3f}")

    tempos = [t.vector[0] for t in tracks]
    floor = sum(1 for v in tempos if v <= 0.02)
    ceiling = sum(1 for v in tempos if v >= 0.98)
    print(f"\nTempo clamping: {floor} at the {MIN_BPM:.0f} BPM floor "
          f"({floor / count:.0%}), {ceiling} at the {MAX_BPM:.0f} BPM ceiling "
          f"({ceiling / count:.0%})")

    buckets = Counter(int(t.bpm // 10) * 10 for t in tracks)
    print("\nBPM distribution:")
    widest = max(buckets.values())
    for low in sorted(buckets):
        bar = "#" * int(buckets[low] / widest * 40)
        print(f"  {low:>3}-{low + 9:<3} {bar} {buckets[low]}")
    lowest = sum(n for bpm, n in buckets.items() if bpm < 70)
    if lowest / count > 0.2:
        print(f"\n  ! {lowest / count:.0%} of tracks sit below 70 BPM. Some of that is real,")
        print("    but a pile-up here usually means the tempo estimator is locking onto")
        print("    the half-time pulse. Tempo carries the heaviest weight, so this is")
        print("    the single largest quality lever available.")

    pairs = [
        distance(tracks[i].vector, tracks[j].vector)
        for i in range(count) for j in range(i + 1, count)
    ]
    pairs.sort()
    near = sum(1 for d in pairs if d <= MAX_STEP)
    print(f"\nPairwise distances over {len(pairs):,} pairs:")
    print(f"  min {pairs[0]:.3f} | p10 {pairs[len(pairs) // 10]:.3f} | "
          f"median {pairs[len(pairs) // 2]:.3f} | max {pairs[-1]:.3f}")
    print(f"  {near / len(pairs):.1%} of pairs are within MAX_STEP ({MAX_STEP}), "
          "so neighbours are easy to find")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="wanda_music.db")
    parser.add_argument("--seed", default="", help="Substring of an artist or title")
    parser.add_argument("--neighbours", type=int, default=0, help="Show N nearest tracks")
    parser.add_argument("--queue", type=int, default=0, help="Build a radio queue of N tracks")
    parser.add_argument("--duplicates", action="store_true", help="Find near-identical vectors")
    parser.add_argument("--diagnose", action="store_true", help="Report axis health")
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"[ERROR] Database not found: {args.db}")
        return 1

    tracks = load_tracks(args.db)
    if not tracks:
        print("No track_features rows yet. Run the indexer first.")
        return 1

    if args.duplicates:
        cmd_duplicates(tracks)
        return 0
    if args.diagnose or not (args.seed or args.neighbours or args.queue):
        cmd_diagnose(tracks)
        return 0

    if not args.seed:
        print("[ERROR] --neighbours and --queue need a --seed.")
        return 1

    seed, matches = find_seed(tracks, args.seed)
    if seed is None:
        print(f"[ERROR] No indexed track matches {args.seed!r}.")
        print("        Only tracks with feature vectors can seed a radio.")
        return 1
    if len(matches) > 1:
        print(f"({len(matches)} matches; using the first. "
              f"Others: {', '.join(m.title[:24] for m in matches[1:4])})\n")

    if args.queue:
        cmd_queue(tracks, seed, args.queue)
    else:
        cmd_neighbours(tracks, seed, args.neighbours or 8)
    return 0


if __name__ == "__main__":
    sys.exit(main())
