"""Test recognition on real, degraded audio -- the question the offline check cannot answer.

`check_embedding_recognition.py` slices its query out of the embedding already stored for the
track, so the query and the target come from the same vector. That measures whether the
catalogue is separable; it cannot measure robustness, because nothing is ever degraded. This
tool closes that gap: it fetches the actual audio, damages it the way a room and a codec would,
re-embeds it, and asks whether the right track still wins.

    python3 tools/field_test_recognition.py --db wanda_music.db --limit 20
    python3 tools/field_test_recognition.py --db wanda_music.db --mix-track --snr 20,10,5,0
    python3 tools/field_test_recognition.py --db wanda_music.db --mix-url "https://..." --snr 10

## This tool is silent

Nothing here plays audio. ffmpeg decodes and encodes to pipes only; there is no ffplay, no
audio device, no output format that could reach one. Degradation is arithmetic on numpy arrays.

## The 60-second window

The index only covers the first 60 seconds of each track (`--seconds 60` on the desktop, and
`indexDeeperWindows` is gone from the app), so a query taken from later in the track has no
counterpart to match and would fail no matter how good the model is. Queries are therefore
drawn from that window, and asking for one outside it is refused rather than silently scored --
otherwise the test would measure the coverage gap and report it as a robustness failure.

## The verdict is the phone's, not rank-1

`EmbeddingRepository` only shows the user a name when the winner clears MIN_SIMILARITY with a
MIN_MARGIN lead. A query that ranks first at 0.03 of margin is a "not found" on the device, so
rank-1 alone overstates recognition. The accepted count is the number that matters.
"""
import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import db_sync
from core.audio_pipeline import DECODE_OK, decode_audio_to_pcm, ffmpeg_path
from core.config import load_env_file
from core.embedder import HOP_SAMPLES, SEGMENT_SAMPLES, compute_embedding, model_available
from core.embedding_match import (
    MIN_MARGIN,
    MIN_SIMILARITY,
    decide,
    load_embeddings,
    rank,
)
from core.stream_resolver import OK as RESOLVE_OK
from core.stream_resolver import AdaptiveLimiter, StreamCache, StreamResolver

SAMPLE_RATE = 8_000

# What the indexer covered. A query drawn past this has nothing to match against.
INDEXED_SECONDS = 60


# --------------------------------------------------------------------------
# Degradations -- arithmetic on decoded arrays, plus one codec round trip
# --------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def mix_at_snr(signal: np.ndarray, interference: np.ndarray, snr_db: float) -> np.ndarray:
    """`signal` with `interference` laid over it at `snr_db`.

    The interference is tiled or trimmed to length and scaled so the ratio of the two RMS levels
    is the requested one. A silent interferer is returned unchanged rather than amplified into
    numerical noise.
    """
    if len(interference) == 0:
        return signal
    if len(interference) < len(signal):
        reps = int(np.ceil(len(signal) / len(interference)))
        interference = np.tile(interference, reps)
    interference = interference[: len(signal)]

    signal_rms, noise_rms = _rms(signal), _rms(interference)
    if signal_rms == 0.0 or noise_rms == 0.0:
        return signal
    wanted = signal_rms / (10.0 ** (snr_db / 20.0))
    mixed = signal + interference * (wanted / noise_rms)

    # Clipped the way a real capture clips, not normalised: rescaling here would hand the model
    # a cleaner signal than a phone microphone ever sees.
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def white_noise(length: int, rng: random.Random) -> np.ndarray:
    state = np.random.default_rng(rng.randrange(2 ** 32))
    return state.standard_normal(length).astype(np.float32)


def opus_round_trip(samples: np.ndarray, bitrate: str = "24k") -> np.ndarray:
    """`samples` after a lossy encode and decode, via pipes.

    Two ffmpeg processes rather than one: encoding and decoding in a single graph would let
    ffmpeg optimise the codec away entirely, which would test nothing.
    """
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return samples

    raw = samples.astype("<f4").tobytes()
    encode = [
        ffmpeg, "-loglevel", "error",
        "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
        "-c:a", "libopus", "-b:a", bitrate, "-f", "ogg", "pipe:1",
    ]
    decode = [
        ffmpeg, "-loglevel", "error", "-i", "pipe:0",
        "-vn", "-sn", "-dn",
        "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1",
    ]
    try:
        encoded = subprocess.run(encode, input=raw, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=120).stdout
        if not encoded:
            return samples
        decoded = subprocess.run(decode, input=encoded, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return samples

    usable = len(decoded) - (len(decoded) % 4)
    if usable <= 0:
        return samples
    return np.frombuffer(decoded[:usable], dtype="<f4").astype(np.float32)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def build_resolver(args) -> StreamResolver:
    navidrome_config = None
    url, user, password = (os.getenv("NAVIDROME_URL"), os.getenv("NAVIDROME_USER"),
                           os.getenv("NAVIDROME_PASS"))
    if url and user and password:
        navidrome_config = {"url": url, "user": user, "pass": password}

    cache_path = os.path.join(args.cache_dir, "stream_urls.json")
    return StreamResolver(
        cache=StreamCache(cache_path),
        limiter=AdaptiveLimiter(ceiling=args.io_workers),
        navidrome_config=navidrome_config,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=args.cookies,
    )


def fetch_samples(track: dict, resolver: StreamResolver, seconds: int):
    """The track's first `seconds` of audio, or `(None, reason)`."""
    target, _, reason = resolver.resolve(track)
    if not target or reason != RESOLVE_OK:
        return None, reason or "NO_STREAM_URL"
    samples, decode_reason = decode_audio_to_pcm(target, max_seconds=seconds)
    if samples is None or decode_reason != DECODE_OK:
        return None, decode_reason
    return samples, ""


def excerpt(samples: np.ndarray, seconds: float, offset: float) -> np.ndarray:
    start = int(offset * SAMPLE_RATE)
    return samples[start: start + int(seconds * SAMPLE_RATE)]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

class Outcome:
    """One query's result, in the terms the phone would report it."""

    def __init__(self, track_id, title, ranked, titles=None):
        accepted, best_id, similarity, margin = decide(ranked)
        titles = titles or {}
        self.title = title
        self.correct = best_id == track_id
        self.accepted = accepted and self.correct
        self.similarity = similarity
        self.margin = margin
        self.matched = titles.get(best_id, best_id)
        # Who the margin was lost to. When a correct match is rejected, this names the row that
        # took the lead away -- usually the same recording stored twice.
        runner_up_id = ranked[1][1] if len(ranked) > 1 else ""
        self.runner_up = titles.get(runner_up_id, runner_up_id)
        # Ranked first but rejected by the gate: the model found it and the thresholds threw it
        # away. Worth separating, because the fix is a threshold, not a better model.
        self.gated_out = self.correct and not accepted


def report(condition: str, outcomes, detail: bool = False) -> None:
    if not outcomes:
        print(f"  {condition:<28} no usable queries")
        return
    n = len(outcomes)
    top1 = sum(o.correct for o in outcomes)
    accepted = sum(o.accepted for o in outcomes)
    gated = sum(o.gated_out for o in outcomes)
    sims = np.array([o.similarity for o in outcomes])
    margins = np.array([o.margin for o in outcomes])
    print(f"  {condition:<28} rank-1 {top1:>3}/{n}  ({top1/n:5.1%})   "
          f"accepted {accepted:>3}/{n}  ({accepted/n:5.1%})   "
          f"sim {sims.mean():.3f}  margin {margins.mean():.3f}"
          + (f"   [{gated} lost to the gate]" if gated else ""))

    if not detail:
        return
    for o in outcomes:
        if o.gated_out:
            print(f"      gated  sim {o.similarity:.3f} margin {o.margin:.3f}  "
                  f"{o.title[:44]:<44} lost the margin to: {o.runner_up[:44]}")
        elif not o.correct:
            print(f"      wrong  sim {o.similarity:.3f}                 "
                  f"{o.title[:44]:<44} matched instead: {o.matched[:44]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="wanda_music.db")
    ap.add_argument("--limit", type=int, default=20, help="how many tracks to test")
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="query length; 6 s is the phone's LISTEN_SECONDS")
    ap.add_argument("--offset", type=float, default=-1,
                    help=f"seconds into the track (default: random within the indexed "
                         f"{INDEXED_SECONDS} s)")
    ap.add_argument("--snr", default="20,10,5,0",
                    help="signal-to-noise ratios in dB to sweep")
    ap.add_argument("--mix-track", action="store_true",
                    help="overlay another library track")
    ap.add_argument("--mix-url", default="",
                    help="overlay the audio of this URL (a video clip works; it is decoded "
                         "with -vn and never played)")
    ap.add_argument("--noise", action="store_true", help="overlay white noise")
    ap.add_argument("--opus", action="store_true", help="add a 24 kbit/s opus round trip")
    ap.add_argument("--clean-only", action="store_true",
                    help="skip every degradation; measures the fetch-and-re-embed path alone")
    ap.add_argument("--detail", action="store_true",
                    help="Name each failure and, for a correct match the gate threw away, "
                         "the row that took the margin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--io-workers", type=int, default=4)
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--cookies-from-browser", default="")
    ap.add_argument("--cookies", default="")
    args = ap.parse_args(argv)

    if args.offset >= 0 and args.offset + args.seconds > INDEXED_SECONDS:
        print(f"[ERROR] --offset {args.offset} + --seconds {args.seconds} runs past the "
              f"{INDEXED_SECONDS} s that are indexed.")
        print("        Nothing beyond that exists in track_embeddings, so the query could only "
              "fail, and the failure would say nothing about robustness.")
        return 2

    if not model_available():
        print("[ERROR] Embedder model not found. Expected models/wanda_embedder.tflite.")
        return 2
    if not ffmpeg_path():
        print("[ERROR] ffmpeg not found; audio cannot be decoded.")
        return 2

    load_env_file()
    rng = random.Random(args.seed)

    conn = db_sync.connect(args.db, readonly=True)
    embeddings = load_embeddings(conn)
    if len(embeddings) < 5:
        print("[ERROR] Not enough embeddings to rank against. Index the library first.")
        return 1

    rows = conn.execute(
        """
        SELECT t.id, t.sourceTrackId, t.source, t.title, t.artist,
               t.streamUri, t.localFilePath
        FROM tracks t
        JOIN track_embeddings te ON te.trackId = t.id
        WHERE t.isLive = 0
        """
    ).fetchall()
    conn.close()

    catalogue = [(tid, vectors) for tid, vectors in embeddings.items() if len(vectors) > 0]
    print(f"[INDEX] {len(catalogue):,} tracks to rank against "
          f"(gate: similarity >= {MIN_SIMILARITY}, margin >= {MIN_MARGIN})")

    tracks = [dict(zip(("id", "sourceTrackId", "source", "title", "artist",
                        "streamUri", "localFilePath"), r)) for r in rows]
    # Every indexed track, not only the sampled ones: a query can lose its margin to any row in
    # the catalogue, and naming it is the point of the detail output.
    titles = {t["id"]: f"{t['artist']} - {t['title']}" for t in tracks}
    rng.shuffle(tracks)
    tracks = tracks[: args.limit]

    resolver = build_resolver(args)
    snrs = [float(s) for s in args.snr.split(",") if s.strip()]

    # Fetched once per track and reused for every condition: re-downloading per SNR would make
    # the sweep network-bound and change nothing about the audio.
    print(f"[FETCH] {len(tracks)} track(s), first {INDEXED_SECONDS} s each ...")
    fetched = []
    for track in tracks:
        started = time.time()
        samples, reason = fetch_samples(track, resolver, INDEXED_SECONDS)
        label = f"{track['artist']} - {track['title']}"
        if samples is None:
            print(f"  skip  {label[:58]:<58} {reason}")
            continue
        fetched.append((track, samples))
        print(f"  ok    {label[:58]:<58} {time.time() - started:5.1f}s")

    if not fetched:
        print("[ERROR] Nothing could be fetched.")
        return 1

    interferer = None
    if args.mix_url:
        interferer, reason = decode_audio_to_pcm(args.mix_url, max_seconds=INDEXED_SECONDS)
        if interferer is None:
            print(f"[ERROR] Could not decode --mix-url: {reason}")
            return 1
        print(f"[MIX]   overlay from URL: {len(interferer) / SAMPLE_RATE:.0f}s decoded")

    def query_for(track, samples):
        if args.offset >= 0:
            offset = args.offset
        else:
            ceiling = max(0.0, min(len(samples) / SAMPLE_RATE, INDEXED_SECONDS) - args.seconds)
            offset = rng.uniform(0.0, ceiling)
        return excerpt(samples, args.seconds, offset)

    def evaluate(clip, track):
        blob = compute_embedding(clip)
        query = np.frombuffer(blob, dtype=">f4").reshape(-1, 128).astype(np.float32)
        if len(query) == 0:
            return None
        return Outcome(track["id"], f"{track['artist']} - {track['title']}",
                       rank(query, catalogue), titles)

    print(f"\n[TEST]  query {args.seconds:.0f}s, drawn from the first {INDEXED_SECONDS}s\n")

    clean = []
    for track, samples in fetched:
        clip = query_for(track, samples)
        if len(clip) < SEGMENT_SAMPLES:
            continue
        outcome = evaluate(clip, track)
        if outcome:
            clean.append(outcome)
    report("clean (re-fetched audio)", clean, args.detail)

    if args.opus:
        outcomes = []
        for track, samples in fetched:
            clip = query_for(track, samples)
            if len(clip) < SEGMENT_SAMPLES:
                continue
            outcome = evaluate(opus_round_trip(clip), track)
            if outcome:
                outcomes.append(outcome)
        report("opus 24k round trip", outcomes, args.detail)

    if not args.clean_only:
        for snr in snrs:
            if args.noise:
                outcomes = []
                for track, samples in fetched:
                    clip = query_for(track, samples)
                    if len(clip) < SEGMENT_SAMPLES:
                        continue
                    noisy = mix_at_snr(clip, white_noise(len(clip), rng), snr)
                    outcome = evaluate(noisy, track)
                    if outcome:
                        outcomes.append(outcome)
                report(f"white noise @ {snr:g} dB", outcomes, args.detail)

            if args.mix_track:
                outcomes = []
                for track, samples in fetched:
                    clip = query_for(track, samples)
                    if len(clip) < SEGMENT_SAMPLES:
                        continue
                    others = [s for t, s in fetched if t["id"] != track["id"]]
                    if not others:
                        continue
                    other = rng.choice(others)
                    mixed = mix_at_snr(clip, excerpt(other, args.seconds, 0.0), snr)
                    outcome = evaluate(mixed, track)
                    if outcome:
                        outcomes.append(outcome)
                report(f"another track @ {snr:g} dB", outcomes, args.detail)

            if interferer is not None:
                outcomes = []
                for track, samples in fetched:
                    clip = query_for(track, samples)
                    if len(clip) < SEGMENT_SAMPLES:
                        continue
                    mixed = mix_at_snr(clip, interferer, snr)
                    outcome = evaluate(mixed, track)
                    if outcome:
                        outcomes.append(outcome)
                report(f"mix-url @ {snr:g} dB", outcomes, args.detail)

    print("\n  rank-1   = the right track scored highest")
    print("  accepted = and cleared the phone's similarity and margin gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
