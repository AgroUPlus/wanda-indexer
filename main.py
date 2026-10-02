"""Wanda desktop indexer.

Computes Shazam-style landmark hashes, 6D KNN features and canonical recording
fingerprints for every track that is missing them, and writes them back into the
Room database the Android app uses.

The work is overwhelmingly network-bound, not CPU-bound: resolving and streaming
a track takes seconds, while the DSP takes ~0.05s. So the pipeline is split into
a small, self-throttling I/O pool feeding a process pool for the maths, rather
than one wide pool of processes all shelling out to yt-dlp at once.
"""
import argparse
import math
import multiprocessing
import os
import signal
import sys
import threading
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    BrokenExecutor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)

from core import config, db_sync, system_info
from core.audio_pipeline import DECODE_OK, decode_audio_to_pcm
from core.progress import ProgressReporter
from core.stream_resolver import OK as RESOLVE_OK
from core.stream_resolver import AdaptiveLimiter, StreamCache, StreamResolver
from sources import navidrome

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

DEFAULT_CACHE_DIR = ".wanda-cache"


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------

def pitch_name_from_coords(key_x: float, key_y: float) -> str:
    """Inverts the circle-of-fifths encoding used for keyX/keyY."""
    if abs(key_x) < 0.05 and abs(key_y) < 0.05:
        return "Unknown"
    angle = math.atan2(key_y, key_x)
    if angle < 0:
        angle += 2 * math.pi
    position = int(round(angle * 12.0 / (2 * math.pi))) % 12
    # position = (pitch * 7) % 12, and 7 is its own inverse mod 12.
    return PITCH_NAMES[(position * 7) % 12]


def track_label(track: dict) -> str:
    artist = track.get("artist") or "Unknown Artist"
    title = track.get("title") or "Unknown Title"
    return f"{artist} - {title}"


# --------------------------------------------------------------------------
# CPU stage (runs in worker processes)
# --------------------------------------------------------------------------

def _init_cpu_worker() -> None:
    """Pin each worker to one BLAS thread.

    NumPy would otherwise spin up a thread pool per process; with 32 processes
    that is 1024 threads fighting over 32 cores.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent owns Ctrl+C


def analyse_samples(samples, needs_landmarks: bool, needs_features: bool,
                    needs_recording: bool, needs_embedding: bool = False,
                    embed_model_path: str = "") -> dict:
    """Runs the DSP over decoded PCM. Executed in a worker process."""
    # Imported here so the parent process never pays for them when --cpu-workers
    # spawns fresh interpreters.
    from core.acoustic_features import extract_features
    from core.fingerprinter import compute_landmarks, compute_recording_fingerprint
    from core.spectrogram import compute_stft, frame_count_for

    timings = {}
    # One STFT, shared by all three consumers instead of three separate ones.
    started = time.time()
    stft = compute_stft(samples, frame_count_for(len(samples)))
    timings["t_stft"] = time.time() - started

    result = {"landmarks": [], "features": None, "recording_fp": b"", "embedding": b""}

    if needs_landmarks:
        started = time.time()
        result["landmarks"] = compute_landmarks(samples, stft)
        timings["t_landmarks"] = time.time() - started

    if needs_features:
        started = time.time()
        result["features"] = extract_features(samples, stft)
        timings["t_features"] = time.time() - started

    if needs_recording:
        started = time.time()
        result["recording_fp"] = compute_recording_fingerprint(samples, stft)
        timings["t_recording"] = time.time() - started

    if needs_embedding:
        # The neural embedder does not use the shared STFT: its mel front-end is
        # baked into the TFLite graph and runs from raw PCM. The interpreter is
        # cached per worker process by core.embedder.
        from core.embedder import compute_embedding
        started = time.time()
        result["embedding"] = compute_embedding(
            samples, model_path=embed_model_path or None
        )
        timings["t_embedding"] = time.time() - started

    result["timings"] = timings
    return result


# --------------------------------------------------------------------------
# I/O stage (runs in threads)
# --------------------------------------------------------------------------

def fetch_audio(track: dict, resolver: StreamResolver, reporter: ProgressReporter,
                key: str, max_seconds: int) -> dict:
    """Resolves a stream URL and decodes it. Both steps release the GIL."""
    outcome = {"track": track, "samples": None, "stream_type": "", "reason": "", "timings": {}}

    # ThreadPoolExecutor names its threads "io_0", "io_1", ...
    reporter.set_worker(key, threading.current_thread().name.replace("_", "#"))
    reporter.stage(key, "resolve")
    started = time.time()
    target, stream_type, reason = resolver.resolve(track)
    outcome["timings"]["t_resolve"] = time.time() - started
    outcome["stream_type"] = stream_type

    if not target or reason != RESOLVE_OK:
        outcome["reason"] = reason if reason != RESOLVE_OK else "NO_STREAM_URL"
        return outcome

    reporter.stage(key, "decode")
    started = time.time()
    samples, decode_reason = decode_audio_to_pcm(target, max_seconds=max_seconds)
    outcome["timings"]["t_decode"] = time.time() - started

    if samples is None or decode_reason != DECODE_OK:
        outcome["reason"] = decode_reason
        return outcome

    outcome["samples"] = samples
    return outcome


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------

class Spinner:
    """Shows that a slow, silent step is still running.

    Long database reads on a DrvFs mount can take a minute or more with nothing
    on screen, which is indistinguishable from a hang.
    """

    def __init__(self, message: str, enabled: bool = True, stream=None):
        self.message = message
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled
        try:
            self.enabled = enabled and self.stream.isatty()
        except (AttributeError, ValueError):
            self.enabled = False
        self._stop = threading.Event()
        self._thread = None
        self._started = 0.0

    def __enter__(self):
        self._started = time.time()
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self):
        frames = "|/-\\"
        index = 0
        while not self._stop.wait(0.25):
            elapsed = time.time() - self._started
            try:
                self.stream.write(
                    f"\r{self.message} {frames[index % len(frames)]} {elapsed:.0f}s\x1b[0K"
                )
                self.stream.flush()
            except (OSError, ValueError):
                return
            index += 1

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            try:
                self.stream.write("\r\x1b[0K")
                self.stream.flush()
            except (OSError, ValueError):
                pass
        return False




def preflight(args, write=print) -> bool:
    """Reports the real environment and DB state. Returns False if unusable."""
    write("=" * 70)
    write("  Wanda Desktop Indexer - preflight")
    write("=" * 70)
    system_info.report()

    if not os.path.exists(args.db):
        write(f"\n[ERROR] Database not found: {args.db}")
        return False

    size_mb = os.path.getsize(args.db) / 1e6
    write(f"\n[DB]    {os.path.abspath(args.db)} ({size_mb:.1f} MB)")

    slow = db_sync.is_slow_mount(args.db)
    if slow:
        write("[DB]    on a slow mount; reads here cost ~100x a local disk")

    with Spinner("[DB]    reading index tables", enabled=slow):
        ok, problems = db_sync.quick_sanity(args.db)

    if not ok:
        write(f"[DB]    CORRUPT -- {problems[0]}")
        for line in problems[1:4]:
            write(f"          {line}")
        write("\n[ERROR] Refusing to run against a corrupt database.")
        write("        Run again with --repair to rebuild it (the original is backed up).")
        return False

    counts = db_sync.count_index_rows(args.db)
    line = (f"[DB]    tracks {counts['tracks']:,} | "
            f"features {counts['track_features']:,} | "
            f"recordings {counts['recording_fingerprints']:,}")
    # Only once there are landmarks to report. A database that has been through the cut-over has
    # no `fingerprints` table at all, and `count_index_rows` returns None for it.
    if counts.get("fingerprints"):
        line += f" | landmarks {counts['fingerprints']:,}"
    if counts.get("track_embeddings") is not None:
        line += f" | embeddings {counts['track_embeddings']:,}"
    write(line)
    if args.check:
        # An explicit check is allowed to pay for the full verification; the
        # indexing path gets it for free on the local working copy instead.
        with Spinner("[DB]    full integrity check", enabled=True):
            ok, problems = db_sync.verify(args.db, thorough=True, log=write,
                                          work_dir=args.work_dir)
        write(f"[DB]    integrity: {'ok' if ok else db_sync.summarise_damage(problems)}")
        if not ok:
            return False

    total_fp, missing_fp = db_sync.sub_hash_gap(args.db)
    if missing_fp:
        write(f"[DB]    {missing_fp:,} of {total_fp:,} recording fingerprints have no sub-hash "
              "index and cannot be matched")
        write("        Run with --backfill-subhashes to rebuild it (no audio needed).")
    elif total_fp:
        write(f"[DB]    recording sub-hash index: complete ({total_fp:,} fingerprints)")

    if not system_info.find_tool("ffmpeg"):
        write("\n[ERROR] ffmpeg is required and was not found on PATH.")
        return False
    if not system_info.find_tool("yt-dlp"):
        write("[WARN]  yt-dlp not found: YouTube Music tracks will be skipped.")
    return True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Wanda desktop fingerprinting / feature indexer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default="wanda_music.db", help="Path to the SQLite database")
    parser.add_argument("--io-workers", type=int, default=6,
                        help="Concurrent network fetches (kept low to avoid HTTP 429)")
    parser.add_argument("--cpu-workers", type=int, default=0,
                        help="DSP worker processes (0 = auto, 1 = run inline)")
    parser.add_argument("--batch-size", type=int, default=25,
                        help="Commit a checkpoint every N tracks")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N tracks (0 = all)")
    parser.add_argument("--source", action="append", default=[], metavar="NAME",
                        help="Only index these sources (e.g. --source NAVIDROME); repeatable")
    parser.add_argument("--seconds", type=int, default=60,
                        help="Seconds of audio to analyse per track")
    parser.add_argument("--embed-model", default="",
                        help="Path to the neural fingerprint model "
                             "(default: models/wanda_embedder.tflite beside this script)")
    parser.add_argument("--no-embed", action="store_true",
                        help="Skip neural audio-embedding extraction even if the model is present")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="Skip the landmark constellation. The neural embedding has replaced "
                             "it for recognition, so computing landmarks costs CPU and grows the "
                             "database by ~18k rows a track for an index nothing reads.")

    parser.add_argument("--navidrome-url", default=os.getenv("NAVIDROME_URL"))
    parser.add_argument("--navidrome-user", default=os.getenv("NAVIDROME_USER"))
    parser.add_argument("--navidrome-pass", default=os.getenv("NAVIDROME_PASS"),
                        help="Prefer the NAVIDROME_PASS env var over this flag")

    parser.add_argument("--cookies-from-browser", default="",
                        help="Pass a browser to yt-dlp (chrome, firefox, edge...) to avoid 429s")
    parser.add_argument("--cookies", default="", help="Netscape cookies file for yt-dlp")
    parser.add_argument("--map-local-path", action="append", default=[], metavar="FROM=TO",
                        help="Rewrite on-device localFilePath prefixes to a local folder")

    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help="Where resolved stream URLs are cached")
    parser.add_argument("--no-cache", action="store_true", help="Ignore the stream URL cache")
    parser.add_argument("--log-file", default="", help="Write structured JSONL progress here")
    parser.add_argument("--plain", action="store_true",
                        help="Force line-by-line output instead of the live dashboard")
    parser.add_argument("--ascii", action="store_true", help="ASCII-only output")

    parser.add_argument("--work-dir", default="",
                        help="Where to put the local working copy (default: ~/.cache/wanda-indexer; "
                             "never /tmp, which is RAM-backed on WSL)")
    parser.add_argument("--no-stage", action="store_true",
                        help="Write directly to the database instead of staging a local copy "
                             "(staging is automatic when the DB is on a slow mount)")
    parser.add_argument("--check", action="store_true", help="Run preflight checks and exit")
    parser.add_argument("--repair", action="store_true",
                        help="Rebuild a corrupt database, preserving all readable rows")
    parser.add_argument("--shrink", action="store_true",
                        help="Convert fingerprints to WITHOUT ROWID (measured ~42%% smaller) "
                             "and VACUUM; requires the app on Android running the matching "
                             "MIGRATION_23_24 (@Database version 24)")
    parser.add_argument("--backfill-subhashes", action="store_true",
                        help="Rebuild the recording sub-hash index from stored fingerprints "
                             "(no audio needed); required for dedupe and inbound Agro sync")
    parser.add_argument("--push", action="store_true",
                        help="Send the finished database to the phone over adb and force-stop "
                             "the app so Room rereads it. The phone's likes, play counts, "
                             "history and drops are read back and merged in first")
    parser.add_argument("--push-overwrite", action="store_true",
                        help="Push without merging, replacing the phone's likes and history "
                             "with this database's copy of them")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be indexed without touching the network")
    return parser.parse_args(argv)


def parse_path_map(entries):
    mapping = []
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--map-local-path expects FROM=TO, got: {entry!r}")
        source, target = entry.split("=", 1)
        mapping.append((source.rstrip("/\\"), os.path.expanduser(target)))
    return mapping


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    # Load .env before argparse, since several defaults read os.getenv.
    env_file = config.load_env_file()
    args = parse_args(argv)
    if env_file:
        print(f"[CONFIG] Loaded credentials from {env_file}")

    if args.repair:
        if not os.path.exists(args.db):
            print(f"[ERROR] Database not found: {args.db}")
            return 1
        ok, problems = db_sync.check_integrity(args.db, thorough=True)
        if ok:
            print("[REPAIR] Database is already healthy; nothing to do.")
            return 0
        print(f"[REPAIR] Damage: {db_sync.summarise_damage(problems)}")
        return 0 if db_sync.repair_db(args.db, work_dir=args.work_dir) else 1

    if args.backfill_subhashes:
        if not os.path.exists(args.db):
            print(f"[ERROR] Database not found: {args.db}")
            return 1
        ok, problems = db_sync.quick_sanity(args.db)
        if not ok:
            print(f"[ERROR] {problems[0]}")
            print("        Run --repair first.")
            return 1
        # Staged for the same reason indexing is: millions of indexed row
        # inserts straight onto a DrvFs mount take hours instead of minutes.
        staging = db_sync.StagedDatabase(args.db, enabled=not args.no_stage,
                                         work_dir=args.work_dir)
        with staging as working_db:
            if staging.corrupt:
                print("\n[ERROR] Database failed its integrity check. Run --repair first.")
                return 1
            db_sync.backfill_sub_hashes(working_db)
            db_sync.finalize_database(working_db)
        return 0

    if args.shrink:
        if not os.path.exists(args.db):
            print(f"[ERROR] Database not found: {args.db}")
            return 1
        ok, problems = db_sync.quick_sanity(args.db)
        if not ok:
            print(f"[ERROR] {problems[0]}")
            print("        Run --repair first.")
            return 1
        before_size = os.path.getsize(args.db)
        staging = db_sync.StagedDatabase(args.db, enabled=not args.no_stage,
                                         work_dir=args.work_dir)
        with staging as working_db:
            if staging.corrupt:
                print("\n[ERROR] Database failed its integrity check. Run --repair first.")
                return 1
            db_sync.shrink_fingerprints_table(working_db)
            db_sync.vacuum(working_db)
            db_sync.finalize_database(working_db)
            ok, problems = db_sync.check_integrity(working_db, thorough=True)
            if not ok:
                print(f"[ERROR] Shrunk database failed integrity check: "
                      f"{db_sync.summarise_damage(problems)}")
                print("        The original database was not modified until this point in the "
                      "staged copy; aborting before writing it back.")
                return 1
        after_size = os.path.getsize(args.db)
        print(f"[SHRINK] {before_size / 1e9:.2f} GB -> {after_size / 1e9:.2f} GB "
              f"({(1 - after_size / before_size) * 100:.0f}% smaller)")
        return 0

    if not preflight(args):
        return 1
    if args.check:
        return 0

    path_map = parse_path_map(args.map_local_path)

    from core import embedder
    args.embed_model = os.path.abspath(args.embed_model) if args.embed_model \
        else embedder.default_model_path()
    want_embeddings = not args.no_embed and embedder.model_available(args.embed_model)
    if args.no_embed:
        print("[INDEXER] Neural embeddings disabled (--no-embed).")
    elif not want_embeddings:
        print(f"[WARN]  Embedding model not found at {args.embed_model}; "
              "embeddings will be skipped. Pass --embed-model or add the file.")
    else:
        print(f"[INDEXER] Neural embedder: {args.embed_model}")

    print("\n[INDEXER] Scanning for tracks that need indexing ...")
    with Spinner("[INDEXER] scanning", enabled=db_sync.is_slow_mount(args.db)):
        pending = db_sync.get_pending_tracks(args.db, want_embeddings=want_embeddings)
    if args.no_landmarks:
        # Cleared here rather than in the query so the pending scan stays one code path. A track
        # whose only outstanding work was landmarks now has none, and decoding it would buy
        # nothing, so it drops out of the run entirely.
        print("[INDEXER] Landmark constellation disabled (--no-landmarks).")
        for track in pending:
            track["needs_landmarks"] = False
        pending = [
            t for t in pending
            if t["needs_features"] or t["needs_recording_fp"] or t.get("needs_embedding")
        ]

    if args.source:
        wanted = {name.upper() for name in args.source}
        pending = [t for t in pending if (t["source"] or "").upper() in wanted]
    if args.limit > 0:
        pending = pending[: args.limit]

    total = len(pending)
    by_source = {}
    for track in pending:
        by_source[track["source"]] = by_source.get(track["source"], 0) + 1
    breakdown = ", ".join(f"{count} {name}" for name, count in sorted(by_source.items()))
    print(f"[INDEXER] {total} track(s) pending" + (f" ({breakdown})" if breakdown else ""))

    if total == 0:
        print("[INDEXER] Everything is already indexed.")
        return 0

    if args.dry_run:
        print("\nFirst tracks that would be processed:")
        for track in pending[:10]:
            needs = [
                name for name, flag in (
                    ("landmarks", track["needs_landmarks"]),
                    ("features", track["needs_features"]),
                    ("recording", track["needs_recording_fp"]),
                    ("embedding", track.get("needs_embedding")),
                ) if flag
            ]
            print(f"  - [{track['source']}] {track_label(track)}  "
                  f"(needs: {', '.join(needs)})")
        if total > 10:
            print(f"  ... and {total - 10} more")
        print("\n[INFO] Dry run only. Re-run without --dry-run to index.")
        return 0

    navidrome_config = None
    if args.navidrome_url and args.navidrome_user and args.navidrome_pass:
        navidrome_config = {
            "url": args.navidrome_url,
            "user": args.navidrome_user,
            "pass": args.navidrome_pass,
        }
    elif any(t["source"] == "NAVIDROME" for t in pending):
        print("[WARN]  Navidrome credentials not set; those tracks will be skipped.")
        print("        Set NAVIDROME_URL / NAVIDROME_USER / NAVIDROME_PASS (see .env.example).")

    navidrome_pending = sum(1 for t in pending if t["source"] == "NAVIDROME")
    if navidrome_config and navidrome_pending:
        # One authenticated ping now beats one opaque decode failure per track.
        ok, message = navidrome.ping(
            navidrome_config["url"], navidrome_config["user"], navidrome_config["pass"]
        )
        if ok:
            print(f"[NAVIDROME] Connected: {message}")
        else:
            print(f"[NAVIDROME] Credentials rejected: {message}")
            print(f"            {navidrome_pending} Navidrome track(s) cannot be indexed.")
            print("            Update NAVIDROME_USER / NAVIDROME_PASS in .env and retry.")
            navidrome_config = None

    cpu_workers = args.cpu_workers or min(8, system_info.cpu_count())
    io_workers = max(1, args.io_workers)

    cache_path = os.path.join(args.cache_dir, "stream_urls.json")
    cache = StreamCache(os.devnull if args.no_cache else cache_path)
    limiter = AdaptiveLimiter(ceiling=io_workers)

    stop_flag = threading.Event()
    resolver = StreamResolver(
        cache=cache, limiter=limiter, navidrome_config=navidrome_config,
        cookies_from_browser=args.cookies_from_browser, cookies_file=args.cookies,
        local_path_map=path_map, stop_flag=stop_flag,
    )

    reporter = ProgressReporter(
        total=total, io_workers=io_workers, cpu_workers=cpu_workers,
        force_plain=args.plain, log_path=args.log_file, ascii_only=args.ascii,
    )
    for source, count in by_source.items():
        reporter.expect(source, count)

    print(f"[START] {total} tracks | {io_workers} network worker(s) | "
          f"{cpu_workers} DSP worker(s) | checkpoint every {args.batch_size}\n")

    # Index against a local copy when the database sits on a slow mount, then
    # write it back once. Inserting millions of indexed landmark rows straight
    # onto a DrvFs path is orders of magnitude slower.
    target_db = args.db
    staging = db_sync.StagedDatabase(args.db, enabled=not args.no_stage,
                                     work_dir=args.work_dir)
    try:
        with staging as working_db:
            if staging.corrupt:
                # The full check runs on the local copy, so damage that the fast
                # preflight probe cannot see still stops the run before any write.
                reporter.stop()
                print("\n[ERROR] The database failed its full integrity check.")
                print("        Run again with --repair to rebuild it.")
                return 1
            args.db = working_db
            exit_code = run_pipeline(args, pending, resolver, limiter, cache, reporter,
                                     io_workers, cpu_workers, stop_flag)
            reporter.stop()
            cache.flush()
            print("[DB] Finalizing database ...")
            db_sync.finalize_database(working_db)
    finally:
        args.db = target_db
        reporter.stop()

    reporter.final_report()
    return exit_code


def run_pipeline(args, pending, resolver, limiter, cache, reporter,
                 io_workers, cpu_workers, stop_flag) -> int:
    buffered_landmarks = {}
    buffered_features = {}
    buffered_recordings = {}
    buffered_embeddings = {}
    buffered_tracks = set()
    buffers = (buffered_landmarks, buffered_features, buffered_recordings,
               buffered_embeddings, buffered_tracks)
    commit_lock = threading.Lock()
    embed_model = getattr(args, "embed_model", "") or ""

    def commit_buffer(force: bool = False) -> None:
        with commit_lock:
            if not buffered_tracks:
                return
            counts = db_sync.batch_insert_index_data(
                args.db, buffered_landmarks, buffered_features, buffered_recordings,
                buffered_embeddings,
            )
            counts["tracks"] = len(buffered_tracks)
            reporter.checkpoint(counts)
            buffered_landmarks.clear()
            buffered_features.clear()
            buffered_recordings.clear()
            buffered_embeddings.clear()
            buffered_tracks.clear()

    # Ctrl+C sets a flag; the loop drains and commits rather than dying mid-write.
    def handle_sigint(signum, frame):
        if stop_flag.is_set():
            reporter.log("[INTERRUPT] Second Ctrl+C: exiting immediately.")
            raise KeyboardInterrupt
        stop_flag.set()
        reporter.log("[INTERRUPT] Finishing in-flight tracks and saving. Ctrl+C again to force.")

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handle_sigint)

    cpu_pool = None
    exit_code = 0
    try:
        with ProgressReporterContext(reporter), ThreadPoolExecutor(
            max_workers=io_workers, thread_name_prefix="io"
        ) as io_pool:
            if cpu_workers > 1:
                cpu_pool = ProcessPoolExecutor(
                    max_workers=cpu_workers, initializer=_init_cpu_worker
                )

            queue = list(pending)
            io_futures = {}
            cpu_futures = {}
            # Bound how much decoded PCM can sit in RAM: 60s @ 8 kHz float32 is
            # ~1.9 MB per track, and the whole library would be gigabytes.
            window = io_workers * 3

            while (queue or io_futures or cpu_futures) and not _should_abort(stop_flag,
                                                                            io_futures,
                                                                            cpu_futures):
                while queue and len(io_futures) + len(cpu_futures) < window and not stop_flag.is_set():
                    track = queue.pop(0)
                    key = track["id"]
                    reporter.track_started(key, "queued", track["source"], track_label(track))
                    future = io_pool.submit(
                        fetch_audio, track, resolver, reporter, key, args.seconds
                    )
                    io_futures[future] = track

                if not io_futures and not cpu_futures:
                    break

                done, _ = wait(list(io_futures) + list(cpu_futures),
                               timeout=0.5, return_when=FIRST_COMPLETED)

                for future in done:
                    if future in io_futures:
                        track = io_futures.pop(future)
                        _handle_fetch_result(
                            future, track, reporter, cpu_pool, cpu_futures, buffers,
                            embed_model,
                        )
                    elif future in cpu_futures:
                        track, timings = cpu_futures.pop(future)
                        cpu_pool = _handle_analysis_result(
                            future, track, timings, reporter,
                            buffered_landmarks, buffered_features, buffered_recordings,
                            buffered_embeddings, buffered_tracks, cpu_pool, cpu_workers,
                        )
                        if len(buffered_tracks) >= args.batch_size:
                            commit_buffer()

            if stop_flag.is_set():
                reporter.log("[INTERRUPT] Stopped early by user request.")
                exit_code = 130

    except KeyboardInterrupt:
        reporter.log("[INTERRUPT] Forced exit.")
        exit_code = 130
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if cpu_pool is not None:
            cpu_pool.shutdown(wait=False, cancel_futures=True)
        try:
            commit_buffer(force=True)
        except Exception as exc:  # a failed final commit must still be reported
            reporter.log(f"[ERROR] Final checkpoint failed: {exc}")
            exit_code = exit_code or 1
        cache.flush()

    if args.push and exit_code == 0:
        # The app still declares the landmark entity, so a database that has had `fingerprints`
        # dropped fails Room's schema check on open — and Room's response to that is to delete
        # the file. Recreate it empty before the database leaves this machine.
        conn = db_sync.connect(args.db)
        try:
            db_sync.ensure_landmarks_table(conn)
            conn.commit()
        finally:
            conn.close()

        # Likes, play counts, history and drops only ever happen on the phone, and they live as
        # columns on `tracks` rather than in tables of their own. Replacing the file to deliver
        # new fingerprints therefore reverts all of it to whatever this copy last captured, so
        # the phone's version is read back and merged in first.
        if not args.push_overwrite:
            phone_copy = args.db + ".phone.tmp"
            if db_sync.pull_database(phone_copy, log=print):
                try:
                    db_sync.merge_user_data(phone_copy, args.db, log=print)
                finally:
                    if os.path.exists(phone_copy):
                        os.remove(phone_copy)
            else:
                print("[MERGE] Could not read the phone's database; refusing to push over its "
                      "likes and history. Re-run with --push-overwrite to push anyway.")
                return 1

        if not db_sync.push_database(args.db, log=print):
            exit_code = 1

    return exit_code


class ProgressReporterContext:
    """Starts the dashboard thread and guarantees it is torn down."""

    def __init__(self, reporter):
        self.reporter = reporter

    def __enter__(self):
        self.reporter.start()
        return self.reporter

    def __exit__(self, *exc):
        return False


def _should_abort(stop_flag, io_futures, cpu_futures) -> bool:
    return stop_flag.is_set() and not io_futures and not cpu_futures


def _handle_fetch_result(future, track, reporter, cpu_pool, cpu_futures, buffers,
                         embed_model: str = "") -> None:
    key = track["id"]
    try:
        outcome = future.result()
    except Exception as exc:
        reporter.track_finished(key, "FAIL", f"FETCH_ERROR: {exc}")
        return

    if outcome["samples"] is None:
        reporter.track_finished(
            key, "FAIL", outcome["reason"],
            {"stream_type": outcome["stream_type"], **outcome["timings"]},
        )
        return

    # "analyse", not "landmarks": this one stage covers every extractor, and naming it after the
    # one that has been removed reported the embedding time as landmark time in the summary.
    reporter.stage(key, "analyse")
    samples = outcome["samples"]
    call = (
        samples,
        bool(track["needs_landmarks"]),
        bool(track["needs_features"]),
        bool(track["needs_recording_fp"]),
        bool(track.get("needs_embedding")),
        embed_model,
    )
    detail = {"stream_type": outcome["stream_type"], **outcome["timings"],
              "samples": int(samples.size)}

    if cpu_pool is None:
        try:
            result = analyse_samples(*call)
        except Exception as exc:
            reporter.track_finished(key, "FAIL", f"DSP_ERROR: {exc}", detail)
            return
        _store_result(track, result, detail, reporter, *buffers)
        return

    cpu_future = cpu_pool.submit(analyse_samples, *call)
    cpu_futures[cpu_future] = (track, detail)


def _handle_analysis_result(future, track, detail, reporter, landmarks_buf, features_buf,
                            recordings_buf, embeddings_buf, tracks_buf, cpu_pool, cpu_workers):
    key = track["id"]
    try:
        result = future.result()
    except BrokenExecutor:
        # A worker died (OOM killer, a segfault in BLAS). Rebuild the pool
        # rather than losing the entire run, and retry this track inline.
        reporter.log("[WARN] DSP worker pool broke; rebuilding it.")
        try:
            cpu_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        cpu_pool = ProcessPoolExecutor(max_workers=cpu_workers, initializer=_init_cpu_worker)
        reporter.track_finished(key, "FAIL", "DSP_WORKER_LOST", detail)
        return cpu_pool
    except Exception as exc:
        reporter.track_finished(key, "FAIL", f"DSP_ERROR: {exc}", detail)
        return cpu_pool

    _store_result(track, result, detail, reporter,
                  landmarks_buf, features_buf, recordings_buf, embeddings_buf, tracks_buf)
    return cpu_pool


def _store_result(track, result, detail, reporter,
                  landmarks_buf, features_buf, recordings_buf, embeddings_buf,
                  tracks_buf) -> None:
    key = track["id"]
    reporter.stage(key, "commit")

    landmarks = result.get("landmarks") or []
    features = result.get("features")
    recording = result.get("recording_fp") or b""
    embedding = result.get("embedding") or b""

    if landmarks:
        landmarks_buf[key] = landmarks
    if features:
        features_buf[key] = features
    if recording:
        recordings_buf[key] = (recording, int(track.get("durationMs") or 0))
    if embedding:
        embeddings_buf[key] = embedding

    if not landmarks and not features and not recording and not embedding:
        reporter.track_finished(key, "FAIL", "NO_OUTPUT", detail)
        return

    tracks_buf.add(key)
    summary = dict(detail)
    summary.update(result.get("timings", {}))
    summary["landmarks"] = len(landmarks)
    if embedding:
        summary["embedding"] = len(embedding) // (4 * 128)
    if features:
        summary["bpm"] = int(round(60 + features["tempo"] * 120))
        summary["energy"] = int(round(features["energy"] * 100))
        summary["key"] = pitch_name_from_coords(features["keyX"], features["keyY"])
    reporter.track_finished(key, "OK", "", summary)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
