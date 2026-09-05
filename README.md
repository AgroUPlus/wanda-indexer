# Wanda Desktop Indexer

Computes **neural audio fingerprints** — plus 6D KNN features, and the legacy
landmark hashes where the database still has a table for them — for your music
library on a PC, and writes them into the same SQLite database the Wanda Android
app uses, so the phone gets instant recognition and Smart Radio without doing
the work itself.

The phone can do this on its own, one streamed track at a time, at roughly 35
seconds a track. Here it is closer to one *per* second, because the network is
the only real cost and six of them run at once: a 1400-track library is about
twenty minutes instead of thirteen hours.

## Quick start

```bash
pip install -r requirements.txt      # numpy, yt-dlp, ai-edge-litert
sudo apt install ffmpeg              # required; brew install ffmpeg on macOS

cp .env.example .env                 # then fill in your Navidrome details

./run.sh --check                     # verify the environment and database
./run.sh --dry-run                   # see what would be indexed
./run.sh                             # index everything
```

On Windows, use `run_windows.bat` (same arguments), or `sync_indexer.bat` for
the guided pull → index → push workflow.

## What it writes

`track_embeddings` is the one that matters: a 128-dimensional vector per half
second of audio, from the same `wanda_embedder.tflite` the app downloads, plus
one summary vector per 30-second chunk that the phone uses to shortlist a
search before it opens anything. Everything else here predates it and is written
only where the database still has the table — a current app has dropped
`fingerprints` and the recording tables entirely.

Two details are contracts with the app, not choices, and `core/embedder.py`
documents both: **whole tracks** (indexing used to stop at 60 seconds, which is
why a clip from a song's third minute could not be recognised and why a reported
position could never exceed 59 seconds), and **int8 storage** at a scale of 255
(a quarter the size, with a worst-case effect on a match score of 0.002 against
thresholds spaced 0.04 apart).

`tools/check_parity.py` is how you confirm the two sides agree — it feeds the
exact clip the phone captured through this code and compares the vectors.

## How it works

Indexing is **network-bound, not CPU-bound**. Resolving and streaming a track
takes seconds; the embedding takes about 1.3s for a minute of audio, and the
legacy DSP about 0.05s. So the pipeline is:

```
   [ I/O pool: 6 threads ]            [ CPU pool: N processes ]
   resolve URL -> ffmpeg decode  -->  neural embedding (whole track)
   (self-throttling, cached)          one STFT -> landmarks
                                              -> 6D features
                                              -> recording fingerprint
                                                    |
                                          checkpoint every 25 tracks
```

Each CPU worker builds its own TFLite interpreter, lazily. An interpreter holds
native state that does not survive `fork`, so one created before the pool starts
is a segfault waiting for the first worker to use it.

- **One STFT per track**, shared by all three analyses instead of three separate
  transforms (`core/spectrogram.py`).
- **Adaptive rate limiting**: an HTTP 429 from YouTube halves network
  concurrency and backs off; sustained success eases it back up.
- **Resolved URLs are cached** in `.wanda-cache/`, so a retry or rerun costs no
  yt-dlp calls.
- **Local staging**: when the database is on a slow mount (a `/mnt/c` path under
  WSL), indexing runs against a local copy and writes back once at the end.
  Inserting millions of indexed rows straight onto DrvFs is orders of magnitude
  slower. Disable with `--no-stage`.

## Feedback

On a terminal you get a live dashboard: overall progress and ETA, what each
worker is doing *right now* and for how long, throttle state, and a running
failure tally by reason.

```
Wanda Indexer  ·  1355 tracks  ·  io 6/6  ·  cpu 8  ·  elapsed 04:12  ·  eta 21:38
[████████████████░░░░░░░░░░░░░░] 512/1355  37.8%   1.90 trk/s
 ok 498   failed 14   committed 500      [CHECKPOINT] 500 tracks written
────────────────────────────────────────────────────────────────────────────────
 io#1   resolve    ytm Melanie Martinez - DEATH                            2.4s
 io#2   decode     nav Sabrina Carpenter - Taste                          11.2s
 io#3   embedding  ytm natori - Overdose                                   1.4s
────────────────────────────────────────────────────────────────────────────────
 failures  RATE_LIMITED 9 · UNAVAILABLE 3 · DECODE_TIMEOUT 2
```

It degrades automatically to one line per track when output is piped to a file,
`TERM` is dumb, `NO_COLOR` is set, or the terminal is too narrow. Force it with
`--plain`, and use `--ascii` where Unicode is not available.

`--log-file run.jsonl` records one JSON object per track — stage timings,
status, failure reason — in either mode.

## Database safety

Writes use WAL with `synchronous = NORMAL`. **Do not reintroduce
`synchronous = OFF` / `journal_mode = MEMORY`**: that combination is what
corrupted the `fingerprints` B-tree in the shipped database, because the indexer
commits checkpoints and is expected to be interrupted.

If a database is already damaged:

```bash
./run.sh --repair
```

This backs up the original (`.corrupt-<timestamp>.bak`, never modified), copies
every readable row into a fresh file by scanning tables in rowid order — which
bypasses corrupt secondary indexes — rebuilds all indexes, verifies the result
with `integrity_check`, and only then swaps it in. It does not need the
`sqlite3` CLI. On the shipped database this recovered all 1,413,962 rows.

`--check` and every push refuse to touch a database that fails its integrity
check.

## Interrupting

Ctrl+C once: in-flight tracks finish, the buffer is committed, the WAL is folded
back in, and the process exits 130. Ctrl+C twice: immediate exit. Either way the
database is left consistent, and completed work up to the last checkpoint is
kept.

## Useful options

| Option | Purpose |
| --- | --- |
| `--check` | Preflight only: hardware, tools, DB integrity, pending counts |
| `--repair` | Rebuild a corrupt database |
| `--dry-run` | List what would be indexed; no network |
| `--source YTMUSIC` | Restrict to one source (repeatable) |
| `--limit N` | Process at most N tracks |
| `--io-workers N` | Network concurrency ceiling (default 6) |
| `--cpu-workers N` | DSP processes (0 = auto, 1 = inline) |
| `--cookies-from-browser chrome` | Pass cookies to yt-dlp; largely eliminates 429s |
| `--map-local-path FROM=TO` | Use already-downloaded audio instead of the network |
| `--no-stage` | Write directly to the database, no local working copy |

### Using downloaded audio

The database stores Android paths (`/data/user/0/…/files/downloads/ytm_*.opus`)
that do not exist on a desktop, so those tracks fall through to the network. If
you pull that folder locally:

```bash
./run.sh --map-local-path /data/user/0/com.wander.android.debug/files/downloads=/home/me/wanda-audio
```

## Tests

```bash
bash tests/run_all.sh            # offline suites
bash tests/run_all.sh --online   # plus a real end-to-end track
```

`tests/test_golden_dsp.py` pins the DSP output against stored hashes. The
Python analyses must stay **bit-compatible** with the Kotlin implementations
(`Fingerprinter.kt`, `FeatureExtractor.kt`, `RecordingFingerprinter.kt`) or the
phone will not match what the desktop computed. If that test fails after a
change, the change is wrong — do not regenerate the fixtures to make it pass
unless the Kotlin side changed too.

## Credentials

Credentials live in `.env` (gitignored), never in the launcher scripts. The
Navidrome password used to be hardcoded in `sync_indexer.bat`; if that file was
ever shared or synced, rotate it.
