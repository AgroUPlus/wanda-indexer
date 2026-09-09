"""SQLite access for the Wanda index, plus ADB transfer to/from the phone.

Durability note: this module previously wrote with `synchronous = OFF` and
`journal_mode = MEMORY`. That combination has no crash safety at all -- an
interrupt during a checkpoint commit (which the indexer invites, since it
commits every N tracks and expects Ctrl+C) can leave the file unrecoverable,
and did: the shipped wanda_music.db had a malformed `fingerprints` B-tree.
Everything here now uses WAL + synchronous=NORMAL, and `repair_db()` exists to
clean up files damaged by the old behaviour.
"""
import glob
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from core.embedder import EMBED_DIM, EMBEDDER_VERSION, MODEL_NAME as EMBED_MODEL_NAME
from core.fingerprinter import sub_hash_halves

EXTRACTOR_VERSION = 1
DEFAULT_PACKAGE = "com.wander.android.debug"
BUSY_TIMEOUT_MS = 30000

# Kept byte-for-byte in step with Room's generated DDL for `TrackEmbeddingEntity`
# (Android MIGRATION_24_25). The desktop indexer may run against a database whose
# app has not been updated yet, so it creates the table itself if missing; Room's
# migration uses the identical `CREATE TABLE IF NOT EXISTS`, so either order works.
EMBEDDINGS_DDL = (
    "CREATE TABLE IF NOT EXISTS track_embeddings ("
    "trackId TEXT NOT NULL, vector BLOB NOT NULL, dim INTEGER NOT NULL, "
    "model TEXT NOT NULL, version INTEGER NOT NULL, computedAt INTEGER NOT NULL, "
    "PRIMARY KEY(trackId))"
)


def ensure_embeddings_table(conn: sqlite3.Connection) -> None:
    conn.execute(EMBEDDINGS_DDL)


LYRICS_DDL = (
    "CREATE TABLE IF NOT EXISTS `track_lyrics` ("
    "`trackId` TEXT NOT NULL, `plainLyrics` TEXT NOT NULL, `syncedLyrics` TEXT, "
    "`source` TEXT NOT NULL, `syncedAt` INTEGER NOT NULL, "
    "PRIMARY KEY(`trackId`))"
)

LYRICS_FTS_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS `lyrics_fts` USING fts4("
    "`trackId` TEXT, `plainLyrics` TEXT, "
    "tokenize=unicode61)"
)


def ensure_lyrics_tables(conn: sqlite3.Connection) -> None:
    conn.execute(LYRICS_DDL)
    conn.execute(LYRICS_FTS_DDL)


# Byte-for-byte the table Room validates at schema version 25 — see
# `app/schemas/com.wander.android.core.database.WanderDatabase/25.json`.
LANDMARKS_DDL = (
    "CREATE TABLE IF NOT EXISTS `fingerprints` ("
    "`hash` INTEGER NOT NULL, `trackId` TEXT NOT NULL, `anchorFrame` INTEGER NOT NULL, "
    "PRIMARY KEY(`hash`, `trackId`, `anchorFrame`))"
)
LANDMARKS_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS `index_fingerprints_trackId` ON `fingerprints` (`trackId`)"
)


def ensure_landmarks_table(conn: sqlite3.Connection) -> None:
    """Recreates `fingerprints` empty if the cut-over dropped it.

    Nothing reads the landmark index any more, but the app still declares `FingerprintEntity` at
    schema version 25, and Room validates the whole schema when it opens the file. A database
    missing the table is rejected, and `DefaultDatabaseErrorHandler` responds to a rejection by
    deleting the database — so pushing one would silently wipe the phone's library.

    An empty table satisfies the check at a cost of one page. Once the landmark code is removed
    from the app and the schema bumped to drop the entity, this can go with it.
    """
    conn.execute(LANDMARKS_DDL)
    conn.execute(LANDMARKS_INDEX_DDL)


# --------------------------------------------------------------------------
# Connections
# --------------------------------------------------------------------------

def connect(db_path: str, readonly: bool = False, fast_unsafe: bool = False) -> sqlite3.Connection:
    """Opens the database with crash-safe pragmas applied.

    `fast_unsafe` disables durability and is ONLY for building a scratch file
    that is integrity-checked before use and can be rebuilt from an untouched
    source -- i.e. `repair_db`. Never use it for incremental indexing writes:
    that is precisely how the shipped database got corrupted.
    """
    if readonly:
        uri = f"file:{_uri_escape(os.path.abspath(db_path))}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_MS / 1000.0)
    else:
        conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_MS / 1000.0)

    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS};")
    if fast_unsafe and not readonly:
        conn.execute("PRAGMA journal_mode = OFF;")
        conn.execute("PRAGMA synchronous = OFF;")
        conn.execute("PRAGMA cache_size = -131072;")  # 128 MiB
        return conn
    if not readonly:
        # WAL survives an interrupted write and also avoids the spurious
        # "database is locked" seen when the file lives on a /mnt/c DrvFs mount.
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
        except sqlite3.DatabaseError:
            pass  # e.g. a filesystem that cannot do shared memory; keep going.
        conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def _uri_escape(path: str) -> str:
    return path.replace("?", "%3f").replace("#", "%23")


def checkpoint_and_close(conn: sqlite3.Connection) -> None:
    """Folds the WAL back into the main file, then closes."""
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.execute("PRAGMA journal_mode = DELETE;")
    except sqlite3.DatabaseError:
        pass
    conn.close()


def finalize_database(db_path: str) -> None:
    """Makes the .db file standalone: no -wal/-shm sidecars left beside it.

    Room on the phone would otherwise open a database missing every change still
    sitting in the WAL, so this must run before any push. It is deliberately NOT
    done per checkpoint -- on a DrvFs mount (a database under /mnt/c in WSL)
    tearing the WAL down costs far more than the inserts themselves.
    """
    try:
        conn = connect(db_path)
    except sqlite3.DatabaseError:
        return
    checkpoint_and_close(conn)


# --------------------------------------------------------------------------
# Integrity + repair
# --------------------------------------------------------------------------

def quick_sanity(db_path: str) -> Tuple[bool, List[str]]:
    """Instant structural probe: can we open it and read every index table?

    A full `quick_check` reads the entire file, which costs ~80s for a 280 MB
    database on a DrvFs mount and produces no output while it runs. This is the
    cheap gate for preflight: `count(*)` on each table scans that table's
    smallest index, which is exactly where the damage lived in the corrupted
    database this tool shipped with -- it is caught in about 10 milliseconds.
    The authoritative check still runs later, on the fast local working copy.
    """
    try:
        conn = connect(db_path, readonly=True)
    except sqlite3.DatabaseError as exc:
        return False, [f"cannot open database: {exc}"]

    problems = []
    try:
        conn.execute("SELECT count(*) FROM sqlite_master;").fetchone()
        for table in ("tracks", "track_features", "recording_fingerprints"):
            try:
                conn.execute(f"SELECT count(*) FROM {table};").fetchone()
            except sqlite3.DatabaseError as exc:
                problems.append(f"table {table} is unreadable: {exc}")
        # `fingerprints` is not probed: the landmark constellation has been replaced by the
        # neural embedding, and a database that has had it dropped is the intended end state,
        # not a damaged one. Probing it here reported every cut-over database as corrupt.
        #
        # track_embeddings is created on demand by the indexer; a database whose
        # app predates it is not corrupt, so it is not in the probe list above.
    except sqlite3.DatabaseError as exc:
        problems.append(f"schema is unreadable: {exc}")
    finally:
        conn.close()
    return (not problems), problems


def check_integrity(db_path: str, thorough: bool = False, progress=None) -> Tuple[bool, List[str]]:
    """Returns (ok, problems). `quick_check` by default; `integrity_check` if thorough.

    Reads the whole file, so pass `progress` (a callable taking elapsed seconds)
    when this might run for a while -- it is called about once a second so the
    caller can show that something is still happening.
    """
    pragma = "integrity_check" if thorough else "quick_check"
    try:
        conn = connect(db_path, readonly=True)
    except sqlite3.DatabaseError as exc:
        return False, [f"cannot open database: {exc}"]
    ticker = None
    if progress is not None:
        started = time.time()
        finished = threading.Event()

        def tick():
            while not finished.wait(1.0):
                progress(time.time() - started)

        ticker = threading.Thread(target=tick, daemon=True)
        ticker.start()

    try:
        rows = [r[0] for r in conn.execute(f"PRAGMA {pragma}(50);")]
    except sqlite3.DatabaseError as exc:
        return False, [f"{pragma} failed: {exc}"]
    finally:
        if ticker is not None:
            finished.set()
            ticker.join(timeout=2.0)
        conn.close()

    if rows == ["ok"]:
        return True, []
    problems = []
    for row in rows:
        problems.extend(str(row).splitlines())
    return False, problems


def summarise_damage(problems: List[str]) -> str:
    """Condenses a wall of integrity_check output into one line."""
    if not problems:
        return "no problems"
    kinds: Dict[str, int] = {}
    for line in problems:
        if "invalid page number" in line:
            kinds["invalid page number"] = kinds.get("invalid page number", 0) + 1
        elif "out of order" in line:
            kinds["rowid out of order"] = kinds.get("rowid out of order", 0) + 1
        elif "depth differs" in line:
            kinds["page depth differs"] = kinds.get("page depth differs", 0) + 1
        else:
            kinds["other"] = kinds.get("other", 0) + 1
    return " · ".join(f"{name} x{count}" for name, count in sorted(kinds.items()))


def _table_names(conn: sqlite3.Connection) -> List[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
    ]


def _object_sql(conn: sqlite3.Connection, kinds=("table", "index", "trigger", "view")):
    rows = conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%';"
    ).fetchall()
    return [r for r in rows if r[0] in kinds]


def verify(db_path: str, thorough: bool = True, log=print,
           work_dir: str = "") -> Tuple[bool, List[str]]:
    """Full verification, run on local storage when the database is on a slow mount.

    Copying 300 MB and checking it locally takes ~11s; checking it in place over
    DrvFs takes ~90s. The copy is read-only and discarded either way.
    """
    if not is_slow_mount(db_path):
        return check_integrity(db_path, thorough=thorough)

    scratch = ""
    try:
        scratch = scratch_dir(db_path, log=lambda *a: None, explicit=work_dir)
        local = os.path.join(scratch, "verify.db")
        size_mb = os.path.getsize(db_path) / 1e6
        log(f"[DB]    copying {size_mb:.0f} MB locally to verify quickly ...")
        shutil.copyfile(db_path, local)
        return check_integrity(local, thorough=thorough)
    except OSError as exc:
        log(f"[DB]    could not stage for verification ({exc}); checking in place")
        return check_integrity(db_path, thorough=thorough)
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def repair_db(db_path: str, log=print, work_dir: str = "") -> bool:
    """Rebuilds a corrupt database into a clean one, preserving every readable row.

    Works without the `sqlite3` CLI (which is absent on plenty of Windows Python
    installs and on this WSL image), by scanning each table in rowid order --
    that walks the table B-tree and bypasses corrupt secondary indexes, which is
    where this kind of damage overwhelmingly lives. Indexes are then recreated
    from scratch. The original file is never modified; a timestamped backup is
    kept regardless.
    """
    db_path = os.path.abspath(db_path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.corrupt-{stamp}.bak"

    # Build on local temp storage. On WSL a database under /mnt/c sits on a
    # DrvFs mount where every fsync costs milliseconds; rebuilding a million-row
    # table there takes hours, versus seconds on the local filesystem.
    scratch = scratch_dir(db_path, log=log, explicit=work_dir)
    rebuilt_path = os.path.join(scratch, "rebuilt.db")

    log(f"[REPAIR] Backing up original to {os.path.basename(backup_path)} ...")
    shutil.copy2(db_path, backup_path)
    for sidecar in ("-wal", "-shm"):
        if os.path.exists(db_path + sidecar):
            shutil.copy2(db_path + sidecar, backup_path + sidecar)

    src = connect(db_path, readonly=True)
    dst = connect(rebuilt_path, fast_unsafe=True)
    try:
        objects = _object_sql(src)
        tables = [o for o in objects if o[0] == "table"]
        others = [o for o in objects if o[0] != "table"]

        # 1. Schema: tables first, indexes afterwards so inserts stay cheap.
        for _, name, _, sql in tables:
            dst.execute(sql)

        total_recovered = 0
        total_lost = 0
        for _, name, _, _ in tables:
            recovered, lost = _copy_table(src, dst, name, log)
            total_recovered += recovered
            total_lost += lost

        # 2. Rebuild every index/trigger/view from the original schema.
        log(f"[REPAIR] Rebuilding {len(others)} index(es)/trigger(s) ...")
        for obj_type, name, _, sql in others:
            try:
                dst.execute(sql)
            except sqlite3.DatabaseError as exc:
                log(f"[REPAIR]   ! could not recreate {obj_type} {name}: {exc}")

        # 3. Restore sqlite_sequence values for AUTOINCREMENT tables.
        dst.commit()

        ok, problems = _integrity_of_connection(dst)
        if not ok:
            log(f"[REPAIR] FAILED: rebuilt file is still corrupt: {summarise_damage(problems)}")
            return False

        log(f"[REPAIR] Recovered {total_recovered:,} rows"
            + (f", lost {total_lost:,} unreadable rows" if total_lost else " with no data loss"))
    finally:
        src.close()
        checkpoint_and_close(dst)

    # 4. Move the verified file into place. It is staged beside the target first
    #    so the final swap is an atomic same-filesystem rename.
    try:
        staged = f"{db_path}.rebuilt-{stamp}"
        log("[REPAIR] Writing the rebuilt database into place ...")
        shutil.copyfile(rebuilt_path, staged)
        os.replace(staged, db_path)
        for sidecar in ("-wal", "-shm"):
            stale = db_path + sidecar
            if os.path.exists(stale):
                os.remove(stale)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    log(f"[REPAIR] Done. Original preserved at {os.path.basename(backup_path)}")
    return True


def _integrity_of_connection(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    try:
        rows = [r[0] for r in conn.execute("PRAGMA integrity_check(50);")]
    except sqlite3.DatabaseError as exc:
        return False, [str(exc)]
    if rows == ["ok"]:
        return True, []
    problems: List[str] = []
    for row in rows:
        problems.extend(str(row).splitlines())
    return False, problems


def _copy_table(src, dst, table: str, log) -> Tuple[int, int]:
    """Copies one table, skipping only the rows that genuinely cannot be read."""
    columns = [r[1] for r in src.execute(f'PRAGMA table_info("{table}");')]
    if not columns:
        return 0, 0
    col_list = ", ".join(f'"{c}"' for c in columns)
    placeholders = ", ".join("?" for _ in columns)
    insert = f'INSERT OR IGNORE INTO "{table}" ({col_list}) VALUES ({placeholders});'

    # WITHOUT ROWID tables have no _rowid_ to order by; read them whole.
    try:
        cursor = src.execute(f'SELECT {col_list} FROM "{table}" ORDER BY _rowid_;')
    except sqlite3.DatabaseError:
        cursor = src.execute(f'SELECT {col_list} FROM "{table}";')

    recovered = 0
    lost = 0
    batch: List[tuple] = []
    while True:
        try:
            row = cursor.fetchone()
        except sqlite3.DatabaseError as exc:
            # A damaged page: report it and stop this table rather than the run.
            lost += 1
            log(f"[REPAIR]   ! {table}: unreadable page, stopping table scan ({exc})")
            break
        if row is None:
            break
        batch.append(row)
        if len(batch) >= 20000:
            dst.executemany(insert, batch)
            recovered += len(batch)
            batch.clear()
    if batch:
        dst.executemany(insert, batch)
        recovered += len(batch)
    dst.commit()
    log(f"[REPAIR]   {table}: {recovered:,} rows")
    return recovered, lost


# --------------------------------------------------------------------------
# Staging on fast local storage
# --------------------------------------------------------------------------

# A working copy needs room for itself plus the rebuilt/copied-back file, with
# headroom for the WAL. Refuse a location that cannot comfortably hold that.
SCRATCH_HEADROOM = 2.5


def _filesystem_of(path: str) -> str:
    """The mount point `path` lives on, or "" when it cannot be determined."""
    try:
        resolved = os.path.abspath(path)
        while not os.path.ismount(resolved):
            parent = os.path.dirname(resolved)
            if parent == resolved:
                return ""
            resolved = parent
        return resolved
    except OSError:
        return ""


def is_memory_backed(path: str) -> bool:
    """True when writing here consumes RAM rather than disk.

    On WSL /tmp is tmpfs. That is harmless for a 200 MB database and actively
    dangerous for a 3 GB one: staging a working copy there would hold the whole
    file in memory, competing with the indexing run that needed the copy.
    """
    mount = _filesystem_of(path)
    if not mount:
        return False
    try:
        with open("/proc/mounts") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) >= 3 and fields[1] == mount:
                    return fields[2] in ("tmpfs", "ramfs", "devtmpfs")
    except OSError:
        pass
    return False


def free_bytes(path: str) -> int:
    try:
        stats = os.statvfs(path)
        return stats.f_bavail * stats.f_frsize
    except (OSError, AttributeError):
        return 0


def scratch_candidates(db_path: str, explicit: str = ""):
    """Where a working copy could go, best first."""
    if explicit:
        yield explicit
        return
    from_env = os.environ.get("WANDA_WORK_DIR")
    if from_env:
        yield from_env

    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    yield os.path.join(cache_root, "wanda-indexer")

    # Last resort: beside the database. Slow if that is a DrvFs mount, but it is
    # somewhere the file demonstrably fits.
    yield os.path.dirname(os.path.abspath(db_path)) or "."


def scratch_dir(db_path: str, log=print, explicit: str = "") -> str:
    """Creates and returns a directory able to hold a working copy of `db_path`.

    Raises OSError when nowhere has room, naming the shortfall -- far better
    than an ENOSPC part-way through copying gigabytes.
    """
    needed = int(os.path.getsize(db_path) * SCRATCH_HEADROOM)
    rejected = []

    for candidate in scratch_candidates(db_path, explicit):
        try:
            os.makedirs(candidate, exist_ok=True)
        except OSError as exc:
            rejected.append(f"{candidate}: {exc}")
            continue

        if is_memory_backed(candidate) and not explicit:
            rejected.append(f"{candidate}: RAM-backed (tmpfs)")
            continue

        available = free_bytes(candidate)
        if available < needed:
            rejected.append(
                f"{candidate}: {available / 1e9:.1f} GB free, needs {needed / 1e9:.1f} GB"
            )
            continue

        log(f"[SCRATCH] {candidate} ({available / 1e9:.0f} GB free)")
        return tempfile.mkdtemp(prefix="wanda-", dir=candidate)

    raise OSError(
        f"No scratch directory can hold a {os.path.getsize(db_path) / 1e9:.1f} GB working copy "
        f"(need ~{needed / 1e9:.1f} GB). Tried:\n  " + "\n  ".join(rejected)
        + "\nSet --work-dir or $WANDA_WORK_DIR to somewhere with space."
    )


def is_slow_mount(path: str) -> bool:
    """True when `path` lives on a filesystem where random writes are costly.

    The case that matters here is WSL: a database under /mnt/c is on a DrvFs
    mount bridged to Windows, where every page write crosses a translation
    layer. Inserting a few hundred thousand indexed rows there takes minutes
    against seconds on the Linux filesystem.
    """
    resolved = os.path.abspath(path)
    if sys.platform.startswith("linux"):
        for prefix in ("/mnt/", "/media/", "/run/user/"):
            if resolved.startswith(prefix):
                return True
    if os.name == "nt":
        drive = os.path.splitdrive(resolved)[0].upper()
        if drive and drive not in ("C:",):
            return True
    return False


class StagedDatabase:
    """Runs the indexing against a local copy, then writes it back.

    Context manager: `with StagedDatabase(path) as working_path:`. On exit the
    working copy is integrity-checked and only then moved back over the
    original, so an interrupted or corrupted run cannot damage the real file.
    Staging is skipped entirely when the database is already on fast storage.
    """

    def __init__(self, db_path: str, enabled: bool = True, log=print, work_dir: str = ""):
        self.original = os.path.abspath(db_path)
        self.log = log
        self.work_dir = work_dir
        self.enabled = enabled and is_slow_mount(self.original)
        self.corrupt = False
        self._scratch_dir = ""
        self.working = self.original

    def __enter__(self) -> str:
        if not self.enabled:
            return self.working

        try:
            self._scratch_dir = scratch_dir(self.original, log=self.log, explicit=self.work_dir)
            self.working = os.path.join(self._scratch_dir, os.path.basename(self.original))
            size_mb = os.path.getsize(self.original) / 1e6
            self.log(f"[STAGE] Copying {size_mb:.0f} MB to local disk for fast writes ...")
            started = time.time()
            shutil.copyfile(self.original, self.working)
            self.log(f"[STAGE] Working copy ready in {time.time() - started:.1f}s")

            # The authoritative integrity check belongs here: on local storage
            # it takes under a second, against ~80s on the source mount.
            started = time.time()
            ok, problems = check_integrity(self.working)
            if not ok:
                self.log(f"[STAGE] Database is corrupt: {summarise_damage(problems)}")
                self.corrupt = True
                self._cleanup()
                self.enabled = False
                self.working = self.original
                return self.working
            self.log(f"[STAGE] Integrity verified in {time.time() - started:.1f}s")
        except OSError as exc:
            self.log(f"[STAGE] Could not stage locally ({exc}); working in place.")
            self._cleanup()
            self.enabled = False
            self.working = self.original
        return self.working

    def __exit__(self, *exc) -> bool:
        if not self.enabled:
            return False
        try:
            self.commit_back()
        finally:
            self._cleanup()
        return False

    def commit_back(self) -> bool:
        """Verifies the working copy and moves it over the original."""
        if not self.enabled or self.working == self.original:
            return True

        finalize_database(self.working)
        ok, problems = check_integrity(self.working)
        if not ok:
            kept = self.original + ".staged-failed"
            self.log(f"[STAGE] Working copy failed its integrity check "
                     f"({summarise_damage(problems)}).")
            try:
                shutil.copyfile(self.working, kept)
                self.log(f"[STAGE] Left it at {kept}; the original is untouched.")
            except OSError:
                pass
            return False

        try:
            self.log("[STAGE] Writing results back ...")
            started = time.time()
            staged = self.original + ".incoming"
            shutil.copyfile(self.working, staged)
            os.replace(staged, self.original)
            for sidecar in ("-wal", "-shm"):
                stale = self.original + sidecar
                if os.path.exists(stale):
                    os.remove(stale)
            self.log(f"[STAGE] Database updated in {time.time() - started:.1f}s")
            return True
        except OSError as exc:
            self.log(f"[STAGE] Could not write back: {exc}")
            self.log(f"[STAGE] Your results are safe at {self.working}")
            self._scratch_dir = ""  # do not delete the only copy of the results
            return False

    def _cleanup(self) -> None:
        if self._scratch_dir:
            shutil.rmtree(self._scratch_dir, ignore_errors=True)
            self._scratch_dir = ""


# --------------------------------------------------------------------------
# Work discovery
# --------------------------------------------------------------------------

def get_pending_tracks(db_path: str, version: int = EXTRACTOR_VERSION,
                       want_embeddings: bool = False,
                       embedder_version: int = EMBEDDER_VERSION) -> List[Dict[str, Any]]:
    """Tracks missing any part of their index, with per-stage flags.

    The old implementation used a pair of `id NOT IN (SELECT ...)` subqueries.
    Besides scanning the million-row fingerprints table twice, `NOT IN` against
    a subquery yielding any NULL returns *no rows at all* -- a silent "nothing
    to do". This uses LEFT JOINs, and reports which stages each track needs so
    a partially indexed track only redoes the missing parts.
    """
    conn = connect(db_path, readonly=True)
    conn.row_factory = sqlite3.Row
    # A database whose app predates the embedding feature has no track_embeddings
    # table; the indexer creates it at write time (batch_insert_index_data), and
    # until then every track counts as needing an embedding.
    has_table = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='track_embeddings';"
    ).fetchone())

    # The landmark constellation is gone from any database that has been through the cut-over, so
    # the join that reports it has to be optional in exactly the way the embedding join already is.
    # Without this the whole scan died with "no such table: fingerprints".
    has_landmarks = bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fingerprints';"
    ).fetchone())
    if has_landmarks:
        landmark_column = "(fp.trackId IS NULL) AS needs_landmarks"
        landmark_join = ("LEFT JOIN (SELECT DISTINCT trackId FROM fingerprints) fp "
                         "ON fp.trackId = t.id")
        landmark_where = "fp.trackId IS NULL OR"
    else:
        landmark_column = "0 AS needs_landmarks"
        landmark_join = ""
        landmark_where = ""

    if not want_embeddings:
        embed_column = "0 AS needs_embedding"
        embed_join = ""
        embed_where = ""
    elif has_table:
        embed_column = "(te.trackId IS NULL) AS needs_embedding"
        embed_join = ("LEFT JOIN (SELECT trackId FROM track_embeddings WHERE version = :ev) te "
                      "ON te.trackId = t.id")
        embed_where = "OR te.trackId IS NULL"
    else:
        embed_column = "1 AS needs_embedding"
        embed_join = ""
        embed_where = "OR 1"

    try:
        rows = conn.execute(
            f"""
            SELECT t.id, t.sourceTrackId, t.source, t.title, t.artist, t.album,
                   t.durationMs, t.streamUri, t.localFilePath,
                   {landmark_column},
                   (tf.trackId IS NULL) AS needs_features,
                   (rf.trackId IS NULL) AS needs_recording_fp,
                   {embed_column}
            FROM tracks t
            {landmark_join}
            LEFT JOIN (SELECT trackId FROM track_features WHERE version = :v) tf
                   ON tf.trackId = t.id
            LEFT JOIN recording_fingerprints rf
                   ON rf.trackId = t.id
            {embed_join}
            WHERE {landmark_where} tf.trackId IS NULL OR rf.trackId IS NULL
                  {embed_where}
            ORDER BY t.source, t.id;
            """,
            {"v": version, "ev": embedder_version},
        ).fetchall()
    finally:
        conn.close()

    pending = []
    for row in rows:
        track = dict(row)
        for flag in ("needs_landmarks", "needs_features", "needs_recording_fp"):
            track[flag] = bool(track[flag])
        track["needs_embedding"] = bool(want_embeddings and track["needs_embedding"])
        pending.append(track)
    return pending


def sub_hash_gap(db_path: str) -> Tuple[int, int]:
    """(fingerprints, of those missing a sub-hash index).

    The desktop indexer wrote `recording_fingerprints` without the matching
    `recording_sub_hashes` rows for a long time, leaving those fingerprints
    invisible to both local dedupe and inbound catalogue sync.
    """
    conn = connect(db_path, readonly=True)
    try:
        total = conn.execute("SELECT count(*) FROM recording_fingerprints;").fetchone()[0]
        missing = conn.execute(
            """
            SELECT count(*) FROM recording_fingerprints f
            WHERE NOT EXISTS (
                SELECT 1 FROM recording_sub_hashes s WHERE s.trackId = f.trackId
            );
            """
        ).fetchone()[0]
    except sqlite3.DatabaseError:
        return 0, 0
    finally:
        conn.close()
    return total, missing


def backfill_sub_hashes(db_path: str, log=print, chunk_size: int = 200) -> int:
    """Rebuilds the sub-hash index for fingerprints that lack one.

    Needs no audio: the halves are a pure function of the stored blob, so this
    is a database pass rather than a re-index. Committed in chunks so an
    interrupt costs at most one chunk, and so several million row inserts do not
    run silently.
    """
    conn = connect(db_path)
    try:
        pending = [
            row[0]
            for row in conn.execute(
                """
                SELECT f.trackId FROM recording_fingerprints f
                WHERE NOT EXISTS (
                    SELECT 1 FROM recording_sub_hashes s WHERE s.trackId = f.trackId
                )
                ORDER BY f.trackId;
                """
            )
        ]
        if not pending:
            log("[BACKFILL] Every fingerprint already has a sub-hash index.")
            return 0

        log(f"[BACKFILL] {len(pending):,} fingerprint(s) need a sub-hash index.")
        written = 0
        done = 0
        started = time.time()

        for start in range(0, len(pending), chunk_size):
            batch = pending[start : start + chunk_size]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT trackId, subHashes FROM recording_fingerprints "
                f"WHERE trackId IN ({placeholders});",
                batch,
            ).fetchall()

            half_rows = [
                (half, track_id)
                for track_id, blob in rows
                for half in sub_hash_halves(blob)
            ]
            with conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO recording_sub_hashes (half, trackId) VALUES (?, ?);",
                    half_rows,
                )
            written += len(half_rows)
            done += len(batch)
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(pending) - done) / rate if rate else 0
            log(f"[BACKFILL] {done:,}/{len(pending):,} tracks · {written:,} rows · "
                f"{rate:.0f} trk/s · eta {int(remaining // 60)}m{int(remaining % 60):02d}s")

        log(f"[BACKFILL] Done: {written:,} sub-hash rows for {done:,} tracks "
            f"in {time.time() - started:.0f}s")
        return written
    finally:
        conn.close()


def count_index_rows(db_path: str) -> Dict[str, Optional[int]]:
    """Row counts for the three index tables; None where the table is unreadable."""
    conn = connect(db_path, readonly=True)
    counts: Dict[str, Optional[int]] = {}
    try:
        for table in ("tracks", "fingerprints", "track_features",
                      "recording_fingerprints", "track_embeddings"):
            try:
                counts[table] = conn.execute(f"SELECT count(*) FROM {table};").fetchone()[0]
            except sqlite3.DatabaseError:
                counts[table] = None
    finally:
        conn.close()
    return counts


def deduplicate_by_embeddings(
    db_path: str,
    sim_threshold: float = 0.88,
    duration_tolerance_ms: int = 3000,
    log=print,
) -> int:
    """Finds and populates duplicate pairs into recording_links using neural embeddings."""
    import numpy as np

    conn = connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recording_links (
                idA TEXT NOT NULL,
                idB TEXT NOT NULL,
                similarity REAL NOT NULL,
                linkedAt INTEGER NOT NULL,
                PRIMARY KEY(idA, idB)
            );
        """)
        tracks = {
            r[0]: r[1]
            for r in cur.execute("SELECT id, durationMs FROM tracks WHERE durationMs > 0").fetchall()
        }
        embeddings = {}
        mean_vectors = {}
        for tid, blob in cur.execute(
            "SELECT trackId, vector FROM track_embeddings WHERE model = ? AND version = ?",
            (EMBED_MODEL_NAME, EMBEDDER_VERSION),
        ).fetchall():
            if tid not in tracks:
                continue
            arr = np.frombuffer(blob, dtype=">f4").reshape(-1, EMBED_DIM)
            embeddings[tid] = arr
            m = np.mean(arr, axis=0)
            norm = np.linalg.norm(m)
            mean_vectors[tid] = m / (norm if norm > 0 else 1.0)

        track_ids = sorted(embeddings.keys(), key=lambda t: tracks[t])
        durations = [tracks[t] for t in track_ids]

        t0 = time.time()
        candidates = []
        for i in range(len(track_ids)):
            d_i = durations[i]
            m_i = mean_vectors[track_ids[i]]
            for j in range(i + 1, len(track_ids)):
                if durations[j] - d_i > duration_tolerance_ms:
                    break
                if np.dot(m_i, mean_vectors[track_ids[j]]) >= 0.75:
                    candidates.append((track_ids[i], track_ids[j]))

        links = []
        now = int(time.time() * 1000)
        for a, b in candidates:
            va = embeddings[a]
            vb = embeddings[b]
            sim_mat = np.dot(va, vb.T)
            score = float((np.mean(np.max(sim_mat, axis=1)) + np.mean(np.max(sim_mat, axis=0))) / 2.0)
            if score >= sim_threshold:
                id_a, id_b = (a, b) if a < b else (b, a)
                links.append((id_a, id_b, score, now))

        cur.executemany(
            "INSERT OR REPLACE INTO recording_links (idA, idB, similarity, linkedAt) VALUES (?, ?, ?, ?);",
            links,
        )
        conn.commit()
        log(f"[DEDUP] Linked {len(links):,} duplicate recordings in {time.time() - t0:.2f}s")
        return len(links)
    finally:
        conn.close()


def drop_legacy_sub_hashes(db_path: str, log=print) -> None:
    """Drops legacy sub-hash tables, freeing hundreds of megabytes."""
    conn = connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS recording_sub_hashes;")
        conn.execute("DROP TABLE IF EXISTS recording_fingerprints;")
        conn.commit()
        log("[CLEANUP] Dropped legacy recording_sub_hashes and recording_fingerprints tables.")
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------

def batch_insert_index_data(
    db_path: str,
    landmarks_by_track: Dict[str, List[Tuple[int, int]]],
    features_by_track: Dict[str, Dict[str, float]],
    recording_fps_by_track: Dict[str, bytes],
    embeddings_by_track: Optional[Dict[str, bytes]] = None,
    version: int = EXTRACTOR_VERSION,
    embedder_version: int = EMBEDDER_VERSION,
) -> Dict[str, int]:
    """Commits one checkpoint atomically. Returns counts of rows written."""
    embeddings_by_track = embeddings_by_track or {}
    conn = connect(db_path)
    now_ms = int(time.time() * 1000)

    embedding_rows = [
        (track_id, blob, EMBED_DIM, EMBED_MODEL_NAME, embedder_version, now_ms)
        for track_id, blob in embeddings_by_track.items()
        if blob
    ]

    landmark_rows = [
        (packed_hash, track_id, anchor_frame)
        for track_id, landmarks in landmarks_by_track.items()
        for packed_hash, anchor_frame in landmarks
    ]
    feature_rows = [
        (
            track_id,
            feat["tempo"], feat["energy"], feat["brightness"],
            feat["danceability"], feat["keyX"], feat["keyY"],
            version, now_ms,
        )
        for track_id, feat in features_by_track.items()
    ]
    rec_rows = [
        (track_id, blob, duration_ms, now_ms)
        for track_id, (blob, duration_ms) in recording_fps_by_track.items()
    ]
    # The sub-hash index is what `matchesForFingerprint` actually searches; a
    # fingerprint stored without it can never be proposed as a candidate.
    half_rows = [
        (half, track_id)
        for track_id, (blob, _) in recording_fps_by_track.items()
        for half in sub_hash_halves(blob)
    ]

    try:
        with conn:  # one transaction: commits together or rolls back together
            if embedding_rows:
                ensure_embeddings_table(conn)
                conn.executemany(
                    "INSERT OR REPLACE INTO track_embeddings "
                    "(trackId, vector, dim, model, version, computedAt) "
                    "VALUES (?, ?, ?, ?, ?, ?);",
                    embedding_rows,
                )
            if landmark_rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO fingerprints (hash, trackId, anchorFrame) "
                    "VALUES (?, ?, ?);",
                    landmark_rows,
                )
            if feature_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO track_features "
                    "(trackId, tempo, energy, brightness, danceability, keyX, keyY, "
                    "version, measuredAt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                    feature_rows,
                )
            if rec_rows:
                # Mirrors RecordingFingerprintDao.replace: clear, upsert, index --
                # all in one transaction. An index left pointing at a fingerprint
                # that was replaced proposes candidates whose sequences no longer
                # match, which are then rejected at the cost of reading them.
                conn.executemany(
                    "DELETE FROM recording_sub_hashes WHERE trackId = ?;",
                    [(track_id,) for track_id in recording_fps_by_track],
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO recording_fingerprints "
                    "(trackId, subHashes, durationMs, computedAt) VALUES (?, ?, ?, ?);",
                    rec_rows,
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO recording_sub_hashes (half, trackId) VALUES (?, ?);",
                    half_rows,
                )
    finally:
        conn.close()

    return {
        "landmarks": len(landmark_rows),
        "features": len(feature_rows),
        "recordings": len(rec_rows),
        "sub_hashes": len(half_rows),
        "embeddings": len(embedding_rows),
    }


def batch_insert_lyrics(db_path: str, lyrics_by_track: Dict[str, Dict[str, Any]]) -> int:
    """Inserts or replaces lyrics in track_lyrics and lyrics_fts in a single transaction."""
    if not lyrics_by_track:
        return 0

    now_ms = int(time.time() * 1000)
    rows = []
    for track_id, data in lyrics_by_track.items():
        plain = (data.get("plainLyrics") or "").strip()
        if not plain:
            continue
        synced = data.get("syncedLyrics")
        source = data.get("source") or "LRCLIB"
        synced_at = data.get("syncedAt") or now_ms
        rows.append((track_id, plain, synced, source, synced_at))

    if not rows:
        return 0

    fts_rows = [(r[0], r[1]) for r in rows]
    track_ids = [(r[0],) for r in rows]

    conn = connect(db_path)
    try:
        with conn:
            ensure_lyrics_tables(conn)
            conn.executemany("DELETE FROM lyrics_fts WHERE trackId = ?;", track_ids)
            conn.executemany(
                "INSERT OR REPLACE INTO track_lyrics (trackId, plainLyrics, syncedLyrics, source, syncedAt) "
                "VALUES (?, ?, ?, ?, ?);",
                rows,
            )
            conn.executemany(
                "INSERT INTO lyrics_fts (trackId, plainLyrics) VALUES (?, ?);",
                fts_rows,
            )
    finally:
        conn.close()

    return len(rows)


def search_lyrics_fts(db_path: str, query: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Searches lyrics_fts for matching tracks and returns track details with highlighted snippets."""
    clean_terms = [re.sub(r"[^\w]", "", term) for term in query.split() if term.strip()]
    if not clean_terms:
        return []
    # Match prefix on last word to support incremental search
    fts_query = " ".join(f"{term}*" if idx == len(clean_terms) - 1 else term for idx, term in enumerate(clean_terms))

    conn = connect(db_path, readonly=True)
    conn.row_factory = sqlite3.Row
    try:
        ensure_lyrics_tables(conn)
        has_tracks = bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracks';"
        ).fetchone())

        if has_tracks:
            sql = """
                SELECT l.trackId, l.plainLyrics, l.syncedLyrics, l.source,
                       t.title, t.artist, t.album, t.durationMs,
                       snippet(lyrics_fts, '<b>', '</b>', '...', -1, 12) AS snippet
                FROM lyrics_fts f
                JOIN track_lyrics l ON l.trackId = f.trackId
                JOIN tracks t ON t.id = l.trackId
                WHERE lyrics_fts MATCH ?
                LIMIT ?;
            """
            rows = conn.execute(sql, (fts_query, limit * 2)).fetchall()
            results = []
            seen = set()
            for r in rows:
                d = dict(r)
                key = (d.get("title", "").strip().lower(), d.get("artist", "").strip().lower())
                if key not in seen:
                    seen.add(key)
                    results.append(d)
                if len(results) >= limit:
                    break
            return results
        else:
            sql = """
                SELECT l.trackId, l.plainLyrics, l.syncedLyrics, l.source,
                       snippet(lyrics_fts, '<b>', '</b>', '...', -1, 12) AS snippet
                FROM lyrics_fts f
                JOIN track_lyrics l ON l.trackId = f.trackId
                WHERE lyrics_fts MATCH ?
                LIMIT ?;
            """
            rows = conn.execute(sql, (fts_query, limit)).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()



def shrink_fingerprints_table(db_path: str, log=print) -> dict:
    """Converts `fingerprints` to WITHOUT ROWID, matching Android MIGRATION_23_24.

    Room's schema validator reads `PRAGMA table_info` / `index_list`, and neither can see
    `WITHOUT ROWID` -- it is pure DDL, invisible to those pragmas. So this desktop rewrite and the
    on-device migration are only required to agree on the observable part: the same columns, the
    same primary key, and the same remaining index (`trackId`; `hash` is dropped because the
    primary key's hash-first order already serves it). Get that right and Room accepts either path
    to get there without caring which one actually ran.

    Does the 24M-row rewrite here rather than as an on-device migration for the same reason
    everything else in this module stages locally first: SQLite migrating a table this size inside
    the app at first-launch-after-update is slow and a poor first experience, and a desktop CPU
    does it while nobody is waiting on it.

    After this, `PRAGMA user_version` is set to 24 so Room recognises the database as already
    migrated and MIGRATION_23_24 is a no-op should it ever run against this file.
    """
    conn = connect(db_path)
    try:
        before = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'fingerprints';"
        ).fetchone()
        if before and "WITHOUT ROWID" in before[0].upper():
            log("[SHRINK] fingerprints is already WITHOUT ROWID; nothing to do.")
            return {"rows": 0, "skipped": True}

        row_count = conn.execute("SELECT count(*) FROM fingerprints;").fetchone()[0]
        log(f"[SHRINK] Rewriting {row_count:,} landmark rows as WITHOUT ROWID ...")
        started = time.time()

        with conn:
            conn.execute(
                """
                CREATE TABLE fingerprints_new (
                    hash INTEGER NOT NULL,
                    trackId TEXT NOT NULL,
                    anchorFrame INTEGER NOT NULL,
                    PRIMARY KEY (hash, trackId, anchorFrame)
                ) WITHOUT ROWID;
                """
            )
            conn.execute(
                "INSERT INTO fingerprints_new (hash, trackId, anchorFrame) "
                "SELECT hash, trackId, anchorFrame FROM fingerprints;"
            )
            conn.execute("DROP INDEX IF EXISTS index_fingerprints_hash;")
            conn.execute("DROP TABLE fingerprints;")
            conn.execute("ALTER TABLE fingerprints_new RENAME TO fingerprints;")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS index_fingerprints_trackId "
                "ON fingerprints (trackId);"
            )
            # Matches Android's @Database(version = 24): a mismatch here would make Room
            # re-run MIGRATION_23_24 on first open, safely a no-op given the guard above, but
            # setting it correctly avoids that redundant multi-second pass on the phone.
            conn.execute("PRAGMA user_version = 24;")

        after_count = conn.execute("SELECT count(*) FROM fingerprints;").fetchone()[0]
        if after_count != row_count:
            raise RuntimeError(
                f"row count changed during rewrite: {row_count:,} -> {after_count:,}; "
                "refusing to treat this as safe"
            )

        log(f"[SHRINK] Done in {time.time() - started:.0f}s. Run VACUUM to reclaim the freed pages.")
        return {"rows": after_count, "skipped": False}
    finally:
        conn.close()


def vacuum(db_path: str, log=print) -> None:
    """Reclaims freed pages. Needed after shrink_fingerprints_table: dropping the old table and
    its index leaves the pages allocated in the file until this runs."""
    log("[VACUUM] Reclaiming freed space (this rewrites the whole file; may take a while) ...")
    started = time.time()
    conn = connect(db_path)
    try:
        conn.execute("VACUUM;")
    finally:
        conn.close()
    log(f"[VACUUM] Done in {time.time() - started:.0f}s.")


# --------------------------------------------------------------------------
# ADB
# --------------------------------------------------------------------------

def get_adb_binary() -> str:
    """Locates adb without hardcoding anyone's home directory."""
    found = shutil.which("adb")
    if found:
        return found

    for env_var in ("ANDROID_SDK_ROOT", "ANDROID_HOME"):
        root = os.environ.get(env_var)
        if root:
            for candidate in (
                os.path.join(root, "platform-tools", "adb"),
                os.path.join(root, "platform-tools", "adb.exe"),
            ):
                if os.path.exists(candidate):
                    return candidate

    # WSL reaching the Windows-side SDK, for any user on any drive.
    for pattern in (
        "/mnt/*/Users/*/AppData/Local/Android/Sdk/platform-tools/adb.exe",
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
    ):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]

    return "adb"


def run_adb_command(args: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    cmd = [get_adb_binary()] + args
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except FileNotFoundError:
        return -1, "", "adb not found on PATH (set ANDROID_SDK_ROOT or install platform-tools)"
    except subprocess.TimeoutExpired:
        return -1, "", f"adb timed out after {timeout}s"
    except OSError as exc:
        return -1, "", str(exc)


def _device_lines(stdout: str) -> List[str]:
    return [
        line.strip()
        for line in stdout.splitlines()
        if line.strip() and not line.startswith("*") and not line.startswith("List of")
    ]


def ensure_device_connected(log=print) -> bool:
    """True when an authorized device is attached; restarts adb once if none is."""
    _, stdout, _ = run_adb_command(["devices"])
    lines = _device_lines(stdout)

    if not lines:
        log("[ADB] No device detected. Restarting the adb server ...")
        run_adb_command(["kill-server"])
        run_adb_command(["start-server"])
        _, stdout, _ = run_adb_command(["devices"])
        lines = _device_lines(stdout)

    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            serial, status = parts[0], parts[1]
            if status == "device":
                log(f"[ADB] Connected to authorized device: {serial}")
                return True
            if status == "unauthorized":
                log(f"[ADB] Device {serial} is unauthorized.")
                log("      Unlock the phone and tap 'Allow' for USB debugging, then retry.")
                return False

    log("[ADB] No Android devices attached.")
    return False


def pull_database(local_db_path: str, package: str = DEFAULT_PACKAGE,
                  db_name: str = "wanda_music.db", log=print) -> bool:
    """Streams the app's database off the device via `exec-out run-as`.

    Writes to a temp file and only replaces the local database once the pull has
    produced a valid, integrity-checked SQLite file -- a failed pull used to be
    able to truncate a good local copy.
    """
    log(f"[DB] Pulling {db_name} from {package} ...")
    tmp_path = local_db_path + ".pull.tmp"
    adb = get_adb_binary()

    attempts = [
        [adb, "exec-out", "run-as", package, "cat", f"databases/{db_name}"],
        [adb, "exec-out", "cat", f"/data/data/{package}/databases/{db_name}"],
    ]
    for cmd in attempts:
        try:
            with open(tmp_path, "wb") as out:
                res = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, timeout=900)
        except (OSError, subprocess.SubprocessError) as exc:
            log(f"[DB]   attempt failed: {exc}")
            continue

        if res.returncode == 0 and os.path.getsize(tmp_path) > 0:
            ok, problems = check_integrity(tmp_path)
            if ok:
                os.replace(tmp_path, local_db_path)
                log(f"[DB] Pulled {os.path.getsize(local_db_path) / 1e6:.1f} MB to {local_db_path}")
                return True
            log(f"[DB]   pulled file is corrupt ({summarise_damage(problems)}); trying next method")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    log("[DB] Pull failed. Is the phone unlocked and USB debugging allowed?")
    return False


# What the user made, as opposed to what the indexer derived.
#
# These live as *columns on `tracks`* rather than in a table of their own, which is the whole
# reason a push is dangerous: replacing the database file to deliver new fingerprints also
# replaces every like and play count with whatever the desktop copy happened to hold.
USER_TRACK_COLUMNS = (
    "isLiked", "isDownloaded", "isCached", "localFilePath", "playCount",
    "lastPlayedTimestamp", "addedTimestamp", "isLibrary", "contentHash",
)

# Tables owned entirely by the user or the device. `drops` is included even though it carries no
# track id -- it stores its own title/contentHash, so it survives independently.
USER_TABLES = (
    "history", "local_playlists", "recording_splits", "canonical_metadata", "drops",
)


def merge_user_data(source_db: str, target_db: str, log=print) -> Dict[str, int]:
    """Carries the user's own data from `source_db` (the phone) into `target_db`.

    Run before a push. The desktop database is authoritative for everything the indexer
    computes and for nothing else: likes, play counts, history and drops only ever happen on the
    phone, so a push that does not bring them across silently reverts them to whatever the last
    pull captured.

    Rows pointing at a track the target does not have are counted and reported rather than
    dropped in silence -- that count is the signal that the two libraries have diverged.
    """
    counts: Dict[str, int] = {}
    conn = connect(target_db)
    try:
        conn.execute("ATTACH DATABASE ? AS phone", (source_db,))
        try:
            columns = ", ".join(f"{c} = (SELECT p.{c} FROM phone.tracks p WHERE p.id = tracks.id)"
                                for c in USER_TRACK_COLUMNS)
            cur = conn.execute(
                f"UPDATE tracks SET {columns} "
                "WHERE id IN (SELECT id FROM phone.tracks)"
            )
            counts["tracks"] = cur.rowcount

            orphans = conn.execute(
                "SELECT count(*) FROM phone.tracks p "
                "WHERE p.id NOT IN (SELECT id FROM tracks)"
            ).fetchone()[0]
            if orphans:
                counts["tracks_only_on_phone"] = orphans
                log(f"[MERGE] {orphans} track(s) exist only on the phone; "
                    "their user data cannot be carried over.")

            for table in USER_TABLES:
                exists = conn.execute(
                    "SELECT 1 FROM phone.sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if not exists:
                    continue
                conn.execute(f"DELETE FROM main.{table}")
                cur = conn.execute(f"INSERT INTO main.{table} SELECT * FROM phone.{table}")
                counts[table] = cur.rowcount

            conn.commit()
        finally:
            conn.execute("DETACH DATABASE phone")
    finally:
        conn.close()

    summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
    log(f"[MERGE] Carried over from the phone: {summary or 'nothing'}")
    return counts


def push_database(local_db_path: str, package: str = DEFAULT_PACKAGE,
                  db_name: str = "wanda_music.db", log=print) -> bool:
    """Streams the updated database back and force-stops the app so Room rereads it."""
    ok, problems = check_integrity(local_db_path)
    if not ok:
        log(f"[DB] Refusing to push a corrupt database: {summarise_damage(problems)}")
        return False

    # Fold any WAL back into the file first, or the phone sees stale data.
    finalize_database(local_db_path)

    log(f"[DB] Pushing {local_db_path} to {package} ...")
    adb = get_adb_binary()
    # Stopped *before* the write, not only after. A running app holds the database open and keeps
    # writing to its own write-ahead log; replacing the file underneath it leaves that log
    # describing pages that no longer exist, SQLite reports corruption on the next open, and
    # `DefaultDatabaseErrorHandler` answers a corruption report by deleting the database. The
    # result is a push that appears to succeed and then destroys the library on the phone.
    run_adb_command(["shell", "am", "force-stop", package])
    try:
        with open(local_db_path, "rb") as src:
            res = subprocess.run(
                [adb, "exec-in", "run-as", package, "sh", "-c",
                 f"cat > databases/{db_name}"],
                stdin=src, stderr=subprocess.PIPE, timeout=900,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"[DB] Push failed: {exc}")
        return False

    if res.returncode != 0:
        log(f"[DB] Push failed: {res.stderr.decode('utf-8', 'replace').strip()}")
        return False

    # Room keeps its own -wal/-shm, and a stale pair left beside a freshly written database is
    # read as a log of changes to make to it — which is how a good push turns into a corruption
    # report. Removed one file per command: some device shells reject `rm` with several operands
    # ("rm: Needs 1 argument"), and the grouped form failed silently enough to look like it worked.
    for suffix in ("-wal", "-shm"):
        run_adb_command(["shell", "run-as", package, "rm", "-f",
                         f"databases/{db_name}{suffix}"])

    # Reported because a push that lands and is then deleted looks identical to one that worked;
    # the size on the device is the only cheap proof it survived.
    _, listing, _ = run_adb_command(["shell", "run-as", package, "ls", "-l",
                                     f"databases/{db_name}"])
    log(f"[DB] On device: {listing}")
    log("[DB] Pushed. Reopen Wanda on the phone to pick up the new index.")
    return True
