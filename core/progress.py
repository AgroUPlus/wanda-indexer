"""Live progress reporting for the indexer.

Two modes, chosen automatically:

  * dashboard -- a pinned, redrawn status block on a real terminal, showing what
    every in-flight worker is doing right now and for how long;
  * plain -- one timestamped line per completed track, used whenever the output
    is a pipe/file, TERM is dumb, colour is disabled, or the terminal is too
    narrow to draw into.

Both modes can additionally write structured JSONL via `log_path`, so a run is
diagnosable after the fact regardless of how it was displayed.

Nothing in here is allowed to kill a run: the whole draw path is guarded, and a
single rendering failure permanently downgrades to plain mode.

Stdlib only, by design -- this has to work on a stock Python on Windows, WSL,
macOS and Linux with nothing installed.
"""
import json
import os
import shutil
import sys
import threading
import time
import unicodedata
from collections import defaultdict, deque
from typing import Dict, Optional

# Pipeline stages in the order a track passes through them.
STAGES = ("queued", "resolve", "decode", "landmarks", "features", "recording", "commit")

_BAR_FULL = "█"
_BAR_EMPTY = "░"
_RULE = "─"


def _supports_ansi(stream) -> bool:
    """True when it is safe to move the cursor around on `stream`."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "").lower() in ("dumb", ""):
        if os.name != "nt":
            return False
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False

    if os.name == "nt":
        # Enable virtual terminal processing; on a console that refuses, fall back.
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-12)  # STD_ERROR_HANDLE
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            if not kernel32.SetConsoleMode(handle, mode.value | 0x0004):
                return False
        except Exception:
            return False
    return True


def _unicode_ok(stream) -> bool:
    encoding = getattr(stream, "encoding", None) or ""
    try:
        _BAR_FULL.encode(encoding or "ascii")
        return True
    except (LookupError, UnicodeEncodeError):
        return False


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def display_width(text: str) -> int:
    """Terminal cells `text` occupies.

    CJK titles are common in this library and each of those characters takes two
    cells; counting them as one wrecks the column alignment.
    """
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
    return width


def _truncate(text: str, width: int) -> str:
    """Truncates to `width` terminal cells, not characters."""
    if width <= 1:
        return ""
    if display_width(text) <= width:
        return text
    out = []
    used = 0
    for char in text:
        char_width = 0 if unicodedata.combining(char) else (
            2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        )
        if used + char_width > width - 1:
            break
        out.append(char)
        used += char_width
    return "".join(out) + "…"


def _pad(text: str, width: int) -> str:
    """Left-justifies to `width` terminal cells."""
    return text + " " * max(0, width - display_width(text))


class _Slot:
    __slots__ = ("worker", "source", "label", "stage", "since", "started")

    def __init__(self, worker, source, label, started):
        self.worker = worker
        self.source = source
        self.label = label
        self.stage = "queued"
        self.since = started
        self.started = started


class ProgressReporter:
    def __init__(self, total: int, io_workers: int, cpu_workers: int,
                 stream=None, force_plain: bool = False, log_path: str = "",
                 ascii_only: bool = False):
        self.total = total
        self.io_workers = io_workers
        self.cpu_workers = cpu_workers
        self.stream = stream if stream is not None else sys.stderr

        self._lock = threading.RLock()
        self._slots: Dict[str, _Slot] = {}
        self._start = time.time()

        self.done = 0
        self.ok = 0
        self.failed = 0
        self.failures: Dict[str, int] = defaultdict(int)
        self.failure_examples: Dict[str, str] = {}
        self.stage_totals: Dict[str, float] = defaultdict(float)
        self.committed = 0
        self.note = ""

        # Per-source exponentially weighted seconds/track: a local file and a
        # throttled YouTube track differ by an order of magnitude, so one global
        # average produces a useless ETA.
        self._rate_by_source: Dict[str, float] = {}
        self._remaining_by_source: Dict[str, int] = defaultdict(int)
        self._recent = deque(maxlen=50)

        width = shutil.get_terminal_size((80, 24)).columns
        self.plain = (
            force_plain
            or not _supports_ansi(self.stream)
            or width < 60
        )
        self.ascii_only = ascii_only or not _unicode_ok(self.stream)
        self._lines_drawn = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._log = None
        if log_path:
            try:
                self._log = open(log_path, "a", encoding="utf-8")
            except OSError:
                self._log = None

    # -- lifecycle ---------------------------------------------------------

    def expect(self, source: str, count: int = 1) -> None:
        with self._lock:
            self._remaining_by_source[source] += count

    def start(self) -> None:
        if self.plain:
            self._write_line(
                f"[START] {self.total} tracks | io workers {self.io_workers} | "
                f"cpu workers {self.cpu_workers} | plain output mode"
            )
            return
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if not self.plain:
            self._safe(self._clear)
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- events ------------------------------------------------------------

    def track_started(self, key: str, worker: str, source: str, label: str) -> None:
        with self._lock:
            self._slots[key] = _Slot(worker, source, label, time.time())

    def set_worker(self, key: str, worker: str) -> None:
        """Records which thread actually picked the track up.

        Tracks are registered by the submitting loop, so the slot starts out
        labelled with the main thread; the pool worker names itself here.
        """
        with self._lock:
            slot = self._slots.get(key)
            if slot is not None:
                slot.worker = worker

    def stage(self, key: str, stage: str) -> None:
        now = time.time()
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                return
            self.stage_totals[slot.stage] += now - slot.since
            slot.stage = stage
            slot.since = now

    def track_finished(self, key: str, status: str, reason: str = "",
                       detail: Optional[dict] = None) -> None:
        now = time.time()
        with self._lock:
            slot = self._slots.pop(key, None)
            elapsed = now - slot.started if slot else 0.0
            source = slot.source if slot else "?"
            label = slot.label if slot else key
            if slot:
                self.stage_totals[slot.stage] += now - slot.since

            self.done += 1
            if status == "OK":
                self.ok += 1
            else:
                self.failed += 1
                bucket = (reason or "UNKNOWN").split(":")[0].strip() or "UNKNOWN"
                self.failures[bucket] += 1
                self.failure_examples.setdefault(bucket, f"{label} [{key}]")

            if self._remaining_by_source.get(source):
                self._remaining_by_source[source] -= 1
            previous = self._rate_by_source.get(source)
            self._rate_by_source[source] = (
                elapsed if previous is None else previous * 0.7 + elapsed * 0.3
            )
            self._recent.append(now)

            self._write_jsonl({
                "ts": round(now, 3), "track": key, "source": source, "label": label,
                "status": status, "reason": reason, "elapsed_s": round(elapsed, 3),
                **(detail or {}),
            })

            if self.plain:
                stamp = time.strftime("%H:%M:%S")
                mark = "ok  " if status == "OK" else "FAIL"
                suffix = f" | {reason}" if reason else ""
                extra = ""
                if detail:
                    bits = [f"{k}={v}" for k, v in detail.items()
                            if k in ("landmarks", "bpm", "energy", "key")]
                    if bits:
                        extra = " | " + " ".join(bits)
                self._write_line(
                    f"[{stamp}] [{self.done}/{self.total}] {mark} [{source}] "
                    f"{label} ({elapsed:.1f}s){extra}{suffix}"
                )

    def checkpoint(self, counts: dict) -> None:
        with self._lock:
            self.committed += counts.get("tracks", 0)
            # Embeddings lead: they are what recognition reads now. Landmarks are reported only
            # while a run still writes them, so a post-cut-over checkpoint does not carry a
            # permanent "0 landmarks" that reads like a failure.
            parts = [f"{counts.get('embeddings', 0)} embeddings",
                     f"{counts.get('features', 0)} feature vectors"]
            if counts.get("landmarks"):
                parts.append(f"{counts['landmarks']:,} landmarks")
            message = (
                f"[CHECKPOINT] {self.committed} tracks written ({', '.join(parts)})"
            )
            self._write_jsonl({"ts": round(time.time(), 3), "event": "checkpoint", **counts})
            if self.plain:
                self._write_line(message)
            else:
                self.note = message

    def set_note(self, note: str) -> None:
        with self._lock:
            self.note = note

    def log(self, message: str) -> None:
        """A one-off message that must survive the redrawing dashboard."""
        with self._lock:
            if self.plain:
                self._write_line(message)
            else:
                self._safe(self._clear)
                self._write_line(message)

    # -- derived numbers ---------------------------------------------------

    def eta_seconds(self) -> float:
        with self._lock:
            remaining = self.total - self.done
            if remaining <= 0:
                return 0.0
            in_flight = max(1, len(self._slots) or self.io_workers)

            # Prefer per-source estimates; fall back to overall throughput.
            total_work = 0.0
            unknown = 0
            for source, count in self._remaining_by_source.items():
                if count <= 0:
                    continue
                per_track = self._rate_by_source.get(source)
                if per_track is None:
                    unknown += count
                else:
                    total_work += per_track * count
            if unknown:
                known = [v for v in self._rate_by_source.values()]
                average = sum(known) / len(known) if known else 5.0
                total_work += average * unknown
            if total_work <= 0:
                elapsed = time.time() - self._start
                rate = self.done / elapsed if elapsed > 0 and self.done else 0
                return remaining / rate if rate else float("inf")
            return total_work / in_flight

    def rate(self) -> float:
        with self._lock:
            if len(self._recent) < 2:
                elapsed = time.time() - self._start
                return self.done / elapsed if elapsed > 0 else 0.0
            span = self._recent[-1] - self._recent[0]
            return (len(self._recent) - 1) / span if span > 0 else 0.0

    # -- rendering ---------------------------------------------------------

    def _render_loop(self) -> None:
        while not self._stop.wait(0.2):
            self._safe(self._draw)

    def _safe(self, func) -> None:
        try:
            func()
        except Exception:
            # A terminal that will not cooperate must not take the run with it.
            self.plain = True
            self._lines_drawn = 0

    def _clear(self) -> None:
        if self._lines_drawn:
            self.stream.write(f"\x1b[{self._lines_drawn}A\x1b[0J")
            self.stream.flush()
            self._lines_drawn = 0

    def _draw(self) -> None:
        lines = self._compose()
        with self._lock:
            if self.plain:
                return
            if self._lines_drawn:
                self.stream.write(f"\x1b[{self._lines_drawn}A")
            body = "".join(line + "\x1b[0K\n" for line in lines)
            self.stream.write(body + "\x1b[0J")
            self.stream.flush()
            self._lines_drawn = len(lines)

    def _compose(self):
        width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 160))
        full, empty, rule = (
            ("#", ".", "-") if self.ascii_only else (_BAR_FULL, _BAR_EMPTY, _RULE)
        )
        sep = " | " if self.ascii_only else "  ·  "

        with self._lock:
            elapsed = time.time() - self._start
            done, total = self.done, max(1, self.total)
            fraction = min(1.0, done / total)
            slots = sorted(self._slots.values(), key=lambda s: (s.worker, s.started))
            in_flight = sum(1 for s in slots if s.stage != "queued")
            failures = dict(self.failures)
            note = self.note
            active_stage = {}
            for slot in slots:
                active_stage[slot.stage] = active_stage.get(slot.stage, 0) + 1

        header = (
            f"Wanda Indexer{sep}{total} tracks{sep}"
            f"io {in_flight}/{self.io_workers}{sep}cpu {self.cpu_workers}{sep}"
            f"elapsed {format_duration(elapsed)}{sep}eta {format_duration(self.eta_seconds())}"
        )

        bar_width = max(10, min(40, width - 40))
        filled = int(bar_width * fraction)
        bar = full * filled + empty * (bar_width - filled)
        progress = (
            f"[{bar}] {done}/{total} {fraction * 100:5.1f}%   {self.rate():.2f} trk/s"
        )

        counts = f" ok {self.ok}   failed {self.failed}   committed {self.committed}"
        if note:
            counts = _truncate(counts + "      " + note, width)

        lines = [_truncate(header, width), _truncate(progress, width), counts, rule * width]

        if slots:
            # Widest label that still fits alongside the fixed-width columns.
            label_width = max(12, width - 34)
            for slot in slots[: max(4, self.io_workers + 2)]:
                age = time.time() - slot.since
                lines.append(_truncate(
                    f" {slot.worker:<6} {slot.stage:<10} {slot.source[:3].lower():<3} "
                    f"{_pad(_truncate(slot.label, label_width), label_width)} {age:5.1f}s",
                    width,
                ))
        else:
            lines.append(" (no tracks in flight)")

        lines.append(rule * width)
        if failures:
            ordered = sorted(failures.items(), key=lambda kv: -kv[1])
            summary = (" · " if not self.ascii_only else " | ").join(
                f"{name} {count}" for name, count in ordered[:6]
            )
            lines.append(_truncate(" failures  " + summary, width))
        else:
            lines.append(" failures  none")
        return lines

    # -- output helpers ----------------------------------------------------

    def _write_line(self, text: str) -> None:
        try:
            self.stream.write(text + "\n")
            self.stream.flush()
        except (OSError, ValueError, UnicodeEncodeError):
            pass

    def _write_jsonl(self, payload: dict) -> None:
        if self._log is None:
            return
        try:
            self._log.write(json.dumps(payload, default=str) + "\n")
            self._log.flush()
        except (OSError, ValueError):
            pass

    # -- final report ------------------------------------------------------

    def final_report(self, write=print) -> None:
        elapsed = time.time() - self._start
        write("=" * 70)
        write(f"Finished in {format_duration(elapsed)} ({elapsed:.1f}s)")
        write(f"  Indexed        : {self.ok}")
        write(f"  Committed      : {self.committed}")
        write(f"  Failed/skipped : {self.failed}")
        if elapsed > 0:
            write(f"  Throughput     : {self.done / elapsed:.2f} tracks/second")

        if self.stage_totals:
            write("  Time by stage  :")
            for name, seconds in sorted(self.stage_totals.items(), key=lambda kv: -kv[1]):
                if seconds < 0.05 or name == "queued":
                    continue
                write(f"      {name:<12} {seconds:8.1f}s")

        if self.failures:
            write("  Failures by reason:")
            for name, count in sorted(self.failures.items(), key=lambda kv: -kv[1]):
                example = self.failure_examples.get(name, "")
                write(f"      {name:<18} {count:>5}   e.g. {_truncate(example, 44)}")
        write("=" * 70)
