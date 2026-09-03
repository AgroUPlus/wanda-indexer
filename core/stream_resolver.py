"""Resolves a track to a playable audio URL, politely.

The old path shelled out to yt-dlp from 32 processes at once with no caching and
no rate-limit awareness. YouTube answers that with HTTP 429, and every 429 became
an opaque "Could not resolve audio stream URL" failure. This module keeps one
adaptive concurrency limit across the whole run, caches resolutions on disk, and
reports *why* something failed.
"""
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
from typing import Dict, Optional, Tuple

from sources.navidrome import build_subsonic_stream_url

# Stable failure reasons, so the final report can group them.
OK = "OK"
RATE_LIMITED = "RATE_LIMITED"
UNAVAILABLE = "UNAVAILABLE"
GEO_BLOCKED = "GEO_BLOCKED"
AGE_RESTRICTED = "AGE_RESTRICTED"
AUTH_FAILED = "AUTH_FAILED"
NO_AUDIO_FORMAT = "NO_AUDIO_FORMAT"
TIMEOUT = "TIMEOUT"
TOOL_MISSING = "TOOL_MISSING"
NO_SOURCE = "NO_SOURCE"
RESOLVE_ERROR = "RESOLVE_ERROR"

_CACHE_VERSION = 1


class StreamCache:
    """Disk-backed map of track -> resolved URL, with expiry awareness.

    Google's CDN URLs carry an `expire=` epoch in the query string; a cached URL
    is only reused while it still has comfortable life left. Reruns after a
    partial failure therefore cost no yt-dlp calls at all.
    """

    def __init__(self, path: str, safety_margin_s: int = 600):
        self.path = path
        self.safety_margin_s = safety_margin_s
        self._lock = threading.Lock()
        self._entries: Dict[str, dict] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path) as handle:
                blob = json.load(handle)
            if blob.get("version") == _CACHE_VERSION:
                self._entries = blob.get("entries", {})
        except (OSError, ValueError):
            self._entries = {}

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None
            expires = entry.get("expires", 0)
            if expires and time.time() + self.safety_margin_s >= expires:
                self._entries.pop(key, None)
                self._dirty = True
                return None
            return entry.get("url")

    def put(self, key: str, url: str) -> None:
        with self._lock:
            self._entries[key] = {"url": url, "expires": _expiry_of(url), "at": int(time.time())}
            self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            payload = {"version": _CACHE_VERSION, "entries": self._entries}
            tmp = self.path + ".tmp"
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(tmp, "w") as handle:
                    json.dump(payload, handle)
                os.replace(tmp, self.path)
                self._dirty = False
            except OSError:
                pass  # a cache we cannot persist is a slow run, not a failed one.


def _expiry_of(url: str) -> int:
    try:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        for key in ("expire", "expires"):
            if key in query:
                return int(query[key][0])
    except (ValueError, TypeError):
        pass
    return 0


class AdaptiveLimiter:
    """Concurrency limit that shrinks on rate-limiting and recovers slowly.

    Starts at `ceiling`. A 429 halves the limit (floor 1) and forces every
    caller to wait out an exponentially growing cooldown; a streak of clean
    successes adds one slot back. This is what keeps a 1000-track YouTube run
    from collapsing into a wall of 429s.
    """

    def __init__(self, ceiling: int, floor: int = 1, recover_after: int = 12):
        self.ceiling = max(1, ceiling)
        self.floor = max(1, min(floor, self.ceiling))
        self.recover_after = recover_after

        self._limit = self.ceiling
        self._active = 0
        self._successes = 0
        self._cooldown_until = 0.0
        self._backoff_s = 0.0
        self._condition = threading.Condition()
        self.throttle_events = 0

    @property
    def limit(self) -> int:
        with self._condition:
            return self._limit

    def acquire(self, stop_flag=None) -> bool:
        with self._condition:
            while True:
                if stop_flag is not None and stop_flag.is_set():
                    return False
                now = time.time()
                if self._active < self._limit and now >= self._cooldown_until:
                    self._active += 1
                    return True
                wait = 0.25
                if now < self._cooldown_until:
                    wait = min(self._cooldown_until - now, 0.5)
                self._condition.wait(wait)

    def release(self) -> None:
        with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify()

    def report_success(self) -> None:
        with self._condition:
            self._successes += 1
            self._backoff_s = 0.0
            if self._successes >= self.recover_after and self._limit < self.ceiling:
                self._limit += 1
                self._successes = 0
                self._condition.notify_all()

    def report_rate_limited(self) -> None:
        with self._condition:
            self.throttle_events += 1
            self._successes = 0
            self._limit = max(self.floor, self._limit // 2)
            self._backoff_s = min(60.0, (self._backoff_s * 2) or 5.0)
            # Jitter so the workers do not resynchronise into another burst.
            self._cooldown_until = time.time() + self._backoff_s * (0.75 + random.random() * 0.5)

    def status(self) -> str:
        with self._condition:
            remaining = max(0.0, self._cooldown_until - time.time())
            if remaining > 0:
                return f"{self._limit}/{self.ceiling} (backoff {remaining:.0f}s)"
            return f"{self._limit}/{self.ceiling}"


_RATE_LIMIT_PATTERNS = re.compile(
    r"429|too many requests|rate.?limit|sign in to confirm.*not a bot", re.I
)
_UNAVAILABLE_PATTERNS = re.compile(
    r"video unavailable|private video|video is private|removed by the uploader|"
    r"does not exist|has been terminated|no longer available|"
    r"unable to (?:extract|download) (?:video|webpage) data", re.I
)
_GEO_PATTERNS = re.compile(
    r"available in your country|geo.?restrict|blocked it in your country|"
    r"not available from your location", re.I
)
_AGE_PATTERNS = re.compile(r"age.?restrict|confirm your age|inappropriate for some users", re.I)
_FORMAT_PATTERNS = re.compile(r"requested format is not available|no video formats found", re.I)


def classify_ytdlp_error(stderr: str) -> str:
    if _RATE_LIMIT_PATTERNS.search(stderr):
        return RATE_LIMITED
    if _UNAVAILABLE_PATTERNS.search(stderr):
        return UNAVAILABLE
    if _GEO_PATTERNS.search(stderr):
        return GEO_BLOCKED
    if _AGE_PATTERNS.search(stderr):
        return AGE_RESTRICTED
    if _FORMAT_PATTERNS.search(stderr):
        return NO_AUDIO_FORMAT
    return RESOLVE_ERROR


class StreamResolver:
    """Turns a track row into (target, stream_type, reason)."""

    def __init__(self, cache: StreamCache, limiter: AdaptiveLimiter,
                 navidrome_config: Optional[dict] = None,
                 cookies_from_browser: str = "", cookies_file: str = "",
                 local_path_map: Optional[list] = None,
                 ytdlp_timeout: int = 60, stop_flag=None):
        self.cache = cache
        self.limiter = limiter
        self.navidrome_config = navidrome_config
        self.cookies_from_browser = cookies_from_browser
        self.cookies_file = cookies_file
        self.local_path_map = local_path_map or []
        self.ytdlp_timeout = ytdlp_timeout
        self.stop_flag = stop_flag
        self._ytdlp = shutil.which("yt-dlp")

    def map_local_path(self, path: str) -> str:
        """Rewrites an on-device path to wherever the file lives on this machine.

        The DB stores Android paths like
        /data/user/0/com.wander.android.debug/files/downloads/ytm_xxx.opus,
        which never exist on the desktop, so every download silently fell
        through to the network.
        """
        if not path:
            return ""
        for prefix, replacement in self.local_path_map:
            if path.startswith(prefix):
                relative = path[len(prefix):].lstrip("/\\")
                return os.path.join(replacement, relative)
        return path

    def resolve(self, track: dict) -> Tuple[Optional[str], str, str]:
        source = (track.get("source") or "").upper()

        local = self.map_local_path(track.get("localFilePath") or "")
        if local and os.path.exists(local):
            return local, "LOCAL FILE", OK

        if source == "NAVIDROME":
            if not self.navidrome_config:
                return None, "NAVIDROME", NO_SOURCE
            url = build_subsonic_stream_url(
                self.navidrome_config["url"],
                self.navidrome_config["user"],
                self.navidrome_config["pass"],
                track["sourceTrackId"],
            )
            return url, "NAVIDROME STREAM", OK

        if source in ("YTMUSIC", "YOUTUBE_MUSIC", "YOUTUBE"):
            return self._resolve_youtube(track["sourceTrackId"])

        stream_uri = track.get("streamUri")
        if stream_uri:
            return stream_uri, "DIRECT URI", OK

        return None, "UNKNOWN", NO_SOURCE

    def _resolve_youtube(self, video_id: str) -> Tuple[Optional[str], str, str]:
        key = f"ytm:{video_id}"
        cached = self.cache.get(key)
        if cached:
            return cached, "YTM CDN (cached)", OK

        if not self._ytdlp:
            return None, "YTM CDN", TOOL_MISSING

        cmd = [
            self._ytdlp,
            # Audio only: the previous call omitted -f and got itag 18, a 360p
            # video muxed with audio -- downloading video frames to discard them.
            "-f", "bestaudio[acodec!=none]/bestaudio/best",
            "--extractor-args", "youtube:player_client=android,web",
            "--no-playlist", "--no-warnings", "--no-progress", "-q",
            "--socket-timeout", "20",
            "-g", f"https://www.youtube.com/watch?v={video_id}",
        ]
        if self.cookies_from_browser:
            cmd[1:1] = ["--cookies-from-browser", self.cookies_from_browser]
        if self.cookies_file:
            cmd[1:1] = ["--cookies", self.cookies_file]

        if not self.limiter.acquire(self.stop_flag):
            return None, "YTM CDN", "ABORTED"
        try:
            res = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=self.ytdlp_timeout,
            )
        except subprocess.TimeoutExpired:
            return None, "YTM CDN", TIMEOUT
        except OSError as exc:
            return None, "YTM CDN", f"{RESOLVE_ERROR}: {exc}"
        finally:
            self.limiter.release()

        if res.returncode == 0:
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.startswith("http"):
                    self.limiter.report_success()
                    self.cache.put(key, line)
                    return line, "YTM CDN STREAM", OK
            return None, "YTM CDN", NO_AUDIO_FORMAT

        reason = classify_ytdlp_error(res.stderr or "")
        if reason == RATE_LIMITED:
            self.limiter.report_rate_limited()
        return None, "YTM CDN", reason
