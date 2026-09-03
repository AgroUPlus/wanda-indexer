"""Standalone YouTube Music stream resolution.

The indexer itself goes through `core.stream_resolver.StreamResolver`, which
adds caching and adaptive rate limiting. This module is the un-throttled
single-shot version, kept for one-off debugging (`tests/test_single_track.py`).
"""
import shutil
import subprocess
from typing import Optional, Tuple

from core.stream_resolver import NO_AUDIO_FORMAT, OK, RESOLVE_ERROR, TIMEOUT, TOOL_MISSING
from core.stream_resolver import classify_ytdlp_error


def get_ytm_stream_url_with_reason(video_id: str, timeout: int = 60) -> Tuple[Optional[str], str]:
    """Returns (url, reason). `reason` is OK on success, else a failure code."""
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        return None, TOOL_MISSING

    cmd = [
        ytdlp,
        "-f", "bestaudio[acodec!=none]/bestaudio/best",
        "--extractor-args", "youtube:player_client=android,web",
        "--no-playlist", "--no-warnings", "--no-progress", "-q",
        "--socket-timeout", "20",
        "-g", f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return None, TIMEOUT
    except OSError as exc:
        return None, f"{RESOLVE_ERROR}: {exc}"

    if res.returncode == 0:
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith("http"):
                return line, OK
        return None, NO_AUDIO_FORMAT

    return None, classify_ytdlp_error(res.stderr or "")


def get_ytm_stream_url(video_id: str) -> Optional[str]:
    """Backwards-compatible wrapper: the URL, or None."""
    url, _ = get_ytm_stream_url_with_reason(video_id)
    return url
