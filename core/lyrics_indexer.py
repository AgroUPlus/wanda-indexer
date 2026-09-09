"""Lyrics fetching, local tag extraction, and search indexing for Wanda."""

import json
import os
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

USER_AGENT = "Wanda/1.5.0 (https://github.com/AgroUPlus/Wanda)"
LRCLIB_BASE_URL = "https://lrclib.net/api/get"
LRC_REGEX = re.compile(r"^\[(\d{2}):(\d{2})\.(\d{2,3})\](.*)$")


def parse_lrc(lrc_content: str) -> List[Tuple[int, str]]:
    """Parses LRC format into a list of (timestamp_ms, line_text) tuples ordered by time."""
    lines: List[Tuple[int, str]] = []
    if not lrc_content:
        return lines

    for line in lrc_content.splitlines():
        line = line.strip()
        match = LRC_REGEX.match(line)
        if match:
            minute = int(match.group(1))
            second = int(match.group(2))
            fraction_str = match.group(3)
            fraction_ms = int(fraction_str) * 10 if len(fraction_str) == 2 else int(fraction_str)
            timestamp_ms = (minute * 60 * 1000) + (second * 1000) + fraction_ms
            text = match.group(4).strip()
            if text:
                lines.append((timestamp_ms, text))
    lines.sort(key=lambda x: x[0])
    return lines


def plain_text_from_lrc(lrc_content: str) -> str:
    """Strips timestamps from LRC format to generate clean plain text lyrics."""
    lines = parse_lrc(lrc_content)
    return "\n".join(text for _, text in lines)


def find_matching_lyric_line(
    plain_lyrics: Optional[str],
    synced_lyrics: Optional[str],
    query: str
) -> Tuple[Optional[int], str]:
    """Finds the most relevant line matching query, returning (timestamp_ms, line_text)."""
    clean_query = query.strip().lower()
    if not clean_query:
        return (None, "")

    # First check synced lyrics for timestamp accuracy
    if synced_lyrics:
        parsed = parse_lrc(synced_lyrics)
        # 1. Exact phrase match
        for ts, text in parsed:
            if clean_query in text.lower():
                return (ts, text)
        # 2. All words match
        words = [w for w in re.split(r"\W+", clean_query) if w]
        if words:
            for ts, text in parsed:
                text_lower = text.lower()
                if all(w in text_lower for w in words):
                    return (ts, text)

    # Fallback to plain lyrics
    if plain_lyrics:
        for line in plain_lyrics.splitlines():
            line_str = line.strip()
            if clean_query in line_str.lower():
                return (None, line_str)

    return (None, "")


def extract_local_tags_lyrics(file_path: str) -> Optional[Dict[str, Any]]:
    """Attempts to read embedded lyrics from an audio file using mutagen."""
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        import mutagen  # type: ignore
    except ImportError:
        return None

    try:
        audio = mutagen.File(file_path)
        if audio is None:
            return None

        # 1. ID3 tags (MP3, AIFF)
        if hasattr(audio, "tags") and audio.tags:
            for key, tag in audio.tags.items():
                if key.startswith("USLT") or key.startswith("SYLT"):
                    text = str(tag.text if hasattr(tag, "text") else tag)
                    if text.strip():
                        return {
                            "plainLyrics": text.strip(),
                            "syncedLyrics": None,
                            "source": "ID3"
                        }

        # 2. FLAC / OGG / Vorbis comments
        if hasattr(audio, "get"):
            for field in ("lyrics", "unsyncedlyrics", "unsynced lyrics"):
                val = audio.get(field)
                if val and isinstance(val, list) and val[0].strip():
                    return {
                        "plainLyrics": val[0].strip(),
                        "syncedLyrics": None,
                        "source": "Vorbis"
                    }

        # 3. MP4 / M4A
        if "\xa9lyr" in audio:
            val = audio["\xa9lyr"]
            text = val[0] if isinstance(val, list) and val else str(val)
            if text.strip():
                return {
                    "plainLyrics": text.strip(),
                    "syncedLyrics": None,
                    "source": "MP4"
                }
    except Exception:
        pass

    return None


def fetch_lrclib_lyrics(
    title: str,
    artist: str,
    album: Optional[str] = None,
    duration_s: Optional[float] = None,
    timeout: float = 6.0
) -> Optional[Dict[str, Any]]:
    """Queries LRCLIB for synced/plain lyrics."""
    if not title or not artist:
        return None

    params = {
        "track_name": title,
        "artist_name": artist,
    }
    if album and album.strip():
        params["album_name"] = album.strip()
    if duration_s and duration_s > 0:
        params["duration"] = str(int(duration_s))

    url = f"{LRCLIB_BASE_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))

            synced = data.get("syncedLyrics") or ""
            plain = data.get("plainLyrics") or ""

            if not plain and synced:
                plain = plain_text_from_lrc(synced)

            if not plain and not synced:
                return None

            return {
                "plainLyrics": plain,
                "syncedLyrics": synced if synced else None,
                "source": "LRCLIB"
            }
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None  # Track not found on LRCLIB
        elif err.code == 429:
            # Respect rate limit retry-after if provided
            retry_after = err.headers.get("Retry-After", "1")
            try:
                time.sleep(min(float(retry_after), 5.0))
            except ValueError:
                pass
        return None
    except Exception:
        return None
