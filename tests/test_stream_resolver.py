"""Cache, adaptive throttling, and error classification."""
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.stream_resolver import (
    AGE_RESTRICTED,
    GEO_BLOCKED,
    NO_AUDIO_FORMAT,
    RATE_LIMITED,
    RESOLVE_ERROR,
    UNAVAILABLE,
    AdaptiveLimiter,
    StreamCache,
    StreamResolver,
    classify_ytdlp_error,
)


def test_error_classification():
    cases = [
        ("ERROR: HTTP Error 429: Too Many Requests", RATE_LIMITED),
        ("Sign in to confirm you're not a bot", RATE_LIMITED),
        ("ERROR: Video unavailable", UNAVAILABLE),
        ("ERROR: This video is private", UNAVAILABLE),
        ("The uploader has not made this video available in your country", GEO_BLOCKED),
        ("Sign in to confirm your age", AGE_RESTRICTED),
        ("ERROR: requested format is not available", NO_AUDIO_FORMAT),
        ("something nobody has seen before", RESOLVE_ERROR),
    ]
    for stderr, expected in cases:
        actual = classify_ytdlp_error(stderr)
        assert actual == expected, f"{stderr!r} -> {actual}, expected {expected}"


def test_cache_round_trip_and_expiry():
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "cache.json")
        cache = StreamCache(path)

        fresh = f"https://cdn.example/video?expire={int(time.time()) + 7200}"
        stale = f"https://cdn.example/video?expire={int(time.time()) + 30}"
        cache.put("ytm:fresh", fresh)
        cache.put("ytm:stale", stale)
        cache.flush()

        assert cache.get("ytm:fresh") == fresh
        # Inside the safety margin, so treated as already expired.
        assert cache.get("ytm:stale") is None
        assert cache.get("ytm:missing") is None

        # Survives a reload.
        assert StreamCache(path).get("ytm:fresh") == fresh


def test_limiter_halves_on_rate_limit_and_recovers():
    limiter = AdaptiveLimiter(ceiling=8, recover_after=2)
    assert limiter.limit == 8

    limiter.report_rate_limited()
    assert limiter.limit == 4, "a 429 should halve the concurrency"
    limiter.report_rate_limited()
    assert limiter.limit == 2
    assert limiter.throttle_events == 2

    # Successes ease the limit back up, one slot at a time.
    for _ in range(4):
        limiter.report_success()
    assert limiter.limit > 2, "clean successes should restore capacity"
    assert limiter.limit <= limiter.ceiling


def test_limiter_never_drops_below_floor():
    limiter = AdaptiveLimiter(ceiling=8, floor=2)
    for _ in range(10):
        limiter.report_rate_limited()
    assert limiter.limit == 2


def test_limiter_enforces_concurrency():
    limiter = AdaptiveLimiter(ceiling=2)
    assert limiter.acquire() and limiter.acquire()

    entered = threading.Event()

    def third():
        limiter.acquire()
        entered.set()

    worker = threading.Thread(target=third, daemon=True)
    worker.start()
    assert not entered.wait(0.4), "a third caller got in past a limit of 2"

    limiter.release()
    assert entered.wait(2.0), "releasing a slot did not admit the waiter"


def test_limiter_acquire_aborts_on_stop_flag():
    limiter = AdaptiveLimiter(ceiling=1)
    assert limiter.acquire()
    stop = threading.Event()
    result = []

    worker = threading.Thread(target=lambda: result.append(limiter.acquire(stop)), daemon=True)
    worker.start()
    time.sleep(0.2)
    stop.set()
    worker.join(timeout=2.0)
    assert result == [False], "acquire must give up when the run is stopping"


def test_local_path_mapping():
    resolver = StreamResolver(
        cache=StreamCache(os.devnull),
        limiter=AdaptiveLimiter(1),
        local_path_map=[("/data/user/0/com.wander.android.debug/files/downloads",
                         "/home/me/wanda-audio")],
    )
    mapped = resolver.map_local_path(
        "/data/user/0/com.wander.android.debug/files/downloads/ytm_abc.opus"
    )
    assert mapped == os.path.join("/home/me/wanda-audio", "ytm_abc.opus"), mapped
    # Unrelated paths pass through untouched.
    assert resolver.map_local_path("/somewhere/else.mp3") == "/somewhere/else.mp3"
    assert resolver.map_local_path("") == ""


def test_navidrome_resolution_needs_credentials():
    resolver = StreamResolver(cache=StreamCache(os.devnull), limiter=AdaptiveLimiter(1))
    track = {"source": "NAVIDROME", "sourceTrackId": "abc", "localFilePath": None}
    url, _, reason = resolver.resolve(track)
    assert url is None and reason == "NO_SOURCE"

    resolver.navidrome_config = {"url": "https://music.example.com", "user": "u", "pass": "p"}
    url, stream_type, reason = resolver.resolve(track)
    assert reason == "OK" and url.startswith("https://music.example.com/rest/stream?")
    assert "id=abc" in url and "t=" in url and "s=" in url
    assert "p=" not in url, "the password must never appear in the URL"


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  ok  {name}")
    print("All stream resolver tests passed.")
