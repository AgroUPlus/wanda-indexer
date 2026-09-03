"""Progress reporter: degradation, accounting, and never-crash guarantees."""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.progress import ProgressReporter, format_duration


class FakeTTY(io.StringIO):
    """A stream that claims to be a terminal."""

    encoding = "utf-8"

    def isatty(self):
        return True


class ExplodingTTY(FakeTTY):
    """A terminal that fails the moment anything is drawn on it."""

    def write(self, text):
        raise OSError("terminal went away")


def test_non_tty_falls_back_to_plain():
    reporter = ProgressReporter(2, 2, 2, stream=io.StringIO())
    assert reporter.plain, "a pipe must not get the cursor-moving dashboard"


def test_no_color_env_forces_plain(monkeypatch_env=os.environ):
    previous = monkeypatch_env.get("NO_COLOR")
    monkeypatch_env["NO_COLOR"] = "1"
    try:
        reporter = ProgressReporter(2, 2, 2, stream=FakeTTY())
        assert reporter.plain, "NO_COLOR must disable the dashboard"
    finally:
        if previous is None:
            monkeypatch_env.pop("NO_COLOR", None)
        else:
            monkeypatch_env["NO_COLOR"] = previous


def test_ascii_mode_emits_no_box_drawing():
    reporter = ProgressReporter(10, 2, 2, stream=FakeTTY(), ascii_only=True)
    reporter.plain = False
    reporter.track_started("a", "io#1", "YTMUSIC", "Artist - Title")
    for line in reporter._compose():
        assert line.isascii(), f"non-ascii output in ascii mode: {line!r}"


def test_plain_mode_logs_one_line_per_track():
    stream = io.StringIO()
    reporter = ProgressReporter(2, 1, 1, stream=stream)
    reporter.track_started("a", "io#1", "YTMUSIC", "Artist - Title")
    reporter.track_finished("a", "OK", "", {"landmarks": 100})
    reporter.track_started("b", "io#1", "YTMUSIC", "Other - Thing")
    reporter.track_finished("b", "FAIL", "RATE_LIMITED")

    output = stream.getvalue()
    assert "[1/2]" in output and "[2/2]" in output
    assert "RATE_LIMITED" in output
    assert reporter.ok == 1 and reporter.failed == 1


def test_failures_are_bucketed_by_reason_prefix():
    reporter = ProgressReporter(3, 1, 1, stream=io.StringIO())
    for key, reason in (("a", "DECODE_FAILED: server said no"),
                        ("b", "DECODE_FAILED: something else"),
                        ("c", "RATE_LIMITED")):
        reporter.track_started(key, "io#1", "YTMUSIC", key)
        reporter.track_finished(key, "FAIL", reason)
    assert reporter.failures["DECODE_FAILED"] == 2, dict(reporter.failures)
    assert reporter.failures["RATE_LIMITED"] == 1


def test_render_failure_downgrades_instead_of_raising():
    reporter = ProgressReporter(5, 2, 2, stream=ExplodingTTY())
    reporter.plain = False
    reporter._safe(reporter._draw)  # must not propagate the OSError
    assert reporter.plain, "a broken terminal must downgrade to plain output"


def test_narrow_terminal_falls_back_to_plain(monkeypatch_env=os.environ):
    previous = monkeypatch_env.get("COLUMNS")
    monkeypatch_env["COLUMNS"] = "40"
    try:
        assert ProgressReporter(2, 2, 2, stream=FakeTTY()).plain
    finally:
        if previous is None:
            monkeypatch_env.pop("COLUMNS", None)
        else:
            monkeypatch_env["COLUMNS"] = previous


def test_duration_formatting():
    assert format_duration(0) == "00:00"
    assert format_duration(61) == "01:01"
    assert format_duration(3661) == "1:01:01"
    assert format_duration(float("inf")) == "--:--"


def test_jsonl_log_is_written(tmp_path=None):
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as workdir:
        log_path = os.path.join(workdir, "run.jsonl")
        reporter = ProgressReporter(1, 1, 1, stream=io.StringIO(), log_path=log_path)
        reporter.track_started("a", "io#1", "YTMUSIC", "Artist - Title")
        reporter.track_finished("a", "OK", "", {"landmarks": 42})
        reporter.stop()

        with open(log_path) as handle:
            record = json.loads(handle.readline())
        assert record["track"] == "a" and record["landmarks"] == 42, record


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  ok  {name}")
    print("All progress tests passed.")
