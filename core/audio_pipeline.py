"""Silent in-memory audio decoding via ffmpeg.

Output contract is fixed: mono, 8 kHz, 32-bit float PCM. Every DSP constant in
`fingerprinter` and `acoustic_features` depends on that sample rate, so it must
not be changed without regenerating the golden vectors and updating the Kotlin
side to match.
"""
import shutil
import subprocess
import time
from typing import Optional, Tuple

import numpy as np

SAMPLE_RATE = 8000
DEFAULT_MAX_SECONDS = 60

DECODE_OK = "OK"
DECODE_TIMEOUT = "DECODE_TIMEOUT"
DECODE_EMPTY = "DECODE_EMPTY"
DECODE_FAILED = "DECODE_FAILED"
FFMPEG_MISSING = "FFMPEG_MISSING"

_FFMPEG = None


def ffmpeg_path() -> str:
    global _FFMPEG
    if _FFMPEG is None:
        _FFMPEG = shutil.which("ffmpeg") or ""
    return _FFMPEG


def _build_command(source_input: str, max_seconds: int) -> list:
    is_network = source_input.startswith(("http://", "https://", "rtmp://", "rtsp://"))

    cmd = [ffmpeg_path(), "-nostdin", "-threads", "1", "-loglevel", "error"]
    if is_network:
        # Survive the mid-transfer resets that CDNs hand out under load.
        cmd += [
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_delay_max", "5",
            "-rw_timeout", "15000000",
        ]
    # -ss/-t before -i: the input is truncated at the source, so a 4-minute
    # track costs one minute of transfer instead of four.
    cmd += ["-ss", "0", "-t", str(max_seconds), "-i", source_input]
    cmd += [
        "-vn", "-sn", "-dn",          # never decode video/subtitle/data streams
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "pipe:1",
    ]
    return cmd


def decode_audio_to_pcm(
    source_input: str,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    retries: int = 2,
    timeout: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """Decodes to mono 8 kHz float32. Returns (samples, reason).

    `samples` is None on failure and `reason` says which failure, so the caller
    can report something better than a blank "decode failed".
    """
    if not ffmpeg_path():
        return None, FFMPEG_MISSING

    cmd = _build_command(source_input, max_seconds)
    deadline = timeout if timeout is not None else max_seconds + 30
    last_reason = DECODE_FAILED

    for attempt in range(max(1, retries)):
        proc = None
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = proc.communicate(timeout=deadline)
            except subprocess.TimeoutExpired:
                # The old code left the ffmpeg process running here; with 32
                # workers on flaky URLs those accumulated for the whole run.
                proc.kill()
                try:
                    proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                last_reason = DECODE_TIMEOUT
                out, err = b"", b""

            if proc.returncode == 0 and out:
                # The buffer may not be a whole number of float32s if ffmpeg was
                # cut off mid-sample; trim rather than let frombuffer raise.
                usable = len(out) - (len(out) % 4)
                if usable > 0:
                    samples = np.frombuffer(out[:usable], dtype=np.float32).copy()
                    if samples.size > 0:
                        return samples, DECODE_OK
                last_reason = DECODE_EMPTY
            elif last_reason != DECODE_TIMEOUT:
                detail = err.decode("utf-8", "replace").strip().splitlines()
                last_reason = DECODE_FAILED
                if detail:
                    last_reason = f"{DECODE_FAILED}: {detail[-1][:120]}"
        except OSError as exc:
            last_reason = f"{DECODE_FAILED}: {exc}"
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()

        if attempt < retries - 1:
            time.sleep(1.0 * (attempt + 1))

    return None, last_reason
