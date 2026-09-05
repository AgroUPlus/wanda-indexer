"""Neural audio fingerprints, byte-compatible with the Android app.

The Wanda app recognises music by comparing a microphone clip against a stored
sequence of 128-dimensional embeddings, one per half-second of every track. It
can compute those itself, but it does it one streamed track at a time on a
phone; this does the same arithmetic on a desktop, over the same database, with
the network as the only bottleneck.

The contract is fixed by `core/audio/fingerprint/AudioEmbedder.kt` in the app
and must not drift from it -- a vector computed differently here is not a worse
match, it is a different alphabet, and every comparison against it is noise:

    model         wanda_embedder.tflite, the `nmfp-triplet` encoder from
                  raraz15/neural-music-fp with its mel front-end in the graph
    input         (1, 8000) float32 -- one second of 8 kHz mono
    output        (1, 128) L2-normalised
    segmentation  1.0 s window, 0.5 s hop, tail zero-padded
    storage       int8, `round(value * 255)`, segment-major, `n * 128` bytes
    identity      model 'nmfp-triplet', version 2, dim 128

`core/audio_pipeline.py` already decodes to exactly the input this wants -- mono
8 kHz float32 -- so nothing resamples here.

The interpreter is not fork-safe and is deliberately not built at import: it is
created lazily, once per process, so the CPU pool's workers each get their own.
"""
import hashlib
import os
import threading
import urllib.request
from typing import Optional

import numpy as np

MODEL_NAME = "nmfp-triplet"
EMBEDDER_VERSION = 2
EMBED_DIM = 128

# Fixed-point scale for a stored component; see `AudioEmbedder.QUANT_SCALE`. The
# model's output is L2-normalised over 128 dimensions, so no component is large
# -- measured over 162,447 real vectors the extreme was 0.433, against the 0.498
# that would clip here. Quantising costs at most 0.002 of match score, and saves
# three quarters of a database that is otherwise ~250 MB for a 1400-track
# library once tracks are indexed whole.
QUANT_SCALE = 255

SAMPLE_RATE = 8000
SEGMENT_SAMPLES = 8000          # 1.0 s
HOP_SAMPLES = 4000              # 0.5 s -- the model's training fingerprint rate

# The asset the app downloads, so both sides run the identical file. Pinned by
# digest: a truncated or substituted model produces plausible vectors that match
# nothing, which is the one failure that would not announce itself.
MODEL_URL = (
    "https://github.com/AgroUPlus/Wanda/releases/download/embedder-v1/wanda_embedder.tflite"
)
MODEL_SHA256 = "50a26e9f19dbaa35de7a1efda1c3b69d471e83f860c53c7df184431e754455aa"
MODEL_SIZE_BYTES = 35_892_012
MODEL_FILENAME = "wanda_embedder.tflite"

_interpreter = None
_interpreter_lock = threading.Lock()


class EmbedderUnavailable(RuntimeError):
    """No TFLite runtime, or no model file. Indexing continues without embeddings."""


def segment(samples: np.ndarray) -> np.ndarray:
    """1 s windows at a 0.5 s hop, tail zero-padded. `(n, 8000)` float32.

    Ported line for line from `AudioEmbedder.segment`, including the two padding
    cases, because a window boundary half a hop out of step would shift every
    stored vector against the ones the phone computes.
    """
    pcm = np.ascontiguousarray(samples, dtype=np.float32)
    if pcm.size < SEGMENT_SAMPLES:
        pcm = np.pad(pcm, (0, SEGMENT_SAMPLES - pcm.size))

    n = 1 + (pcm.size - SEGMENT_SAMPLES) // HOP_SAMPLES
    remainder = pcm.size - ((n - 1) * HOP_SAMPLES + SEGMENT_SAMPLES)
    if remainder > 0:
        pcm = np.pad(pcm, (0, HOP_SAMPLES - remainder))
        n += 1

    # A strided view, not n copies: a 20-minute track is 2400 windows of 8000
    # floats, and materialising them would be 77 MB per track per worker.
    return np.lib.stride_tricks.as_strided(
        pcm,
        shape=(n, SEGMENT_SAMPLES),
        strides=(pcm.strides[0] * HOP_SAMPLES, pcm.strides[0]),
        writeable=False,
    )


def pack(vectors: np.ndarray) -> bytes:
    """Segment vectors as the BLOB the app reads: one int8 per component.

    `np.rint`, not `astype(np.int8)`, and clipped before the cast. Truncation
    would bias every component towards zero, and an unclipped cast *wraps* --
    0.6 * 255 = 153 becomes -103, a large positive number stored as a large
    negative one. Neither announces itself; both simply stop matching.
    """
    v = np.asarray(vectors, dtype=np.float32)
    q = np.clip(np.rint(v * QUANT_SCALE), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    """Inverse of `pack`, for tests and for `tools/check_recognition.py`."""
    q = np.frombuffer(blob, dtype=np.int8).reshape(-1, EMBED_DIM)
    return q.astype(np.float32) / QUANT_SCALE


# Segments per summary chunk -- 60 at a 0.5 s hop, so thirty seconds. See
# `EmbeddingRepository.SUMMARY_CHUNK_SEGMENTS`; the value was chosen by
# measurement, and the two must agree or the phone's shortlist is reading
# summaries of a different length than it expects.
SUMMARY_CHUNK_SEGMENTS = 60


def summary(vectors: np.ndarray) -> np.ndarray:
    """One L2-normalised mean per 30 s chunk -- the app's search index.

    The app shortlists a recognition by these before opening any track's
    segments, so a row without them still matches but costs the phone the read
    the column exists to avoid. A trailing part-chunk is folded into the one
    before it rather than kept, matching `EmbeddingRepository.summaryOf`: a mean
    over three seconds would be compared against the same threshold as a mean
    over thirty.

    One per track was the original design and is measurably too blunt -- a clip
    resembles one moment of a song, not its average. On a real 1374-track index
    a whole-track mean put the true track as low as rank 524; per 30 s chunk, no
    lower than rank 2.
    """
    v = np.asarray(vectors, dtype=np.float32)
    if len(v) == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    chunks = max(1, len(v) // SUMMARY_CHUNK_SEGMENTS)
    out = np.empty((chunks, EMBED_DIM), dtype=np.float32)
    for c in range(chunks):
        start = c * SUMMARY_CHUNK_SEGMENTS
        stop = len(v) if c == chunks - 1 else start + SUMMARY_CHUNK_SEGMENTS
        m = v[start:stop].mean(axis=0)
        n = float(np.linalg.norm(m))
        out[c] = m / n if n > 0 else m
    return out


def model_path(cache_dir: str = ".wanda-cache", explicit: str = "") -> str:
    """The model file, downloading and verifying it once if it is not here yet."""
    if explicit:
        if not os.path.isfile(explicit):
            raise EmbedderUnavailable(f"model file not found: {explicit}")
        return explicit

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, MODEL_FILENAME)
    if os.path.isfile(path) and os.path.getsize(path) == MODEL_SIZE_BYTES:
        return path

    tmp = path + ".part"
    try:
        with urllib.request.urlopen(MODEL_URL) as response, open(tmp, "wb") as out:
            digest = hashlib.sha256()
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                digest.update(block)
                out.write(block)
        if digest.hexdigest() != MODEL_SHA256:
            raise EmbedderUnavailable(
                f"model checksum mismatch (got {digest.hexdigest()})"
            )
        os.replace(tmp, path)
    except EmbedderUnavailable:
        _unlink(tmp)
        raise
    except OSError as exc:
        _unlink(tmp)
        raise EmbedderUnavailable(f"could not download the model: {exc}") from exc
    return path


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _load_interpreter(path: str):
    """The first runtime that imports. Neither ships on every platform."""
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
            except ImportError as exc:
                raise EmbedderUnavailable(
                    "no TFLite runtime; pip install ai-edge-litert (or tflite-runtime)"
                ) from exc
    # One thread: the CPU pool is already the parallelism, and letting each of N
    # worker processes start its own thread pool is how a 32-core machine ends
    # up with a thousand threads fighting over it. Same reasoning as
    # `main._init_cpu_worker`.
    return Interpreter(model_path=path, num_threads=1)


def is_available(cache_dir: str = ".wanda-cache", explicit: str = "") -> bool:
    """Whether embedding can run at all, without paying to find out twice."""
    try:
        interpreter(cache_dir, explicit)
        return True
    except EmbedderUnavailable:
        return False


def interpreter(cache_dir: str = ".wanda-cache", explicit: str = ""):
    """This process's interpreter, built on first use.

    Lazy and per-process on purpose. A TFLite interpreter holds native state
    that does not survive `fork`, so one built in the parent before the pool
    starts is a segfault waiting for the first worker to use it.
    """
    global _interpreter
    if _interpreter is None:
        with _interpreter_lock:
            if _interpreter is None:
                itp = _load_interpreter(model_path(cache_dir, explicit))
                itp.allocate_tensors()
                _interpreter = itp
    return _interpreter


def embed(samples: np.ndarray, cache_dir: str = ".wanda-cache",
          explicit: str = "") -> np.ndarray:
    """`(n, 128)` L2-normalised segment embeddings for mono 8 kHz float PCM."""
    windows = segment(samples)
    if windows.shape[0] == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    itp = interpreter(cache_dir, explicit)
    in_detail = itp.get_input_details()[0]
    out_detail = itp.get_output_details()[0]

    out = np.empty((windows.shape[0], EMBED_DIM), dtype=np.float32)
    for i in range(windows.shape[0]):
        itp.set_tensor(in_detail["index"], windows[i].reshape(1, SEGMENT_SAMPLES))
        itp.invoke()
        out[i] = itp.get_tensor(out_detail["index"])[0]

    # Re-normalise, as the app does: fp16 round-off inside the graph leaves the
    # vector fractionally off the unit sphere, and every later cosine inherits
    # the error as a per-track bias.
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    np.divide(out, norms, out=out, where=norms > 0)
    return out
