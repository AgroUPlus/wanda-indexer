"""Neural audio-fingerprint embeddings (TFLite).

Replaces the hand-built landmark constellation with a compact CNN: each 1-second
segment of 8 kHz PCM becomes a 128-d L2-normalised vector, and a track is stored
as the sequence of its segment vectors. Matching is then cosine similarity over a
short window instead of a landmark-vote SQL join.

The model (`models/wanda_embedder.tflite`) is the `nmfp-triplet` encoder from
raraz15/neural-music-fp (ISMIR 2025, AGPL-3.0) with its mel front-end folded into
the graph, so the exact same file runs here and in the Android app -- there is no
Python/Kotlin numerical parity to keep in sync, unlike `fingerprinter`.

Input contract mirrors `audio_pipeline`: mono, 8 kHz, float32.
"""
import os
import threading
from typing import Optional

import numpy as np

# Bump when the model file or the segmentation below changes: it invalidates
# every stored vector, exactly like EXTRACTOR_VERSION does for the DSP stages.
EMBEDDER_VERSION = 1
MODEL_NAME = "nmfp-triplet"

EMBED_DIM = 128
SEGMENT_SAMPLES = 8000          # 1.0 s at 8 kHz -- the model's fixed input length
HOP_SAMPLES = 4000             # 0.5 s, the fingerprint rate the model was trained at

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "wanda_embedder.tflite",
)

# One interpreter per process, created lazily. TFLite interpreters are not
# thread-safe, so callers in a threaded context must serialise on `_LOCK`;
# the indexer runs this in single-threaded worker processes, so it is free.
_INTERPRETERS: dict = {}
_LOCK = threading.Lock()


def default_model_path() -> str:
    return _DEFAULT_MODEL


def model_available(model_path: Optional[str] = None) -> bool:
    return os.path.isfile(model_path or _DEFAULT_MODEL)


def _load_runtime():
    """Returns an `Interpreter` class from whichever TFLite runtime is installed.

    `ai-edge-litert` is the current name, `tflite-runtime` the old one; full
    TensorFlow is the fallback so a dev box with it already installed just works.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter
    except ImportError:
        pass
    try:
        from tensorflow.lite import Interpreter  # type: ignore
        return Interpreter
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RuntimeError(
            "No TFLite runtime found. Install 'ai-edge-litert' (see requirements.txt)."
        ) from exc


def get_interpreter(model_path: Optional[str] = None):
    path = os.path.abspath(model_path or _DEFAULT_MODEL)
    cached = _INTERPRETERS.get(path)
    if cached is not None:
        return cached
    with _LOCK:
        cached = _INTERPRETERS.get(path)
        if cached is None:
            Interpreter = _load_runtime()
            cached = Interpreter(model_path=path)
            cached.allocate_tensors()
            _INTERPRETERS[path] = cached
    return cached


def _segment(samples: np.ndarray) -> np.ndarray:
    """Fixed-length 1 s windows at a 0.5 s hop; the tail is zero-padded, not dropped.

    A track shorter than one segment (rare, but jingles exist) still gets one
    padded segment so it is never silently skipped.
    """
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.size < SEGMENT_SAMPLES:
        samples = np.pad(samples, (0, SEGMENT_SAMPLES - samples.size))
    n = 1 + (samples.size - SEGMENT_SAMPLES) // HOP_SAMPLES
    remainder = samples.size - ((n - 1) * HOP_SAMPLES + SEGMENT_SAMPLES)
    if remainder > 0:
        pad = HOP_SAMPLES - remainder
        samples = np.pad(samples, (0, pad))
        n += 1
    windows = np.lib.stride_tricks.sliding_window_view(samples, SEGMENT_SAMPLES)
    return windows[::HOP_SAMPLES][:n]


def compute_embedding(samples: np.ndarray, interpreter=None,
                      model_path: Optional[str] = None) -> bytes:
    """Per-segment 128-d vectors for one track, as a big-endian float32 BLOB.

    Layout: `n_segments` rows of `EMBED_DIM` values, row-major, `>f4`. That is
    ~61 KB for a 60 s track (against ~2.5 MB of landmark rows), and keeping the
    per-segment matrix -- rather than mean-pooling -- lets a 6 s microphone clip
    be matched against any 6 s window of the track.
    """
    if interpreter is None:
        interpreter = get_interpreter(model_path)
    windows = _segment(samples)
    if windows.size == 0:
        return b""

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    in_index, out_index = in_detail["index"], out_detail["index"]

    out = np.empty((len(windows), EMBED_DIM), dtype=np.float32)
    for i, window in enumerate(windows):
        interpreter.set_tensor(in_index, window[np.newaxis, :].astype(np.float32))
        interpreter.invoke()
        out[i] = interpreter.get_tensor(out_index)[0]

    # The model L2-normalises already; redo it so the fp16 round-off cannot leave
    # a vector slightly off the unit sphere and bias later cosine scores.
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    np.divide(out, norms, out=out, where=norms > 0)
    return out.astype(">f4").tobytes()


def unpack_embedding(blob: bytes) -> np.ndarray:
    """Inverse of `compute_embedding`: (n_segments, EMBED_DIM) float32."""
    flat = np.frombuffer(blob, dtype=">f4")
    return flat.reshape(-1, EMBED_DIM).astype(np.float32)
