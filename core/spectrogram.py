"""Shared short-time Fourier transform.

`fingerprinter` and `acoustic_features` each used to walk the signal frame by
frame in Python and call `np.fft.rfft` per frame -- three full STFTs of the same
audio per track. They all use the identical geometry (1024-sample Hann window,
256-sample hop, 8 kHz mono), so the transform is computed once here and the
callers derive whatever they need from it.

Bit-compatibility note: this returns the raw *complex* spectrum rather than
magnitudes or power. Each caller then applies its own original expression --
`np.abs(z)` uses hypot, `re**2 + im**2` does not, and the two disagree in the
last ulp. Sharing the complex result keeps every downstream value identical to
the per-frame implementation while still paying for only one FFT.
"""
import numpy as np

FRAME_SIZE = 1024
HOP_SIZE = 256
BIN_COUNT = FRAME_SIZE // 2

# Periodic-symmetric Hann matching Wanda's Kotlin windowing (divisor N-1).
HANN_WINDOW = (
    0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(FRAME_SIZE) / (FRAME_SIZE - 1))
).astype(np.float32)


def frame_count_for(sample_count: int) -> int:
    """Number of whole frames, matching `(len - FRAME) // HOP + 1`."""
    if sample_count < FRAME_SIZE:
        return 0
    return (sample_count - FRAME_SIZE) // HOP_SIZE + 1


def compute_stft(samples: np.ndarray, frames: int = -1) -> np.ndarray:
    """Complex STFT, shape (frames, BIN_COUNT + 1).

    `frames` defaults to every whole frame in the signal. Callers that want
    fewer (FeatureExtractor drops the final frame) pass their own count and get
    a prefix -- the transform is per-frame, so a prefix is exactly what the
    old per-frame loop would have produced.
    """
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    available = frame_count_for(len(samples))
    if frames < 0:
        frames = available
    frames = min(frames, available)
    if frames <= 0:
        return np.zeros((0, BIN_COUNT + 1), dtype=np.complex128)

    # Strided view over the overlapping frames -- no copy of the signal.
    windows = np.lib.stride_tricks.sliding_window_view(samples, FRAME_SIZE)[
        : frames * HOP_SIZE : HOP_SIZE
    ]
    return np.fft.rfft(windows * HANN_WINDOW, n=FRAME_SIZE, axis=-1)


def power_from(stft: np.ndarray) -> np.ndarray:
    """Power spectrum over the first BIN_COUNT bins, as `re**2 + im**2`."""
    head = stft[:, :BIN_COUNT]
    return np.real(head) ** 2 + np.imag(head) ** 2
