import math
import numpy as np
from typing import List, Optional, Tuple

from core.spectrogram import (
    BIN_COUNT,
    FRAME_SIZE,
    HANN_WINDOW,
    HOP_SIZE,
    compute_stft,
    frame_count_for,
    power_from,
)

BAND_EDGES = [1, 10, 20, 40, 80, 160, 512]
THRESHOLD_DECAY = 0.08
FAN_OUT = 4

FREQ_BITS = 8
FREQ_MASK = (1 << FREQ_BITS) - 1
DELTA_BITS = 6
MAX_DELTA_FRAMES = (1 << DELTA_BITS) - 1
MIN_DELTA_FRAMES = 1

_HANN_WINDOW = HANN_WINDOW  # re-exported for backwards compatibility

def quantise_frequency(bin_idx: int) -> int:
    return (bin_idx >> 1) & FREQ_MASK

def pack_landmark(anchor_bin: int, target_bin: int, delta_frames: int) -> int:
    # Packed into signed 32-bit int as in Wanda/JVM
    unsigned_val = (
        (quantise_frequency(anchor_bin) << (FREQ_BITS + DELTA_BITS))
        | (quantise_frequency(target_bin) << DELTA_BITS)
        | (delta_frames & MAX_DELTA_FRAMES)
    )
    if unsigned_val >= 0x80000000:
        return unsigned_val - 0x100000000
    return unsigned_val

def compute_landmarks(samples: np.ndarray, stft: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """
    Computes Shazam-style landmarks (hash, anchor_frame) matching Wanda's Fingerprinter.kt.

    `stft` may be a spectrum already computed by `core.spectrogram.compute_stft`;
    it is shared with the feature extractor so the FFT is paid for once.
    """
    frame_count = frame_count_for(len(samples))
    if frame_count <= 0:
        return []

    if stft is None:
        stft = compute_stft(samples, frame_count)
    stft = stft[:frame_count]

    # Power & log magnitude: ln(sqrt(power) + 1e-9), all frames at once.
    magnitudes = np.log(np.sqrt(power_from(stft)) + 1e-9)

    band_count = len(BAND_EDGES) - 1

    # Per band, the strongest bin of every frame. The threshold below is a
    # sequential recurrence, but locating each frame's peak is not.
    best_bins = np.empty((band_count, frame_count), dtype=np.int64)
    best_mags = np.empty((band_count, frame_count), dtype=np.float64)
    for band in range(band_count):
        start_bin, end_bin = BAND_EDGES[band], BAND_EDGES[band + 1]
        band_mags = magnitudes[:, start_bin:end_bin]
        rel = np.argmax(band_mags, axis=1)
        best_bins[band] = rel + start_bin
        best_mags[band] = band_mags[np.arange(frame_count), rel]

    # Adaptive threshold: a peak must beat the decaying running maximum of its
    # band. Frame-ordered, one band at a time, exactly as Fingerprinter.kt does.
    peaks = []
    thresholds = np.full(band_count, -float("inf"), dtype=np.float32)
    for frame in range(frame_count):
        for band in range(band_count):
            best_mag = float(best_mags[band, frame])
            if best_mag >= thresholds[band]:
                peaks.append((frame, int(best_bins[band, frame])))
                thresholds[band] = best_mag
            thresholds[band] -= THRESHOLD_DECAY

    # Pair each peak with the next FAN_OUT peaks inside the delta window.
    # Peaks are already sorted by frame.
    landmarks = []
    num_peaks = len(peaks)
    for i in range(num_peaks):
        anchor_frame, anchor_bin = peaks[i]
        fanned = 0
        j = i + 1
        while j < num_peaks and fanned < FAN_OUT:
            target_frame, target_bin = peaks[j]
            delta = target_frame - anchor_frame
            j += 1
            if delta < MIN_DELTA_FRAMES:
                continue
            if delta > MAX_DELTA_FRAMES:
                break
            packed = pack_landmark(anchor_bin, target_bin, delta)
            landmarks.append((packed, anchor_frame))
            fanned += 1

    return landmarks


# Canonical Recording Fingerprint (Haitsma-Kalker)
REC_BAND_COUNT = 33
REC_BITS = REC_BAND_COUNT - 1
REC_MIN_FREQ = 100.0
REC_MAX_FREQ = 3800.0
REC_TIME_DELTA_FRAMES = 4

_bins_per_hz = FRAME_SIZE / 8000.0
_ratio = (REC_MAX_FREQ / REC_MIN_FREQ) ** (1.0 / REC_BAND_COUNT)
REC_BAND_EDGES = [
    max(1, min(BIN_COUNT - 1, int(REC_MIN_FREQ * (_ratio ** i) * _bins_per_hz)))
    for i in range(REC_BAND_COUNT + 1)
]

def compute_recording_fingerprint(samples: np.ndarray, stft=None) -> bytes:
    """
    Computes Wanda's RecordingFingerprinter sub-hashes and packs into byte array for Room BLOB.

    `stft` may be shared with `compute_landmarks` / `extract_features`.
    """
    frame_count = frame_count_for(len(samples))
    if frame_count <= REC_TIME_DELTA_FRAMES:
        return b""

    if stft is None:
        stft = compute_stft(samples, frame_count)
    power = power_from(stft[:frame_count])

    # Log energy per logarithmically-spaced band. Bands whose edges collapse to
    # the same bin sum to zero and floor at 1e-9, as in the Kotlin original.
    energies = np.zeros((frame_count, REC_BITS + 1), dtype=np.float64)
    for band in range(REC_BAND_COUNT):
        b_start = REC_BAND_EDGES[band]
        b_end = REC_BAND_EDGES[band + 1]
        band_sum = np.sum(power[:, b_start:b_end], axis=1)
        energies[:, band] = np.log(np.maximum(band_sum, 1e-9))

    # Haitsma-Kalker: one bit per adjacent band pair, sign of the change in the
    # band-to-band energy gradient across REC_TIME_DELTA_FRAMES.
    gradient = energies[:, :REC_BITS] - energies[:, 1 : REC_BITS + 1]
    bits = (gradient[REC_TIME_DELTA_FRAMES:] - gradient[:-REC_TIME_DELTA_FRAMES]) > 0

    weights = (np.uint32(1) << np.arange(REC_BITS, dtype=np.uint32))
    hashes = bits.astype(np.uint32) @ weights

    # Big-endian 4-byte ints, as Room expects.
    return hashes.astype(">u4").tobytes()


# --------------------------------------------------------------------------
# Sub-hash index
# --------------------------------------------------------------------------

# Keeps a low half from colliding with a high half of the same value. Must stay
# identical to RecordingIdentityRepository.HIGH_HALF_TAG in the Android app.
HIGH_HALF_TAG = 1 << 16


def sub_hash_halves(fingerprint) -> set:
    """Both halves of every sub-hash, each tagged with the end it came from.

    Mirrors `RecordingIdentityRepository.halvesOf`. This is the inverted index
    the phone actually searches: `matchesForFingerprint` proposes candidates
    from `recording_sub_hashes` and only then compares full sequences. A
    fingerprint written without these rows can never be proposed, so storing one
    without the other silently disables recognition for that track.

    Why halves rather than whole hashes: a whole 32-bit sub-hash survives a
    re-container but finds *nothing* once the audio has been through a lossy
    encoder. Split in two, one damaged half leaves the other intact.

    A set, not a list: a repeated hash indexes once, so a passage that repeats
    does not outweigh one that does not.

    Accepts the stored BLOB (big-endian uint32, as written by
    `compute_recording_fingerprint`) or any array of ints.
    """
    if isinstance(fingerprint, (bytes, bytearray, memoryview)):
        raw = bytes(fingerprint)
        usable = len(raw) - (len(raw) % 4)
        if usable <= 0:
            return set()
        hashes = np.frombuffer(raw[:usable], dtype=">u4")
    else:
        hashes = np.asarray(fingerprint)
        if hashes.size == 0:
            return set()

    values = hashes.astype(np.int64)
    low = values & 0xFFFF
    high = ((values >> 16) & 0xFFFF) | HIGH_HALF_TAG
    return set(low.tolist()) | set(high.tolist())
