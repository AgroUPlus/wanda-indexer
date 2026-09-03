import math
import numpy as np
from typing import Optional, Dict

from core.spectrogram import (
    BIN_COUNT,
    FRAME_SIZE,
    HANN_WINDOW,
    HOP_SIZE,
    compute_stft,
)

SAMPLE_RATE = 8000
MIN_FRAMES = 64

MIN_BPM = 60.0
MAX_BPM = 180.0
DEFAULT_BPM = 110.0
QUIET_DB = -40.0
LOUD_DB = -6.0
MIN_PITCH_HZ = 65.0
MAX_PITCH_HZ = 2000.0
TONALITY_GAIN = 3.5
FIFTHS_STEP = 7
FRAMES_PER_SECOND = SAMPLE_RATE / HOP_SIZE

_HANN_WINDOW = HANN_WINDOW  # re-exported for backwards compatibility

def normalise_tempo(bpm: float) -> float:
    return float(np.clip((bpm - MIN_BPM) / (MAX_BPM - MIN_BPM), 0.0, 1.0))

def key_point(pitch_class: int, strength: float) -> (float, float):
    position = (pitch_class * FIFTHS_STEP) % 12
    angle = 2.0 * math.pi * position / 12.0
    scale = max(0.0, min(1.0, strength))
    return (float(math.cos(angle) * scale), float(math.sin(angle) * scale))

def extract_features(samples: np.ndarray, stft=None) -> Optional[Dict[str, float]]:
    """
    Measures the 6D acoustic features matching FeatureExtractor.kt exactly.

    `stft` may be a spectrum already computed by `core.spectrogram.compute_stft`;
    it is shared with the fingerprinter so the FFT is paid for once.
    """
    if len(samples) < FRAME_SIZE * MIN_FRAMES:
        return None

    count = (len(samples) - FRAME_SIZE) // HOP_SIZE
    if count < MIN_FRAMES:
        return None

    # Magnitude spectrogram. FeatureExtractor.kt drops the final frame, so this
    # uses `count` frames where the fingerprinter uses one more.
    if stft is None:
        stft = compute_stft(samples, count)
    frames = np.abs(stft[:count, :BIN_COUNT]).astype(np.float32)

    # Onset Envelope (Spectral Flux)
    # diff across time: current - previous
    diff = np.diff(frames, axis=0)
    onsets = np.sum(np.maximum(diff, 0.0), axis=1)

    # Tempo & Pulse (Autocorrelation)
    mean_onset = float(np.mean(onsets))
    centred = onsets - mean_onset
    min_lag = int(60.0 * FRAMES_PER_SECOND / MAX_BPM)
    max_lag = min(int(60.0 * FRAMES_PER_SECOND / MIN_BPM), len(onsets) // 2)

    bpm = DEFAULT_BPM
    pulse = 0.0

    if max_lag > min_lag:
        best_lag = min_lag
        best_score = -float('inf')
        total_score = 0.0

        for lag in range(min_lag, max_lag + 1):
            score = float(np.mean(centred[:len(centred) - lag] * centred[lag:]))
            total_score += score
            if score > best_score:
                best_score = score
                best_lag = lag

        average = total_score / (max_lag - min_lag + 1)
        energy_variance = float(np.mean(centred ** 2))
        if energy_variance > 0:
            pulse = float(np.clip((best_score - average) / energy_variance, 0.0, 1.0))
        bpm = float(60.0 * FRAMES_PER_SECOND / best_lag)

    # Energy (RMS in dB)
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms <= 0:
        energy_db_norm = 0.0
    else:
        db = 20.0 * math.log10(rms)
        energy_db_norm = float(np.clip((db - QUIET_DB) / (LOUD_DB - QUIET_DB), 0.0, 1.0))

    # Brightness (Spectral Centroid)
    bin_indices = np.arange(BIN_COUNT)
    weighted_sum = np.sum(frames * bin_indices, axis=1)
    mag_sum = np.sum(frames, axis=1)
    valid_mask = mag_sum > 0
    if np.any(valid_mask):
        centroids = weighted_sum[valid_mask] / mag_sum[valid_mask]
        brightness = float(np.clip(np.mean(centroids) / BIN_COUNT, 0.0, 1.0))
    else:
        brightness = 0.0

    # Chroma
    chroma = np.zeros(12, dtype=np.float32)
    bin_hz = SAMPLE_RATE / FRAME_SIZE
    freqs = np.arange(BIN_COUNT) * bin_hz
    pitch_mask = (freqs >= MIN_PITCH_HZ) & (freqs <= MAX_PITCH_HZ)
    pitch_indices = np.where(pitch_mask)[0]
    
    midis = 69.0 + 12.0 * np.log2(freqs[pitch_indices] / 440.0)
    pitch_classes = (np.floor(midis).astype(int) % 12 + 12) % 12

    # Accumulated frame by frame with unbuffered adds: float32 summation is
    # order-dependent, and this is the order FeatureExtractor.kt uses.
    for frame in frames:
        np.add.at(chroma, pitch_classes, frame[pitch_indices])

    # Dominant Key
    total_chroma = float(np.sum(chroma))
    if total_chroma <= 0:
        pitch_class, tonality = 0, 0.0
    else:
        best_pc = int(np.argmax(chroma))
        share = chroma[best_pc] / total_chroma
        tonality = float(np.clip((share - 1.0 / 12.0) / (1.0 - 1.0 / 12.0), 0.0, 1.0) * TONALITY_GAIN)
        pitch_class, tonality = best_pc, float(min(1.0, tonality))

    key_x, key_y = key_point(pitch_class, tonality)

    if energy_db_norm <= 0 or math.isnan(bpm) or math.isnan(brightness):
        return None

    return {
        "tempo": normalise_tempo(bpm),
        "energy": energy_db_norm,
        "brightness": brightness,
        "danceability": pulse,
        "keyX": key_x,
        "keyY": key_y
    }
