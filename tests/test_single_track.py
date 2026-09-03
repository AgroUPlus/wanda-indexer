"""End-to-end check on one real track. Requires network + yt-dlp + ffmpeg.

    python3 tests/test_single_track.py [VIDEO_ID]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.acoustic_features import extract_features
from core.audio_pipeline import DECODE_OK, decode_audio_to_pcm
from core.fingerprinter import compute_landmarks, compute_recording_fingerprint
from core.spectrogram import compute_stft, frame_count_for
from main import pitch_name_from_coords
from sources.ytm import get_ytm_stream_url_with_reason

DEFAULT_VIDEO_ID = "--zACCrlmFQ"  # natori - Overdose


def main(video_id: str) -> int:
    print(f"[1/3] Resolving stream URL for {video_id} ...")
    started = time.time()
    url, reason = get_ytm_stream_url_with_reason(video_id)
    if not url:
        print(f"  FAILED: {reason}")
        if reason == "RATE_LIMITED":
            print("  YouTube is rate limiting this machine. Retry later, or pass")
            print("  --cookies-from-browser to the indexer.")
        return 1
    print(f"  ok in {time.time() - started:.2f}s")

    print("[2/3] Decoding the first 60s in memory ...")
    started = time.time()
    samples, decode_reason = decode_audio_to_pcm(url, max_seconds=60)
    if samples is None or decode_reason != DECODE_OK:
        print(f"  FAILED: {decode_reason}")
        return 1
    print(f"  ok in {time.time() - started:.2f}s ({len(samples):,} samples)")

    print("[3/3] Running the DSP ...")
    started = time.time()
    stft = compute_stft(samples, frame_count_for(len(samples)))
    landmarks = compute_landmarks(samples, stft)
    features = extract_features(samples, stft)
    recording = compute_recording_fingerprint(samples, stft)
    print(f"  ok in {time.time() - started:.3f}s")

    print()
    print(f"  landmarks   : {len(landmarks):,}")
    print(f"  recording fp: {len(recording):,} bytes")
    if features:
        print(f"  tempo       : {int(round(60 + features['tempo'] * 120))} BPM")
        print(f"  energy      : {int(round(features['energy'] * 100))}%")
        print(f"  key         : {pitch_name_from_coords(features['keyX'], features['keyY'])}")
    else:
        print("  features    : not extracted (track too short or silent)")

    assert landmarks, "no landmarks produced"
    assert recording, "no recording fingerprint produced"
    print("\nEnd-to-end test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_ID))
