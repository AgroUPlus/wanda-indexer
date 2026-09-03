# Wanda Indexer & App — Roadmap & What's Left To Be Done

This document tracks the current verified state of the Wanda indexing pipeline, what was accomplished, and the concrete tasks remaining across both the Android application and the desktop indexer.

---

## 1. Accomplished & Verified So Far

### Recording Fingerprint Pipeline & Alignment Fix
- **Alignment issue identified & solved**: Codec round-trips (MP3 64k/128k, Opus 48k) retain >0.92 correlation, but small timing shifts (even 0.5s–3.0s intro or trimming) collapsed matches from 1.000 down to coin-flip (~0.50).
- **Vote-based alignment**: Implemented offset voting and refinement in both Python (`tools/check_recognition.py`) and Kotlin (`RecordingFingerprinter.kt` in the Android repo).
- **Sub-hash index coverage**: Fixed missing index entries; previously 1,238 / 1,355 tracks lacked index entries (91% invisible). Now 100% of tracks (1,355/1,355) are indexed in `recording_sub_hashes`.
- **Duplicate detection results**: Real library duplicate detection improved from 32 (or 44/60) to 43 of 44 duplicate pairs (59/60 app candidate pairs), with the only miss being an authentic alternate recording (*Mantis Lords*).

---

## 2. Outstanding Tasks & Next Steps

### A. Android App Side (Kotlin / Gradle / Device)
1. **Compile & verify Kotlin alignment changes**:
   - The Kotlin implementation (`RecordingFingerprinter.kt` / `RecordingFingerprinterTest`) is reviewed but uncompiled in the Android repo.
   - Run tests on Windows/Android build environment:
     ```cmd
     gradlew clean
     gradlew test --tests "*RecordingFingerprinterTest*"
     ```
2. **Build debug APK & install on connected device**:
   - Device identified: Pixel 10 (Android 17, `61291FDCR000TP`).
   - Clean build on Windows (to avoid mixed state from earlier WSL attempts):
     ```cmd
     gradlew assembleDebug
     ```
   - Install APK via Windows `adb` to avoid WSL USB bridge issues:
     ```cmd
     "%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe" install -r app/build/outputs/apk/debug/app-debug.apk
     ```
3. **Verify duplicates merge on phone**:
   - Check duplicate pairs that failed previously:
     - *Bella Poarch – Stay Gone*
     - *Ado – Value*
     - *Ariana Grande – Touch It*
   - Verify that duplicate entries merge into a single entity and inbound Agro metadata resolves properly.

---

## 3. Database Footprint & Schema Optimization
The SQLite database currently measures **~3.38 GB for 1,355 tracks (~2.5 MB/track)**, which is heavy for pushing over USB to mobile storage.

1. **`WITHOUT ROWID` on `fingerprints` table**:
   - Landmark table contains ~24M rows (2.9 GB).
   - Switching primary index to `WITHOUT ROWID` saves duplicate B-tree overhead, cutting size down to ~2.0 GB.
2. **Landmark density reduction**:
   - Tune peak picker / landmark density thresholds to reduce landmark rows per second without sacrificing Shazam-like microphone match accuracy.
   - Target size: low hundreds of MB.
3. **Sub-hash query scaling**:
   - The `HAVING COUNT(*) >= 4` sub-hash filter currently clears for almost any track pair at scale. While fast at 1,355 tracks, at large library scales (e.g. 1M songs), candidate generation needs tighter sub-hash bucketing or min-hash filtering to avoid scanning entire libraries.

---

## 4. Feature Learning & Smart Radio Optimization (GPU / Stage A & B)
Currently, Smart Radio relies on 6 handcrafted acoustic features (tempo, energy, brightness, danceability, key x/y) weighted by fixed coefficients:
- `TEMPO_WEIGHT = 1.6`
- `ENERGY_WEIGHT = 1.3`
- `BRIGHTNESS_WEIGHT = 0.8` (measured to only span 28% of range, contributing minimal signal)
- `DANCE_WEIGHT = 1.0`
- `KEY_WEIGHT = 0.6`

### Stage A — Weight Fitting (CPU, Minutes)
- Use ground-truth pairs from the library (4,515 track pairs from the same album and artist).
- Optimize feature weights via metric learning (e.g., contrastive loss / triplet loss or logistic regression) to maximize intra-album/artist similarity and spread unrelated tracks.
- Normalize or rescale brightness to utilize its dynamic range.

### Stage B — Learned Neural Audio Embeddings (RTX 4000 GPU, Hours)
- Replace or augment handcrafted features with a compact learned embedding (~64–96 dimensions) trained using self-supervised contrastive learning (SimCLR / MoCo / triplet audio).
- Augmentations: pitch shift, EQ distortion, Gaussian noise, codec re-encodes (MP3/Opus).
- Benefits:
  - Dramatically improves Smart Radio similarity quality.
  - Potential to shrink fingerprint footprint from ~2.5 MB/track to < 3 KB/track (approaching Now Playing efficiency).
