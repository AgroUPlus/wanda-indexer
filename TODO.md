# Wanda Indexer & App — Roadmap & Architecture Blueprint

This document tracks the current verified state of the Wanda indexing pipeline, what was accomplished, and the concrete engineering roadmap across both the Android application and the desktop indexer.

---

## 1. Accomplished & Verified So Far

### Recording Fingerprint Pipeline & Alignment Fix
- **Alignment issue identified & solved**: Codec round-trips (MP3 64k/128k, Opus 48k) retain >0.92 correlation, but small timing shifts (even 0.5s–3.0s intro or trimming) collapsed matches from 1.000 down to coin-flip (~0.50).
- **Vote-based alignment**: Implemented offset voting and refinement in both Python (`tools/check_recognition.py`) and Kotlin (`RecordingFingerprinter.kt` in the Android repo).
- **Sub-hash index coverage**: Fixed missing index entries; previously 1,238 / 1,355 tracks lacked index entries (91% invisible). Now 100% of tracks (1,355/1,355) are indexed in `recording_sub_hashes`.
- **Duplicate detection results**: Real library duplicate detection improved from 32 (or 44/60) to 43 of 44 duplicate pairs (59/60 app candidate pairs), with the only miss being an authentic alternate recording (*Mantis Lords*).
- **Git Repositories & PRs**:
  - `wanda-indexer`: Initialized, cleaned (`.gitignore`, `.env.example`, tools, tests), pushed to [`AgroUPlus/wanda-indexer:main`](https://github.com/AgroUPlus/wanda-indexer).
  - `Wanda` (Android): Committed vote-based alignment, tests, and `MIGRATION_23_24` (`WITHOUT ROWID`); opened PR [#67](https://github.com/AgroUPlus/Wanda/pull/67).

---

## 2. Outstanding Immediate Tasks

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

## 3. Short-Term Database Optimization (Existing SQLite Architecture)
The SQLite database currently measures **~3.38 GB for 1,355 tracks (~2.5 MB/track)**, which is heavy for pushing over USB to mobile storage.

1. **`WITHOUT ROWID` on `fingerprints` table**:
   - Landmark table contains ~24M rows (2.9 GB).
   - Switching primary index to `WITHOUT ROWID` saves duplicate B-tree overhead, cutting size down to ~2.0 GB. (Added in Android via `MIGRATION_23_24`).
2. **Landmark density reduction**:
   - Tune peak picker / landmark density thresholds to reduce landmark rows per second without sacrificing Shazam-like microphone match accuracy.
   - Target size: low hundreds of MB.
3. **Sub-hash query scaling**:
   - The `HAVING COUNT(*) >= 4` sub-hash filter currently clears for almost any track pair at scale. While fast at 1,355 tracks, at large library scales (e.g. 1M songs), candidate generation needs tighter sub-hash bucketing or min-hash filtering to avoid scanning entire libraries.

---

## 4. Next-Gen Neural Audio Embeddings (Unified Stage B Roadmap)

### Overview
Instead of handcrafting DSP rules (STFT peak detection $\rightarrow$ 18,000 landmark hashes/song $\rightarrow$ 32 filterbanks $\rightarrow$ 6 hand-picked acoustic features), replace the entire feature extraction and landmark subsystem with a compact **Convolutional Neural Network (CNN)** running as a **5–10 MB TensorFlow Lite (`.tflite`) or ONNX file**.

### Concrete Open-Source Candidate Architectures
1. **Google NNFP (Neural Audio Fingerprint)**:
   - The open academic re-implementation of the exact architecture Google uses for Sound Search and Now Playing.
   - Outputs a sequence of compact 64-bit or 128-bit hash vectors per second.
   - Robust against heavy background noise, mic distortions, and MP3/Opus compression.
2. **MERT / Discogs-EffNet / MusiCNN**:
   - Audio representations trained on millions of songs.
   - Directly maps audio segments into a dense 64-dim to 128-dim vector.
   - Dual capability: acts as both an acoustic fingerprint and a semantic mood/vibe vector for Smart Radio.
3. **CLMR (Contrastive Learning for Music Representation)**:
   - Self-supervised contrastive framework (SimCLR adapted to audio).
   - Can be easily fine-tuned or trained on consumer hardware (RTX 4070 Super) in an afternoon using audio augmentations (pitch shift, EQ, noise, MP3 round-trips).

### Comparison: Current System vs. Neural Embeddings

| Metric / Feature | Current System | Neural Embedding Model |
| :--- | :--- | :--- |
| **Size per track** | **~2.5 MB / track** (3.38 GB for 1,355 tracks) | **~256 B – 2 KB / track** (~350 KB for 1,355 tracks) |
| **Scale to 10M tracks** | **~25 Terabytes** (impossible on mobile) | **~25 to 30 Gigabytes** (fits easily in RAM / storage) |
| **DB Rows per track** | ~18,000 rows (24M rows for 1,355 tracks) | **1 single row** per track (`BLOB` vector column) |
| **Sync Time to Phone** | Minutes over USB (pushing 3.4 GB) | **< 1 second** |
| **Phone Inference Speed** | 200–800ms (SQL queries across millions of rows) | **~15ms** (running on Pixel 10 Tensor NPU) |
| **Code Maintenance** | Fragile bit-level math parity between Python & Kotlin | **Zero parity bugs**: exact same `.tflite` model runs on both |

### How It Interacts With Hum-to-Search
- **Acoustic Fingerprints vs. Hum-to-Search**:
  - **Acoustic Fingerprinting Models** (NNFP, MERT) listen for audio recordings (timbre, drums, mastering, vocals). They **cannot** recognize human humming, because a hum lacks the arrangement, instruments, and timbre of the studio master.
  - **Wanda's Architecture cleanly decouples them**:
    - `RecognitionEngine.LANDMARK` / Audio Identity: Replace with the **Neural Embedding Model**.
    - `RecognitionEngine.MELODY` / Hum-to-Search: Retain Wanda's existing **`ContourMatcher.kt`** (Dynamic Time Warping on relative pitch intervals) or adopt **Google SPICE** (Self-Play-based Pitch Extraction for melody matching).

### Step-by-Step Implementation Strategy
1. **Desktop Pipeline (`wanda-indexer`)**:
   - Integrate an ONNX / TFLite runtime in Python.
   - Compute the 64/128-dim vector per track during ingestion and store as a SQLite `BLOB` in a new `track_embeddings` table.
2. **Android Application (`Wanda`)**:
   - Add `org.tensorflow:tensorflow-lite:2.14.0` or `com.microsoft.onnxruntime:onnxruntime-android`.
   - Place `wanda_embedder.tflite` in `app/src/main/assets/`.
   - Update `RecognitionRepository` to evaluate microphone clips via the TFLite interpreter and run cosine similarity across loaded library vectors.

### Status (2026-09-04)

- **Model chosen**: `nmfp-triplet` from [raraz15/neural-music-fp](https://github.com/raraz15/neural-music-fp)
  (ISMIR 2025, AGPL-3.0 — matches Wanda-main's AGPL-3.0). 8 kHz / 1 s / 0.5 s hop / 128-d, exactly
  the existing pipeline's audio shape. Rejected MERT (CC-BY-NC + too heavy), pfann (no license).
- **`models/wanda_embedder.tflite` built**: mel front-end folded into the graph (raw PCM
  `(1,8000)` → `(1,128)`), builtins-only fp16, ~35 MB, cosine `1.0000` vs the float32 TF model.
  Rebuild: `tools/build_embedder_tflite.py`. Provenance/licence: `models/README.md`.
- **Desktop pipeline done** (`core/embedder.py`, `core/db_sync.py`, `main.py`): fourth extractor
  alongside landmarks/features/recording, writes `track_embeddings`, `--embed-model` /
  `--no-embed` flags, all existing tests pass. Landmarks still written in parallel (dual-index).
- **Android side scaffolded, uncompiled**: `TrackEmbeddingEntity`/`Dao`, `MIGRATION_24_25`
  (`@Database` → 25), `AudioEmbedder` (LiteRT), `EmbeddingRepository`, new
  `RecognitionEngine.EMBEDDING` path in `RecognitionRepository`, 5th measurement in
  `FingerprintIndexWorker`, `litert` gradle deps.
- **Model is downloaded at runtime, not bundled** (`EmbeddingModelManager`): offered on the
  first-run "Song recognition" card and in Settings → Fingerprints, fetched into `filesDir/`
  and SHA-256-verified. Host as a GitHub Release asset (`embedder-v1` tag) — LFS bandwidth is
  metered, release assets aren't. Keeps the APK small; recognition-by-embedding is simply off
  until the download completes.
- **Next**: publish the release asset; `gradlew assembleDebug` (generates schema `25.json`);
  tune `EmbeddingRepository` `MIN_SIMILARITY` / `MIN_MARGIN` on the 44-duplicate-pair
  benchmark (`tools/check_embedding_recognition.py`); then drop landmark writes.
