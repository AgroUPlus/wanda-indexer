# wanda_embedder.tflite

Neural audio-fingerprint model. Takes **raw 8 kHz mono float PCM `(1, 8000)`** (one 1-second
segment) and emits a **128-d L2-normalised embedding `(1, 128)`**. The mel front-end is folded
into the graph, so this exact file runs unchanged in:

- the desktop indexer — `core/embedder.py`
- the Android app — `app/src/main/assets/wanda_embedder.tflite`

There is therefore **no Python/Kotlin numerical parity to maintain**, unlike the landmark
fingerprinter.

## Provenance

- Encoder: **`nmfp-triplet`** from [raraz15/neural-music-fp](https://github.com/raraz15/neural-music-fp)
  ("Enhancing Neural Audio Fingerprint Robustness to Audio Degradation for Music Identification",
  ISMIR 2025) — the NNFP architecture (Chang et al., ICASSP 2021), trained with a triplet loss
  against codec / room-IR / mic-IR / background-noise degradation.
- Checkpoint `ckpt-100`, Zenodo record `15719945`.
- Front-end: the repo's Essentia mel-spectrogram (`n_fft=1024`, `hop=256`, `n_mels=256`,
  `f_min=160`, `f_max=4000`, 80 dB dynamic range, scaled to `[-1,1]`) reimplemented with
  `tf.signal` ops — matches Essentia to cosine `1.0000` on the sanity clips.
- Conversion: float16, **builtins-only** (no Flex/`SELECT_TF_OPS` delegate needed). TFLite
  output matches the float32 TF model to cosine `1.0000`.

Rebuild with `tools/build_embedder_tflite.py` (see its docstring).

## Segmentation contract (`EMBEDDER_VERSION = 1`)

1-second segments (8000 samples), 0.5-second hop (4000 samples), tail zero-padded. A track is
stored as its sequence of segment vectors, big-endian float32, in `track_embeddings.vector`.

## License

`neural-music-fp` (code **and** weights) is **AGPL-3.0**. The Wanda Android app is AGPL-3.0, so
the combination is compliant; this derived model inherits AGPL-3.0. Do not relicense.

## Size & hosting

~35 MB (`SHA-256 50a26e9f…55aa`, 35 892 012 bytes).

- **Desktop indexer**: keep the file here at `models/wanda_embedder.tflite` (or point
  `--embed-model` elsewhere). If it's awkward in git, add it to `.gitattributes` as LFS or
  fetch it in CI.
- **Android app**: **not bundled in the APK.** `EmbeddingModelManager` downloads it at runtime
  — offered during first-run setup (the "Song recognition" card) and from
  Settings → Fingerprints — into `filesDir/models/`, verified against the pinned SHA-256.
  Host it as a **GitHub Release asset** (LFS bandwidth is metered at 1 GiB/month on the free
  tier; release assets are not):

  ```
  gh release create embedder-v1 models/wanda_embedder.tflite \
    --repo AgroUPlus/Wanda --title "Recognition model v1" --notes "nmfp-triplet, 8kHz/128-d"
  ```

  When the model is rebuilt, bump `MODEL_URL` / `SHA256` / `SIZE_BYTES` in
  `EmbeddingModelManager.kt` **and** `EMBEDDER_VERSION` (invalidates stored vectors) together.
