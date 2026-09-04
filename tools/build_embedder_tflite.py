"""Build models/wanda_embedder.tflite from the raraz15/neural-music-fp checkpoint.

Folds the repo's Essentia mel front-end into an in-graph tf.signal equivalent and prepends it
to the FingerPrinter encoder, so the exported .tflite takes raw 8 kHz mono PCM (1, 8000) and
emits a 128-d L2-normalised embedding (1, 128) -- the identical file then runs unchanged in the
Android app (assets/wanda_embedder.tflite) and in core/embedder.py.

Result: builtins-only (no Flex delegate), float16, ~35 MB. Mel front-end matches Essentia to
cosine 1.0000; TFLite matches the float32 TF model to cosine 1.0000 on the sanity clips.

Usage:
    # one-time setup (needs Python <=3.11)
    python -m venv tfenv && tfenv/bin/pip install "tensorflow-cpu==2.15.*" "numpy<2" essentia
    git clone https://github.com/raraz15/neural-music-fp
    # download nmfp-triplet.zip from Zenodo record 15719945, unzip so that
    #   <nmfp_repo>/logs/nmfp/fma-nmfp_deg/checkpoint/nmfp-triplet/ckpt-100.*  exists
    tfenv/bin/python tools/build_embedder_tflite.py <nmfp_repo> models/wanda_embedder.tflite

License note: neural-music-fp (code + weights) is AGPL-3.0. The Android app is AGPL-3.0, so the
combination is fine; keep the derived .tflite under the same terms.
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import essentia.standard as es

FS = 8000
SEG = 8000
N_FFT = 1024
HOP = 256
N_MELS = 256
F_MIN = 160.0
F_MAX = 4000.0
DR = 80.0
AMIN = 1e-5


def essentia_melspec(audio):
    audio = audio.astype(np.float32)
    fg = lambda x: es.FrameGenerator(x, frameSize=N_FFT, hopSize=HOP, startFromZero=False,
                                     lastFrameToEndOfFile=True, validFrameThresholdRatio=0)
    window = es.Windowing(type="hann", normalized=False, size=N_FFT,
                          symmetric=False, zeroPhase=False)
    spec = es.Spectrum(size=N_FFT)
    mb = es.MelBands(highFrequencyBound=F_MAX, inputSize=N_FFT // 2 + 1, log=False,
                     lowFrequencyBound=F_MIN, normalize="unit_tri", numberBands=N_MELS,
                     sampleRate=FS, type="magnitude", warpingFormula="slaneyMel",
                     weighting="linear")
    m = np.array([mb(spec(window(f))) for f in fg(audio)])
    m = np.where(m > AMIN, m, AMIN)
    m = 20 * np.log10(m / np.max(m))
    m = np.where(m > -DR, m, -DR)
    return (1 + m / (DR / 2)).T.astype(np.float32)


def essentia_mel_matrix():
    mb = es.MelBands(highFrequencyBound=F_MAX, inputSize=N_FFT // 2 + 1, log=False,
                     lowFrequencyBound=F_MIN, normalize="unit_tri", numberBands=N_MELS,
                     sampleRate=FS, type="magnitude", warpingFormula="slaneyMel",
                     weighting="linear")
    nbins = N_FFT // 2 + 1
    w = np.zeros((N_MELS, nbins), np.float32)
    for i in range(nbins):
        e = np.zeros(nbins, np.float32)
        e[i] = 1.0
        w[:, i] = mb(e)
    return w


def essentia_window():
    return es.Windowing(type="hann", normalized=False, size=N_FFT,
                        symmetric=False, zeroPhase=False)(np.ones(N_FFT, np.float32)).astype(np.float32)


class Frontend(tf.keras.layers.Layer):
    def __init__(self, mel_w, win, n_frames, **kw):
        super().__init__(**kw)
        self.mel_w = tf.constant(mel_w.T, tf.float32)
        self.win = tf.constant(win, tf.float32)
        self.n_frames = int(n_frames)

    def call(self, pcm):
        x = tf.pad(pcm, [[0, 0], [N_FFT // 2, N_FFT]])
        frames = tf.signal.frame(x, N_FFT, HOP)[:, : self.n_frames, :]
        mag = tf.abs(tf.signal.rfft(frames * self.win))
        mel = tf.maximum(tf.matmul(mag, self.mel_w), AMIN)
        peak = tf.reduce_max(mel, axis=[1, 2], keepdims=True)
        db = tf.maximum(20.0 * tf.math.log(mel / peak) / tf.math.log(10.0), -DR)
        return tf.transpose(1.0 + db / (DR / 2.0), [0, 2, 1])[..., tf.newaxis]


def main():
    nmfp_repo = sys.argv[1] if len(sys.argv) > 1 else "neural-music-fp"
    out = sys.argv[2] if len(sys.argv) > 2 else "models/wanda_embedder.tflite"
    sys.path.insert(0, nmfp_repo)
    ckpt_dir = os.path.join(nmfp_repo, "logs/nmfp/fma-nmfp_deg/checkpoint/nmfp-triplet")

    from nmfp.model.nnfp import FingerPrinter

    n_frames = essentia_melspec(np.zeros(SEG, np.float32)).shape[1]
    frontend = Frontend(essentia_mel_matrix(), essentia_window(), n_frames)

    fp = FingerPrinter(emb_sz=128, fc_unit_dim=[32, 1], norm="layer_norm2d", mixed_precision=False)
    fp.trainable = False
    tf.train.Checkpoint(model=fp).restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()

    class Embedder(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, SEG], tf.float32)])
        def __call__(self, pcm):
            return fp(frontend(pcm))

    m = Embedder()
    m(tf.zeros([1, SEG]))

    rng = np.random.default_rng(0)
    cos = []
    for _ in range(8):
        a = (rng.standard_normal(SEG) * 0.1).astype(np.float32)
        ref = fp(essentia_melspec(a)[None, :, :, None]).numpy()[0]
        got = m(a[None]).numpy()[0]
        cos.append(float(ref @ got))
    print("frontend parity cosine: min=%.4f mean=%.4f" % (min(cos), float(np.mean(cos))))

    conv = tf.lite.TFLiteConverter.from_concrete_functions(
        [m.__call__.get_concrete_function()], m)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "wb").write(tfl)
    print("wrote %s (%d KB)" % (out, len(tfl) // 1024))

    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    cos = []
    for _ in range(8):
        a = (rng.standard_normal((1, SEG)) * 0.1).astype(np.float32)
        it.set_tensor(i["index"], a)
        it.invoke()
        q = it.get_tensor(o["index"])[0]
        f = m(a).numpy()[0]
        cos.append(float(q / np.linalg.norm(q) @ (f / np.linalg.norm(f))))
    print("tflite vs tf cosine: min=%.4f mean=%.4f" % (min(cos), float(np.mean(cos))))


if __name__ == "__main__":
    main()
