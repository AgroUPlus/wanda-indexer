#!/usr/bin/env python3
"""Do the phone and this machine compute the same vectors for the same audio?

The one question neither side can answer alone. A track measured here is
compared, on the phone, against a clip measured there -- so a disagreement about
segment boundaries or quantisation does not degrade the match, it destroys it,
while both sides go on looking perfectly correct in isolation.

Debug builds of the app write two files after every recognition attempt:

    files/capture/last-listen.f32            the exact float PCM it was given
    files/capture/last-listen-embedding.i8   the vectors it computed from it

This feeds the first through `core/embedder.py` and compares the result with the
second. Identical input, so anything but a near-perfect match is a real
divergence, not a difference in what was heard.

    ./tools/check_parity.py                  # pulls both files over adb
    ./tools/check_parity.py --pcm a --vec b  # or compare files you already have

Open the recogniser on the phone once before running it, so the files exist.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import db_sync, embedder

CAPTURE_PCM = "files/capture/last-listen.f32"
CAPTURE_VEC = "files/capture/last-listen-embedding.i8"


def pull(package: str, remote: str, local: str) -> bool:
    adb = db_sync.get_adb_binary()
    if not adb:
        print("[ERROR] adb not found on PATH.")
        return False
    with open(local, "wb") as out:
        proc = subprocess.run(
            [adb, "exec-out", "run-as", package, "cat", remote],
            stdout=out, stderr=subprocess.PIPE,
        )
    if proc.returncode != 0 or os.path.getsize(local) == 0:
        print(f"[ERROR] could not read {remote}: "
              f"{proc.stderr.decode(errors='replace').strip() or 'empty file'}")
        print("        The app must be a debug build, and the recogniser must have "
              "been opened at least once.")
        return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", default=db_sync.DEFAULT_PACKAGE)
    parser.add_argument("--pcm", default="", help="Local float32 PCM instead of pulling it")
    parser.add_argument("--vec", default="", help="Local int8 vectors instead of pulling them")
    parser.add_argument("--embedder-model", default="")
    parser.add_argument("--cache-dir", default=".wanda-cache")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        pcm_path = args.pcm or os.path.join(tmp, "clip.f32")
        vec_path = args.vec or os.path.join(tmp, "clip.i8")
        if not args.pcm and not pull(args.package, CAPTURE_PCM, pcm_path):
            return 1
        if not args.vec and not pull(args.package, CAPTURE_VEC, vec_path):
            return 1

        # Little-endian float32, as MicRecorder.dumpForDiagnosis writes it.
        pcm = np.fromfile(pcm_path, dtype="<f4")
        theirs = embedder.unpack(open(vec_path, "rb").read())

        print(f"clip:   {len(pcm):,} samples ({len(pcm) / embedder.SAMPLE_RATE:.1f}s at "
              f"{embedder.SAMPLE_RATE} Hz)")
        try:
            ours = embedder.embed(pcm, args.cache_dir, args.embedder_model)
        except embedder.EmbedderUnavailable as exc:
            print(f"[ERROR] {exc}")
            return 1

        print(f"phone:  {theirs.shape[0]} segments")
        print(f"here:   {ours.shape[0]} segments")
        if ours.shape[0] != theirs.shape[0]:
            print("[FAIL] Different segment counts -- the segmentation has diverged. "
                  "Compare `segment()` here with `AudioEmbedder.segment`.")
            return 1

        # Cosine per segment. Both sides are unit vectors, so this is the dot
        # product, and anything below ~0.999 is a real disagreement rather than
        # the quantiser: the *input* is identical, so only the arithmetic can
        # differ.
        cos = np.sum(ours * theirs, axis=1)
        print(f"cosine: min {cos.min():.6f}  mean {cos.mean():.6f}  max {cos.max():.6f}")

        worst = int(np.argmin(cos))
        if cos.min() >= 0.999:
            print("[OK]   The two engines agree.")
            return 0
        if cos.min() >= 0.99:
            print(f"[WARN] Close but not exact; worst is segment {worst}. Expected when the "
                  "model file differs between the two sides -- check the SHA-256.")
            return 0
        print(f"[FAIL] Segment {worst} disagrees ({cos.min():.4f}). Suspect, in order: the "
              "quantisation scale, the segment hop, and the model file.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
