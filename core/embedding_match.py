"""Scoring a clip against the neural fingerprint index, the way the phone does it.

This is the desktop half of `EmbeddingRepository.match` in the Android app. Both sides run the
identical `wanda_embedder.tflite`, so keeping the *scoring* identical too is what lets a number
measured here predict what the phone will do — a tool that ranked differently would report a
robustness the device does not have.

Kept in `core/` rather than in one of the tools because two of them need it: the offline
separability check (`tools/check_embedding_recognition.py`) and the degraded-audio field test
(`tools/field_test_recognition.py`). They previously carried their own copies, which is exactly
how two tools start disagreeing about what a match is.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.embedder import EMBED_DIM, EMBEDDER_VERSION, MODEL_NAME

# The gate `EmbeddingRepository` applies on the phone. A query that ranks first but does not
# clear both of these is still a "no match" to the user, so any tool reporting rank-1 alone
# overstates how well recognition works.
MIN_SIMILARITY = 0.55
MIN_MARGIN = 0.04


def unpack(blob: bytes) -> np.ndarray:
    """The stored BLOB as `(n_segments, EMBED_DIM)` unit rows.

    Big-endian float32, matching `core.embedder.compute_embedding` and `AudioEmbedder.pack`.
    """
    return np.frombuffer(blob, dtype=">f4").reshape(-1, EMBED_DIM).astype(np.float32)


def load_embeddings(conn) -> Dict[str, np.ndarray]:
    """Every stored embedding for the current model, keyed by track id.

    Filtered on model and version: rows written by an earlier embedder are not comparable, and
    silently mixing them would poison every score.
    """
    rows = conn.execute(
        "SELECT trackId, vector FROM track_embeddings WHERE model = ? AND version = ?",
        (MODEL_NAME, EMBEDDER_VERSION),
    ).fetchall()
    return {tid: unpack(blob) for tid, blob in rows}


def score(query: np.ndarray, track: np.ndarray) -> float:
    """How well `query` resembles `track`: the mean over query segments of the best cosine.

    High only when the clip's whole sequence has a counterpart somewhere in the track, so a
    coincidental timbre match on a single segment cannot carry it. Rows are already
    L2-normalised, so the dot product is the cosine.
    """
    if len(query) == 0 or len(track) == 0:
        return 0.0
    return float((query @ track.T).max(axis=1).mean())


# Kept under its historical name because the offline check reports it as a self-consistency
# statistic rather than as a retrieval score.
best_cosine = score


def rank(query: np.ndarray,
         catalogue: Sequence[Tuple[str, np.ndarray]]) -> List[Tuple[float, str]]:
    """Every catalogue entry scored against `query`, best first."""
    return sorted(((score(query, vectors), tid) for tid, vectors in catalogue), reverse=True)


def decide(ranked: Sequence[Tuple[float, str]]) -> Tuple[bool, str, float, float]:
    """Apply the phone's gate to a ranking.

    Returns `(accepted, track_id, similarity, margin)`. `accepted` is what the user would
    actually see: a name, or "not found".
    """
    if not ranked:
        return False, "", 0.0, 0.0
    best_score, best_id = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = best_score - runner_up
    accepted = best_score >= MIN_SIMILARITY and margin >= MIN_MARGIN
    return accepted, best_id, best_score, margin
