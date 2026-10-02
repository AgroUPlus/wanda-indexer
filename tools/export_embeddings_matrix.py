#!/usr/bin/env python3
"""Exports an offline pairwise similarity matrix across indexed tracks.

Usage:
    python tools/export_embeddings_matrix.py [DATABASE] [--limit N] [--output OUT.csv]
"""
import argparse
import csv
import sqlite3
import numpy as np

from core.embedding_match import load_catalogue, score


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export similarity matrix for indexed embeddings")
    parser.add_argument("database", nargs="?", default="wanda_music.db", help="Path to database")
    parser.add_argument("--limit", type=int, default=50, help="Max tracks to evaluate (default: 50)")
    parser.add_argument("--output", default="similarity_matrix.csv", help="Output CSV path")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.database)
    catalogue = load_catalogue(conn)
    if not catalogue:
        print("No embeddings found in catalogue.")
        return

    titles = dict(conn.execute("SELECT id, artist || ' - ' || title FROM tracks").fetchall())
    keys = list(catalogue.keys())[: args.limit]
    n = len(keys)
    print(f"Calculating similarity matrix for {n} tracks...")

    matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        vi = catalogue[keys[i]]
        for j in range(i, n):
            vj = catalogue[keys[j]]
            s = score(vi, vj)
            matrix[i, j] = s
            matrix[j, i] = s

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "title"] + [titles.get(k, k) for k in keys])
        for i in range(n):
            row = [keys[i], titles.get(keys[i], keys[i])] + [f"{matrix[i, j]:.4f}" for j in range(n)]
            writer.writerow(row)

    print(f"Similarity matrix exported to {args.output}")


if __name__ == "__main__":
    main()
