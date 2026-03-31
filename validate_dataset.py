from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isl_shared import FEATURE_DIM, SEQUENCE_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate processed ISL dataset quality")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--min-samples-per-class", type=int, default=20)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Missing dataset: {args.dataset}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Missing labels: {args.labels}")

    data = np.load(args.dataset)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    labels_map = json.loads(args.labels.read_text(encoding="utf-8"))

    if X.ndim != 3:
        raise RuntimeError(f"X must be rank 3 [N,T,D], got shape {X.shape}")
    n, t, d = X.shape
    if t != SEQUENCE_LENGTH:
        raise RuntimeError(f"Expected sequence length {SEQUENCE_LENGTH}, got {t}")
    if d != FEATURE_DIM:
        raise RuntimeError(f"Expected feature dim {FEATURE_DIM}, got {d}")

    if np.isnan(X).any() or np.isinf(X).any():
        raise RuntimeError("Dataset has NaN/Inf values")

    unique, counts = np.unique(y, return_counts=True)
    if len(unique) != len(labels_map):
        raise RuntimeError(
            f"Label mismatch: dataset has {len(unique)} classes but labels map has {len(labels_map)}"
        )

    class_counts = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}
    missing = [i for i in range(len(labels_map)) if i not in class_counts]
    if missing:
        raise RuntimeError(f"Missing classes in dataset for indices: {missing}")

    min_count = int(counts.min())
    max_count = int(counts.max())
    hand_presence_ratio = float(np.mean(np.any(np.abs(X[:, :, :126]) > 1e-6, axis=2)))

    low_classes = [int(cls) for cls, cnt in class_counts.items() if cnt < args.min_samples_per_class]

    print(f"Samples: {n}")
    print(f"Classes: {len(unique)}")
    print(f"Min per class: {min_count}")
    print(f"Max per class: {max_count}")
    print(f"Hand-presence ratio: {hand_presence_ratio:.3f}")

    if low_classes:
        print(f"WARN: classes below {args.min_samples_per_class} samples: {low_classes}")
    else:
        print("Class-count check: OK")

    print("Dataset validation: OK")


if __name__ == "__main__":
    main()
