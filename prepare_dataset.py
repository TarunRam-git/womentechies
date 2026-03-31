from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from isl_shared import BASE_FEATURE_DIM, FEATURE_DIM, SEQUENCE_LENGTH, HolisticFeatureExtractor, rebuild_sequence_with_relational


VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".webm", ".m4v"}


def clean_label(folder_name: str) -> str:
    label = re.sub(r"^\s*\d+\.\s*", "", folder_name)
    label = label.strip().lower()
    label = label.replace(" ", "_")
    label = label.replace("(", "").replace(")", "")
    label = re.sub(r"[^a-z0-9_]+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label


def collect_videos(raw_root: Path) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        label = clean_label(path.parent.name)
        if label:
            items.append((label, path))
    return items


def _sample_frame_indices(total_frames: int) -> np.ndarray | None:
    if total_frames <= 0:
        return None
    usable = int(total_frames * 0.9)
    if usable < SEQUENCE_LENGTH:
        return None
    return np.linspace(0, usable - 1, SEQUENCE_LENGTH, dtype=np.int32)


def extract_video_sequence(
    video_path: Path,
    extractor: HolisticFeatureExtractor,
    min_hand_frames: int,
    min_face_hand_frames: int,
) -> tuple[np.ndarray | None, float, bool]:
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_idx = _sample_frame_indices(total_frames)

    if sampled_idx is not None:
        sampled_set = set(sampled_idx.tolist())
        features: list[np.ndarray] = []
        hand_counts: list[int] = []
        face_hand_flags: list[bool] = []
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx in sampled_set:
                    feat, details = extractor.extract_with_details(frame)
                    if feat.shape[0] != FEATURE_DIM:
                        return None, 0.0, False
                    features.append(feat)
                    hand_counts.append(details.hand_count)
                    face_hand_flags.append(details.hand_count > 0 and details.face_detected and details.hand_face_distance is not None)
                    if len(features) == SEQUENCE_LENGTH:
                        break
                frame_idx += 1
        finally:
            cap.release()

        if len(features) != SEQUENCE_LENGTH:
            return None, 0.0, False

        hand_presence = np.array(hand_counts, dtype=np.float32) > 0
        face_hand_presence = np.array(face_hand_flags, dtype=np.float32) > 0
        hand_ratio = float(np.mean(hand_presence))
        if int(np.sum(hand_presence)) < min_hand_frames:
            return None, hand_ratio, True
        if int(np.sum(face_hand_presence)) < min_face_hand_frames:
            return None, hand_ratio, True
        return np.stack(features).astype(np.float32), hand_ratio, False

    features_all: list[np.ndarray] = []
    hand_counts_all: list[int] = []
    face_hand_flags_all: list[bool] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            feat, details = extractor.extract_with_details(frame)
            if feat.shape[0] != FEATURE_DIM:
                return None, 0.0, False
            features_all.append(feat)
            hand_counts_all.append(details.hand_count)
            face_hand_flags_all.append(details.hand_count > 0 and details.face_detected and details.hand_face_distance is not None)
    finally:
        cap.release()

    if len(features_all) < SEQUENCE_LENGTH:
        return None, 0.0, False

    usable = int(len(features_all) * 0.9)
    if usable < SEQUENCE_LENGTH:
        return None, 0.0, False

    arr = np.stack(features_all[:usable]).astype(np.float32)
    hand_arr = np.array(hand_counts_all[:usable], dtype=np.float32)
    face_hand_arr = np.array(face_hand_flags_all[:usable], dtype=np.float32)
    idx = np.linspace(0, usable - 1, SEQUENCE_LENGTH, dtype=np.int32)
    sampled_hand = hand_arr[idx] > 0
    sampled_face_hand = face_hand_arr[idx] > 0
    hand_ratio = float(np.mean(sampled_hand))
    if int(np.sum(sampled_hand)) < min_hand_frames:
        return None, hand_ratio, True
    if int(np.sum(sampled_face_hand)) < min_face_hand_frames:
        return None, hand_ratio, True
    return arr[idx], hand_ratio, False


def augment_noise(seq: np.ndarray, std: float = 0.01) -> np.ndarray:
    base = seq[:, :BASE_FEATURE_DIM].astype(np.float32)
    noise = np.random.normal(0.0, std, size=base.shape).astype(np.float32)
    noisy = (base + noise).astype(np.float32)
    return rebuild_sequence_with_relational(noisy)


def augment_time_warp(seq: np.ndarray, start_frac: float, end_frac: float) -> np.ndarray:
    seq_full = seq[:, :FEATURE_DIM].astype(np.float32)
    n = seq.shape[0]
    start = int(max(0, min(n - 2, round(start_frac * (n - 1)))))
    end = int(max(start + 1, min(n - 1, round(end_frac * (n - 1)))))
    sub = seq_full[start : end + 1]
    if sub.shape[0] < 2:
        return seq_full.copy()
    idx = np.linspace(0, sub.shape[0] - 1, SEQUENCE_LENGTH, dtype=np.int32)
    return sub[idx].astype(np.float32)


def augment_shift(seq: np.ndarray, shift: float = 0.05) -> np.ndarray:
    shifted = seq[:, :BASE_FEATURE_DIM].copy().astype(np.float32)
    shifted[:, 126:] += np.random.uniform(-shift, shift, shifted[:, 126:].shape).astype(np.float32)
    shifted[:, 126:] = np.clip(shifted[:, 126:], 0.0, 1.0)
    return rebuild_sequence_with_relational(shifted)


def _mirror_triplets(block: np.ndarray, relative_x: bool) -> np.ndarray:
    points = block.reshape(block.shape[0], -1, 3).copy()
    zero_mask = np.all(np.isclose(points, 0.0), axis=2)
    if relative_x:
        points[:, :, 0] = -points[:, :, 0]
    else:
        points[:, :, 0] = 1.0 - points[:, :, 0]
    points[zero_mask] = 0.0
    return points.reshape(block.shape).astype(np.float32)


def augment_mirror(seq: np.ndarray) -> np.ndarray:
    base = seq[:, :BASE_FEATURE_DIM].astype(np.float32)
    rh = base[:, :63]
    lh = base[:, 63:126]
    pose = base[:, 126:150]
    face = base[:, 150:180]

    rh_m = _mirror_triplets(lh, relative_x=True)
    lh_m = _mirror_triplets(rh, relative_x=True)
    pose_m = _mirror_triplets(pose, relative_x=False)
    face_m = _mirror_triplets(face, relative_x=False)
    mirrored_base = np.concatenate([rh_m, lh_m, pose_m, face_m], axis=1).astype(np.float32)
    return rebuild_sequence_with_relational(mirrored_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ISL sequence dataset from raw videos")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/dataset.npz"))
    parser.add_argument("--labels-out", type=Path, default=Path("models/dynamic_labels.json"))
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--min-hand-frames", type=int, default=10)
    parser.add_argument("--min-face-hand-frames", type=int, default=8)
    parser.add_argument("--min-videos-per-class", type=int, default=3)
    args = parser.parse_args()

    videos = collect_videos(args.raw)
    if not videos:
        raise RuntimeError(f"No videos found under {args.raw}")

    counts = Counter(label for label, _ in videos)
    kept_labels = {label for label, cnt in counts.items() if cnt >= args.min_videos_per_class}
    dropped = sorted([(label, cnt) for label, cnt in counts.items() if label not in kept_labels], key=lambda x: x[0])
    videos = [(label, path) for label, path in videos if label in kept_labels]
    if not videos:
        raise RuntimeError("No videos remain after class-frequency filtering")
    if dropped:
        dropped_text = ", ".join([f"{label}:{cnt}" for label, cnt in dropped])
        print(f"Dropped low-frequency classes (<{args.min_videos_per_class} videos): {dropped_text}")

    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    classes = sorted({label for label, _ in videos})
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    extractor = HolisticFeatureExtractor()
    X: list[np.ndarray] = []
    y: list[int] = []
    skipped = 0
    skipped_low_hands = 0
    hand_ratios: list[float] = []

    try:
        total_videos = len(videos)
        for i, (label, video_path) in enumerate(videos, start=1):
            seq, hand_ratio, is_low_hands = extract_video_sequence(
                video_path,
                extractor,
                min_hand_frames=args.min_hand_frames,
                min_face_hand_frames=args.min_face_hand_frames,
            )
            if seq is None:
                skipped += 1
                if is_low_hands:
                    skipped_low_hands += 1
                continue

            hand_ratios.append(hand_ratio)

            target = class_to_idx[label]
            X.append(seq)
            y.append(target)

            X.append(augment_noise(seq))
            y.append(target)

            X.append(augment_time_warp(seq, 0.00, 0.75))
            y.append(target)

            X.append(augment_time_warp(seq, 0.25, 1.00))
            y.append(target)

            X.append(augment_mirror(seq))
            y.append(target)

            X.append(augment_shift(seq))
            y.append(target)

            if args.log_every > 0 and (i % args.log_every == 0 or i == total_videos):
                mean_hand_ratio = float(np.mean(hand_ratios)) if hand_ratios else 0.0
                print(
                    f"Processed {i}/{total_videos} videos | kept samples: {len(X)} | "
                    f"skipped videos: {skipped} | mean hand-presence: {mean_hand_ratio:.2f}"
                )
    finally:
        extractor.close()

    if not X:
        raise RuntimeError("No usable sequences after preprocessing")

    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.array(y, dtype=np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X_arr, y=y_arr)

    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    idx_to_label = {str(idx): label for label, idx in class_to_idx.items()}
    args.labels_out.write_text(json.dumps(idx_to_label, indent=2), encoding="utf-8")

    print(f"Saved dataset to: {args.out}")
    print(f"Saved labels to: {args.labels_out}")
    mean_hand_ratio = float(np.mean(hand_ratios)) if hand_ratios else 0.0
    print(
        f"Classes: {len(classes)} | Samples: {len(X_arr)} | Skipped videos: {skipped} "
        f"(low-hands: {skipped_low_hands}) | Mean hand-presence: {mean_hand_ratio:.2f}"
    )


if __name__ == "__main__":
    main()
