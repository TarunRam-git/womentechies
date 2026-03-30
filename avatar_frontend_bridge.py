from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from avatar_render import (
    load_sign_templates,
    point_valid,
    reconstruct_hand_absolute,
    remap_points,
    resolve_defaults,
    stretch_hand_for_visibility,
)


def _pt_to_list(pt: np.ndarray) -> list[float]:
    if not point_valid(pt):
        return [0.0, 0.0, 0.0, 0.0]
    return [float(pt[0]), float(pt[1]), float(pt[2]), 1.0]


def feature_to_landmarks(frame_feat: np.ndarray) -> dict:
    right_hand_rel = frame_feat[:63].reshape(21, 3)
    left_hand_rel = frame_feat[63:126].reshape(21, 3)
    pose8 = frame_feat[126:150].reshape(8, 3)

    left_shoulder = pose8[0]
    right_shoulder = pose8[1]
    left_wrist = pose8[4]
    right_wrist = pose8[5]

    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])) if point_valid(left_shoulder) and point_valid(right_shoulder) else 0.12
    hand_scale = float(np.clip(shoulder_width * 0.58, 0.07, 0.18))

    right_hand_abs = reconstruct_hand_absolute(right_hand_rel, right_wrist, hand_scale)
    left_hand_abs = reconstruct_hand_absolute(left_hand_rel, left_wrist, hand_scale)

    right_hand_abs = stretch_hand_for_visibility(right_hand_abs, right_wrist, gain=1.45)
    left_hand_abs = stretch_hand_for_visibility(left_hand_abs, left_wrist, gain=1.45)

    anchor_points = []
    for pt in [left_shoulder, right_shoulder, pose8[6], pose8[7]]:
        if point_valid(pt):
            anchor_points.append(pt[:2])
    if anchor_points:
        anchor_center = np.mean(np.array(anchor_points, dtype=np.float32), axis=0)
    else:
        anchor_center = np.array([0.5, 0.5], dtype=np.float32)

    target_center = np.array([0.5, 0.56], dtype=np.float32)
    center3 = np.array([anchor_center[0], anchor_center[1], 0.0], dtype=np.float32)
    target3 = np.array([target_center[0], target_center[1], 0.0], dtype=np.float32)

    pose8 = remap_points(pose8, center3, 1.20, target3)
    right_hand_abs = remap_points(right_hand_abs, center3, 1.20, target3)
    left_hand_abs = remap_points(left_hand_abs, center3, 1.20, target3)

    return {
        "pose": [_pt_to_list(p) for p in pose8],
        "left_hand": [_pt_to_list(p) for p in left_hand_abs],
        "right_hand": [_pt_to_list(p) for p in right_hand_abs],
    }


def main() -> None:
    default_dataset, default_labels = resolve_defaults()

    parser = argparse.ArgumentParser(description="Generate frontend-friendly avatar motion JSON from sentence")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--labels", type=Path, default=default_labels)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--blend-frames", type=int, default=10)
    args = parser.parse_args()

    seq, labels, spans = load_sign_templates(
        args.dataset,
        args.labels,
        args.text,
        blend_frames=args.blend_frames,
        return_spans=True,
    )

    frames = [feature_to_landmarks(seq[i]) for i in range(seq.shape[0])]
    motion = {
        "engine": "frontend-avatar",
        "fps": float(args.fps),
        "text": args.text,
        "tokens": labels,
        "token_spans": [{"start": int(s), "end": int(e), "token": t} for s, e, t in spans],
        "total_frames": len(frames),
        "frames": frames,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(motion), encoding="utf-8")
    print(str(args.out))


if __name__ == "__main__":
    main()
