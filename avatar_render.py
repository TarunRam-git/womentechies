from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from isl_shared import BASE_FEATURE_DIM, SEQUENCE_LENGTH


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

SIMPLE_HAND_CONNECTIONS = [
    (0, 5), (5, 9), (9, 13), (13, 17), (17, 0),
    (0, 4), (5, 8), (9, 12), (13, 16), (17, 20),
]

FINGER_CONNECTION_GROUPS = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],
    [(0, 5), (5, 6), (6, 7), (7, 8)],
    [(0, 9), (9, 10), (10, 11), (11, 12)],
    [(0, 13), (13, 14), (14, 15), (15, 16)],
    [(0, 17), (17, 18), (18, 19), (19, 20)],
]

LEFT_FINGER_COLORS = [
    (75, 160, 255),
    (90, 220, 255),
    (110, 255, 180),
    (70, 240, 120),
    (255, 220, 170),
]

RIGHT_FINGER_COLORS = [
    (255, 150, 90),
    (255, 190, 90),
    (255, 230, 90),
    (170, 250, 120),
    (120, 220, 255),
]

POSE_CONNECTIONS = [
    (0, 1),  # shoulders
    (0, 2), (2, 4),  # left arm
    (1, 3), (3, 5),  # right arm
    (0, 6), (1, 7),  # torso sides
    (6, 7),  # hips
]


def canonical_label(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def tokenize_sentence(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [w for w in cleaned.split() if w]


def parse_sentence_to_labels(text: str, known_labels: set[str]) -> list[str]:
    direct = [canonical_label(t) for t in text.split() if t.strip()]
    if direct and all(t in known_labels for t in direct):
        return direct

    words = tokenize_sentence(text)
    if not words:
        return []

    max_parts = max((len(lb.split("_")) for lb in known_labels), default=1)
    out: list[str] = []
    i = 0
    while i < len(words):
        matched = None
        max_try = min(max_parts, len(words) - i)
        for n in range(max_try, 0, -1):
            candidate = "_".join(words[i : i + n])
            if candidate in known_labels:
                matched = candidate
                i += n
                break
        if matched is None:
            raise KeyError(f"Unknown word/phrase near: '{words[i]}'")
        out.append(matched)
    return out


def resolve_defaults() -> tuple[Path, Path]:
    return Path("data/processed/avatar_direction2_templates.npz"), Path("models/avatar_direction2_labels.json")


def smooth_sequence_temporal(sequence: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[0] < 2:
        return sequence.astype(np.float32)
    out = sequence.astype(np.float32).copy()
    for i in range(1, out.shape[0]):
        out[i] = (alpha * out[i]) + ((1.0 - alpha) * out[i - 1])
    return out


def build_transition_frames(start_frame: np.ndarray, end_frame: np.ndarray, n_frames: int) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((0, start_frame.shape[0]), dtype=np.float32)
    out: list[np.ndarray] = []
    for i in range(1, n_frames + 1):
        t = i / float(n_frames + 1)
        t = t * t * (3.0 - 2.0 * t)
        frame = ((1.0 - t) * start_frame) + (t * end_frame)
        out.append(frame.astype(np.float32))
    return np.stack(out).astype(np.float32)


def load_sign_templates(
    dataset_path: Path,
    labels_path: Path,
    text: str,
    blend_frames: int = 10,
    return_spans: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[np.ndarray, list[str], list[tuple[int, int, str]]]:
    data = np.load(dataset_path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    labels_map = json.loads(labels_path.read_text(encoding="utf-8"))
    label_to_idx = {canonical_label(v): int(k) for k, v in labels_map.items()}

    known_labels = set(label_to_idx.keys())
    tokens = parse_sentence_to_labels(text, known_labels)
    if not tokens:
        raise ValueError("No input tokens parsed. Example: --text happy")

    templates: list[np.ndarray] = []
    resolved: list[str] = []

    for token in tokens:
        if token not in label_to_idx:
            available = ", ".join(sorted(label_to_idx.keys())[:20])
            raise KeyError(f"Unknown sign '{token}'. Available examples: {available}")

        cls = label_to_idx[token]
        cls_seq = X[y == cls]
        if cls_seq.shape[0] == 0:
            raise RuntimeError(f"No samples found for '{token}' in dataset")

        template = cls_seq[-1].astype(np.float32)
        if template.shape[0] != SEQUENCE_LENGTH:
            idx = np.linspace(0, template.shape[0] - 1, SEQUENCE_LENGTH, dtype=np.int32)
            template = template[idx]

        if template.shape[1] > BASE_FEATURE_DIM:
            template = template[:, :BASE_FEATURE_DIM]

        templates.append(smooth_sequence_temporal(template, alpha=0.32))
        resolved.append(token)

    full_sequence: list[np.ndarray] = []
    token_spans: list[tuple[int, int, str]] = []
    cursor = 0
    for i, t in enumerate(templates):
        full_sequence.append(t)
        token_spans.append((cursor, cursor + t.shape[0], resolved[i]))
        cursor += t.shape[0]
        if i < len(templates) - 1:
            trans = build_transition_frames(t[-1], templates[i + 1][0], max(0, int(blend_frames)))
            if trans.shape[0] > 0:
                full_sequence.append(trans)
                cursor += trans.shape[0]

    seq = np.concatenate(full_sequence, axis=0).astype(np.float32)
    if return_spans:
        return seq, resolved, token_spans
    return seq, resolved


def point_valid(pt: np.ndarray) -> bool:
    return not np.all(np.isclose(pt[:2], 0.0, atol=1e-6))


def to_canvas(pt: np.ndarray, width: int, height: int, pad: int) -> tuple[int, int]:
    x = float(np.clip(pt[0], 0.0, 1.0))
    y = float(np.clip(pt[1], 0.0, 1.0))
    cx = int(pad + x * (width - 2 * pad))
    cy = int(pad + y * (height - 2 * pad))
    return cx, cy


def draw_connections(frame: np.ndarray, pts: np.ndarray, connections: list[tuple[int, int]], color: tuple[int, int, int], thickness: int) -> None:
    h, w, _ = frame.shape
    for a, b in connections:
        if not point_valid(pts[a]) or not point_valid(pts[b]):
            continue
        p1 = to_canvas(pts[a], w, h, pad=26)
        p2 = to_canvas(pts[b], w, h, pad=26)
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_points(frame: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], radius: int) -> None:
    h, w, _ = frame.shape
    for p in pts:
        if not point_valid(p):
            continue
        cv2.circle(frame, to_canvas(p, w, h, pad=26), radius, color, -1, cv2.LINE_AA)


def reconstruct_hand_absolute(hand_rel: np.ndarray, wrist_abs: np.ndarray, hand_scale: float) -> np.ndarray:
    hand_abs = np.zeros_like(hand_rel)
    if not point_valid(wrist_abs):
        return hand_abs
    hand_abs[:, :2] = wrist_abs[:2] + (hand_rel[:, :2] * hand_scale)
    hand_abs[:, 2] = hand_rel[:, 2]
    hand_abs[0, :2] = wrist_abs[:2]
    return hand_abs.astype(np.float32)


def stretch_hand_for_visibility(hand_abs: np.ndarray, wrist_abs: np.ndarray, gain: float = 1.75) -> np.ndarray:
    if not point_valid(wrist_abs):
        return hand_abs
    out = hand_abs.copy()
    for i in range(1, out.shape[0]):
        if not point_valid(out[i]):
            continue
        vec = out[i, :2] - wrist_abs[:2]
        out[i, :2] = wrist_abs[:2] + (vec * gain)
    return out.astype(np.float32)


def remap_points(points: np.ndarray, center: np.ndarray, scale: float, target_center: np.ndarray) -> np.ndarray:
    out = points.copy().astype(np.float32)
    for i in range(out.shape[0]):
        if not point_valid(out[i]):
            continue
        out[i, :2] = ((out[i, :2] - center[:2]) * scale) + target_center[:2]
    return out


def ellipsize(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def draw_finger_groups(frame: np.ndarray, hand_pts: np.ndarray, color_groups: list[tuple[int, int, int]], thickness: int) -> None:
    for conns, color in zip(FINGER_CONNECTION_GROUPS, color_groups):
        draw_connections(frame, hand_pts, conns, color, thickness)

    if point_valid(hand_pts[0]):
        draw_connections(frame, hand_pts, [(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)], (185, 185, 185), thickness)


def render_frame(base_frame: np.ndarray, frame_feat: np.ndarray, text: str, current_label: str) -> np.ndarray:
    frame = base_frame.copy()
    overlay = frame.copy()

    right_hand_rel = frame_feat[:63].reshape(21, 3)
    left_hand_rel = frame_feat[63:126].reshape(21, 3)
    pose = frame_feat[126:150].reshape(8, 3)
    face = frame_feat[150:180].reshape(10, 3)

    left_shoulder = pose[0]
    right_shoulder = pose[1]
    left_wrist = pose[4]
    right_wrist = pose[5]

    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])) if point_valid(left_shoulder) and point_valid(right_shoulder) else 0.12
    hand_scale = float(np.clip(shoulder_width * 0.58, 0.07, 0.18))

    right_hand_abs = reconstruct_hand_absolute(right_hand_rel, right_wrist, hand_scale)
    left_hand_abs = reconstruct_hand_absolute(left_hand_rel, left_wrist, hand_scale)

    right_hand_abs = stretch_hand_for_visibility(right_hand_abs, right_wrist, gain=1.45)
    left_hand_abs = stretch_hand_for_visibility(left_hand_abs, left_wrist, gain=1.45)

    anchor_points = []
    for pt in [left_shoulder, right_shoulder, pose[6], pose[7]]:
        if point_valid(pt):
            anchor_points.append(pt[:2])
    if anchor_points:
        anchor_center = np.mean(np.array(anchor_points, dtype=np.float32), axis=0)
    else:
        anchor_center = np.array([0.5, 0.5], dtype=np.float32)

    target_center = np.array([0.5, 0.56], dtype=np.float32)
    pose = remap_points(pose, np.array([anchor_center[0], anchor_center[1], 0.0], dtype=np.float32), 1.20, np.array([target_center[0], target_center[1], 0.0], dtype=np.float32))
    face = remap_points(face, np.array([anchor_center[0], anchor_center[1], 0.0], dtype=np.float32), 1.20, np.array([target_center[0], target_center[1] - 0.14, 0.0], dtype=np.float32))
    right_hand_abs = remap_points(right_hand_abs, np.array([anchor_center[0], anchor_center[1], 0.0], dtype=np.float32), 1.20, np.array([target_center[0], target_center[1], 0.0], dtype=np.float32))
    left_hand_abs = remap_points(left_hand_abs, np.array([anchor_center[0], anchor_center[1], 0.0], dtype=np.float32), 1.20, np.array([target_center[0], target_center[1], 0.0], dtype=np.float32))

    draw_connections(overlay, pose, POSE_CONNECTIONS, (0, 0, 255), 6)
    draw_finger_groups(overlay, left_hand_abs, LEFT_FINGER_COLORS, 5)
    draw_finger_groups(overlay, right_hand_abs, RIGHT_FINGER_COLORS, 5)
    draw_connections(overlay, left_hand_abs, SIMPLE_HAND_CONNECTIONS, (220, 240, 255), 3)
    draw_connections(overlay, right_hand_abs, SIMPLE_HAND_CONNECTIONS, (220, 255, 220), 3)

    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0.0, frame)

    draw_points(frame, pose, (210, 210, 255), 6)
    draw_points(frame, right_hand_abs, (250, 250, 250), 4)
    draw_points(frame, left_hand_abs, (250, 250, 250), 4)

    face_valid = np.array([point_valid(p) for p in face], dtype=bool)
    if np.any(face_valid):
        face_pts = face[face_valid]
        center = np.mean(face_pts[:, :2], axis=0)
        spread = float(np.max(np.linalg.norm(face_pts[:, :2] - center, axis=1)))
        rx = int(max(26, min(70, spread * base_frame.shape[1] * 1.8)))
        ry = int(max(34, min(92, spread * base_frame.shape[0] * 2.2)))
    else:
        shoulders = []
        if point_valid(left_shoulder):
            shoulders.append(left_shoulder[:2])
        if point_valid(right_shoulder):
            shoulders.append(right_shoulder[:2])
        if shoulders:
            s_center = np.mean(np.array(shoulders, dtype=np.float32), axis=0)
            center = np.array([s_center[0], s_center[1] - 0.2], dtype=np.float32)
        else:
            center = np.array([0.5, 0.22], dtype=np.float32)
        rx, ry = 42, 58

    h, w, _ = frame.shape
    hc = to_canvas(np.array([center[0], center[1], 0.0], dtype=np.float32), w, h, pad=26)
    cv2.ellipse(frame, hc, (rx, ry), 0, 0, 360, (0, 0, 200), 4, cv2.LINE_AA)
    eye_y = hc[1] - int(ry * 0.25)
    eye_dx = int(rx * 0.38)
    cv2.circle(frame, (hc[0] - eye_dx, eye_y), 5, (0, 0, 200), -1, cv2.LINE_AA)
    cv2.circle(frame, (hc[0] + eye_dx, eye_y), 5, (0, 0, 200), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (hc[0], hc[1] + int(ry * 0.26)), (int(rx * 0.34), int(ry * 0.16)), 0, 0, 180, (0, 0, 200), 4, cv2.LINE_AA)

    panel_w = min(frame.shape[1] - 10, max(340, int(frame.shape[1] * 0.86)))
    cv2.rectangle(frame, (10, 10), (panel_w, 86), (20, 20, 20), -1)
    cv2.putText(frame, f"Avatar sign: {ellipsize(current_label, 34)}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Text: {ellipsize(text, 56)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (10, frame.shape[0] - 38), (300, frame.shape[0] - 10), (18, 18, 18), -1)
    cv2.putText(frame, "q/esc: quit  r: replay", (20, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2, cv2.LINE_AA)

    return frame


def render_sentence(
    text: str,
    dataset_path: Path,
    labels_path: Path,
    fps: float = 15.0,
    size: int = 420,
    blend_frames: int = 10,
    fullscreen: bool = False,
) -> None:
    sequence, labels, token_spans = load_sign_templates(
        dataset_path,
        labels_path,
        text,
        blend_frames=blend_frames,
        return_spans=True,
    )

    canvas_size = int(max(280, min(960, size)))
    base = np.full((canvas_size, canvas_size, 3), 32, dtype=np.uint8)
    cv2.rectangle(base, (0, 0), (canvas_size - 1, canvas_size - 1), (60, 60, 60), 1)

    window = "ISL Avatar Renderer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window, 960, 640)

    frame_idx = 0
    last = time.perf_counter()

    while True:
        now = time.perf_counter()
        if now - last >= (1.0 / max(fps, 1.0)):
            frame_idx = (frame_idx + 1) % sequence.shape[0]
            last = now

        current_label = labels[-1]
        for s, e, token in token_spans:
            if s <= frame_idx < e:
                current_label = token
                break

        frame = render_frame(base, sequence[frame_idx], text, current_label)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("r"):
            frame_idx = 0

    cv2.destroyWindow(window)


def main() -> None:
    default_dataset, default_labels = resolve_defaults()

    parser = argparse.ArgumentParser(description="Offline lightweight avatar sign renderer")
    parser.add_argument("--text", type=str, default="", help="Sign text, e.g. 'happy' or 'good morning happy'")
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--labels", type=Path, default=default_labels)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--size", type=int, default=420)
    parser.add_argument("--blend-frames", type=int, default=10, help="Transition frames between words")
    parser.add_argument("--fullscreen", action="store_true", help="Open renderer in fullscreen mode")
    args = parser.parse_args()

    if not args.dataset.exists() or not args.labels.exists():
        raise FileNotFoundError(
            "Missing user-recorded avatar data. First run: "
            "py -3.12 direction2_avatar_capture.py --restart --labels models/labels_10class_combined.json "
            "--out data/processed/avatar_direction2_templates.npz --labels-out models/avatar_direction2_labels.json"
        )

    input_text = args.text.strip()
    if not input_text:
        input_text = input("Type sentence for avatar signing (example: good morning happy): ").strip()

    print(f"Using avatar dataset: {args.dataset}")
    print(f"Using avatar labels: {args.labels}")

    render_sentence(
        text=input_text,
        dataset_path=args.dataset,
        labels_path=args.labels,
        fps=args.fps,
        size=args.size,
        blend_frames=args.blend_frames,
        fullscreen=args.fullscreen,
    )


if __name__ == "__main__":
    main()
