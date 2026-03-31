import json
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .pose_player import PlaybackEvent


def get_opencv_gui_backend() -> str:
    try:
        build_info = cv2.getBuildInformation()
    except Exception:
        return "UNKNOWN"

    marker = "GUI:"
    idx = build_info.find(marker)
    if idx < 0:
        return "UNKNOWN"

    after = build_info[idx + len(marker):].splitlines()
    if not after:
        return "UNKNOWN"
    return after[0].strip().upper()


def ensure_live_gui_available():
    gui_backend = get_opencv_gui_backend()
    if gui_backend in {"NONE", "UNKNOWN"}:
        raise RuntimeError(
            "OpenCV GUI backend is unavailable (GUI: NONE). "
            "Live avatar window cannot be displayed. "
            "Run: python -m src.main --fix-opencv"
        )


# Minimal body graph for readable 2D avatar rendering from MediaPipe pose landmarks.
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
]

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (0, 5),
    (5, 6),
    (5, 9),
    (9, 10),
    (9, 13),
    (13, 14),
    (13, 17),
    (17, 18),
    (0, 17),
]

POSE_SMOOTH_ALPHA = 0.7
HAND_SMOOTH_ALPHA = 0.76


def _to_px(point: List[float], width: int, height: int) -> Tuple[int, int] | None:
    if len(point) < 2:
        return None
    x = int(float(point[0]) * width)
    y = int(float(point[1]) * height)
    return x, y


def _is_visible(point: List[float], visibility_threshold: float = 0.2) -> bool:
    if len(point) < 4:
        return True
    return float(point[3]) >= visibility_threshold


def _draw_graph(
    canvas: np.ndarray,
    landmarks: List[List[float]],
    edges: Iterable[Tuple[int, int]],
    color: Tuple[int, int, int],
    radius: int,
    line_thickness: int,
):
    h, w = canvas.shape[:2]

    for a, b in edges:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        pa = landmarks[a]
        pb = landmarks[b]
        if not _is_visible(pa) or not _is_visible(pb):
            continue
        p1 = _to_px(pa, w, h)
        p2 = _to_px(pb, w, h)
        if p1 is None or p2 is None:
            continue
        cv2.line(canvas, p1, p2, (24, 24, 24), line_thickness + 2, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, color, line_thickness, cv2.LINE_AA)

    for lm in landmarks:
        if not _is_visible(lm):
            continue
        p = _to_px(lm, w, h)
        if p is None:
            continue
        cv2.circle(canvas, p, radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(canvas, p, radius, color, -1, cv2.LINE_AA)


def _make_gradient_background(width: int, height: int) -> np.ndarray:
    top = np.array([14, 18, 26], dtype=np.float32)
    bottom = np.array([34, 40, 54], dtype=np.float32)
    alpha = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1, 1)
    grad = top * (1.0 - alpha) + bottom * alpha
    return np.repeat(grad, width, axis=1).astype(np.uint8)


def _blend_landmarks(prev: List[List[float]], curr: List[List[float]], alpha: float = 0.7) -> List[List[float]]:
    if not prev or len(prev) != len(curr):
        return curr

    blended: List[List[float]] = []
    for p_prev, p_curr in zip(prev, curr):
        n = min(len(p_prev), len(p_curr))
        merged = [float(alpha * p_prev[i] + (1.0 - alpha) * p_curr[i]) for i in range(n)]
        if len(p_curr) > n:
            merged.extend(p_curr[n:])
        blended.append(merged)
    return blended


def _smooth_frame(prev_frame: dict | None, frame: dict) -> dict:
    if prev_frame is None:
        return frame

    out = dict(frame)
    for key in ("pose", "left_hand", "right_hand"):
        prev = prev_frame.get(key, []) if isinstance(prev_frame, dict) else []
        curr = frame.get(key, []) if isinstance(frame, dict) else []
        if isinstance(prev, list) and isinstance(curr, list) and prev and curr:
            alpha = POSE_SMOOTH_ALPHA if key == "pose" else HAND_SMOOTH_ALPHA
            out[key] = _blend_landmarks(prev, curr, alpha=alpha)
    return out


def render_avatar_motion_to_video(
    motion_json_path: Path,
    out_video_path: Path,
    width: int = 960,
    height: int = 720,
    fps_override: float | None = None,
):
    with motion_json_path.open("r", encoding="utf-8") as f:
        motion = json.load(f)

    frames = motion.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames found in motion JSON: {motion_json_path}")

    fps = float(fps_override or motion.get("fps", 30.0) or 30.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video for writing: {out_video_path}")

    bg = _make_gradient_background(width, height)
    prev_frame: dict | None = None

    try:
        for frame in frames:
            canvas = bg.copy()
            smoothed = _smooth_frame(prev_frame, frame if isinstance(frame, dict) else {})
            _render_single_frame(canvas, smoothed)
            prev_frame = smoothed
            writer.write(canvas)
    finally:
        writer.release()


def _retime_frames(frames: List[dict], speed: float) -> List[dict]:
    if not frames:
        return []

    safe_speed = max(speed, 0.1)
    target_len = max(1, int(round(len(frames) / safe_speed)))
    if target_len == len(frames):
        return frames

    retimed: List[dict] = []
    for i in range(target_len):
        src_idx = min(int(i * len(frames) / target_len), len(frames) - 1)
        retimed.append(frames[src_idx])
    return retimed


def _render_single_frame(canvas: np.ndarray, frame: dict):
    pose = frame.get("pose", []) if isinstance(frame, dict) else []
    left_hand = frame.get("left_hand", []) if isinstance(frame, dict) else []
    right_hand = frame.get("right_hand", []) if isinstance(frame, dict) else []

    if isinstance(pose, list) and pose:
        _draw_graph(
            canvas=canvas,
            landmarks=pose,
            edges=POSE_CONNECTIONS,
            color=(80, 210, 255),
            radius=3,
            line_thickness=2,
        )

    if isinstance(left_hand, list) and left_hand:
        _draw_graph(
            canvas=canvas,
            landmarks=left_hand,
            edges=HAND_CONNECTIONS,
            color=(80, 255, 120),
            radius=2,
            line_thickness=2,
        )

    if isinstance(right_hand, list) and right_hand:
        _draw_graph(
            canvas=canvas,
            landmarks=right_hand,
            edges=HAND_CONNECTIONS,
            color=(255, 180, 70),
            radius=2,
            line_thickness=2,
        )


def _placeholder_frames(count: int) -> List[dict]:
    n = max(1, count)
    return [{}] * n


def play_events_live(
    events: List[PlaybackEvent],
    width: int = 960,
    height: int = 720,
    fps: float = 30.0,
    window_title: str = "ISL Avatar Live",
):
    ensure_live_gui_available()

    all_frames: List[dict] = []

    for event in events:
        if not event.pose_path:
            all_frames.extend(_placeholder_frames(int(round(fps * 0.35))))
            continue

        pose_path = Path(event.pose_path)
        if not pose_path.exists():
            all_frames.extend(_placeholder_frames(int(round(fps * 0.35))))
            continue

        with pose_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        frames = payload.get("frames", [])
        if not isinstance(frames, list) or not frames:
            all_frames.extend(_placeholder_frames(int(round(fps * 0.35))))
            continue

        all_frames.extend(_retime_frames(frames, event.speed))

    if not all_frames:
        raise ValueError("No pose frames available to render in live mode")

    delay_ms = max(1, int(round(1000.0 / max(fps, 1.0))))
    bg = _make_gradient_background(width, height)
    prev_frame: dict | None = None

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, width, height)

    for frame in all_frames:
        canvas = bg.copy()
        smoothed = _smooth_frame(prev_frame, frame)
        _render_single_frame(canvas, smoothed)
        cv2.imshow(window_title, canvas)
        prev_frame = smoothed

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyWindow(window_title)