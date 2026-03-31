import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .pose_player import PlaybackEvent

HAND_FINGER_CHAINS: List[List[int]] = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]


def get_opencv_gui_backend() -> str:
    try:
        build_info = cv2.getBuildInformation()
    except Exception:
        return "UNKNOWN"

    marker = "GUI:"
    idx = build_info.find(marker)
    if idx < 0:
        return "UNKNOWN"

    after = build_info[idx + len(marker) :].splitlines()
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


def _to_px(point: List[float], width: int, height: int) -> Tuple[int, int] | None:
    if len(point) < 2:
        return None
    return int(float(point[0]) * width), int(float(point[1]) * height)


def _make_background(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), (30, 34, 41), dtype=np.uint8)


def _draw_reference_avatar_base(canvas: np.ndarray):
    h, w = canvas.shape[:2]
    red = (0, 0, 255)
    lw = max(2, int(min(w, h) * 0.006))
    cx = int(w * 0.5)

    # Head and face lines.
    head_cy = int(h * 0.18)
    head_rx = int(w * 0.105)
    head_ry = int(h * 0.122)
    cv2.ellipse(canvas, (cx, head_cy), (head_rx, head_ry), 0, 0, 360, red, lw, cv2.LINE_AA)

    brow_y = int(head_cy - head_ry * 0.42)
    eye_y = int(head_cy - head_ry * 0.2)
    eye_dx = int(head_rx * 0.43)
    eye_rx = int(head_rx * 0.24)
    eye_ry = int(head_ry * 0.09)
    cv2.ellipse(canvas, (cx - eye_dx, brow_y), (eye_rx, eye_ry), -8, 190, 350, red, lw, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx + eye_dx, brow_y), (eye_rx, eye_ry), 8, 190, 350, red, lw, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx - eye_dx, eye_y), (eye_rx, eye_ry), 0, 0, 360, red, lw, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx + eye_dx, eye_y), (eye_rx, eye_ry), 0, 0, 360, red, lw, cv2.LINE_AA)

    mouth_y = int(head_cy + head_ry * 0.47)
    cv2.ellipse(canvas, (cx, mouth_y), (int(head_rx * 0.3), int(head_ry * 0.14)), 0, 0, 360, red, lw, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, int(mouth_y + head_ry * 0.06)), (int(head_rx * 0.2), int(head_ry * 0.08)), 0, 0, 180, red, lw, cv2.LINE_AA)

    # Red wireframe body.
    top_y = int(h * 0.49)
    side_y = int(h * 0.79)
    bottom_y = int(h * 0.99)

    left_top = (int(w * 0.26), top_y)
    right_top = (int(w * 0.74), top_y)
    left_side = (int(w * 0.12), side_y)
    right_side = (int(w * 0.88), side_y)
    left_bottom = (int(w * 0.39), bottom_y)
    right_bottom = (int(w * 0.61), bottom_y)

    cv2.line(canvas, left_top, right_top, red, lw, cv2.LINE_AA)
    cv2.line(canvas, left_top, left_side, red, lw, cv2.LINE_AA)
    cv2.line(canvas, right_top, right_side, red, lw, cv2.LINE_AA)
    cv2.line(canvas, left_top, left_bottom, red, lw, cv2.LINE_AA)
    cv2.line(canvas, right_top, right_bottom, red, lw, cv2.LINE_AA)
    cv2.line(canvas, left_bottom, right_bottom, red, lw, cv2.LINE_AA)


def _draw_reference_hand(canvas: np.ndarray, landmarks: List[List[float]], left_side: bool = False):
    if not landmarks:
        return

    h, w = canvas.shape[:2]
    lw = max(2, int(min(w, h) * 0.005))

    finger_colors = [
        (255, 176, 102),
        (255, 225, 160),
        (90, 240, 120),
        (96, 219, 242),
        (110, 154, 255),
    ]
    if left_side:
        finger_colors = list(reversed(finger_colors))

    palm_color = (244, 204, 161)

    for a, b in ((0, 5), (5, 9), (9, 13), (13, 17), (0, 17)):
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        p1 = _to_px(landmarks[a], w, h)
        p2 = _to_px(landmarks[b], w, h)
        if p1 is not None and p2 is not None:
            cv2.line(canvas, p1, p2, palm_color, lw, cv2.LINE_AA)

    for chain, color in zip(HAND_FINGER_CHAINS, finger_colors):
        for i in range(len(chain) - 1):
            a = chain[i]
            b = chain[i + 1]
            if a >= len(landmarks) or b >= len(landmarks):
                continue
            p1 = _to_px(landmarks[a], w, h)
            p2 = _to_px(landmarks[b], w, h)
            if p1 is not None and p2 is not None:
                cv2.line(canvas, p1, p2, color, lw, cv2.LINE_AA)

    for idx, lm in enumerate(landmarks):
        p = _to_px(lm, w, h)
        if p is None:
            continue
        r = max(2, lw - 1)
        c = (240, 240, 236) if idx == 0 else (232, 232, 232)
        cv2.circle(canvas, p, r, c, -1, cv2.LINE_AA)


def _draw_connected_arm(canvas: np.ndarray, shoulder_px: Tuple[int, int], wrist_px: Tuple[int, int]):
    h, w = canvas.shape[:2]
    mx = int((shoulder_px[0] + wrist_px[0]) * 0.5)
    my = int((shoulder_px[1] + wrist_px[1]) * 0.5)
    bend = max(10, int(min(w, h) * 0.035))
    direction = -1 if wrist_px[0] < shoulder_px[0] else 1
    elbow_px = (mx + direction * bend, my - bend)

    red = (0, 0, 255)
    lw = max(2, int(min(w, h) * 0.006))
    cv2.line(canvas, shoulder_px, elbow_px, red, lw, cv2.LINE_AA)
    cv2.line(canvas, elbow_px, wrist_px, red, lw, cv2.LINE_AA)


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


def _lerp_landmarks(a: List[List[float]], b: List[List[float]], t: float) -> List[List[float]]:
    if not a and not b:
        return []
    if not a:
        return b
    if not b:
        return a
    if len(a) != len(b):
        return b

    out: List[List[float]] = []
    for pa, pb in zip(a, b):
        n = min(len(pa), len(pb))
        mixed = [float((1.0 - t) * pa[i] + t * pb[i]) for i in range(n)]
        if len(pb) > n:
            mixed.extend(pb[n:])
        out.append(mixed)
    return out


def _interpolate_frame(a: dict, b: dict, t: float) -> dict:
    out: dict = {}
    for key in ("pose", "left_hand", "right_hand"):
        av = a.get(key, []) if isinstance(a, dict) else []
        bv = b.get(key, []) if isinstance(b, dict) else []
        if isinstance(av, list) and isinstance(bv, list):
            out[key] = _lerp_landmarks(av, bv, t)
    return out


def _densify_frames(frames: List[dict], factor: int = 2) -> List[dict]:
    if factor <= 1 or len(frames) <= 1:
        return frames

    dense: List[dict] = []
    inserts = factor - 1
    for i in range(len(frames) - 1):
        a = frames[i] if isinstance(frames[i], dict) else {}
        b = frames[i + 1] if isinstance(frames[i + 1], dict) else {}
        dense.append(a)
        for j in range(1, inserts + 1):
            t = j / float(factor)
            dense.append(_interpolate_frame(a, b, t))

    dense.append(frames[-1] if isinstance(frames[-1], dict) else {})
    return dense


def _smooth_frame(prev_frame: dict | None, frame: dict) -> dict:
    if prev_frame is None:
        return frame

    out = dict(frame)
    for key in ("pose", "left_hand", "right_hand"):
        prev = prev_frame.get(key, []) if isinstance(prev_frame, dict) else []
        curr = frame.get(key, []) if isinstance(frame, dict) else []
        if key == "pose" and isinstance(prev, list) and prev and isinstance(curr, list) and not curr:
            out[key] = prev
            continue
        if isinstance(prev, list) and isinstance(curr, list) and prev and curr:
            out[key] = _blend_landmarks(prev, curr)
    return out


def _hand_motion_score(curr_hand: List[List[float]], prev_hand: List[List[float]], width: int, height: int) -> float:
    if not curr_hand or not prev_hand:
        return 0.0

    wrist_c = _to_px(curr_hand[0], width, height) if len(curr_hand) > 0 else None
    wrist_p = _to_px(prev_hand[0], width, height) if len(prev_hand) > 0 else None
    if wrist_c is None or wrist_p is None:
        return 0.0

    score = float(np.hypot(float(wrist_c[0] - wrist_p[0]), float(wrist_c[1] - wrist_p[1])))
    for idx in (4, 8, 12, 16, 20):
        if idx < len(curr_hand) and idx < len(prev_hand):
            pc = _to_px(curr_hand[idx], width, height)
            pp = _to_px(prev_hand[idx], width, height)
            if pc is not None and pp is not None:
                score += 0.35 * float(np.hypot(float(pc[0] - pp[0]), float(pc[1] - pp[1])))
    return score


def _select_active_hand_side(
    left_hand: List[List[float]],
    right_hand: List[List[float]],
    prev_left_hand: List[List[float]],
    prev_right_hand: List[List[float]],
    width: int,
    height: int,
) -> str | None:
    has_left = bool(left_hand)
    has_right = bool(right_hand)

    if has_left and not has_right:
        return "left"
    if has_right and not has_left:
        return "right"
    if not has_left and not has_right:
        return None

    left_score = _hand_motion_score(left_hand, prev_left_hand, width, height)
    right_score = _hand_motion_score(right_hand, prev_right_hand, width, height)

    if left_score < 1.0 and right_score < 1.0:
        return "right"
    return "left" if left_score >= right_score else "right"


def _render_single_frame(canvas: np.ndarray, frame: dict, prev_frame: dict | None = None):
    _draw_reference_avatar_base(canvas)

    left_hand = frame.get("left_hand", []) if isinstance(frame, dict) else []
    right_hand = frame.get("right_hand", []) if isinstance(frame, dict) else []
    prev_left_hand = prev_frame.get("left_hand", []) if isinstance(prev_frame, dict) else []
    prev_right_hand = prev_frame.get("right_hand", []) if isinstance(prev_frame, dict) else []

    h, w = canvas.shape[:2]
    left_shoulder = (int(w * 0.26), int(h * 0.49))
    right_shoulder = (int(w * 0.74), int(h * 0.49))

    active_side = _select_active_hand_side(
        left_hand if isinstance(left_hand, list) else [],
        right_hand if isinstance(right_hand, list) else [],
        prev_left_hand if isinstance(prev_left_hand, list) else [],
        prev_right_hand if isinstance(prev_right_hand, list) else [],
        w,
        h,
    )

    # Show only the moving upper hand and keep it connected to its shoulder.
    if active_side == "left" and isinstance(left_hand, list) and left_hand:
        left_wrist = _to_px(left_hand[0], w, h)
        if left_wrist is not None:
            _draw_connected_arm(canvas, left_shoulder, left_wrist)
        _draw_reference_hand(canvas, left_hand, left_side=True)
    elif active_side == "right" and isinstance(right_hand, list) and right_hand:
        right_wrist = _to_px(right_hand[0], w, h)
        if right_wrist is not None:
            _draw_connected_arm(canvas, right_shoulder, right_wrist)
        _draw_reference_hand(canvas, right_hand, left_side=False)


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


def _placeholder_frames(count: int) -> List[dict]:
    return [{}] * max(1, count)


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
    interp_factor = 2
    dense_frames = _densify_frames(frames, factor=interp_factor)
    effective_fps = fps * interp_factor

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_video_path), fourcc, effective_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output video for writing: {out_video_path}")

    bg = _make_background(width, height)
    prev_frame: dict | None = None

    try:
        for frame in dense_frames:
            canvas = bg.copy()
            smoothed = _smooth_frame(prev_frame, frame if isinstance(frame, dict) else {})
            _render_single_frame(canvas, smoothed, prev_frame)
            prev_frame = smoothed
            writer.write(canvas)
    finally:
        writer.release()


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

    interp_factor = 2
    all_frames = _densify_frames(all_frames, factor=interp_factor)
    effective_fps = fps * interp_factor
    delay_ms = max(1, int(round(1000.0 / max(effective_fps, 1.0))))

    bg = _make_background(width, height)
    prev_frame: dict | None = None

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, width, height)

    for frame in all_frames:
        canvas = bg.copy()
        smoothed = _smooth_frame(prev_frame, frame if isinstance(frame, dict) else {})
        _render_single_frame(canvas, smoothed, prev_frame)
        cv2.imshow(window_title, canvas)
        prev_frame = smoothed

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyWindow(window_title)
