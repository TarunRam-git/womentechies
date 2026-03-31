from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from avatar_render import render_frame, smooth_sequence_temporal
from isl_shared import BASE_FEATURE_DIM, FEATURE_DIM, SEQUENCE_LENGTH, HolisticFeatureExtractor


WINDOW_NAME = "Direction2 Avatar Capture"


@dataclass
class CaptureConfig:
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    capture_seconds: float = 2.0
    countdown_seconds: int = 2
    min_hand_frames: int = 12
    min_face_hand_frames: int = 10
    review_fps: float = 15.0
    review_size: int = 520


def normalize_label(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def is_window_closed(name: str) -> bool:
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def fit_to_window(frame: np.ndarray, window_name: str) -> tuple[np.ndarray, int, int]:
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
    except cv2.error:
        return frame, frame.shape[1], frame.shape[0]
    if win_w <= 0 or win_h <= 0:
        return frame, frame.shape[1], frame.shape[0]
    if frame.shape[1] == win_w and frame.shape[0] == win_h:
        return frame, win_w, win_h
    return cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR), win_w, win_h


def show_frame(window_name: str, frame: np.ndarray, state: dict) -> None:
    shown, win_w, win_h = fit_to_window(frame, window_name)
    state["source_w"] = int(frame.shape[1])
    state["source_h"] = int(frame.shape[0])
    state["shown_w"] = int(win_w)
    state["shown_h"] = int(win_h)
    cv2.imshow(window_name, shown)


def draw_button(frame: np.ndarray, rect: tuple[int, int, int, int], text: str, color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.putText(frame, text, (x0 + 16, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def overlay_lines(frame: np.ndarray, lines: list[str], start_y: int = 36) -> None:
    panel_h = max(40, (len(lines) * 28) + 20)
    panel_w = min(frame.shape[1] - 16, max(320, int(frame.shape[1] * 0.72)))
    cv2.rectangle(frame, (8, max(0, start_y - 30)), (8 + panel_w, max(8, start_y - 30 + panel_h)), (18, 18, 18), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (16, start_y + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)


def safe_button_rect(w: int, h: int, right_margin: int, bottom_margin: int, btn_w: int, btn_h: int) -> tuple[int, int, int, int]:
    x1 = max(16 + btn_w, w - right_margin)
    y1 = max(16 + btn_h, h - bottom_margin)
    x0 = max(8, x1 - btn_w)
    y0 = max(8, y1 - btn_h)
    return x0, y0, x1, y1


def load_classifications(labels_path: Path) -> list[str]:
    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    ordered = [normalize_label(raw[k]) for k in sorted(raw.keys(), key=lambda x: int(x))]
    return ordered


def on_mouse(event, x, y, flags, state) -> None:
    if event != cv2.EVENT_LBUTTONDOWN or not isinstance(state, dict):
        return
    source_w = int(state.get("source_w", 0))
    source_h = int(state.get("source_h", 0))
    shown_w = int(state.get("shown_w", 0))
    shown_h = int(state.get("shown_h", 0))

    if source_w > 0 and source_h > 0 and shown_w > 0 and shown_h > 0:
        fx = int((x / max(1, shown_w)) * source_w)
        fy = int((y / max(1, shown_h)) * source_h)
    else:
        fx, fy = x, y

    for name, rect in state.get("buttons", {}).items():
        x0, y0, x1, y1 = rect
        if x0 <= fx <= x1 and y0 <= fy <= y1:
            state["action"] = name
            break


def wait_for_start(
    cap: cv2.VideoCapture,
    current_label: str,
    next_label: str | None,
    state: dict,
) -> bool:
    state["action"] = ""
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        start_rect = (w - 220, 16, w - 24, 64)
        state["buttons"] = {"start": start_rect}

        draw_button(frame, start_rect, "START", (0, 170, 0))
        overlay_lines(
            frame,
            [
                f"Current classification: {current_label}",
                f"Next classification: {next_label if next_label is not None else 'done'}",
                "Record one sequence for this classification",
                "Click START or press 's' | q to quit",
            ],
            start_y=90,
        )

        show_frame(WINDOW_NAME, frame, state)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or is_window_closed(WINDOW_NAME):
            return False
        if key == ord("s") or state.get("action") == "start":
            state["action"] = ""
            return True


def record_single_sequence(
    cap: cv2.VideoCapture,
    extractor: HolisticFeatureExtractor,
    cfg: CaptureConfig,
    label: str,
    state: dict,
) -> tuple[np.ndarray | None, bool]:
    start_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        left = cfg.countdown_seconds - int(time.time() - start_t)
        overlay_lines(frame, [f"Get ready for: {label}", f"Starting in: {max(0, left)}"], start_y=40)
        show_frame(WINDOW_NAME, frame, state)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or is_window_closed(WINDOW_NAME):
            return None, True
        if (time.time() - start_t) >= cfg.countdown_seconds:
            break

    seq: list[np.ndarray] = []
    hand_count: list[int] = []
    face_hand_count: list[bool] = []
    rec_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        feat, details = extractor.extract_with_details(frame)
        extractor.draw_holistic(frame, details.results)

        if feat.shape[0] == FEATURE_DIM:
            seq.append(feat.astype(np.float32))
            hand_count.append(details.hand_count)
            face_hand_count.append(details.hand_count > 0 and details.face_detected and details.hand_face_distance is not None)

        elapsed = time.time() - rec_start
        overlay_lines(
            frame,
            [
                f"Recording: {label}",
                f"Captured frames: {len(seq)}",
                f"Time left: {max(0.0, cfg.capture_seconds - elapsed):.1f}s",
                "q to cancel",
            ],
            start_y=36,
        )

        show_frame(WINDOW_NAME, frame, state)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or is_window_closed(WINDOW_NAME):
            return None, True
        if elapsed >= cfg.capture_seconds:
            break

    if len(seq) < 8:
        return None, False

    seq_arr = np.stack(seq).astype(np.float32)
    idx = np.linspace(0, len(seq_arr) - 1, SEQUENCE_LENGTH, dtype=np.int32)
    sampled = seq_arr[idx]
    sampled = smooth_sequence_temporal(sampled, alpha=0.30)

    hand_arr = np.array(hand_count, dtype=np.float32)[idx] > 0
    face_hand_arr = np.array(face_hand_count, dtype=np.float32)[idx] > 0

    if int(np.sum(hand_arr)) < cfg.min_hand_frames:
        return None, False
    if int(np.sum(face_hand_arr)) < cfg.min_face_hand_frames:
        return None, False

    return sampled, False


def review_avatar_sequence(
    sequence: np.ndarray,
    label: str,
    next_label: str | None,
    cfg: CaptureConfig,
    state: dict,
) -> str:
    size = int(max(320, min(960, cfg.review_size)))
    base = np.full((size, size, 3), 32, dtype=np.uint8)
    cv2.rectangle(base, (0, 0), (size - 1, size - 1), (60, 60, 60), 1)

    frame_idx = 0
    last = time.perf_counter()

    while True:
        now = time.perf_counter()
        if now - last >= (1.0 / max(cfg.review_fps, 1.0)):
            frame_idx = (frame_idx + 1) % sequence.shape[0]
            last = now

        current = sequence[frame_idx]
        if current.shape[0] > BASE_FEATURE_DIM:
            current = current[:BASE_FEATURE_DIM]

        frame = render_frame(base, current, text=label, current_label=label)

        h, w, _ = frame.shape
        btn_h = 48
        btn_w = max(140, min(220, int(w * 0.30)))
        reject_rect = safe_button_rect(w, h, right_margin=24, bottom_margin=14, btn_w=btn_w, btn_h=btn_h)
        approve_rect = safe_button_rect(
            w,
            h,
            right_margin=24 + btn_w + 16,
            bottom_margin=14,
            btn_w=btn_w,
            btn_h=btn_h,
        )
        state["buttons"] = {"approve": approve_rect, "reject": reject_rect}

        draw_button(frame, approve_rect, "APPROVE", (0, 150, 0))
        draw_button(frame, reject_rect, "RE-RECORD", (30, 80, 210))

        overlay_lines(
            frame,
            [
                f"Review classification: {label}",
                f"Next classification: {next_label if next_label is not None else 'done'}",
                "a: approve | r: re-record | q/esc: quit",
            ],
            start_y=34,
        )

        show_frame(WINDOW_NAME, frame, state)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27) or is_window_closed(WINDOW_NAME):
            return "quit"
        if key == ord("a") or state.get("action") == "approve":
            state["action"] = ""
            return "approve"
        if key == ord("r") or state.get("action") == "reject":
            state["action"] = ""
            return "reject"


def save_progress(out_dataset: Path, out_labels: Path, labels: list[str], sequences: list[np.ndarray], y: list[int]) -> None:
    if not sequences:
        return
    X_arr = np.stack(sequences).astype(np.float32)
    y_arr = np.array(y, dtype=np.int64)

    out_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dataset, X=X_arr, y=y_arr)

    out_labels.parent.mkdir(parents=True, exist_ok=True)
    idx_to_label = {str(i): labels[i] for i in range(len(labels))}
    out_labels.write_text(json.dumps(idx_to_label, indent=2), encoding="utf-8")


def load_progress(out_dataset: Path, out_labels: Path, labels: list[str]) -> tuple[list[np.ndarray], list[int]]:
    if not out_dataset.exists() or not out_labels.exists():
        return [], []

    saved_labels = json.loads(out_labels.read_text(encoding="utf-8"))
    expected = {str(i): labels[i] for i in range(len(labels))}
    if saved_labels != expected:
        return [], []

    data = np.load(out_dataset)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    if X.ndim != 3 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        return [], []

    seqs = [X[i] for i in range(X.shape[0])]
    ys = [int(v) for v in y.tolist()]
    return seqs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description="Direction 2: record avatar templates per classification with approve/re-record")
    parser.add_argument("--labels", type=Path, default=Path("models/labels_10class_combined.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/avatar_direction2_templates.npz"))
    parser.add_argument("--labels-out", type=Path, default=Path("models/avatar_direction2_labels.json"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--capture-seconds", type=float, default=2.0)
    parser.add_argument("--min-hand-frames", type=int, default=12)
    parser.add_argument("--min-face-hand-frames", type=int, default=10)
    parser.add_argument("--review-size", type=int, default=520)
    parser.add_argument("--review-fps", type=float, default=15.0)
    parser.add_argument("--restart", action="store_true", help="Ignore previous saved progress and restart from class 1")
    parser.add_argument("--fullscreen", action="store_true", help="Open window in fullscreen mode")
    args = parser.parse_args()

    if not args.labels.exists():
        raise FileNotFoundError(f"Missing labels file: {args.labels}")

    class_labels = load_classifications(args.labels)
    if not class_labels:
        raise RuntimeError("No classifications found in labels file")

    cfg = CaptureConfig(
        camera_index=args.camera_index,
        capture_seconds=args.capture_seconds,
        min_hand_frames=args.min_hand_frames,
        min_face_hand_frames=args.min_face_hand_frames,
        review_size=args.review_size,
        review_fps=args.review_fps,
    )

    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    extractor = HolisticFeatureExtractor()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    state: dict = {"action": "", "buttons": {}}
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

    if args.restart:
        saved_sequences: list[np.ndarray] = []
        y_values: list[int] = []
    else:
        saved_sequences, y_values = load_progress(args.out, args.labels_out, class_labels)

    completed = {int(v) for v in y_values}
    next_start = 0
    while next_start < len(class_labels) and next_start in completed:
        next_start += 1

    if next_start > 0:
        print(f"Resuming from classification {next_start + 1}/{len(class_labels)}")

    try:
        for class_idx in range(next_start, len(class_labels)):
            label = class_labels[class_idx]
            next_label = class_labels[class_idx + 1] if (class_idx + 1) < len(class_labels) else None

            while True:
                start_ok = wait_for_start(cap, label, next_label, state)
                if not start_ok:
                    print("Stopped by user")
                    return

                seq, aborted = record_single_sequence(cap, extractor, cfg, label, state)
                if aborted:
                    print("Stopped by user")
                    return

                if seq is None:
                    ok, frame = cap.read()
                    if ok:
                        frame = cv2.flip(frame, 1)
                        overlay_lines(frame, ["Capture rejected (low visibility).", "Press START to record again."], start_y=52)
                        show_frame(WINDOW_NAME, frame, state)
                        cv2.waitKey(800)
                    continue

                decision = review_avatar_sequence(seq, label, next_label, cfg, state)
                if decision == "quit":
                    print("Stopped by user")
                    return
                if decision == "reject":
                    continue

                saved_sequences.append(seq)
                y_values.append(class_idx)
                save_progress(args.out, args.labels_out, class_labels, saved_sequences, y_values)

                ok, frame = cap.read()
                if ok:
                    frame = cv2.flip(frame, 1)
                    overlay_lines(
                        frame,
                        [
                            f"Saved: {label}",
                            f"Completed: {len(saved_sequences)}/{len(class_labels)}",
                            f"Next: {next_label if next_label is not None else 'done'}",
                        ],
                        start_y=54,
                    )
                    show_frame(WINDOW_NAME, frame, state)
                    cv2.waitKey(700)
                break
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved Direction2 templates: {args.out}")
    print(f"Saved Direction2 labels: {args.labels_out}")
    print(f"Saved classifications: {len(saved_sequences)}")


if __name__ == "__main__":
    main()
