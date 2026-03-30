from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from isl_shared import FEATURE_DIM, SEQUENCE_LENGTH, HolisticFeatureExtractor


WINDOW_NAME = "Collect Webcam Dataset"
START_BUTTON = (760, 14, 940, 58)


def is_window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def draw_start_button(frame: np.ndarray) -> None:
    x0, y0, x1, y1 = START_BUTTON
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 180, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.putText(
        frame,
        "START",
        (x0 + 42, y0 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def wait_for_manual_start(
    cap: cv2.VideoCapture,
    label: str,
    total_samples: int,
    class_idx: int,
    class_total: int,
    next_label: str | None,
    start_state: dict[str, bool],
) -> bool:
    start_state["requested"] = False
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)

        draw_start_button(frame)
        overlay_text(
            frame,
            [
                f"Paused -> Classification: {label} ({class_idx}/{class_total})",
                f"This run will capture {total_samples} samples",
                f"Upcoming classification: {next_label if next_label is not None else 'done'}",
                "Click START or press 's' to run this classification",
                "Press q to abort",
            ],
            start_y=90,
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            start_state["requested"] = True
        if key == ord("q") or is_window_closed(WINDOW_NAME):
            return True
        if start_state.get("requested", False):
            return False


def _on_mouse(event, x, y, flags, param) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if not isinstance(param, dict):
        return
    x0, y0, x1, y1 = START_BUTTON
    if x0 <= x <= x1 and y0 <= y <= y1:
        param["requested"] = True


@dataclass
class CollectConfig:
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    samples_per_label: int = 20
    min_hand_frames: int = 12
    min_face_hand_frames: int = 10
    capture_seconds: float = 2.0


def parse_labels(raw: str) -> list[str]:
    labels = [x.strip().lower().replace(" ", "_") for x in raw.split(",") if x.strip()]
    if not labels:
        raise ValueError("No labels parsed. Pass comma-separated labels.")
    return labels


def overlay_text(frame: np.ndarray, lines: list[str], start_y: int = 28) -> None:
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, start_y + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def countdown(frame: np.ndarray, sec_left: int, label: str, sample_idx: int, total_samples: int) -> None:
    text = f"Get ready: {label} [{sample_idx}/{total_samples}] in {sec_left}"
    cv2.putText(
        frame,
        text,
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def record_one_sequence(
    cap: cv2.VideoCapture,
    extractor: HolisticFeatureExtractor,
    label: str,
    sample_idx: int,
    total_samples: int,
    overall_idx: int,
    overall_total: int,
    next_label: str | None,
    min_hand_frames: int,
    min_face_hand_frames: int,
    capture_seconds: float,
    countdown_seconds: int = 2,
) -> tuple[np.ndarray | None, float, float, bool]:
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        elapsed = time.time() - start
        left = countdown_seconds - int(elapsed)
        countdown(frame, max(0, left), label, sample_idx, total_samples)
        overlay_text(
            frame,
            [
                f"Current: {label} ({overall_idx}/{overall_total})",
                f"Next: {next_label if next_label is not None else 'done'}",
            ],
            start_y=82,
        )
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or is_window_closed(WINDOW_NAME):
            return None, 0.0, 0.0, True
        if elapsed >= countdown_seconds:
            break

    seq: list[np.ndarray] = []
    hand_counts: list[int] = []
    face_hand_flags: list[bool] = []
    capture_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        feat, details = extractor.extract_with_details(frame)
        extractor.draw_holistic(frame, details.results)

        if feat.shape[0] == FEATURE_DIM:
            seq.append(feat)
            hand_counts.append(details.hand_count)
            face_hand_flags.append(details.hand_count > 0 and details.face_detected and details.hand_face_distance is not None)

        elapsed = time.time() - capture_start
        remaining = max(0.0, capture_seconds - elapsed)

        overlay_text(
            frame,
            [
                f"Recording: {label}",
                f"Sample: {sample_idx}/{total_samples}",
                f"Overall: {overall_idx}/{overall_total}",
                f"Captured frames: {len(seq)}",
                f"Time left: {remaining:.1f}s",
                f"Next: {next_label if next_label is not None else 'done'}",
                f"Face+hand frames: {int(np.sum(face_hand_flags))}",
                "Press q to abort",
            ],
            start_y=26,
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or is_window_closed(WINDOW_NAME):
            return None, 0.0, 0.0, True

        if elapsed >= capture_seconds:
            break

    if len(seq) < 8:
        return None, 0.0, 0.0, False

    seq_arr = np.stack(seq).astype(np.float32)
    hand_arr = np.array(hand_counts, dtype=np.float32)
    face_hand_arr = np.array(face_hand_flags, dtype=np.float32)
    idx = np.linspace(0, len(seq_arr) - 1, SEQUENCE_LENGTH, dtype=np.int32)
    seq_arr = seq_arr[idx]
    hand_sampled = hand_arr[idx] > 0
    face_hand_sampled = face_hand_arr[idx] > 0

    hand_ratio = float(np.mean(hand_sampled))
    face_hand_ratio = float(np.mean(face_hand_sampled))
    if int(np.sum(hand_sampled)) < min_hand_frames:
        return None, hand_ratio, face_hand_ratio, False
    if int(np.sum(face_hand_sampled)) < min_face_hand_frames:
        return None, hand_ratio, face_hand_ratio, False
    return seq_arr, hand_ratio, face_hand_ratio, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled webcam ISL sequences")
    parser.add_argument("--labels", type=str, required=True, help="Comma-separated labels, e.g. hello,thank_you,yes,no")
    parser.add_argument("--samples-per-label", type=int, default=20)
    parser.add_argument("--min-hand-frames", type=int, default=12)
    parser.add_argument("--min-face-hand-frames", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("data/processed/webcam_dataset.npz"))
    parser.add_argument("--labels-out", type=Path, default=Path("models/webcam_labels.json"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--capture-seconds", type=float, default=2.0)
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    cfg = CollectConfig(
        camera_index=args.camera_index,
        samples_per_label=args.samples_per_label,
        min_hand_frames=args.min_hand_frames,
        min_face_hand_frames=args.min_face_hand_frames,
        capture_seconds=args.capture_seconds,
    )

    cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    extractor = HolisticFeatureExtractor()

    class_to_idx = {label: i for i, label in enumerate(labels)}
    X: list[np.ndarray] = []
    y: list[int] = []
    per_label_done = {label: 0 for label in labels}

    def save_progress() -> None:
        if not X:
            return
        X_arr = np.stack(X).astype(np.float32)
        y_arr = np.array(y, dtype=np.int64)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, X=X_arr, y=y_arr)
        args.labels_out.parent.mkdir(parents=True, exist_ok=True)
        idx_to_label = {str(i): label for label, i in class_to_idx.items()}
        args.labels_out.write_text(json.dumps(idx_to_label, indent=2), encoding="utf-8")

    cv2.namedWindow(WINDOW_NAME)
    start_state: dict[str, bool] = {"requested": False}
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse, start_state)

    aborted = False
    try:
        for class_idx, label in enumerate(labels, start=1):
            next_label = labels[class_idx] if class_idx < len(labels) else None

            aborted = wait_for_manual_start(
                cap=cap,
                label=label,
                total_samples=cfg.samples_per_label,
                class_idx=class_idx,
                class_total=len(labels),
                next_label=next_label,
                start_state=start_state,
            )
            if aborted:
                break

            while per_label_done[label] < cfg.samples_per_label:
                sample_idx = per_label_done[label] + 1

                seq, hand_ratio, face_hand_ratio, aborted = record_one_sequence(
                    cap=cap,
                    extractor=extractor,
                    label=label,
                    sample_idx=sample_idx,
                    total_samples=cfg.samples_per_label,
                    overall_idx=class_idx,
                    overall_total=len(labels),
                    next_label=next_label,
                    min_hand_frames=cfg.min_hand_frames,
                    min_face_hand_frames=cfg.min_face_hand_frames,
                    capture_seconds=cfg.capture_seconds,
                )
                if aborted:
                    aborted = True
                    break

                if seq is None:
                    frame_ok, frame = cap.read()
                    if frame_ok:
                        frame = cv2.flip(frame, 1)
                        overlay_text(
                            frame,
                            [
                                f"Rejected (hands: {hand_ratio:.2f}, face+hand: {face_hand_ratio:.2f})",
                                f"Retry current classification: {label}",
                                f"Upcoming classification: {next_label if next_label is not None else 'done'}",
                            ],
                            start_y=48,
                        )
                        cv2.imshow(WINDOW_NAME, frame)
                        cv2.waitKey(900)
                        if is_window_closed(WINDOW_NAME):
                            aborted = True
                            break
                    continue

                X.append(seq)
                y.append(class_to_idx[label])
                per_label_done[label] += 1

                frame_ok, frame = cap.read()
                if frame_ok:
                    frame = cv2.flip(frame, 1)
                    overlay_text(
                        frame,
                        [
                            f"Captured: {label} ({per_label_done[label]}/{cfg.samples_per_label})",
                            f"Upcoming classification: {next_label if next_label is not None else 'done'}",
                        ],
                        start_y=48,
                    )
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(350)
                    if is_window_closed(WINDOW_NAME):
                        aborted = True
                        break
                if aborted:
                    break

            if aborted:
                break

            save_progress()
            frame_ok, frame = cap.read()
            if frame_ok:
                frame = cv2.flip(frame, 1)
                overlay_text(
                    frame,
                    [
                        f"Saved classification: {label}",
                        f"Saved {cfg.samples_per_label} samples to disk: {args.out.name}",
                        f"Upcoming classification: {next_label if next_label is not None else 'done'}",
                    ],
                    start_y=48,
                )
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(800)
                if is_window_closed(WINDOW_NAME):
                    aborted = True
                    break

            if aborted:
                break
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()

    if aborted:
        print("Capture stopped by user")
        return

    if not X:
        raise RuntimeError("No sequences collected")

    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.array(y, dtype=np.int64)

    print(f"Saved webcam dataset: {args.out}")
    print(f"Saved labels: {args.labels_out}")
    print(f"Collected sequences: {len(X_arr)} | Classes: {len(labels)}")


if __name__ == "__main__":
    main()
