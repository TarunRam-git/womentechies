from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyttsx3

from isl_shared import FEATURE_DIM, SEQUENCE_LENGTH, HolisticFeatureExtractor, softmax


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    confidence_threshold: float = 0.75
    min_margin_threshold: float = 0.07
    min_hand_presence_ratio: float = 0.55
    cooldown_sec: float = 1.5
    prob_ema_alpha: float = 0.62
    stability_window: int = 7
    stable_frames_required: int = 5
    consensus_ratio_threshold: float = 0.72
    repeat_token_cooldown_sec: float = 2.2
    draw_landmarks: bool = True
    headless: bool = False
    state_out_path: Path | None = None
    model_path: Path = Path("models/dynamic_lstm.onnx")
    labels_path: Path = Path("models/dynamic_labels.json")


def resolve_best_compatible_model(default_model: Path, labels_path: Path) -> Path:
    if not labels_path.exists():
        return default_model

    labels_map = json.loads(labels_path.read_text(encoding="utf-8"))
    expected_classes = len(labels_map)
    models_dir = default_model.parent
    best_model = default_model
    best_score = float("-inf")

    for meta_path in models_dir.glob("dynamic_meta*.json"):
        stem_suffix = meta_path.stem.replace("dynamic_meta", "")
        onnx_name = f"dynamic_lstm{stem_suffix}.onnx"
        onnx_path = models_dir / onnx_name
        if not onnx_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if int(meta.get("feature_dim", -1)) != FEATURE_DIM:
                continue
            if int(meta.get("num_classes", -1)) != expected_classes:
                continue
            score = float(meta.get("val_accuracy", 0.0))
            if score > best_score:
                best_score = score
                best_model = onnx_path
        except Exception:
            continue

    return best_model


class SpeechWorker:
    def __init__(self) -> None:
        self._q: queue.Queue[str] = queue.Queue(maxsize=24)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def speak(self, text: str) -> None:
        if not text:
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            return

    def stop(self) -> None:
        self._stop.set()
        self._q.put("__STOP__")
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        while not self._stop.is_set():
            msg = self._q.get()
            if msg == "__STOP__":
                break
            engine.say(msg)
            engine.runAndWait()


class RealtimeISLApp:
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self._window_name = "Offline ISL -> Text/Speech"
        self._extractor = HolisticFeatureExtractor()
        try:
            import onnxruntime as ort
        except Exception as ex:
            raise RuntimeError(
                "onnxruntime is not available in this Python environment. "
                "Use Python 3.11/3.12 and install: pip install -r requirements.txt"
            ) from ex
        self._session = ort.InferenceSession(cfg.model_path.as_posix(), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._labels = json.loads(cfg.labels_path.read_text(encoding="utf-8"))
        self._validate_model_contract()

        self._buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self._hand_presence: deque[int] = deque(maxlen=SEQUENCE_LENGTH)
        self._top_history: deque[int] = deque(maxlen=max(3, int(cfg.stability_window)))
        self._speech = SpeechWorker()

        self._last_accept_t = 0.0
        self._last_token_t = 0.0
        self._sentence_tokens: list[str] = []
        self._candidate_idx: int | None = None
        self._candidate_streak = 0
        self._prob_ema: np.ndarray | None = None
        self._display_text = "-"
        self._display_conf = 0.0
        self._display_margin = 0.0
        self._display_consensus = 0.0
        self._display_streak = 0
        self._status = "collecting"
        self._fps = 0.0

    def _write_state(self) -> None:
        if self._cfg.state_out_path is None:
            return
        payload = {
            "text": self._display_text,
            "status": self._status,
            "confidence": float(self._display_conf),
            "margin": float(self._display_margin),
            "consensus": float(self._display_consensus),
            "sentence_tokens": list(self._sentence_tokens),
            "sentence": " ".join(self._sentence_tokens).replace("_", " "),
            "model": str(self._cfg.model_path),
            "labels": str(self._cfg.labels_path),
            "updated_at": time.time(),
        }
        out_path = self._cfg.state_out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_path.write_text(json.dumps(payload), encoding="utf-8")
        except PermissionError:
            return

    def _validate_model_contract(self) -> None:
        input_meta = self._session.get_inputs()[0]
        shape = input_meta.shape
        if len(shape) != 3:
            raise RuntimeError(f"Model input must be 3D [batch, time, feat], got {shape}")

        time_dim = shape[1]
        feat_dim = shape[2]
        if isinstance(time_dim, int) and time_dim != SEQUENCE_LENGTH:
            raise RuntimeError(
                f"Model expects sequence length {time_dim}, app uses {SEQUENCE_LENGTH}. Retrain or update config."
            )
        if isinstance(feat_dim, int) and feat_dim != FEATURE_DIM:
            raise RuntimeError(
                f"Model expects feature dim {feat_dim}, app uses {FEATURE_DIM}. Regenerate dataset and retrain."
            )

        out_meta = self._session.get_outputs()[0]
        out_shape = out_meta.shape
        if len(out_shape) == 2 and isinstance(out_shape[1], int):
            if out_shape[1] != len(self._labels):
                raise RuntimeError(
                    f"Model outputs {out_shape[1]} classes but labels file has {len(self._labels)} classes"
                )

    def run(self) -> None:
        cap = cv2.VideoCapture(self._cfg.camera_index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self._cfg.camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)

        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        if not self._cfg.headless:
            cv2.namedWindow(self._window_name)

        prev = time.perf_counter()

        try:
            self._write_state()
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                now = time.perf_counter()
                dt = now - prev
                prev = now
                if dt > 0:
                    self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

                feat, details = self._extractor.extract_with_details(frame)
                if feat.shape[0] == FEATURE_DIM:
                    self._buffer.append(feat)
                    self._hand_presence.append(details.hand_count)

                if self._cfg.draw_landmarks and not self._cfg.headless:
                    self._extractor.draw_holistic(frame, details.results)

                if len(self._buffer) == SEQUENCE_LENGTH:
                    self._infer_continuous(now)
                else:
                    self._status = "collecting"
                    self._write_state()

                if not self._cfg.headless:
                    self._draw_overlay(frame)
                    cv2.imshow(self._window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if self._is_window_closed():
                        break
                    if key == ord("c"):
                        self._sentence_tokens.clear()
                        self._write_state()
                    if key == ord("x") and self._sentence_tokens:
                        self._sentence_tokens.pop()
                        self._write_state()
                else:
                    time.sleep(0.005)
        finally:
            cap.release()
            if not self._cfg.headless:
                cv2.destroyAllWindows()
            self._speech.stop()
            self._extractor.close()

    def _is_window_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _infer_continuous(self, now: float) -> None:
        if (now - self._last_accept_t) < self._cfg.cooldown_sec:
            self._status = "cooldown"
            self._write_state()
            return

        if len(self._hand_presence) < SEQUENCE_LENGTH:
            self._status = "collecting"
            self._write_state()
            return

        hand_presence_ratio = float(np.mean(np.array(self._hand_presence, dtype=np.float32) > 0.0))
        if hand_presence_ratio < self._cfg.min_hand_presence_ratio:
            self._status = "low_hands"
            self._display_conf = 0.0
            self._display_margin = 0.0
            self._display_consensus = 0.0
            self._candidate_idx = None
            self._candidate_streak = 0
            self._write_state()
            return

        seq = np.stack(self._buffer).astype(np.float32)
        inp = np.expand_dims(seq, axis=0)
        logits = self._session.run(None, {self._input_name: inp})[0]
        probs = softmax(logits[0])
        if self._prob_ema is None:
            self._prob_ema = probs.astype(np.float32)
        else:
            alpha = float(np.clip(self._cfg.prob_ema_alpha, 0.05, 1.0))
            self._prob_ema = (alpha * probs) + ((1.0 - alpha) * self._prob_ema)
        smoothed_probs = self._prob_ema

        idx = int(np.argmax(smoothed_probs))
        conf = float(smoothed_probs[idx])
        sorted_probs = np.sort(smoothed_probs)
        second_best = float(sorted_probs[-2]) if sorted_probs.shape[0] > 1 else 0.0
        margin = conf - second_best
        self._display_margin = margin
        self._top_history.append(idx)

        if self._candidate_idx == idx:
            self._candidate_streak += 1
        else:
            self._candidate_idx = idx
            self._candidate_streak = 1

        if self._top_history:
            consensus = float(np.mean(np.array(self._top_history, dtype=np.int64) == idx))
        else:
            consensus = 0.0
        self._display_consensus = consensus
        self._display_streak = int(self._candidate_streak)

        if conf < self._cfg.confidence_threshold:
            self._status = "low_conf"
            self._display_conf = conf
            self._write_state()
            return

        if margin < self._cfg.min_margin_threshold:
            self._status = "low_margin"
            self._display_conf = conf
            self._write_state()
            return

        if self._candidate_streak < int(max(1, self._cfg.stable_frames_required)):
            self._status = "unstable"
            self._display_conf = conf
            self._write_state()
            return

        if consensus < float(np.clip(self._cfg.consensus_ratio_threshold, 0.0, 1.0)):
            self._status = "no_consensus"
            self._display_conf = conf
            self._write_state()
            return

        label = self._labels.get(str(idx), str(idx))
        self._display_text = label
        self._display_conf = conf
        self._status = "accepted"
        self._last_accept_t = now

        if self._sentence_tokens and self._sentence_tokens[-1] == label and (now - self._last_token_t) < self._cfg.repeat_token_cooldown_sec:
            self._status = "repeat_block"
            self._write_state()
            return

        if (not self._sentence_tokens) or self._sentence_tokens[-1] != label or (now - self._last_token_t) > 1.2:
            self._sentence_tokens.append(label)
            if len(self._sentence_tokens) > 18:
                self._sentence_tokens = self._sentence_tokens[-18:]
            self._last_token_t = now

        self._speech.speak(label.replace("_", " "))
        self._write_state()

    def _draw_overlay(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape

        fill_ratio = len(self._buffer) / SEQUENCE_LENGTH
        bar_w = int(w * 0.5)
        bar_h = 18
        bar_x0, bar_y0 = 16, 16
        bar_x1 = bar_x0 + bar_w
        bar_y1 = bar_y0 + bar_h

        cv2.rectangle(frame, (12, h - 158), (w - 12, h - 8), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"FPS: {self._fps:.1f}",
            (24, h - 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"TEXT: {self._display_text}",
            (24, h - 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"CONF: {self._display_conf:.2f}",
            (24, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 210, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"MARGIN: {self._display_margin:.2f}  STABLE: {self._display_streak}/{self._cfg.stable_frames_required}",
            (24, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (170, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"CONSENSUS: {self._display_consensus:.2f}  STATUS: {self._status}",
            (24, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (170, 220, 255),
            2,
            cv2.LINE_AA,
        )

        sentence = " ".join(self._sentence_tokens).replace("_", " ")
        if len(sentence) > 70:
            sentence = sentence[-70:]

        cv2.putText(
            frame,
            f"SENTENCE: {sentence if sentence else '-'}",
            (16, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Continuous mode | c: clear sentence | x: delete last",
            (16, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )

        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x1, bar_y1), (120, 120, 120), 2)
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + int(bar_w * fill_ratio), bar_y1), (0, 200, 255), -1)
        cv2.putText(
            frame,
            f"Buffer: {len(self._buffer)}/{SEQUENCE_LENGTH}",
            (bar_x1 + 16, bar_y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run realtime ISL inference")
    parser.add_argument("--model", type=Path, default=None, help="Path to ONNX model")
    parser.add_argument("--labels", type=Path, default=None, help="Path to labels JSON")
    parser.add_argument("--confidence", type=float, default=0.75, help="Minimum confidence to accept prediction")
    parser.add_argument("--margin", type=float, default=0.07, help="Minimum top1-top2 probability margin")
    parser.add_argument("--cooldown", type=float, default=1.5, help="Seconds to wait after accepted token")
    parser.add_argument("--ema-alpha", type=float, default=0.62, help="EMA smoothing factor for probabilities")
    parser.add_argument("--stable-frames", type=int, default=5, help="Consecutive frames needed before accepting token")
    parser.add_argument("--stability-window", type=int, default=7, help="History window size for consensus")
    parser.add_argument("--consensus", type=float, default=0.72, help="Required ratio of same label in history window")
    parser.add_argument("--repeat-cooldown", type=float, default=2.2, help="Minimum gap to repeat same token")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to write live sentence/state JSON")
    parser.add_argument("--headless", action="store_true", help="Run webcam inference without opening OpenCV window")
    args = parser.parse_args()

    default_10_model = Path("models/dynamic_lstm_10class_combined.onnx")
    default_10_labels = Path("models/labels_10class_combined.json")

    if args.model is not None and args.labels is not None:
        cfg = AppConfig(model_path=args.model, labels_path=args.labels)
    elif default_10_model.exists() and default_10_labels.exists():
        cfg = AppConfig(model_path=default_10_model, labels_path=default_10_labels)
    else:
        cfg = AppConfig()
        cfg.model_path = resolve_best_compatible_model(cfg.model_path, cfg.labels_path)

    cfg.confidence_threshold = float(args.confidence)
    cfg.min_margin_threshold = float(args.margin)
    cfg.cooldown_sec = float(args.cooldown)
    cfg.prob_ema_alpha = float(args.ema_alpha)
    cfg.stable_frames_required = int(args.stable_frames)
    cfg.stability_window = int(args.stability_window)
    cfg.consensus_ratio_threshold = float(args.consensus)
    cfg.repeat_token_cooldown_sec = float(args.repeat_cooldown)
    cfg.state_out_path = args.out_json
    cfg.headless = bool(args.headless)

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Missing model: {cfg.model_path}")
    if not cfg.labels_path.exists():
        raise FileNotFoundError(f"Missing labels: {cfg.labels_path}")

    print(f"Using model: {cfg.model_path}")
    print(f"Using labels: {cfg.labels_path}")

    app = RealtimeISLApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
