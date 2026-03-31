from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


SEQUENCE_LENGTH = 30
POSE_IDX = [11, 12, 13, 14, 15, 16, 23, 24]
FACE_IDX = [61, 291, 13, 14, 70, 300, 159, 145, 386, 374]
HAND_DIM = 21 * 3
POSE_DIM = len(POSE_IDX) * 3
FACE_DIM = len(FACE_IDX) * 3
BASE_FEATURE_DIM = (HAND_DIM * 2) + POSE_DIM + FACE_DIM
RELATIONAL_DIM = 17
FEATURE_DIM = BASE_FEATURE_DIM + RELATIONAL_DIM
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


@dataclass
class ExtractConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1
    smooth_landmarks: bool = True


@dataclass
class FrameDetails:
    results: Any
    hand_count: int
    face_detected: bool
    face_center_xy: np.ndarray | None
    wrist_center_xy: np.ndarray | None
    hand_face_distance: float | None


def apply_clahe_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


class HolisticFeatureExtractor:
    def __init__(self, cfg: ExtractConfig | None = None) -> None:
        self._cfg = cfg or ExtractConfig()
        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self._cfg.model_complexity,
            smooth_landmarks=self._cfg.smooth_landmarks,
            refine_face_landmarks=False,
            min_detection_confidence=self._cfg.min_detection_confidence,
            min_tracking_confidence=self._cfg.min_tracking_confidence,
        )

    def close(self) -> None:
        self._holistic.close()

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        feature, _ = self.extract_with_details(frame_bgr)
        return feature

    def extract_with_details(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, FrameDetails]:
        enhanced = apply_clahe_bgr(frame_bgr)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(rgb)
        feature = compose_feature_vector(results)

        wrists = []
        hand_count = 0
        if results.left_hand_landmarks is not None:
            hand_count += 1
            lw = results.left_hand_landmarks.landmark[0]
            wrists.append([lw.x, lw.y])
        if results.right_hand_landmarks is not None:
            hand_count += 1
            rw = results.right_hand_landmarks.landmark[0]
            wrists.append([rw.x, rw.y])

        wrist_center_xy = None
        if wrists:
            wrist_center_xy = np.mean(np.array(wrists, dtype=np.float32), axis=0)

        face_center_xy = None
        if results.face_landmarks is not None:
            points = np.array(
                [[results.face_landmarks.landmark[idx].x, results.face_landmarks.landmark[idx].y] for idx in FACE_IDX],
                dtype=np.float32,
            )
            face_center_xy = np.mean(points, axis=0)

        hand_face_distance = None
        if wrist_center_xy is not None and face_center_xy is not None:
            hand_face_distance = float(np.linalg.norm(wrist_center_xy - face_center_xy))

        details = FrameDetails(
            results=results,
            hand_count=hand_count,
            face_detected=results.face_landmarks is not None,
            face_center_xy=face_center_xy,
            wrist_center_xy=wrist_center_xy,
            hand_face_distance=hand_face_distance,
        )
        return feature, details

    @staticmethod
    def draw_holistic(frame_bgr: np.ndarray, results: Any) -> None:
        draw = mp.solutions.drawing_utils
        styles = mp.solutions.drawing_styles
        holistic = mp.solutions.holistic

        draw.draw_landmarks(
            frame_bgr,
            results.face_landmarks,
            holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=styles.get_default_face_mesh_contours_style(),
        )
        draw.draw_landmarks(
            frame_bgr,
            results.pose_landmarks,
            holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
        )
        draw.draw_landmarks(
            frame_bgr,
            results.left_hand_landmarks,
            holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_hand_landmarks_style(),
        )
        draw.draw_landmarks(
            frame_bgr,
            results.right_hand_landmarks,
            holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=styles.get_default_hand_landmarks_style(),
        )


def compose_feature_vector(results: Any) -> np.ndarray:
    right_hand = _extract_and_normalize_hand(results.right_hand_landmarks)
    left_hand = _extract_and_normalize_hand(results.left_hand_landmarks)
    pose = _extract_pose_with_hand_fallback(
        results.pose_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks,
    )
    face = _extract_face(results.face_landmarks)
    base = np.concatenate([right_hand, left_hand, pose, face], axis=0).astype(np.float32)
    rel = compute_relational_from_base(base)
    feature = np.concatenate([base, rel], axis=0).astype(np.float32)
    return feature


def rebuild_sequence_with_relational(seq: np.ndarray) -> np.ndarray:
    if seq.ndim != 2:
        raise ValueError("Expected sequence with shape [T, D]")
    if seq.shape[1] == FEATURE_DIM:
        base = seq[:, :BASE_FEATURE_DIM].astype(np.float32)
    elif seq.shape[1] == BASE_FEATURE_DIM:
        base = seq.astype(np.float32)
    else:
        raise ValueError(f"Expected feature dim {BASE_FEATURE_DIM} or {FEATURE_DIM}, got {seq.shape[1]}")

    rel_list = [compute_relational_from_base(frame) for frame in base]
    rel = np.stack(rel_list).astype(np.float32)
    return np.concatenate([base, rel], axis=1).astype(np.float32)


def compute_relational_from_base(base: np.ndarray) -> np.ndarray:
    if base.shape[0] < BASE_FEATURE_DIM:
        return np.zeros(RELATIONAL_DIM, dtype=np.float32)

    pose = base[126:150].reshape(8, 3).astype(np.float32)
    face = base[150:180].reshape(10, 3).astype(np.float32)

    left_shoulder = pose[0]
    right_shoulder = pose[1]
    left_elbow = pose[2]
    right_elbow = pose[3]
    left_wrist = pose[4]
    right_wrist = pose[5]

    face_center = np.mean(face, axis=0)
    mouth_center = np.mean(face[[2, 3]], axis=0)

    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    scale = shoulder_width if shoulder_width > 1e-4 else 1.0

    left_wrist_face = (left_wrist - face_center) / scale
    right_wrist_face = (right_wrist - face_center) / scale
    left_wrist_mouth = (left_wrist - mouth_center) / scale
    right_wrist_mouth = (right_wrist - mouth_center) / scale

    wrist_distance = float(np.linalg.norm(left_wrist - right_wrist) / scale)
    left_wrist_shoulder = float(np.linalg.norm(left_wrist - left_shoulder) / scale)
    right_wrist_shoulder = float(np.linalg.norm(right_wrist - right_shoulder) / scale)

    left_angle = _joint_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = _joint_angle(right_shoulder, right_elbow, right_wrist)

    rel = np.concatenate(
        [
            left_wrist_face,
            right_wrist_face,
            left_wrist_mouth,
            right_wrist_mouth,
            np.array([wrist_distance, left_wrist_shoulder, right_wrist_shoulder, left_angle, right_angle], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return rel


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.float32:
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return np.float32(0.0)
    cos_val = float(np.dot(ba, bc) / denom)
    cos_val = max(-1.0, min(1.0, cos_val))
    return np.float32(np.arccos(cos_val) / np.pi)


def _extract_and_normalize_hand(hand_landmarks: Any) -> np.ndarray:
    if hand_landmarks is None:
        return np.zeros(HAND_DIM, dtype=np.float32)

    arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = arr[0].copy()
    arr = arr - wrist
    max_dist = float(np.max(np.linalg.norm(arr, axis=1)))
    if max_dist > 1e-6:
        arr = arr / max_dist
    return arr.reshape(-1).astype(np.float32)


def _extract_pose(pose_landmarks: Any) -> np.ndarray:
    if pose_landmarks is None:
        return np.zeros(POSE_DIM, dtype=np.float32)
    vals = []
    for idx in POSE_IDX:
        lm = pose_landmarks.landmark[idx]
        vals.extend([lm.x, lm.y, lm.z])
    return np.array(vals, dtype=np.float32)


def _extract_pose_with_hand_fallback(pose_landmarks: Any, left_hand_landmarks: Any, right_hand_landmarks: Any) -> np.ndarray:
    pose = _extract_pose(pose_landmarks).reshape(len(POSE_IDX), 3)
    if left_hand_landmarks is not None:
        lw = left_hand_landmarks.landmark[0]
        pose[4] = np.array([lw.x, lw.y, lw.z], dtype=np.float32)
    if right_hand_landmarks is not None:
        rw = right_hand_landmarks.landmark[0]
        pose[5] = np.array([rw.x, rw.y, rw.z], dtype=np.float32)
    return pose.reshape(-1).astype(np.float32)


def _extract_face(face_landmarks: Any) -> np.ndarray:
    if face_landmarks is None:
        return np.zeros(FACE_DIM, dtype=np.float32)
    vals = []
    for idx in FACE_IDX:
        lm = face_landmarks.landmark[idx]
        vals.extend([lm.x, lm.y, lm.z])
    return np.array(vals, dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)
