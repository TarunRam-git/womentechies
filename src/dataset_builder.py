import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from tqdm import tqdm

from .config import PathConfig


VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


@dataclass
class ClipSample:
    gloss: str
    phrase: str
    video_path: Path


def _parse_gloss_from_folder(name: str) -> tuple[str, str]:
    clean = re.sub(r"^\s*\d+\.\s*", "", name).strip()
    return clean.upper(), clean


def discover_samples(raw_data_dir: Path) -> List[ClipSample]:
    samples: List[ClipSample] = []

    for gloss_dir in raw_data_dir.rglob("*"):
        if not gloss_dir.is_dir():
            continue

        videos = [p for p in gloss_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        if not videos:
            continue

        gloss, phrase = _parse_gloss_from_folder(gloss_dir.name)
        for video in videos:
            samples.append(ClipSample(gloss=gloss, phrase=phrase, video_path=video))

    return samples


def _ensure_task_model(url: str, model_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 0:
        return
    print(f"Downloading model: {model_path.name}")
    urlretrieve(url, str(model_path))


def _extract_with_legacy_solutions(video_path: Path, out_json: Path):
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(str(video_path))
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
    )

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        def pack_landmarks(lm):
            if lm is None:
                return []
            return [[p.x, p.y, p.z, p.visibility] for p in lm.landmark]

        row = {
            "pose": pack_landmarks(res.pose_landmarks),
            "left_hand": pack_landmarks(res.left_hand_landmarks),
            "right_hand": pack_landmarks(res.right_hand_landmarks),
        }
        frames.append(row)

    cap.release()
    holistic.close()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"fps": fps, "frames": frames}, f)


def _extract_with_tasks_api(video_path: Path, out_json: Path):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    models_dir = out_json.parents[2] / "models" / "mediapipe"
    pose_model = models_dir / "pose_landmarker_lite.task"
    hand_model = models_dir / "hand_landmarker.task"
    _ensure_task_model(POSE_MODEL_URL, pose_model)
    _ensure_task_model(HAND_MODEL_URL, hand_model)

    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((frame_idx * 1000.0) / fps)
        frame_idx += 1

        pose_res = pose_landmarker.detect_for_video(mp_image, ts)
        hand_res = hand_landmarker.detect_for_video(mp_image, ts)

        pose = []
        if pose_res.pose_landmarks:
            pose = [[p.x, p.y, p.z, getattr(p, "visibility", 0.0)] for p in pose_res.pose_landmarks[0]]

        left_hand = []
        right_hand = []
        if hand_res.hand_landmarks:
            for idx, landmarks in enumerate(hand_res.hand_landmarks):
                packed = [[p.x, p.y, p.z, 1.0] for p in landmarks]
                label = ""
                if hand_res.handedness and idx < len(hand_res.handedness) and hand_res.handedness[idx]:
                    label = hand_res.handedness[idx][0].category_name.lower()
                if label == "left":
                    left_hand = packed
                elif label == "right":
                    right_hand = packed
                else:
                    if not left_hand:
                        left_hand = packed
                    elif not right_hand:
                        right_hand = packed

        frames.append({"pose": pose, "left_hand": left_hand, "right_hand": right_hand})

    cap.release()
    pose_landmarker.close()
    hand_landmarker.close()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"fps": fps, "frames": frames}, f)


def extract_pose_json(video_path: Path, out_json: Path):
    if hasattr(mp, "solutions"):
        return _extract_with_legacy_solutions(video_path, out_json)
    return _extract_with_tasks_api(video_path, out_json)


def build_dataset(paths: PathConfig):
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(paths.raw_data_dir)
    manifest_path = paths.manifests_dir / "pose_manifest.jsonl"
    dict_path = paths.manifests_dir / "gloss_dict.json"

    gloss_dict: Dict[str, str] = {}

    with manifest_path.open("w", encoding="utf-8") as mf:
        for sample in tqdm(samples, desc="Extracting pose"):
            out_json = paths.pose_dir / sample.gloss / (sample.video_path.stem + ".json")
            extract_pose_json(sample.video_path, out_json)

            gloss_dict[sample.phrase.lower()] = sample.gloss
            tokens = re.findall(r"[A-Za-z0-9']+", sample.phrase.lower())
            if len(tokens) == 1:
                gloss_dict[tokens[0]] = sample.gloss

            row = {
                "gloss": sample.gloss,
                "phrase": sample.phrase,
                "video_path": str(sample.video_path),
                "pose_json": str(out_json),
                "fps": 30.0,
            }
            mf.write(json.dumps(row) + "\n")

    with dict_path.open("w", encoding="utf-8") as f:
        json.dump(gloss_dict, f, indent=2)

    print(f"Built manifest: {manifest_path}")
    print(f"Built gloss dictionary: {dict_path}")


if __name__ == "__main__":
    build_dataset(PathConfig())
