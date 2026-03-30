import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


FALLBACK_GLOSS_ALIASES = {
    "HELP": "PLEASED",
}


@dataclass
class PoseClip:
    gloss: str
    pose_path: Path
    fps: float


@dataclass
class PlaybackEvent:
    gloss: str
    pose_path: str
    start_time: float
    duration: float
    speed: float


class PoseLookup:
    def __init__(self, manifest_path: Path):
        self.by_gloss: Dict[str, List[PoseClip]] = {}
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    gloss = row["gloss"].upper()
                    clip = PoseClip(gloss=gloss, pose_path=Path(row["pose_json"]), fps=float(row.get("fps", 30.0)))
                    self.by_gloss.setdefault(gloss, []).append(clip)

    def pick(self, gloss: str) -> PoseClip | None:
        requested = gloss.upper()
        options = self.by_gloss.get(requested, [])
        if not options and requested in FALLBACK_GLOSS_ALIASES:
            options = self.by_gloss.get(FALLBACK_GLOSS_ALIASES[requested], [])
        if not options:
            return None
        return random.choice(options)


class PlaybackQueueBuilder:
    def __init__(self, lookup: PoseLookup, default_duration: float = 0.8):
        self.lookup = lookup
        self.default_duration = default_duration

    def build(self, glosses: List[str], speed: float = 1.0) -> List[PlaybackEvent]:
        events: List[PlaybackEvent] = []
        t = 0.0

        for gloss in glosses:
            clip = self.lookup.pick(gloss)
            pose_path = str(clip.pose_path) if clip else ""
            duration = self.default_duration / max(speed, 0.1)
            events.append(
                PlaybackEvent(
                    gloss=gloss,
                    pose_path=pose_path,
                    start_time=t,
                    duration=duration,
                    speed=speed,
                )
            )
            t += duration

        return events


def _load_pose_clip_frames(pose_path: Path) -> tuple[float, List[dict]]:
    if not pose_path.exists():
        return 30.0, []

    with pose_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fps = float(data.get("fps", 30.0))
    frames = data.get("frames", [])
    if not isinstance(frames, list):
        return fps, []
    return fps, frames


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


def export_avatar_motion(events: List[PlaybackEvent], out_path: Path, output_fps: float = 30.0):
    frame_cursor = 0
    merged_frames: List[dict] = []
    segments: List[dict] = []

    cache: Dict[str, tuple[float, List[dict]]] = {}

    for event in events:
        if not event.pose_path:
            segments.append(
                {
                    "gloss": event.gloss,
                    "start_frame": frame_cursor,
                    "end_frame": frame_cursor,
                    "source_pose": "",
                    "missing": True,
                }
            )
            continue

        pose_path = Path(event.pose_path)
        key = str(pose_path)
        if key not in cache:
            cache[key] = _load_pose_clip_frames(pose_path)

        src_fps, src_frames = cache[key]
        clip_frames = _retime_frames(src_frames, event.speed)

        start = frame_cursor
        merged_frames.extend(clip_frames)
        frame_cursor += len(clip_frames)
        end = frame_cursor

        segments.append(
            {
                "gloss": event.gloss,
                "start_frame": start,
                "end_frame": end,
                "source_pose": key,
                "source_fps": src_fps,
                "missing": len(clip_frames) == 0,
            }
        )

    motion = {
        "engine": "avatar",
        "fps": output_fps,
        "total_frames": len(merged_frames),
        "segments": segments,
        "frames": merged_frames,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(motion, f, indent=2)


def export_threejs_timeline(events: List[PlaybackEvent], out_path: Path):
    timeline = {
        "engine": "threejs",
        "interpolation": "linear",
        "events": [e.__dict__ for e in events],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)


def export_unity_timeline(events: List[PlaybackEvent], out_path: Path):
    timeline = {
        "engine": "unity",
        "blend": "linear",
        "events": [e.__dict__ for e in events],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)
