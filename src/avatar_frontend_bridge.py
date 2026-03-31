from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PathConfig
from .gloss_mapper import GlossMapper
from .pose_player import PlaybackQueueBuilder, PoseLookup, export_avatar_motion


def _normalize_phrase(text: str) -> str:
    return " ".join(text.strip().lower().split())


def rebuild_assets_from_pose(paths: PathConfig) -> tuple[Path, Path]:
    pose_root = paths.pose_dir
    if not pose_root.exists():
        raise FileNotFoundError(f"Missing processed pose directory: {pose_root}")

    paths.manifests_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = paths.manifests_dir / "pose_manifest.jsonl"
    gloss_dict_path = paths.manifests_dir / "gloss_dict.json"

    rows: list[dict] = []
    gloss_dict: dict[str, str] = {}

    for gloss_dir in sorted(p for p in pose_root.iterdir() if p.is_dir()):
        folder_name = gloss_dir.name.strip()
        if not folder_name or folder_name.startswith("_"):
            continue
        if folder_name.lower() == "direction2":
            continue

        gloss = folder_name.upper()
        phrase = _normalize_phrase(folder_name.replace("_", " "))
        gloss_dict[phrase] = gloss
        if " " in phrase:
            gloss_dict[phrase.replace(" ", "_")] = gloss

        pose_files = sorted(gloss_dir.glob("*.json"))
        for pose_json in pose_files:
            fps = 30.0
            try:
                payload = json.loads(pose_json.read_text(encoding="utf-8"))
                fps = float(payload.get("fps", 30.0))
            except Exception:
                fps = 30.0

            rows.append(
                {
                    "gloss": gloss,
                    "phrase": phrase,
                    "video_path": str(pose_json),
                    "pose_json": str(pose_json),
                    "fps": fps,
                }
            )

    if not rows:
        raise FileNotFoundError(f"No pose clips found under: {pose_root}")

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    gloss_dict_path.write_text(json.dumps(gloss_dict, indent=2), encoding="utf-8")
    return manifest_path, gloss_dict_path


def render_text_to_motion(text: str, out_path: Path, speed: float = 1.0, refresh_assets: bool = True) -> dict:
    paths = PathConfig()
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = paths.manifests_dir / "pose_manifest.jsonl"
    gloss_dict_path = paths.manifests_dir / "gloss_dict.json"

    if refresh_assets or (not manifest_path.exists()) or (not gloss_dict_path.exists()):
        manifest_path, gloss_dict_path = rebuild_assets_from_pose(paths)

    mapper = GlossMapper(
        gloss_dict_path=gloss_dict_path,
        oov_log_path=paths.logs_dir / "oov_words.log",
    )
    lookup = PoseLookup(manifest_path)
    queue_builder = PlaybackQueueBuilder(lookup)

    gloss_result = mapper.map_text(text)
    filtered_glosses: list[str] = []
    for gloss in gloss_result.glosses:
        if not gloss or gloss.startswith("FS_"):
            continue
        if lookup.pick(gloss) is None:
            continue
        filtered_glosses.append(gloss)

    events = queue_builder.build(filtered_glosses, speed=float(speed))
    export_avatar_motion(events, out_path, output_fps=30.0)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    data["engine"] = "src-avatar"
    data["text"] = text
    data["tokens"] = filtered_glosses
    data["unknown_tokens"] = gloss_result.oov_words
    out_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frontend avatar motion via src pipeline")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render_text_to_motion(
        text=args.text,
        out_path=args.out,
        speed=args.speed,
        refresh_assets=not args.no_refresh,
    )
    print(str(args.out))


if __name__ == "__main__":
    main()
