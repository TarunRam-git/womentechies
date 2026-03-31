from __future__ import annotations

import argparse
from pathlib import Path

from avatar_render import render_sentence, resolve_defaults


def main() -> None:
    default_dataset, default_labels = resolve_defaults()

    parser = argparse.ArgumentParser(description="Type sentence and render avatar motion")
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--labels", type=Path, default=default_labels)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--size", type=int, default=420)
    parser.add_argument("--blend-frames", type=int, default=10)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    if not args.dataset.exists() or not args.labels.exists():
        raise FileNotFoundError(
            "Missing user-recorded avatar data. First run: "
            "py -3.12 direction2_avatar_capture.py --restart --labels models/labels_10class_combined.json "
            "--out data/processed/avatar_direction2_templates.npz --labels-out models/avatar_direction2_labels.json"
        )

    print("Type a sentence and press Enter to render.")
    print("Commands: /q quit")

    while True:
        text = input("\nSentence> ").strip()
        if not text:
            continue
        if text.lower() in {"/q", "q", "quit", "exit"}:
            break

        try:
            render_sentence(
                text=text,
                dataset_path=args.dataset,
                labels_path=args.labels,
                fps=args.fps,
                size=args.size,
                blend_frames=args.blend_frames,
                fullscreen=args.fullscreen,
            )
        except Exception as ex:
            print(f"Error: {ex}")


if __name__ == "__main__":
    main()
