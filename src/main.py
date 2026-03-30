import argparse
import asyncio
import json
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from time import time

from .config import AudioConfig, PathConfig, STTConfig
from .gloss_mapper import GlossMapper
from .pose_player import (
    PlaybackQueueBuilder,
    PoseLookup,
    export_avatar_motion,
    export_threejs_timeline,
)
async def run_pipeline(
    language: str,
    speed: float,
    model_name: str,
    device: str,
    live_avatar: bool,
    render_size: str,
    window_title: str,
):
    from .audio_capture import capture_audio
    from .stt_whisper import WhisperSTT

    play_events_live = None
    width, height = _parse_render_size(render_size)
    if live_avatar:
        from .avatar_renderer import play_events_live as _play_events_live

        play_events_live = _play_events_live

    paths = PathConfig()
    paths.raw_audio_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    stop_event = asyncio.Event()

    print(f"Loading Whisper model '{model_name}' on device '{device}'...")
    print("First run may take time while model files are downloaded and cached.")
    stt = WhisperSTT(STTConfig(language=language, model_name=model_name, device=device))
    print("Whisper model loaded.")

    mapper = GlossMapper(
        gloss_dict_path=paths.manifests_dir / "gloss_dict.json",
        oov_log_path=paths.logs_dir / "oov_words.log",
    )
    lookup = PoseLookup(paths.manifests_dir / "pose_manifest.jsonl")
    queue_builder = PlaybackQueueBuilder(lookup)

    async def consumer():
        while not stop_event.is_set():
            chunk = await audio_queue.get()
            stt_result = stt.transcribe_chunk(chunk)
            if not stt_result.text:
                continue

            gloss_result = mapper.map_text(stt_result.text)
            events = queue_builder.build(gloss_result.glosses, speed=speed)
            out_timeline = paths.processed_dir / "timelines" / f"timeline_{int(chunk.created_at)}.json"
            export_threejs_timeline(events, out_timeline)

            print("\\n=== PIPELINE OUTPUT ===")
            print(f"wav: {stt_result.wav_path}")
            print(f"stt: {stt_result.text}")
            print(f"confidence: {stt_result.confidence:.2f}")
            print(f"gloss: {gloss_result.glosses}")
            print(f"timeline: {out_timeline}")

            if live_avatar and play_events_live is not None:
                await asyncio.to_thread(
                    play_events_live,
                    events,
                    width,
                    height,
                    30.0,
                    window_title,
                )

            audit_path = paths.logs_dir / "pipeline_audit.jsonl"
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "wav": stt_result.wav_path,
                    "text": stt_result.text,
                    "confidence": stt_result.confidence,
                    "glosses": gloss_result.glosses,
                    "oov": gloss_result.oov_words,
                    "timeline": str(out_timeline),
                }) + "\\n")

    producer_task = asyncio.create_task(
        capture_audio(AudioConfig(), paths.raw_audio_dir, audio_queue, stop_event)
    )
    consumer_task = asyncio.create_task(consumer())

    print("Microphone pipeline running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(0.25)
    except (KeyboardInterrupt, asyncio.CancelledError):
        stop_event.set()
    finally:
        stop_event.set()
        consumer_task.cancel()
        with suppress(asyncio.CancelledError):
            await consumer_task
        await producer_task


def _render_motion_json(motion_path: Path, out_video: Path | None, size: str):
    from .avatar_renderer import render_avatar_motion_to_video

    width, height = _parse_render_size(size)

    out_path = out_video or motion_path.with_suffix(".mp4")
    render_avatar_motion_to_video(
        motion_json_path=motion_path,
        out_video_path=out_path,
        width=width,
        height=height,
    )
    print(f"rendered_video: {out_path}")


def _parse_render_size(size: str) -> tuple[int, int]:
    if "x" not in size.lower():
        raise ValueError("--render-size must be in WIDTHxHEIGHT format, e.g. 960x720")

    w_str, h_str = size.lower().split("x", maxsplit=1)
    width = int(w_str)
    height = int(h_str)
    if width <= 0 or height <= 0:
        raise ValueError("Render size must be positive")
    return width, height


def run_text_to_avatar(
    text: str,
    speed: float,
    out_timeline: Path | None,
    out_motion: Path | None,
    render_video: bool,
    out_video: Path | None,
    render_size: str,
):
    paths = PathConfig()
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    mapper = GlossMapper(
        gloss_dict_path=paths.manifests_dir / "gloss_dict.json",
        oov_log_path=paths.logs_dir / "oov_words.log",
    )
    lookup = PoseLookup(paths.manifests_dir / "pose_manifest.jsonl")
    queue_builder = PlaybackQueueBuilder(lookup)

    gloss_result = mapper.map_text(text)
    events = queue_builder.build(gloss_result.glosses, speed=speed)

    ts = int(time())
    timeline_path = out_timeline or (paths.processed_dir / "timelines" / f"text_timeline_{ts}.json")
    motion_path = out_motion or (paths.processed_dir / "timelines" / f"text_avatar_motion_{ts}.json")

    export_threejs_timeline(events, timeline_path)
    export_avatar_motion(events, motion_path)

    print("\n=== TEXT TO AVATAR OUTPUT ===")
    print(f"text: {text}")
    print(f"gloss: {gloss_result.glosses}")
    print(f"oov: {gloss_result.oov_words}")
    print(f"timeline: {timeline_path}")
    print(f"avatar_motion: {motion_path}")

    if render_video:
        _render_motion_json(motion_path, out_video, render_size)


def run_live_text_avatar(text: str, speed: float, render_size: str, window_title: str):
    from .avatar_renderer import play_events_live

    width, height = _parse_render_size(render_size)

    paths = PathConfig()
    mapper = GlossMapper(
        gloss_dict_path=paths.manifests_dir / "gloss_dict.json",
        oov_log_path=paths.logs_dir / "oov_words.log",
    )
    lookup = PoseLookup(paths.manifests_dir / "pose_manifest.jsonl")
    queue_builder = PlaybackQueueBuilder(lookup)

    gloss_result = mapper.map_text(text)
    events = queue_builder.build(gloss_result.glosses, speed=speed)

    print("\n=== LIVE TEXT TO AVATAR ===")
    print(f"text: {text}")
    print(f"gloss: {gloss_result.glosses}")
    print("press 'q' in the avatar window to stop playback")

    play_events_live(
        events=events,
        width=width,
        height=height,
        window_title=window_title,
    )


def run_live_text_console(speed: float, render_size: str, window_title: str):
    print("Live text avatar mode. Type text and press Enter. Type 'exit' to quit.")
    while True:
        try:
            text = input("text> ").strip()
        except EOFError:
            print("No interactive stdin detected. Use --text \"...\" for single input or run in an interactive terminal.")
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            break
        run_live_text_avatar(text=text, speed=speed, render_size=render_size, window_title=window_title)


def fix_opencv_for_live_mode():
    steps = [
        [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python-headless", "opencv-contrib-python-headless"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "opencv-python==4.11.0.86"],
    ]

    for cmd in steps:
        print("running:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError("OpenCV fix step failed")

    verify = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        check=False,
        capture_output=True,
        text=True,
    )
    if "opencv-python-headless" in verify.stdout.lower():
        raise RuntimeError(
            "opencv-python-headless is still installed. Close all Python/VS Code terminals and run: "
            f"{sys.executable} -m pip uninstall -y opencv-python-headless"
        )

    print("OpenCV live-window fix completed. Re-run your live avatar command.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Realtime speech to ISL gloss to avatar timeline")
    p.add_argument("--language", default="en", choices=["en", "hi", "ta"], help="Whisper language")
    p.add_argument("--speed", type=float, default=1.0, help="Sign playback speed")
    p.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    p.add_argument(
        "--text",
        default="",
        help="Direct text input for text-to-avatar conversion",
    )
    p.add_argument(
        "--text-file",
        default="",
        help="Path to a UTF-8 text file to convert to avatar motion",
    )
    p.add_argument(
        "--out-timeline",
        default="",
        help="Optional custom output path for timeline JSON",
    )
    p.add_argument(
        "--out-motion",
        default="",
        help="Optional custom output path for avatar motion JSON",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Render generated avatar motion JSON to MP4",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Play avatar in realtime window (no video file generation)",
    )
    p.add_argument(
        "--live-console",
        action="store_true",
        help="Interactive realtime mode: type text repeatedly and watch avatar actions",
    )
    p.add_argument(
        "--render-motion",
        default="",
        help="Render an existing avatar motion JSON file to MP4",
    )
    p.add_argument(
        "--out-video",
        default="",
        help="Optional custom output path for rendered MP4",
    )
    p.add_argument(
        "--render-size",
        default="960x720",
        help="Output render resolution in WIDTHxHEIGHT format",
    )
    p.add_argument(
        "--window-title",
        default="ISL Avatar Live",
        help="Window title for realtime avatar playback",
    )
    p.add_argument(
        "--save-json",
        action="store_true",
        help="When using --text, keep JSON outputs instead of live-only playback",
    )
    p.add_argument(
        "--fix-opencv",
        action="store_true",
        help="Repair OpenCV installation for live window rendering",
    )
    p.add_argument(
        "--no-live-avatar",
        action="store_true",
        help="Disable live avatar playback in microphone pipeline mode",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.render_motion:
        _render_motion_json(
            motion_path=Path(args.render_motion),
            out_video=Path(args.out_video) if args.out_video else None,
            size=args.render_size,
        )
        raise SystemExit(0)

    if args.fix_opencv:
        fix_opencv_for_live_mode()
        raise SystemExit(0)

    if args.live_console:
        run_live_text_console(
            speed=args.speed,
            render_size=args.render_size,
            window_title=args.window_title,
        )
        raise SystemExit(0)

    text_input = (args.text or "").strip()
    if args.text_file:
        text_input = Path(args.text_file).read_text(encoding="utf-8").strip()

    if text_input:
        live_only = bool(args.live) or (
            not args.render
            and not args.save_json
            and not args.out_timeline
            and not args.out_motion
        )

        if live_only:
            try:
                run_live_text_avatar(
                    text=text_input,
                    speed=args.speed,
                    render_size=args.render_size,
                    window_title=args.window_title,
                )
            except RuntimeError as e:
                print(f"Live avatar failed: {e}")
                print("Try: python -m src.main --fix-opencv")
                raise
        else:
            run_text_to_avatar(
                text=text_input,
                speed=args.speed,
                out_timeline=Path(args.out_timeline) if args.out_timeline else None,
                out_motion=Path(args.out_motion) if args.out_motion else None,
                render_video=bool(args.render),
                out_video=Path(args.out_video) if args.out_video else None,
                render_size=args.render_size,
            )
    else:
        try:
            asyncio.run(
                run_pipeline(
                    language=args.language,
                    speed=args.speed,
                    model_name=args.model,
                    device=args.device,
                    live_avatar=not args.no_live_avatar,
                    render_size=args.render_size,
                    window_title=args.window_title,
                )
            )
        except KeyboardInterrupt:
            print("Pipeline stopped.")
