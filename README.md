# Offline Speech-to-ISL Avatar Pipeline

This project uses your video dataset in `data/raw` to build a local pipeline:

1. Audio capture from microphone (16 kHz mono, VAD chunking, silence split, noise gate)
2. Offline STT using faster-whisper
3. Text to ISL gloss mapping (rules + dictionary + fingerspelling fallback)
4. Gloss to pose playback timeline for avatar rendering engines (Three.js/Unity)

## Dataset Use

Your existing folder labels are treated as gloss classes, for example:

- `45. we` -> `WE`
- `55. Thank you` -> `THANK YOU`

Each `.MOV` clip is processed into pose keypoint JSON and stored in `data/processed/pose/...`.

## Install

```bash
pip install -r requirements.txt
```

Download a Whisper model once (first run of `faster-whisper` will fetch it).

## Step 1: Build Pose Dataset + Gloss Dictionary

```bash
python -m src.dataset_builder
```

Outputs:

- `data/processed/manifests/pose_manifest.jsonl`
- `data/processed/manifests/gloss_dict.json`
- `data/processed/pose/<GLOSS>/<clip>.json`

## Step 2: Optional Text-to-Gloss ML Model

```bash
python -m src.train_text2gloss
```

Output:

- `data/processed/models/text_to_gloss_model.pkl`

## Step 3: Run Realtime Pipeline

```bash
python -m src.main --language en --speed 1.0
```

This now runs the full chain together:

- audio capture from mic
- speech-to-text
- text-to-gloss
- live avatar playback window

If you want pipeline without live avatar window:

```bash
python -m src.main --language en --speed 1.0 --no-live-avatar
```

Supported languages for Whisper decode:

- `en`
- `hi`
- `ta`

Pipeline outputs for each chunk:

- raw wav chunk in `data/processed/audio_raw`
- decoded text
- gloss sequence (e.g. `["BOY", "EAT"]`)
- timeline JSON for Three.js in `data/processed/timelines`

## Text-to-Avatar Conversion (Direct Input)

Convert typed text directly to gloss timeline and merged avatar motion:

```bash
python -m src.main --text "hello how are you" --speed 1.0
```

Optional file input and custom output paths:

```bash
python -m src.main --text-file input.txt --out-timeline out_timeline.json --out-motion out_motion.json
```

Text mode outputs:

- timeline JSON for event sequencing
- avatar motion JSON with merged per-frame pose landmarks for rendering

## Realtime Avatar From Text (No Video Generation)

Play avatar actions live in a window from typed text:

```bash
python -m src.main --text "hello how are you"
```

Run interactive realtime mode (type text repeatedly):

```bash
python -m src.main --live-console
```

In live mode, press `q` in the avatar window to stop current playback.

Known multi-word phrases from your gloss dictionary (for example `good morning`) are mapped first.
If any token has no pose clip, live mode uses a neutral fallback frame instead of crashing.

Custom help phrase mappings are included:

- `i need help` -> `I HELP`
- `need help` -> `HELP`
- `help me` -> `HELP I`

Use custom window size if needed:

```bash
python -m src.main --text "thank you" --render-size 1280x720
```

If you still want JSON outputs with `--text`, add:

```bash
python -m src.main --text "hello" --save-json
```

If live window does not appear and you see `GUI: NONE` OpenCV error, run:

```bash
python -m src.main --fix-opencv
```

Then retry:

```bash
python -m src.main --live-console
```

## Render JSON to Avatar Video (MP4)

Render an existing motion JSON file into an avatar video:

```bash
python -m src.main --render-motion data/processed/timelines/text_avatar_motion_1774899054.json --out-video data/processed/timelines/avatar_preview.mp4
```

Or generate from text and render in one command:

```bash
python -m src.main --text "hello how are you" --render --out-video data/processed/timelines/avatar_from_text.mp4
```

Optional render size:

```bash
python -m src.main --render-motion data/processed/timelines/text_avatar_motion_1774899054.json --render-size 1280x720
```

## Audio Targets Implemented

- sample rate: 16 kHz mono
- chunking: 2-4 seconds with VAD
- silence detection with hangover window
- noise gate at `-40 dB`
- async queue from audio to STT

Latency target of `50 ms` is measured for chunk enqueue path and reported.

## Notes for Avatar Renderer

The generated timeline JSON includes per-gloss pose clip path, start time, duration, and speed. You can consume this in:

- Three.js skeleton player
- Unity animation queue
- Blender import script

The current interpolation metadata is linear and can be extended with cubic blending.
