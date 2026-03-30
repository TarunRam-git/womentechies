from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 30
    min_chunk_seconds: float = 2.0
    max_chunk_seconds: float = 4.0
    silence_hangover_ms: int = 500
    vad_mode: int = 2
    noise_gate_db: float = -40.0


@dataclass(frozen=True)
class STTConfig:
    model_name: str = "small"
    compute_type: str = "int8"
    device: str = "auto"
    language: str = "en"
    confidence_threshold: float = 0.45


@dataclass(frozen=True)
class PathConfig:
    workspace_root: Path = Path(__file__).resolve().parents[1]
    raw_data_dir: Path = workspace_root / "data" / "raw"
    processed_dir: Path = workspace_root / "data" / "processed"
    raw_audio_dir: Path = processed_dir / "audio_raw"
    pose_dir: Path = processed_dir / "pose"
    manifests_dir: Path = processed_dir / "manifests"
    logs_dir: Path = processed_dir / "logs"
