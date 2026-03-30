import asyncio
import math
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import webrtcvad

from .config import AudioConfig


@dataclass
class AudioChunk:
    samples: np.ndarray
    sample_rate: int
    created_at: float
    wav_path: Path


class AudioChunker:
    def __init__(self, config: AudioConfig, out_dir: Path, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.config = config
        self.out_dir = out_dir
        self.queue = queue
        self.loop = loop
        self.vad = webrtcvad.Vad(config.vad_mode)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.frame_size = int(config.sample_rate * config.frame_ms / 1000)
        self.min_frames = int(config.min_chunk_seconds * 1000 / config.frame_ms)
        self.max_frames = int(config.max_chunk_seconds * 1000 / config.frame_ms)
        self.silence_hangover_frames = int(config.silence_hangover_ms / config.frame_ms)

        self.voice_frames = []
        self.silence_frames = 0
        self.latency_stats_ms = []

    def _dbfs(self, frame: np.ndarray) -> float:
        rms = np.sqrt(np.mean(np.square(frame.astype(np.float32))))
        if rms <= 0:
            return -120.0
        return 20.0 * math.log10(rms / 32768.0)

    def _is_speech_frame(self, frame: np.ndarray) -> bool:
        if self._dbfs(frame) < self.config.noise_gate_db:
            return False
        return self.vad.is_speech(frame.tobytes(), self.config.sample_rate)

    def _flush_chunk(self) -> Optional[AudioChunk]:
        if not self.voice_frames:
            return None

        chunk = np.concatenate(self.voice_frames).astype(np.int16)
        if len(self.voice_frames) < self.min_frames:
            self.voice_frames.clear()
            self.silence_frames = 0
            return None

        now = time.time()
        chunk_id = int(now * 1000)
        wav_path = self.out_dir / f"chunk_{chunk_id}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(self.config.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.config.sample_rate)
            wf.writeframes(chunk.tobytes())

        self.voice_frames.clear()
        self.silence_frames = 0
        return AudioChunk(samples=chunk, sample_rate=self.config.sample_rate, created_at=now, wav_path=wav_path)

    def process_block(self, indata: np.ndarray):
        mono = indata[:, 0]
        total_samples = len(mono)
        cursor = 0

        while cursor + self.frame_size <= total_samples:
            frame = mono[cursor: cursor + self.frame_size]
            cursor += self.frame_size
            is_speech = self._is_speech_frame(frame)

            if is_speech:
                self.voice_frames.append(frame.copy())
                self.silence_frames = 0
                if len(self.voice_frames) >= self.max_frames:
                    self._enqueue_chunk()
            else:
                if self.voice_frames:
                    self.silence_frames += 1
                    if self.silence_frames >= self.silence_hangover_frames:
                        self._enqueue_chunk()

    def _enqueue_chunk(self):
        started = time.perf_counter()
        chunk = self._flush_chunk()
        if chunk is None:
            return
        self.loop.call_soon_threadsafe(self.queue.put_nowait, chunk)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.latency_stats_ms.append(elapsed_ms)

    def latency_report(self) -> float:
        if not self.latency_stats_ms:
            return 0.0
        return float(np.mean(self.latency_stats_ms))


async def capture_audio(config: AudioConfig, out_dir: Path, queue: asyncio.Queue, stop_event: asyncio.Event):
    loop = asyncio.get_running_loop()
    chunker = AudioChunker(config=config, out_dir=out_dir, queue=queue, loop=loop)

    def callback(indata, frames, time_info, status):
        del frames, time_info
        if status:
            return
        chunker.process_block(indata.copy())

    with sd.InputStream(
        samplerate=config.sample_rate,
        channels=config.channels,
        dtype="int16",
        blocksize=chunker.frame_size,
        callback=callback,
    ):
        while not stop_event.is_set():
            await asyncio.sleep(0.05)

    if chunker.voice_frames:
        chunker._enqueue_chunk()

    mean_latency = chunker.latency_report()
    if mean_latency > 50.0:
        print(f"[WARN] Audio chunk latency {mean_latency:.1f} ms exceeds target 50 ms")
    else:
        print(f"[OK] Audio chunk latency {mean_latency:.1f} ms")
