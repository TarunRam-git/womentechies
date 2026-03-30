import re
from dataclasses import dataclass
from typing import List

from faster_whisper import WhisperModel

from .audio_capture import AudioChunk
from .config import STTConfig


FILLER_WORDS = {
    "uh", "um", "hmm", "erm", "like", "you know", "actually", "basically"
}


@dataclass
class STTResult:
    text: str
    confidence: float
    words: List[dict]
    wav_path: str


class WhisperSTT:
    def __init__(self, config: STTConfig):
        self.config = config
        self.model = WhisperModel(
            model_size_or_path=config.model_name,
            device=config.device,
            compute_type=config.compute_type,
        )

    def _clean_text(self, text: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
        filtered = [tok for tok in tokens if tok not in FILLER_WORDS]
        return " ".join(filtered)

    def transcribe_chunk(self, chunk: AudioChunk) -> STTResult:
        segments, info = self.model.transcribe(
            chunk.samples.astype("float32") / 32768.0,
            language=self.config.language,
            word_timestamps=True,
            beam_size=5,
            vad_filter=False,
        )

        words = []
        text_parts = []
        conf_values = []

        for seg in segments:
            text_parts.append(seg.text.strip())
            if seg.avg_logprob is not None:
                conf_values.append(float(seg.avg_logprob))
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    })

        raw_text = " ".join([t for t in text_parts if t]).strip()
        clean = self._clean_text(raw_text)

        if conf_values:
            avg_prob_proxy = sum(conf_values) / len(conf_values)
            confidence = max(0.0, min(1.0, 1.0 + (avg_prob_proxy / 5.0)))
        else:
            confidence = 0.0

        if confidence < self.config.confidence_threshold:
            clean = ""

        return STTResult(text=clean, confidence=confidence, words=words, wav_path=str(chunk.wav_path))
