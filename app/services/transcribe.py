from __future__ import annotations

import os
from typing import Optional, Tuple

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "default"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Tuple[str, Optional[str], Optional[float]]:
        segments, info = self.model.transcribe(audio_path, language=language)
        texts = []
        for seg in segments:
            texts.append(seg.text.strip())
        full_text = " ".join([t for t in texts if t])
        return full_text, info.language, info.duration


