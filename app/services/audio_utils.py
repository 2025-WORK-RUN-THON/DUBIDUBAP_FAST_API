from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np


def load_mono_audio(path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    y, orig_sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


