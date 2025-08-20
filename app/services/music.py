from __future__ import annotations

from typing import Dict

import numpy as np
import librosa


MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASS_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]


def estimate_key_from_chroma(chroma: np.ndarray) -> tuple[str, str, float]:
    """Estimate musical key from chroma using Krumhansl profiles."""
    chroma_mean = chroma.mean(axis=1)
    # Normalize
    if np.max(chroma_mean) > 0:
        chroma_mean = chroma_mean / np.max(chroma_mean)

    best_key = "C"
    best_mode = "major"
    best_score = -1.0

    for shift in range(12):
        major_score = float(np.correlate(np.roll(MAJOR_PROFILE, shift), chroma_mean))
        minor_score = float(np.correlate(np.roll(MINOR_PROFILE, shift), chroma_mean))
        if major_score > best_score:
            best_score = major_score
            best_key = PITCH_CLASS_NAMES[shift]
            best_mode = "major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = PITCH_CLASS_NAMES[shift]
            best_mode = "minor"

    return best_key, best_mode, best_score


def analyze_music(y: np.ndarray, sr: int) -> Dict[str, float | str]:
    """Analyze BPM and key/mode from audio waveform."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except Exception:
        tempo = 0.0

    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key, mode, confidence = estimate_key_from_chroma(chroma)
    except Exception:
        key, mode, confidence = "C", "major", 0.0

    return {
        "bpm": float(tempo),
        "key": key,
        "mode": mode,
        "key_confidence": float(confidence),
    }


