from __future__ import annotations

from typing import Dict

import numpy as np
import librosa
from app.core.config import settings


def compute_pitch_profile(y: np.ndarray, sr: int) -> Dict[str, float]:
    try:
        # Simple VAD: energy-based mask to reduce silence
        frame_length = 1024
        hop_length = 256
        
        # RMS 계산
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        thresh = float(np.percentile(rms, 60))
        mask_frames = rms > thresh
        
        # 프레임 분할
        y_frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        
        # 배열 크기 일치 확인 및 안전한 처리
        if mask_frames.size == y_frames.shape[1] and mask_frames.any():
            try:
                y_kept = y_frames[:, mask_frames].ravel(order='F')
                if y_kept.size > 0:
                    y = y_kept
            except (IndexError, ValueError) as e:
                # 프레임 처리 실패 시 원본 오디오 사용
                print(f"VAD 프레임 처리 실패, 원본 오디오 사용: {e}")
        else:
            # 크기가 일치하지 않으면 원본 오디오 사용
            print(f"VAD 마스크 크기 불일치: mask={mask_frames.size}, frames={y_frames.shape[1]}")
        
        # 음정 분석
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz(settings.PITCH_MIN_NOTE), 
            fmax=librosa.note_to_hz(settings.PITCH_MAX_NOTE),
            sr=sr
        )
        
        f0 = f0[~np.isnan(f0)]
        if f0.size == 0:
            return {"mean_hz": 0.0, "std_hz": 0.0, "median_hz": 0.0}
        
        return {
            "mean_hz": float(np.mean(f0)),
            "std_hz": float(np.std(f0)),
            "median_hz": float(np.median(f0)),
        }
        
    except Exception as e:
        print(f"음정 분석 중 오류 발생: {e}")
        return {"mean_hz": 0.0, "std_hz": 0.0, "median_hz": 0.0}


