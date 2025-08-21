"""
실제 YouTube 음악 분석 시스템
오디오 다운로드 → 가사 추출 → 감성 분석 → 음향 특성 분석
"""

import os
import logging
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import yt_dlp
import whisper
import librosa
import numpy as np
import torchcrepe

logger = logging.getLogger(__name__)


class AudioProcessor:
    """YouTube 오디오 처리 및 분석 서비스"""
    
    def __init__(self):
        self.whisper_model = None
        self.audio_temp_dir = Path("temp_audio")
        self.audio_temp_dir.mkdir(exist_ok=True)
        
    async def initialize_models(self):
        """모델 초기화 (지연 로딩)"""
        try:
            # Whisper 모델 로드 (base 모델 사용)
            logger.info("Whisper 모델 로딩 중...")
            self.whisper_model = whisper.load_model("base")
            
            logger.info("✅ Whisper 모델 로딩 완료")
            return True
            
        except Exception as e:
            logger.error(f"모델 초기화 실패: {e}")
            return False
    
    async def download_audio(self, youtube_url: str) -> Optional[str]:
        """YouTube에서 오디오 다운로드"""
        try:
            output_path = self.audio_temp_dir / f"audio_{hash(youtube_url)}.wav"
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(output_path.with_suffix('.%(ext)s')),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            
            if output_path.exists():
                logger.info(f"✅ 오디오 다운로드 완료: {output_path}")
                return str(output_path)
            else:
                logger.error(f"오디오 파일 생성 실패: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"YouTube 오디오 다운로드 실패 {youtube_url}: {e}")
            return None
    
    async def extract_lyrics(self, audio_path: str) -> Dict:
        """Whisper로 가사 추출"""
        try:
            if not self.whisper_model:
                await self.initialize_models()
            
            logger.info(f"가사 추출 중: {audio_path}")
            result = self.whisper_model.transcribe(audio_path, language="ko")
            
            lyrics_data = {
                "text": result["text"].strip(),
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"], 
                        "text": seg["text"].strip()
                    }
                    for seg in result["segments"]
                ],
                "language": result["language"],
                "confidence": np.mean([seg.get("no_speech_prob", 0) for seg in result["segments"]])
            }
            
            logger.info(f"✅ 가사 추출 완료: {len(lyrics_data['text'])}자")
            return lyrics_data
            
        except Exception as e:
            logger.error(f"가사 추출 실패: {e}")
            return {"text": "", "segments": [], "language": "unknown", "confidence": 0.0}
    
    async def analyze_audio_emotion(self, audio_path: str) -> Dict:
        """음향 특성 기반 간단한 감성 분석"""
        try:
            # 오디오 로드
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 기본 음향 특성 추출
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # 간단한 감성 추정 (휴리스틱 기반)
            if tempo > 140 and spectral_centroid > 2000:
                emotion = "energetic"
                confidence = 0.8
            elif tempo < 90 and spectral_centroid < 1500:
                emotion = "calm"
                confidence = 0.7
            elif tempo > 120 and zero_crossing_rate > 0.1:
                emotion = "happy"
                confidence = 0.75
            else:
                emotion = "neutral"
                confidence = 0.6
            
            return {
                "dominant_emotion": emotion,
                "confidence": confidence,
                "analysis": {
                    "tempo": float(tempo),
                    "spectral_centroid": float(spectral_centroid),
                    "zero_crossing_rate": float(zero_crossing_rate)
                }
            }
            
        except Exception as e:
            logger.error(f"감성 분석 실패: {e}")
            return {"dominant_emotion": "neutral", "confidence": 0.5, "analysis": {}}
    
    async def analyze_audio_features(self, audio_path: str) -> Dict:
        """음향 특성 분석 (BPM, 키, 모드 등) - TorchCrepe 사용"""
        try:
            # librosa로 오디오 로드
            y, sr = librosa.load(audio_path, sr=22050)
            
            # BPM 추정
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # TorchCrepe로 음정 추정
            try:
                # 16kHz로 리샘플링 (TorchCrepe 권장)
                y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
                
                # 음정 추적 (f0 추정)
                frequency, confidence = torchcrepe.predict(
                    y_16k, 
                    16000, 
                    hop_length=160,
                    fmin=50, 
                    fmax=550,
                    model='tiny',
                    batch_size=512
                )
                
                # 유효한 음정만 필터링 (신뢰도 0.5 이상)
                valid_freqs = frequency[confidence > 0.5]
                
                if len(valid_freqs) > 0:
                    # 주요 음정 추정
                    median_freq = np.median(valid_freqs)
                    
                    # 음정을 노트로 변환
                    A4_freq = 440.0
                    note_number = 12 * np.log2(median_freq / A4_freq) + 69
                    note_index = int(round(note_number)) % 12
                    
                    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    estimated_key = key_names[note_index]
                    pitch_confidence = float(np.mean(confidence[confidence > 0.5]))
                else:
                    # 폴백: 크로마 기반 키 추정
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    chroma_mean = np.mean(chroma, axis=1)
                    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    estimated_key = key_names[np.argmax(chroma_mean)]
                    pitch_confidence = 0.6
                    
            except Exception as crepe_error:
                logger.warning(f"TorchCrepe 분석 실패, 크로마로 폴백: {crepe_error}")
                # 폴백: 크로마 기반 키 추정
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                estimated_key = key_names[np.argmax(chroma_mean)]
                pitch_confidence = 0.5
            
            # 간단한 모드 추정 (장조/단조)
            if tempo > 120:
                mode = "major"
                mode_confidence = 0.7
            else:
                mode = "minor" 
                mode_confidence = 0.6
            
            return {
                "bpm": float(tempo),
                "key": estimated_key,
                "mode": mode,
                "mode_confidence": float(mode_confidence),
                "pitch_confidence": pitch_confidence,
                "duration": float(len(y) / sr),
                "beats_count": len(beats)
            }
            
        except Exception as e:
            logger.error(f"음향 특성 분석 실패: {e}")
            return {
                "bpm": 120.0,
                "key": "C",
                "mode": "major",
                "mode_confidence": 0.5,
                "pitch_confidence": 0.5,
                "duration": 0.0,
                "beats_count": 0
            }
    
    async def process_youtube_video(self, youtube_url: str, video_id: str) -> Dict:
        """YouTube 비디오 완전 분석 파이프라인"""
        logger.info(f"🎵 YouTube 비디오 분석 시작: {video_id}")
        
        results = {
            "video_id": video_id,
            "url": youtube_url,
            "status": "processing",
            "lyrics": {},
            "emotion": {},
            "audio_features": {},
            "processing_time": 0
        }
        
        import time
        start_time = time.time()
        
        try:
            # 1. 오디오 다운로드
            logger.info("1️⃣ 오디오 다운로드 중...")
            audio_path = await self.download_audio(youtube_url)
            if not audio_path:
                results["status"] = "download_failed"
                return results
            
            # 2. 병렬 처리로 분석 수행
            logger.info("2️⃣ 병렬 분석 시작...")
            tasks = [
                self.extract_lyrics(audio_path),
                self.analyze_audio_emotion(audio_path),
                self.analyze_audio_features(audio_path)
            ]
            
            lyrics_data, emotion_data, features_data = await asyncio.gather(*tasks)
            
            results.update({
                "status": "completed",
                "lyrics": lyrics_data,
                "emotion": emotion_data,
                "audio_features": features_data,
                "processing_time": time.time() - start_time
            })
            
            # 3. 임시 파일 정리
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            logger.info(f"✅ 분석 완료: {video_id} ({results['processing_time']:.2f}초)")
            return results
            
        except Exception as e:
            logger.error(f"비디오 분석 실패 {video_id}: {e}")
            results.update({
                "status": "failed",
                "error": str(e),
                "processing_time": time.time() - start_time
            })
            return results
    
    def cleanup_temp_files(self):
        """임시 파일 정리"""
        try:
            for file in self.audio_temp_dir.glob("*"):
                file.unlink()
            logger.info("임시 오디오 파일 정리 완료")
        except Exception as e:
            logger.warning(f"임시 파일 정리 실패: {e}")


# 전역 인스턴스
audio_processor = AudioProcessor()