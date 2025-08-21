#!/usr/bin/env python3
"""
149개 YouTube 영상 데이터 처리 및 데이터베이스 구축 스크립트
- 영상 다운로드 및 오디오 추출
- 자막/transcript 추출
- 음악 분석 (BPM, 키, 모드)
- 감정 분석
- 임베딩 생성
- 데이터베이스 저장
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any
import time
from datetime import datetime

from app.db import get_session, init_db
from app.models import Video, Embedding, AnalysisSummary
from app.services.ytdl import download_audio
from app.services.transcribe import Transcriber
from app.services.text_embed import embed_text
from app.services.music import analyze_music
from app.services.emotion import summarize_audio_emotion
from app.services.embeddings import to_bytes
# from app.core.logging import logger

# 데이터 경로
DATA_DIR = Path("data")
ENRICHED_JSON = DATA_DIR / "yt_enriched.json"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIO_DIR = DATA_DIR / "audio"
RESULTS_FILE = PROCESSED_DIR / "processed_embeddings_final.json"

# 디렉토리 생성
PROCESSED_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('youtube_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_single_video(video_data: Dict, session, transcriber) -> Dict[str, Any]:
    """단일 YouTube 영상 처리"""
    video_id = video_data["id"]
    url = video_data["url"]
    title = video_data["title"]
    
    logger.info(f"Processing video: {title} ({video_id})")
    
    try:
        # 1. 오디오 다운로드
        audio_path = AUDIO_DIR / f"{video_id}.m4a"
        if not audio_path.exists():
            logger.info(f"Downloading audio for {video_id}")
            try:
                downloaded_path = download_audio(url, str(AUDIO_DIR), video_id)
                if not Path(downloaded_path).exists():
                    logger.error(f"Failed to download audio for {video_id}")
                    return {"id": video_id, "status": "failed", "error": "download_failed"}
                audio_path = Path(downloaded_path)
            except Exception as e:
                logger.error(f"Download error for {video_id}: {e}")
                return {"id": video_id, "status": "failed", "error": f"download_error: {e}"}
        
        # 2. 음성 전사
        logger.info(f"Transcribing audio for {video_id}")
        transcript, language, duration = transcriber.transcribe(str(audio_path))
        if not transcript:
            logger.warning(f"No transcript generated for {video_id}")
            transcript = ""
        
        # 3. 음악 분석
        logger.info(f"Analyzing music features for {video_id}")
        import librosa
        y, sr = librosa.load(str(audio_path), sr=None)
        music_features = analyze_music(y, sr)
        
        # 4. 감정 분석
        logger.info(f"Analyzing audio emotion for {video_id}")
        audio_emotion = summarize_audio_emotion(y, sr)
        
        # 5. 임베딩 생성
        logger.info(f"Generating embeddings for {video_id}")
        # 제목 임베딩
        title_embedding = embed_text(title)
        # 가사/transcript 임베딩
        lyrics_embedding = embed_text(transcript) if transcript else embed_text(title)
        
        # 6. 데이터베이스에 저장
        logger.info(f"Saving to database for {video_id}")
        
        # Video 엔티티 생성
        video = Video(
            id=video_id,
            url=url,
            title=title,
            channel_title=video_data.get("channelTitle", ""),
            published_at=video_data.get("publishedAt", ""),
            views=video_data.get("views", 0),
            like_count=video_data.get("likeCount", 0),
            duration_sec=video_data.get("durationSec", 0),
            category_id=video_data.get("categoryId", ""),
            language=video_data.get("language", "ko"),
            transcript=transcript
        )
        
        # 기존 비디오 체크
        existing = session.get(Video, video_id)
        if existing:
            # 업데이트
            for key, value in video.model_dump(exclude={"id"}).items():
                setattr(existing, key, value)
            video = existing
        else:
            session.add(video)
        
        session.flush()  # ID 생성을 위해
        
        # Embedding 엔티티들 생성
        embeddings = [
            Embedding(
                video_id=video_id,
                kind="title",
                vector=to_bytes(title_embedding)
            ),
            Embedding(
                video_id=video_id,
                kind="lyrics",
                vector=to_bytes(lyrics_embedding)
            )
        ]
        
        for emb in embeddings:
            # 기존 임베딩 삭제
            from sqlmodel import select
            existing_emb = session.exec(select(Embedding).where(
                Embedding.video_id == video_id,
                Embedding.kind == emb.kind
            )).first()
            if existing_emb:
                session.delete(existing_emb)
            session.add(emb)
        
        # AnalysisSummary 엔티티들 생성
        summaries = []
        
        # 음악 분석 요약
        if music_features:
            summaries.append(AnalysisSummary(
                video_id=video_id,
                kind="music_summary",
                data_json=json.dumps(music_features)
            ))
        
        # 감정 분석 요약
        if audio_emotion:
            summaries.append(AnalysisSummary(
                video_id=video_id,
                kind="audio_emotion",
                data_json=json.dumps(audio_emotion)
            ))
        
        # 가사 후크 후보들 (간단한 구현)
        if transcript:
            words = transcript.split()
            if len(words) > 10:
                # 자주 나오는 구문 찾기 (간단한 구현)
                phrases = []
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i:i+3])
                    if len(phrase) > 5:
                        phrases.append([phrase, 1])  # [phrase, count]
                
                if phrases:
                    summaries.append(AnalysisSummary(
                        video_id=video_id,
                        kind="lyrics_summary",
                        data_json=json.dumps({"hook_candidates": phrases[:10]})
                    ))
        
        for summary in summaries:
            # 기존 요약 삭제
            existing_summary = session.exec(select(AnalysisSummary).where(
                AnalysisSummary.video_id == video_id,
                AnalysisSummary.kind == summary.kind
            )).first()
            if existing_summary:
                session.delete(existing_summary)
            session.add(summary)
        
        session.commit()
        
        logger.info(f"Successfully processed {video_id}")
        
        # 결과 요약
        result = {
            "id": video_id,
            "title": title,
            "status": "success",
            "transcript_length": len(transcript),
            "music_features": music_features,
            "audio_emotion": audio_emotion,
            "embeddings": {
                "title": title_embedding.tolist(),
                "lyrics": lyrics_embedding.tolist()
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {video_id}: {str(e)}")
        session.rollback()
        return {"id": video_id, "status": "failed", "error": str(e)}

def main():
    """메인 처리 함수"""
    logger = setup_logging()
    
    logger.info("Starting YouTube data processing pipeline")
    
    # 데이터베이스 초기화
    init_db()
    
    # enriched JSON 로드
    if not ENRICHED_JSON.exists():
        logger.error(f"Enriched JSON not found: {ENRICHED_JSON}")
        return
    
    with open(ENRICHED_JSON, "r", encoding="utf-8") as f:
        videos_data = json.load(f)
    
    logger.info(f"Found {len(videos_data)} videos to process")
    
    # 전체 결과 수집
    all_results = []
    session = next(get_session())
    transcriber = Transcriber(model_size="small")
    
    try:
        # 각 비디오 처리
        for i, video_data in enumerate(videos_data, 1):
            logger.info(f"Progress: {i}/{len(videos_data)}")
            
            result = process_single_video(video_data, session, transcriber)
            all_results.append(result)
            
            # 성공/실패 상태 로깅
            if result.get("status") == "success":
                logger.info(f"✅ {result['id']}: {result['title']}")
            else:
                logger.error(f"❌ {result['id']}: {result.get('error', 'unknown error')}")
            
            # 중간 저장 (매 5개마다)
            if i % 5 == 0:
                logger.info(f"Intermediate save at {i} videos")
                with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 최종 결과 저장
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 통계 출력
        successful = [r for r in all_results if r.get("status") == "success"]
        failed = [r for r in all_results if r.get("status") == "failed"]
        
        logger.info(f"""
=== Processing Complete ===
Total Videos: {len(videos_data)}
Successful: {len(successful)}
Failed: {len(failed)}
Results saved to: {RESULTS_FILE}
""")
        
        # 데이터베이스 상태 확인
        from sqlmodel import select
        video_count = len(session.exec(select(Video)).all())
        embedding_count = len(session.exec(select(Embedding)).all())
        summary_count = len(session.exec(select(AnalysisSummary)).all())
        
        logger.info(f"""
=== Database Status ===
Videos: {video_count}
Embeddings: {embedding_count}
Analysis Summaries: {summary_count}
""")
        
    finally:
        session.close()

if __name__ == "__main__":
    main()