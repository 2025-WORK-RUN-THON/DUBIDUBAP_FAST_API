#!/usr/bin/env python3
"""
실제 YouTube 오디오 분석을 사용한 데이터 프로세서
Whisper + 감성분석 + 음향특성 분석을 통한 실제 음악 데이터 생성
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any

# 기본 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from app.db import get_session, init_db
from app.models import Video, Embedding, AnalysisSummary
from app.services.text_embed import embed_text
from app.services.embeddings import to_bytes
from app.services.audio_processor import audio_processor

async def process_videos_with_real_analysis():
    """실제 오디오 분석을 사용한 YouTube 비디오 데이터 처리"""
    logger.info("🎵 실제 오디오 분석 기반 비디오 처리 시작")
    
    # 데이터베이스 초기화
    init_db()
    
    # enriched JSON 로드
    enriched_file = Path("data/yt_enriched.json")
    if not enriched_file.exists():
        logger.error("data/yt_enriched.json not found")
        return
    
    with open(enriched_file, "r", encoding="utf-8") as f:
        videos_data = json.load(f)
    
    logger.info(f"총 {len(videos_data)}개 비디오 처리 예정")
    
    # AI 모델 초기화
    logger.info("🤖 AI 모델 초기화 중...")
    model_init_success = await audio_processor.initialize_models()
    if not model_init_success:
        logger.error("AI 모델 초기화 실패 - 더미 데이터로 폴백")
        use_real_analysis = False
    else:
        use_real_analysis = True
    
    session = next(get_session())
    processed_count = 0
    failed_count = 0
    
    try:
        for i, video_data in enumerate(videos_data, 1):
            video_id = video_data["id"]
            title = video_data["title"]
            url = video_data["url"]
            
            logger.info(f"🎵 처리 중 {i}/{len(videos_data)}: {title}")
            
            # Video 엔티티 생성/업데이트
            from sqlmodel import select
            existing = session.exec(
                select(Video).where(Video.video_id == video_id)
            ).first()
            
            if existing:
                existing.title = title
                existing.url = url
                existing.view_count = video_data.get("views", 0)
                video = existing
                logger.info(f"📝 기존 비디오 업데이트: {video_id}")
            else:
                video = Video(
                    video_id=video_id,
                    url=url,
                    title=title,
                    view_count=video_data.get("views", 0)
                )
                session.add(video)
                logger.info(f"➕ 새 비디오 생성: {video_id}")
            
            session.flush()
            
            try:
                # 임베딩 생성 (기존과 동일)
                logger.info(f"🧠 임베딩 생성: {title}")
                title_embedding = embed_text(title)
                
                tags = video_data.get("tags", [])
                if tags:
                    tags_text = " ".join(tags[:5])
                    lyrics_text = f"{title} {tags_text}"
                else:
                    lyrics_text = title
                
                lyrics_embedding = embed_text(lyrics_text)
                
                # 기존 임베딩 삭제
                existing_embeddings = session.exec(
                    select(Embedding).where(Embedding.video_id == video.id)
                ).all()
                for emb in existing_embeddings:
                    session.delete(emb)
                
                # 새 임베딩 추가
                embeddings = [
                    Embedding(
                        video_id=video.id,
                        kind="title",
                        vector=to_bytes(title_embedding)
                    ),
                    Embedding(
                        video_id=video.id,
                        kind="lyrics",
                        vector=to_bytes(lyrics_embedding)
                    )
                ]
                
                for emb in embeddings:
                    session.add(emb)
                
                # 실제 오디오 분석 수행
                if use_real_analysis and i <= 5:  # 처음 5개만 실제 분석 (테스트)
                    logger.info(f"🎧 실제 오디오 분석 시작: {video_id}")
                    
                    analysis_result = await audio_processor.process_youtube_video(url, video_id)
                    
                    if analysis_result["status"] == "completed":
                        # 실제 분석 데이터 사용
                        music_summary = {
                            "bpm": analysis_result["audio_features"]["bpm"],
                            "key": analysis_result["audio_features"]["key"],
                            "mode": analysis_result["audio_features"]["mode"],
                            "confidence": analysis_result["audio_features"]["mode_confidence"],
                            "duration": analysis_result["audio_features"]["duration"],
                            "real_analysis": True
                        }
                        
                        emotion_analysis = {
                            "dominant_emotion": analysis_result["emotion"]["dominant_emotion"],
                            "confidence": analysis_result["emotion"]["confidence"],
                            "emotions": analysis_result["emotion"]["analysis"],
                            "real_analysis": True
                        }
                        
                        # 가사 데이터 (Whisper)
                        lyrics_analysis = {
                            "lyrics": analysis_result["lyrics"]["text"],
                            "language": analysis_result["lyrics"]["language"],
                            "segments": len(analysis_result["lyrics"]["segments"]),
                            "confidence": analysis_result["lyrics"]["confidence"],
                            "real_analysis": True
                        }
                        
                        logger.info(f"✅ 실제 분석 완료: {video_id} (BPM: {music_summary['bpm']:.1f}, 키: {music_summary['key']} {music_summary['mode']})")
                        processed_count += 1
                        
                    else:
                        # 분석 실패시 더미 데이터 사용
                        logger.warning(f"⚠️ 오디오 분석 실패, 더미 데이터 사용: {video_id}")
                        music_summary = generate_dummy_music_summary(i)
                        emotion_analysis = generate_dummy_emotion_analysis(i)
                        lyrics_analysis = generate_dummy_lyrics_analysis(i)
                        failed_count += 1
                        
                else:
                    # 더미 데이터 사용 (나머지 비디오들)
                    music_summary = generate_dummy_music_summary(i)
                    emotion_analysis = generate_dummy_emotion_analysis(i)
                    lyrics_analysis = generate_dummy_lyrics_analysis(i)
                
                # 기존 분석 요약 삭제
                existing_summaries = session.exec(
                    select(AnalysisSummary).where(AnalysisSummary.video_id == video.id)
                ).all()
                for summary in existing_summaries:
                    session.delete(summary)
                
                # 새 분석 요약 추가
                summaries = [
                    AnalysisSummary(
                        video_id=video.id,
                        kind="music_summary",
                        data_json=json.dumps(music_summary)
                    ),
                    AnalysisSummary(
                        video_id=video.id,
                        kind="audio_emotion",
                        data_json=json.dumps(emotion_analysis)
                    ),
                    AnalysisSummary(
                        video_id=video.id,
                        kind="lyrics_analysis",
                        data_json=json.dumps(lyrics_analysis)
                    )
                ]
                
                for summary in summaries:
                    session.add(summary)
                
                logger.info(f"📊 분석 데이터 저장 완료: {video_id}")
                
            except Exception as e:
                logger.error(f"❌ 비디오 처리 오류 {video_id}: {e}")
                failed_count += 1
                continue
        
        # 모든 변경사항 커밋
        session.commit()
        
        # 최종 통계
        from sqlmodel import select, func
        final_video_count = session.exec(select(func.count(Video.id))).one()
        final_embedding_count = session.exec(select(func.count(Embedding.id))).one()
        final_summary_count = session.exec(select(func.count(AnalysisSummary.id))).one()
        
        logger.info(f"""
🎉 === 처리 완료 ===
총 비디오: {final_video_count}
임베딩: {final_embedding_count}
분석 요약: {final_summary_count}
실제 분석 성공: {processed_count}
분석 실패/더미: {failed_count}
""")
        
    except Exception as e:
        logger.error(f"❌ 전체 처리 실패: {e}")
        session.rollback()
        raise
    finally:
        session.close()
        # 임시 파일 정리
        audio_processor.cleanup_temp_files()


def generate_dummy_music_summary(index: int) -> Dict:
    """더미 음악 요약 데이터 생성"""
    return {
        "bpm": round(120 + (index * 5) % 60, 2),
        "key": ["C", "D", "E", "F", "G", "A", "B"][index % 7],
        "mode": ["major", "minor"][index % 2],
        "confidence": 0.7 + (index % 3) * 0.1,
        "real_analysis": False
    }


def generate_dummy_emotion_analysis(index: int) -> Dict:
    """더미 감성 분석 데이터 생성"""
    emotions = ["happy", "energetic", "calm", "melancholic", "excited"]
    return {
        "dominant_emotion": emotions[index % len(emotions)],
        "confidence": 0.6 + (index % 4) * 0.1,
        "emotions": {
            "happy": 0.3 + (index % 3) * 0.2,
            "energetic": 0.4 + (index % 4) * 0.15,
            "calm": 0.2 + (index % 2) * 0.3
        },
        "real_analysis": False
    }


def generate_dummy_lyrics_analysis(index: int) -> Dict:
    """더미 가사 분석 데이터 생성"""
    return {
        "lyrics": f"더미 가사 데이터 {index}",
        "language": "ko",
        "segments": index % 10 + 5,
        "confidence": 0.8,
        "real_analysis": False
    }


if __name__ == "__main__":
    asyncio.run(process_videos_with_real_analysis())