#!/usr/bin/env python3
"""
서버 업로드 스크립트 (EC2에서 실행)
Step 2에서 분석한 결과를 서버 DB에 저장
"""

import json
import logging
from pathlib import Path

# 프로젝트 모듈 import
import sys
sys.path.append('/opt/trendy-lyrics/current')

from app.db import get_session, init_db
from app.models import Video, Embedding, AnalysisSummary
from app.services.text_embed import embed_text
from app.services.embeddings import to_bytes
from sqlmodel import select

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_analysis_results():
    """분석 결과를 서버 DB에 업로드"""
    
    # 분석 결과 파일 로드
    analysis_file = "analyzed_results_20250821_232951.json"
    
    if not Path(analysis_file).exists():
        logger.error(f"분석 파일이 없습니다: {analysis_file}")
        return False
    
    with open(analysis_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    videos = data.get("videos", [])
    metadata = data.get("metadata", {})
    
    logger.info(f"📁 업로드할 분석 결과: {len(videos)}개")
    logger.info(f"📊 성공률: {metadata.get('success_rate', 0):.1f}%")
    
    # DB 초기화 및 세션 생성
    init_db()
    session = next(get_session())
    
    uploaded_count = 0
    
    try:
        for i, video_data in enumerate(videos, 1):
            video_id = video_data["video_id"]
            title = video_data["title"]
            
            logger.info(f"📤 업로드 {i}/{len(videos)}: {title}")
            
            # Video 엔티티 생성/업데이트
            existing = session.exec(select(Video).where(Video.video_id == video_id)).first()
            
            if existing:
                existing.title = title
                existing.url = video_data["url"]
                existing.view_count = video_data.get("views", 0)
                video = existing
                logger.debug("기존 비디오 업데이트")
            else:
                video = Video(
                    video_id=video_id,
                    url=video_data["url"],
                    title=title,
                    view_count=video_data.get("views", 0)
                )
                session.add(video)
                logger.debug("새 비디오 생성")
            
            session.flush()
            
            # 임베딩 생성 (서버에서 동일한 모델 사용)
            title_embedding = embed_text(title)
            
            # 검색 키워드를 lyrics로 활용
            search_query = video_data.get("search_query", "")
            lyrics_text = f"{title} {search_query}"
            lyrics_embedding = embed_text(lyrics_text)
            
            # 기존 데이터 삭제 (SQLModel 방식)
            existing_embeddings = session.exec(
                select(Embedding).where(Embedding.video_id == video.id)
            ).all()
            for emb in existing_embeddings:
                session.delete(emb)
            
            existing_summaries = session.exec(
                select(AnalysisSummary).where(AnalysisSummary.video_id == video.id)
            ).all()
            for summary in existing_summaries:
                session.delete(summary)
            
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
            
            # 분석 요약 추가
            summaries = [
                AnalysisSummary(
                    video_id=video.id,
                    kind="music_summary",
                    data_json=json.dumps(video_data["music_summary"])
                ),
                AnalysisSummary(
                    video_id=video.id,
                    kind="audio_emotion",
                    data_json=json.dumps(video_data["emotion_analysis"])
                ),
                AnalysisSummary(
                    video_id=video.id,
                    kind="lyrics_analysis",
                    data_json=json.dumps(video_data["lyrics_analysis"])
                )
            ]
            
            for summary in summaries:
                session.add(summary)
            
            uploaded_count += 1
            
            # 주기적 커밋 (메모리 관리)
            if i % 10 == 0:
                session.commit()
                logger.info(f"💾 중간 저장: {i}개 처리 완료")
        
        # 최종 커밋
        session.commit()
        
        logger.info(f"""
🎉 업로드 완료!
   ✅ 성공: {uploaded_count}개
   💾 데이터베이스에 저장 완료
""")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 업로드 실패: {e}")
        session.rollback()
        return False
    
    finally:
        session.close()

if __name__ == "__main__":
    success = upload_analysis_results()
    if success:
        logger.info("✅ 서버 업로드 완료!")
    else:
        logger.error("❌ 서버 업로드 실패!")