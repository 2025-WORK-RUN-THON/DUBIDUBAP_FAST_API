#!/usr/bin/env python3
"""
EC2 서버의 데이터베이스 상태 확인 스크립트
"""

import logging
from app.db import get_session, init_db
from app.models import Video, Embedding, AnalysisSummary
from sqlmodel import select, func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_db_status():
    """데이터베이스 상태 확인"""
    logger.info("Checking database status...")
    
    init_db()
    session = next(get_session())
    
    try:
        # 기본 통계
        video_count = session.exec(select(func.count(Video.id))).one()
        embedding_count = session.exec(select(func.count(Embedding.id))).one()
        summary_count = session.exec(select(func.count(AnalysisSummary.id))).one()
        
        logger.info(f"""
=== 데이터베이스 상태 ===
Videos: {video_count}
Embeddings: {embedding_count}
Analysis Summaries: {summary_count}
""")
        
        # 분석 요약 종류별 개수
        summary_kinds = session.exec(
            select(AnalysisSummary.kind, func.count(AnalysisSummary.id))
            .group_by(AnalysisSummary.kind)
        ).all()
        
        if summary_kinds:
            logger.info("=== 분석 요약 종류별 ===")
            for kind, count in summary_kinds:
                logger.info(f"{kind}: {count}")
        else:
            logger.warning("분석 요약 데이터가 없습니다!")
        
        # 샘플 비디오 확인
        sample_videos = session.exec(select(Video).limit(3)).all()
        if sample_videos:
            logger.info("=== 샘플 비디오 ===")
            for video in sample_videos:
                logger.info(f"- {video.title} ({video.video_id})")
        
        # 임베딩 종류별 개수
        embedding_kinds = session.exec(
            select(Embedding.kind, func.count(Embedding.id))
            .group_by(Embedding.kind)
        ).all()
        
        if embedding_kinds:
            logger.info("=== 임베딩 종류별 ===")
            for kind, count in embedding_kinds:
                logger.info(f"{kind}: {count}")
        
        return {
            "videos": video_count,
            "embeddings": embedding_count,
            "summaries": summary_count,
            "has_analysis_data": summary_count > 0
        }
        
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return None
    finally:
        session.close()

if __name__ == "__main__":
    check_db_status()