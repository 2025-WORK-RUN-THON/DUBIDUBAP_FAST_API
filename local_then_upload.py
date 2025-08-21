#!/usr/bin/env python3
"""
로컬에서 분석 후 서버 업로드 전략
1. 로컬에서 오디오 분석 수행
2. 결과를 JSON으로 저장
3. 서버에 분석 결과만 업로드
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.services.audio_processor import audio_processor

async def analyze_locally_then_prepare_upload():
    """로컬 분석 후 업로드용 데이터 준비"""
    
    # enriched JSON 로드
    enriched_file = Path("data/yt_enriched.json")
    with open(enriched_file, "r", encoding="utf-8") as f:
        videos_data = json.load(f)
    
    # AI 모델 초기화
    logger.info("🤖 로컬 AI 모델 초기화...")
    await audio_processor.initialize_models()
    
    # 분석 결과 저장용
    analysis_results = []
    
    # 처음 3개만 테스트
    for i, video_data in enumerate(videos_data[:3], 1):
        video_id = video_data["id"]
        url = video_data["url"]
        title = video_data["title"]
        
        logger.info(f"🎵 로컬 분석 {i}/3: {title}")
        
        # 실제 오디오 분석
        result = await audio_processor.process_youtube_video(url, video_id)
        
        if result["status"] == "completed":
            # 서버 업로드용 데이터 구조
            upload_data = {
                "video_id": video_id,
                "url": url,
                "title": title,
                "view_count": video_data.get("views", 0),
                "tags": video_data.get("tags", []),
                
                # 분석 결과
                "music_summary": {
                    "bpm": result["audio_features"]["bpm"],
                    "key": result["audio_features"]["key"],
                    "mode": result["audio_features"]["mode"],
                    "mode_confidence": result["audio_features"]["mode_confidence"],
                    "pitch_confidence": result["audio_features"]["pitch_confidence"],
                    "duration": result["audio_features"]["duration"],
                    "real_analysis": True
                },
                
                "emotion_analysis": {
                    "dominant_emotion": result["emotion"]["dominant_emotion"],
                    "confidence": result["emotion"]["confidence"],
                    "analysis": result["emotion"]["analysis"],
                    "real_analysis": True
                },
                
                "lyrics_analysis": {
                    "lyrics": result["lyrics"]["text"],
                    "language": result["lyrics"]["language"],
                    "segments": len(result["lyrics"]["segments"]),
                    "confidence": result["lyrics"]["confidence"],
                    "real_analysis": True
                },
                
                "processing_time": result["processing_time"]
            }
            
            analysis_results.append(upload_data)
            logger.info(f"✅ 분석 완료: BPM {result['audio_features']['bpm']:.1f}")
        
        else:
            logger.warning(f"⚠️ 분석 실패: {video_id}")
    
    # 결과 저장
    output_file = Path("data/local_analysis_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📁 분석 결과 저장: {output_file}")
    logger.info(f"총 {len(analysis_results)}개 분석 완료")
    
    # 업로드 스크립트 생성
    create_upload_script(analysis_results)
    
    # 임시 파일 정리
    audio_processor.cleanup_temp_files()

def create_upload_script(analysis_results: List[Dict]):
    """서버 업로드용 스크립트 생성"""
    
    upload_script = f'''#!/usr/bin/env python3
"""
서버 업로드 스크립트 (EC2에서 실행)
로컬에서 분석한 결과를 서버 DB에 저장
"""

import json
import logging
from app.db import get_session, init_db
from app.models import Video, Embedding, AnalysisSummary
from app.services.text_embed import embed_text
from app.services.embeddings import to_bytes
from sqlmodel import select

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_analysis_results():
    """로컬 분석 결과를 서버 DB에 업로드"""
    
    # 분석 결과 로드
    analysis_data = {json.dumps(analysis_results, ensure_ascii=False, indent=2)}
    
    init_db()
    session = next(get_session())
    
    try:
        for data in analysis_data:
            video_id = data["video_id"]
            title = data["title"]
            
            logger.info(f"📤 업로드: {{title}}")
            
            # Video 생성/업데이트
            existing = session.exec(select(Video).where(Video.video_id == video_id)).first()
            
            if existing:
                existing.title = title
                existing.url = data["url"]  
                existing.view_count = data["view_count"]
                video = existing
            else:
                video = Video(
                    video_id=video_id,
                    url=data["url"],
                    title=title,
                    view_count=data["view_count"]
                )
                session.add(video)
            
            session.flush()
            
            # 임베딩 생성 (서버에서)
            title_embedding = embed_text(title)
            
            tags = data.get("tags", [])
            lyrics_text = f"{{title}} {{' '.join(tags[:5])}}" if tags else title
            lyrics_embedding = embed_text(lyrics_text)
            
            # 기존 데이터 삭제
            session.exec(select(Embedding).where(Embedding.video_id == video.id)).delete()
            session.exec(select(AnalysisSummary).where(AnalysisSummary.video_id == video.id)).delete()
            
            # 새 임베딩 추가
            embeddings = [
                Embedding(video_id=video.id, kind="title", vector=to_bytes(title_embedding)),
                Embedding(video_id=video.id, kind="lyrics", vector=to_bytes(lyrics_embedding))
            ]
            
            for emb in embeddings:
                session.add(emb)
            
            # 분석 요약 추가
            summaries = [
                AnalysisSummary(video_id=video.id, kind="music_summary", data_json=json.dumps(data["music_summary"])),
                AnalysisSummary(video_id=video.id, kind="audio_emotion", data_json=json.dumps(data["emotion_analysis"])),
                AnalysisSummary(video_id=video.id, kind="lyrics_analysis", data_json=json.dumps(data["lyrics_analysis"]))
            ]
            
            for summary in summaries:
                session.add(summary)
            
            logger.info(f"✅ 업로드 완료: {{video_id}}")
        
        session.commit()
        logger.info("🎉 모든 분석 결과 업로드 완료!")
        
    except Exception as e:
        logger.error(f"❌ 업로드 실패: {{e}}")
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    upload_analysis_results()
'''
    
    script_file = Path("upload_to_server.py")
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(upload_script)
    
    logger.info(f"📋 업로드 스크립트 생성: {script_file}")

if __name__ == "__main__":
    asyncio.run(analyze_locally_then_prepare_upload())