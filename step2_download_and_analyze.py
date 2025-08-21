#!/usr/bin/env python3
"""
Step 2: 수집된 YouTube URL 다운로드 및 분석
Step 1에서 수집한 URL들을 실제로 다운로드하고 오디오 분석 수행
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import argparse

# 기본 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from app.services.audio_processor import audio_processor

class YouTubeAnalyzer:
    """YouTube 비디오 다운로드 및 분석기"""
    
    def __init__(self):
        self.analysis_results = []
        self.failed_analyses = []
        self.output_dir = Path("data/analyzed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_collection(self, collection_file: str) -> List[Dict]:
        """Step 1에서 수집한 URL 데이터 로드"""
        
        file_path = Path("data/collected") / collection_file
        if not file_path.exists():
            # 현재 디렉토리에서도 확인
            file_path = Path(collection_file)
            if not file_path.exists():
                raise FileNotFoundError(f"수집 파일을 찾을 수 없습니다: {collection_file}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        videos = data.get("videos", [])
        metadata = data.get("metadata", {})
        
        logger.info(f"📁 수집 파일 로드: {file_path}")
        logger.info(f"📊 총 {len(videos)}개 비디오 (수집일: {metadata.get('collected_at', 'N/A')})")
        
        return videos
    
    async def analyze_batch(
        self, 
        videos: List[Dict], 
        batch_size: int = 5,
        max_videos: Optional[int] = None
    ) -> List[Dict]:
        """비디오 배치 분석"""
        
        if max_videos:
            videos = videos[:max_videos]
            logger.info(f"🎯 분석 대상을 {max_videos}개로 제한")
        
        logger.info(f"🎵 총 {len(videos)}개 비디오 분석 시작 (배치 크기: {batch_size})")
        
        # AI 모델 초기화
        logger.info("🤖 AI 모델 로딩...")
        model_success = await audio_processor.initialize_models()
        if not model_success:
            logger.error("❌ AI 모델 로딩 실패")
            return []
        
        # 배치별 처리
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(videos) + batch_size - 1) // batch_size
            
            logger.info(f"🔄 배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개)")
            
            # 배치 내 병렬 처리
            tasks = []
            for video in batch:
                task = self.analyze_single_video(video)
                tasks.append(task)
            
            # 배치 실행
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ 배치 오류: {batch[j]['title']}: {result}")
                    self.failed_analyses.append({
                        "video": batch[j],
                        "error": str(result),
                        "failed_at": datetime.now().isoformat()
                    })
                elif result:
                    self.analysis_results.append(result)
            
            # 진행상황 출력
            success_count = len(self.analysis_results)
            fail_count = len(self.failed_analyses)
            processed = success_count + fail_count
            
            logger.info(f"📊 배치 완료: 성공 {success_count}, 실패 {fail_count} (전체 {processed}/{len(videos)})")
            
            # 배치 간 휴식
            if i + batch_size < len(videos):
                await asyncio.sleep(2)
        
        # 최종 정리
        audio_processor.cleanup_temp_files()
        
        logger.info(f"""
🎉 전체 분석 완료!
   ✅ 성공: {len(self.analysis_results)}개
   ❌ 실패: {len(self.failed_analyses)}개
   📊 성공률: {len(self.analysis_results) / len(videos) * 100:.1f}%
""")
        
        return self.analysis_results
    
    async def analyze_single_video(self, video: Dict) -> Optional[Dict]:
        """단일 비디오 분석"""
        
        video_id = video["id"]
        url = video["url"]
        title = video["title"]
        
        try:
            logger.info(f"🎵 분석 시작: {title}")
            
            # 오디오 분석 실행
            result = await audio_processor.process_youtube_video(url, video_id)
            
            if result["status"] == "completed":
                # 원본 메타데이터와 분석 결과 병합
                analysis_data = {
                    # 원본 정보
                    "video_id": video_id,
                    "url": url,
                    "title": title,
                    "description": video.get("description", ""),
                    "channelTitle": video.get("channelTitle", ""),
                    "publishedAt": video.get("publishedAt", ""),
                    "views": video.get("views", 0),
                    "likes": video.get("likes", 0),
                    "comments": video.get("comments", 0),
                    "duration": video.get("duration", ""),
                    "search_query": video.get("search_query", ""),
                    
                    # 분석 결과
                    "music_summary": {
                        "bpm": result["audio_features"]["bpm"],
                        "key": result["audio_features"]["key"],
                        "mode": result["audio_features"]["mode"],
                        "mode_confidence": result["audio_features"]["mode_confidence"],
                        "pitch_confidence": result["audio_features"]["pitch_confidence"],
                        "duration_analyzed": result["audio_features"]["duration"],
                        "beats_count": result["audio_features"]["beats_count"],
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
                        "segments_count": len(result["lyrics"]["segments"]),
                        "confidence": result["lyrics"]["confidence"],
                        "real_analysis": True
                    },
                    
                    # 메타데이터
                    "processing_time": result["processing_time"],
                    "analyzed_at": datetime.now().isoformat(),
                    "analysis_version": "1.0"
                }
                
                logger.info(f"✅ 분석 완료: {title} | BPM: {result['audio_features']['bpm']:.1f} | 키: {result['audio_features']['key']} {result['audio_features']['mode']}")
                return analysis_data
            
            else:
                logger.warning(f"⚠️ 분석 실패: {title} - {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 분석 오류: {title}: {e}")
            return None
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """분석 결과 저장"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analyzed_results_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # 메타데이터 포함한 최종 데이터
        output_data = {
            "metadata": {
                "analyzed_at": datetime.now().isoformat(),
                "total_analyzed": len(self.analysis_results),
                "total_failed": len(self.failed_analyses),
                "success_rate": len(self.analysis_results) / (len(self.analysis_results) + len(self.failed_analyses)) * 100 if self.analysis_results or self.failed_analyses else 0,
                "analysis_version": "1.0",
                "tools_used": ["yt-dlp", "whisper", "torchcrepe", "librosa"]
            },
            "videos": self.analysis_results,
            "failed_analyses": self.failed_analyses
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 분석 결과 저장: {output_path}")
        
        # 통계 출력
        if self.analysis_results:
            bpms = [v["music_summary"]["bpm"] for v in self.analysis_results]
            avg_bpm = sum(bpms) / len(bpms)
            
            emotions = [v["emotion_analysis"]["dominant_emotion"] for v in self.analysis_results]
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])
            
            logger.info(f"""
📈 분석 통계:
   평균 BPM: {avg_bpm:.1f}
   가장 많은 감정: {most_common_emotion[0]} ({most_common_emotion[1]}개)
   언어 분포: {len([v for v in self.analysis_results if v['lyrics_analysis']['language'] == 'ko'])}개 한국어
""")
        
        return str(output_path)
    
    def create_upload_script(self, analysis_file: str) -> str:
        """서버 업로드용 스크립트 생성"""
        
        upload_script_template = '''#!/usr/bin/env python3
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
    analysis_file = "ANALYSIS_FILE_PLACEHOLDER"
    
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
            
            # 기존 데이터 삭제
            session.query(Embedding).filter(Embedding.video_id == video.id).delete()
            session.query(AnalysisSummary).filter(AnalysisSummary.video_id == video.id).delete()
            
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
'''
        
        # 스크립트 파일명 생성
        script_filename = f"upload_{Path(analysis_file).stem}_to_server.py"
        script_path = self.output_dir / script_filename
        
        # 실제 분석 파일명으로 치환
        upload_script = upload_script_template.replace("ANALYSIS_FILE_PLACEHOLDER", analysis_file)
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(upload_script)
        
        logger.info(f"📋 서버 업로드 스크립트 생성: {script_path}")
        return str(script_path)


def main():
    """메인 함수"""
    
    parser = argparse.ArgumentParser(description="YouTube 비디오 다운로드 및 분석")
    parser.add_argument("collection_file", help="Step 1에서 생성된 수집 파일")
    parser.add_argument("--max-videos", type=int, help="분석할 최대 비디오 수 (테스트용)")
    parser.add_argument("--batch-size", type=int, default=3, help="배치 크기 (기본: 3)")
    
    args = parser.parse_args()
    
    analyzer = YouTubeAnalyzer()
    
    try:
        # 수집 파일 로드
        videos = analyzer.load_collection(args.collection_file)
        
        if not videos:
            logger.error("❌ 분석할 비디오가 없습니다")
            return
        
        # 분석 실행
        results = asyncio.run(analyzer.analyze_batch(
            videos=videos,
            batch_size=args.batch_size,
            max_videos=args.max_videos
        ))
        
        if results:
            # 결과 저장
            analysis_file = analyzer.save_results()
            
            # 서버 업로드 스크립트 생성
            upload_script = analyzer.create_upload_script(Path(analysis_file).name)
            
            logger.info(f"""
🎯 Step 2 완료!
   📁 분석 결과: {analysis_file}
   📋 업로드 스크립트: {upload_script}
   
🚀 다음 단계 (서버에서 실행):
   scp {Path(analysis_file).name} {Path(upload_script).name} ec2-user@3.36.70.96:/opt/trendy-lyrics/current/
   ssh -i ~/Desktop/keypair/umc-hackathon.pem ec2-user@3.36.70.96
   cd /opt/trendy-lyrics/current && python {Path(upload_script).name}
""")
        
        else:
            logger.error("❌ 분석된 비디오가 없습니다")
    
    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()