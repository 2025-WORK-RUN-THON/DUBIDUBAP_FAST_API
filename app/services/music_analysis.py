"""
음악 분석 서비스
임베딩 기반 유사도 검색 및 트렌드 분석
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


class MusicAnalysisService:
    """음악 분석 및 검색 서비스"""
    
    def __init__(self):
        self.embeddings_data: List[Dict] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.data_loaded = False
        
    async def load_embeddings_data(self, data_path: str = "data/processed/processed_embeddings_final.json"):
        """임베딩 데이터 로드"""
        try:
            data_file = Path(data_path)
            if not data_file.exists():
                logger.warning(f"임베딩 데이터 파일이 없습니다: {data_path}")
                return False
            
            with open(data_file, 'r', encoding='utf-8') as f:
                self.embeddings_data = json.load(f)
            
            # 임베딩 매트릭스 생성
            embeddings = [item['embedding'] for item in self.embeddings_data if item.get('embedding')]
            if embeddings:
                self.embeddings_matrix = np.array(embeddings)
                self.data_loaded = True
                
                logger.info(
                    f"✅ 임베딩 데이터 로드 완료: {len(self.embeddings_data)}개 비디오, "
                    f"임베딩 차원: {self.embeddings_matrix.shape}"
                )
                return True
            else:
                logger.error("유효한 임베딩 데이터가 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"임베딩 데이터 로드 실패: {e}")
            return False
    
    async def create_query_embedding(self, text: str) -> Optional[np.ndarray]:
        """쿼리 텍스트에서 임베딩 생성"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"쿼리 임베딩 생성 실패: {e}")
            return None
    
    async def find_similar_music(
        self, 
        query: str, 
        search_type: str = "combined",
        limit: int = 10,
        threshold: float = 0.3
    ) -> List[Dict]:
        """유사한 음악 검색"""
        
        if not self.data_loaded or self.embeddings_matrix is None:
            logger.warning("임베딩 데이터가 로드되지 않았습니다")
            return []
        
        # 쿼리 임베딩 생성
        query_embedding = await self.create_query_embedding(query)
        if query_embedding is None:
            return []
        
        # 코사인 유사도 계산
        similarities = cosine_similarity([query_embedding], self.embeddings_matrix)[0]
        
        # 임계값 이상인 결과만 필터링
        valid_indices = np.where(similarities >= threshold)[0]
        if len(valid_indices) == 0:
            logger.info(f"임계값 {threshold} 이상인 결과가 없습니다")
            return []
        
        # 상위 결과 선별
        top_indices = valid_indices[np.argsort(similarities[valid_indices])[-limit:][::-1]]
        
        results = []
        for idx in top_indices:
            video = self.embeddings_data[idx].copy()
            video['similarity_score'] = float(similarities[idx])
            
            # 응답에서 임베딩 제거 (크기 때문에)
            video.pop('embedding', None)
            
            results.append(video)
        
        logger.info(f"유사도 검색 완료: {len(results)}개 결과 (최고 점수: {max(similarities):.3f})")
        return results
    
    async def get_music_recommendations(self, video_id: str, limit: int = 5) -> List[Dict]:
        """특정 비디오 기반 추천"""
        
        if not self.data_loaded:
            return []
        
        # 해당 비디오 찾기
        target_video = None
        target_idx = None
        
        for idx, video in enumerate(self.embeddings_data):
            if video['video_id'] == video_id:
                target_video = video
                target_idx = idx
                break
        
        if target_video is None:
            logger.warning(f"비디오를 찾을 수 없습니다: {video_id}")
            return []
        
        # 해당 비디오의 임베딩으로 유사도 계산
        target_embedding = np.array(target_video['embedding']).reshape(1, -1)
        similarities = cosine_similarity(target_embedding, self.embeddings_matrix)[0]
        
        # 자기 자신 제외하고 상위 결과 선별
        similarities[target_idx] = -1  # 자기 자신 제외
        top_indices = np.argsort(similarities)[-limit:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 유효한 유사도만
                video = self.embeddings_data[idx].copy()
                video['similarity_score'] = float(similarities[idx])
                video.pop('embedding', None)
                recommendations.append(video)
        
        logger.info(f"추천 생성 완료: {video_id} -> {len(recommendations)}개")
        return recommendations
    
    async def analyze_music_trends(self, period: str = "all") -> Dict:
        """음악 트렌드 분석"""
        
        if not self.data_loaded:
            return {"error": "데이터가 로드되지 않았습니다"}
        
        # 조회수 기반 인기 순위
        popular_videos = sorted(
            self.embeddings_data, 
            key=lambda x: x['views'], 
            reverse=True
        )[:10]
        
        # 총 통계
        total_views = sum(video['views'] for video in self.embeddings_data)
        avg_views = total_views / len(self.embeddings_data)
        
        # 템포 분석 (오디오 특성이 있는 경우)
        tempos = [
            video['audio_features'].get('tempo', 0) 
            for video in self.embeddings_data 
            if video.get('audio_features', {}).get('tempo')
        ]
        
        avg_tempo = np.mean(tempos) if tempos else 0
        
        # 키워드 분석 (제목에서)
        keywords = {}
        for video in self.embeddings_data:
            title_words = video['title'].split()
            for word in title_words:
                if len(word) > 1:  # 한 글자 제외
                    keywords[word] = keywords.get(word, 0) + 1
        
        # 상위 키워드
        top_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        
        trend_analysis = {
            "period": period,
            "total_videos": len(self.embeddings_data),
            "total_views": total_views,
            "average_views": int(avg_views),
            "popular_videos": [
                {
                    "title": video["title"],
                    "views": video["views"],
                    "video_id": video["video_id"]
                }
                for video in popular_videos
            ],
            "audio_analysis": {
                "average_tempo": avg_tempo,
                "tempo_count": len(tempos)
            },
            "trending_keywords": [
                {"keyword": word, "count": count}
                for word, count in top_keywords
            ]
        }
        
        logger.info(f"트렌드 분석 완료: {len(self.embeddings_data)}개 비디오")
        return trend_analysis
    
    async def generate_music_summary(self, similar_videos: List[Dict]) -> Dict:
        """유사 음악 기반 특성 요약"""
        
        if not similar_videos:
            return {
                "bpm": 120.0,
                "key": "C",
                "mode": "major",
                "genre": "pop"
            }
        
        # 평균 BPM 계산
        tempos = [
            video.get('audio_features', {}).get('tempo', 120)
            for video in similar_videos
        ]
        avg_bpm = np.mean([t for t in tempos if t > 0]) if tempos else 120.0
        
        # 조회수 기반 가중 평균
        total_views = sum(video['views'] for video in similar_videos)
        if total_views > 0:
            weighted_bpm = sum(
                video.get('audio_features', {}).get('tempo', 120) * video['views']
                for video in similar_videos
            ) / total_views
            avg_bmp = weighted_bpm if weighted_bpm > 0 else avg_bpm
        
        # 장르 추정 (간단한 BPM 기반)
        if avg_bpm < 90:
            genre = "ballad"
        elif avg_bpm < 120:
            genre = "pop"
        elif avg_bpm < 140:
            genre = "dance"
        else:
            genre = "electronic"
        
        return {
            "bpm": round(float(avg_bpm), 2),
            "key": "C",  # 향후 오디오 분석으로 개선
            "mode": "major",
            "genre": genre,
            "confidence": min(len(similar_videos) / 5.0, 1.0)  # 0-1 신뢰도
        }


# 전역 서비스 인스턴스
music_analysis_service = MusicAnalysisService()