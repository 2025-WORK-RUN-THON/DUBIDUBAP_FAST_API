#!/usr/bin/env python3
"""
Step 1: YouTube 키워드 기반 URL 수집 자동화
로고송, CM송, 광고음악 등의 키워드로 YouTube Data API 검색 (3분 이내 영상만 필터링)
"""

import os
import json
import logging
import time
import re ### <-- NEW/MODIFIED ###
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import requests

# 기본 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeCollector:
    """YouTube Data API를 사용한 비디오 수집기"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.collected_videos = []
        self.seen_video_ids = set()
        
        # 결과 저장 폴더
        self.output_dir = Path("data/collected")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod ### <-- NEW/MODIFIED ###
    def _parse_duration(duration_str: str) -> Optional[int]: ### <-- NEW/MODIFIED ###
        """ISO 8601 duration 형식을 초 단위로 변환합니다.""" ### <-- NEW/MODIFIED ###
        if not duration_str: ### <-- NEW/MODIFIED ###
            return None ### <-- NEW/MODIFIED ###
        
        # Regex to parse ISO 8601 duration format (PT#H#M#S) ### <-- NEW/MODIFIED ###
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str) ### <-- NEW/MODIFIED ###
        if not match: ### <-- NEW/MODIFIED ###
            return None ### <-- NEW/MODIFIED ###
        
        hours = int(match.group(1)) if match.group(1) else 0 ### <-- NEW/MODIFIED ###
        minutes = int(match.group(2)) if match.group(2) else 0 ### <-- NEW/MODIFIED ###
        seconds = int(match.group(3)) if match.group(3) else 0 ### <-- NEW/MODIFIED ###
        
        delta = timedelta(hours=hours, minutes=minutes, seconds=seconds) ### <-- NEW/MODIFIED ###
        return int(delta.total_seconds()) ### <-- NEW/MODIFIED ###
    
    def search_videos(
        self, 
        query: str, 
        max_results: int = 50,
        order: str = "relevance",
        published_after: Optional[str] = None
    ) -> List[Dict]:
        """YouTube 검색 API 호출"""
        
        url = f"{self.base_url}/search"
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": min(max_results, 50),
            "order": order,
            "key": self.api_key,
            "regionCode": "KR",
            "relevanceLanguage": "ko"
        }
        
        if published_after:
            params["publishedAfter"] = published_after
        
        try:
            logger.info(f"🔍 검색중: '{query}' (최대 {max_results}개)")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            videos = []
            
            for item in data.get("items", []):
                video_id = item["id"]["videoId"]
                
                if video_id in self.seen_video_ids:
                    continue
                
                video_info = {
                    "id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "channelTitle": item["snippet"]["channelTitle"],
                    "publishedAt": item["snippet"]["publishedAt"],
                    "thumbnails": item["snippet"]["thumbnails"],
                    "search_query": query,
                    "collected_at": datetime.now().isoformat()
                }
                
                videos.append(video_info)
                self.seen_video_ids.add(video_id)
            
            logger.info(f"✅ 검색 완료: {len(videos)}개 (중복 제거 후)")
            return videos
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 검색 실패 '{query}': {e}")
            return []
    
    def get_video_statistics(self, video_ids: List[str]) -> Dict[str, Dict]:
        """비디오 통계 정보 가져오기 (조회수, 좋아요, 댓글, **영상 길이** 등)"""
        
        if not video_ids:
            return {}
        
        url = f"{self.base_url}/videos"
        all_stats = {}
        
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            params = {
                "part": "statistics,contentDetails",
                "id": ",".join(batch_ids),
                "key": self.api_key
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                for item in data.get("items", []):
                    video_id = item["id"]
                    stats = item.get("statistics", {})
                    content_details = item.get("contentDetails", {})
                    
                    all_stats[video_id] = {
                        "views": int(stats.get("viewCount", 0)),
                        "likes": int(stats.get("likeCount", 0)),
                        "comments": int(stats.get("commentCount", 0)),
                        "duration": content_details.get("duration", ""),
                        "definition": content_details.get("definition", "sd")
                    }
                
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ 통계 수집 실패: {e}")
                continue
        
        logger.info(f"📊 통계 수집 완료: {len(all_stats)}개")
        return all_stats
    
    def collect_by_keywords(
        self, 
        keywords: List[str], 
        target_count: int = 100,
        min_views: int = 1000,
        max_duration_sec: int = 180 ### <-- NEW/MODIFIED ###
    ) -> List[Dict]:
        """키워드 리스트로 비디오 수집"""
        
        logger.info(f"🎯 목표: 총 {target_count}개 수집 (최소 조회수: {min_views:,}, 최대 길이: {max_duration_sec}초)") ### <-- NEW/MODIFIED ###
        
        per_keyword = max(target_count // len(keywords), 10)
        
        for keyword in keywords:
            if len(self.collected_videos) >= target_count:
                break
            
            videos = self.search_videos(keyword, max_results=per_keyword * 2)
            
            if videos:
                video_ids = [v["id"] for v in videos]
                stats = self.get_video_statistics(video_ids)
                
                collected_count_for_keyword = 0 ### <-- NEW/MODIFIED ###
                for video in videos:
                    video_id = video["id"]
                    if video_id in stats:
                        video.update(stats[video_id])
                        
                        # 조회수 필터
                        if video.get("views", 0) < min_views:
                            logger.debug(f"📉 조회수 부족: {video['title']} ({video.get('views', 0):,})")
                            continue
                        
                        # ### 영상 길이 필터 ### <-- NEW/MODIFIED ###
                        duration_sec = self._parse_duration(video.get("duration", "")) ### <-- NEW/MODIFIED ###
                        if duration_sec is None or duration_sec > max_duration_sec: ### <-- NEW/MODIFIED ###
                            logger.debug(f"⏳ 영상 길이 초과: {video['title']} ({duration_sec}초)") ### <-- NEW/MODIFIED ###
                            continue ### <-- NEW/MODIFIED ###
                        
                        # 모든 필터 통과
                        video["duration_seconds"] = duration_sec ### <-- NEW/MODIFIED ###
                        self.collected_videos.append(video)
                        collected_count_for_keyword += 1 ### <-- NEW/MODIFIED ###

                logger.info(f"💾 '{keyword}' 수집: {collected_count_for_keyword}개 추가") ### <-- NEW/MODIFIED ###
                
            time.sleep(1)
        
        self.collected_videos.sort(key=lambda x: x.get("views", 0), reverse=True)
        
        if len(self.collected_videos) > target_count:
            self.collected_videos = self.collected_videos[:target_count]
        
        logger.info(f"🎉 최종 수집: {len(self.collected_videos)}개")
        return self.collected_videos
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """결과를 JSON 파일로 저장"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_collection_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        output_data = {
            "metadata": {
                "collected_at": datetime.now().isoformat(),
                "total_count": len(self.collected_videos),
                "collection_method": "youtube_data_api_v3",
                "keywords_used": list(set(v.get("search_query") for v in self.collected_videos)),
                "min_views_filter": True,
                "max_duration_filter_sec": 180 ### <-- NEW/MODIFIED ###
            },
            "videos": self.collected_videos
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 결과 저장: {output_path}")
        
        if not self.collected_videos: ### <-- NEW/MODIFIED ###
            logger.warning("⚠️ 수집된 비디오가 없어 통계를 생성할 수 없습니다.") ### <-- NEW/MODIFIED ###
            return str(output_path) ### <-- NEW/MODIFIED ###

        total_views = sum(v.get("views", 0) for v in self.collected_videos)
        avg_views = total_views // len(self.collected_videos) if self.collected_videos else 0
        
        logger.info(f"""
📈 수집 통계:
   총 비디오: {len(self.collected_videos):,}개
   총 조회수: {total_views:,}
   평균 조회수: {avg_views:,}
   최고 조회수: {max(v.get('views', 0) for v in self.collected_videos):,}
""")
        
        return str(output_path)
    
    def validate_collection(self) -> Dict:
        """수집 결과 검증"""
        
        if not self.collected_videos:
            return {"valid": False, "errors": ["수집된 비디오가 없습니다"]}
        
        errors = []
        warnings = []
        
        urls = [v["url"] for v in self.collected_videos]
        if len(urls) != len(set(urls)):
            errors.append("중복된 URL이 발견되었습니다")
        
        low_views = [v for v in self.collected_videos if v.get("views", 0) < 1000]
        if len(low_views) > len(self.collected_videos) * 0.3:
            warnings.append(f"조회수가 낮은 비디오가 많습니다: {len(low_views)}개")
        
        old_videos = []
        one_year_ago = datetime.now() - timedelta(days=365)
        
        for video in self.collected_videos:
            try:
                published = datetime.fromisoformat(video["publishedAt"].replace("Z", "+00:00"))
                if published < one_year_ago:
                    old_videos.append(video)
            except:
                pass
        
        if len(old_videos) > len(self.collected_videos) * 0.5:
            warnings.append(f"1년 이상 된 비디오가 많습니다: {len(old_videos)}개")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "total": len(self.collected_videos),
                "low_views": len(low_views),
                "old_videos": len(old_videos)
            }
        }


def main():
    """메인 실행 함수"""
    
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("❌ YOUTUBE_API_KEY 환경변수가 설정되지 않았습니다")
        logger.error("   Google Cloud Console에서 YouTube Data API v3 키를 발급받으세요")
        logger.error("   export YOUTUBE_API_KEY='your_api_key_here'")
        return
    
    keywords = [
        "로고송", "CM송", "광고음악", "브랜드송", "기업 음악",
        "광고 BGM", "상업 음악", "마케팅 음악", "브랜드 뮤직",
        "기업 로고송", "TV 광고 음악", "라디오 광고",
        "YouTube 광고 음악", "브랜딩 음악", "제품 광고 음악"
    ]
    
    collector = YouTubeCollector(api_key)
    
    try:
        videos = collector.collect_by_keywords(
            keywords=keywords,
            target_count=100,
            min_views=5000,
            max_duration_sec=180  # 3분 (3 * 60 = 180초) ### <-- NEW/MODIFIED ###
        )
        
        if videos:
            output_file = collector.save_results("collected_urls_100.json")
            validation = collector.validate_collection()
            
            if validation["valid"]:
                logger.info("✅ 수집 검증 통과!")
            else:
                logger.warning("⚠️ 검증 오류:")
                for error in validation["errors"]:
                    logger.error(f"   {error}")
            
            if validation["warnings"]:
                logger.warning("⚠️ 경고사항:")
                for warning in validation["warnings"]:
                    logger.warning(f"   {warning}")
            
            logger.info(f"""
🎯 Step 1 완료!
   📁 저장 위치: {output_file}
   📊 수집 개수: {len(videos)}개
   
🚀 다음 단계: 
   python step2_download_and_analyze.py {Path(output_file).name}
""")
            
        else:
            logger.error("❌ 조건에 맞는 비디오를 수집하지 못했습니다.")
    
    except KeyboardInterrupt:
        logger.info("⏹️ 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}", exc_info=True)


if __name__ == "__main__":
    main()