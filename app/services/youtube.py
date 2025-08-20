from __future__ import annotations

from typing import List, Tuple

from googleapiclient.discovery import build
import time
from app.utils.rate_limit import get_limiter


def fetch_popular_ad_videos(
    api_key: str,
    query: str,
    min_view_count: int = 1_000_000,
    target_count: int = 20,
) -> List[Tuple[str, str]]:
    youtube = build("youtube", "v3", developerKey=api_key)
    limiter = get_limiter("youtube_search", max_calls=8, per_seconds=1.0)

    video_urls: List[Tuple[str, str]] = []
    processed_ids: set[str] = set()
    next_token = None

    while len(video_urls) < target_count:
        # Rate-limit search.list
        while not limiter.allow():
            time.sleep(limiter.sleep_for_next_allowed())
        search_response = (
            youtube.search()
            .list(q=query, part="id", type="video", order="viewCount", maxResults=50, pageToken=next_token)
            .execute()
        )

        video_ids: list[str] = []
        for item in search_response.get("items", []):
            vid = item["id"]["videoId"]
            if vid not in processed_ids:
                video_ids.append(vid)

        if not video_ids:
            break

        # Rate-limit videos.list
        while not limiter.allow():
            time.sleep(limiter.sleep_for_next_allowed())
        video_response = (
            youtube.videos()
            .list(part="statistics,snippet", id=",".join(video_ids))
            .execute()
        )

        for video in video_response.get("items", []):
            view_count = int(video["statistics"].get("viewCount", 0))
            if view_count >= min_view_count:
                video_id = video["id"]
                title = video["snippet"]["title"]
                url = f"https://www.youtube.com/watch?v={video_id}"
                video_urls.append((url, title))
                processed_ids.add(video_id)

                if len(video_urls) >= target_count:
                    break

        if len(video_urls) >= target_count:
            break

        next_token = search_response.get("nextPageToken")
        if not next_token:
            break

    return video_urls


