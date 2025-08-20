from __future__ import annotations

from typing import Dict, List, Tuple

import re
from collections import Counter


def normalize_korean_text(text: str) -> str:
    # 간단 정규화: 공백/특수문자 정리
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def count_brand_mentions(text: str, brand_name: str) -> int:
    """브랜드명 언급 횟수 (대소문자 무시, 단순 포함 기반).

    한국어 형태소 분석은 생략하고, 부분일치 카운트로 구현.
    """
    if not brand_name:
        return 0
    text_norm = normalize_korean_text(text).lower()
    brand_norm = brand_name.strip().lower()
    if not brand_norm:
        return 0
    # 겹치지 않게 카운트
    return len(re.findall(re.escape(brand_norm), text_norm))


def detect_repeated_phrases(text: str, min_ngram: int = 2, max_ngram: int = 6, top_k: int = 5) -> List[Tuple[str, int]]:
    """가사 내 반복 구절(Hook) 후보 탐지: n-그램 빈도 기반.

    - 너무 짧은 토큰 제거 (1글자 토큰 제외)
    - 공백/구두점 정리 후 분석
    """
    text_norm = normalize_korean_text(text)
    # 구두점 제거
    text_norm = re.sub(r"[\.,!?;:\-\(\)\[\]\{\}\"\']", " ", text_norm)
    tokens = [t for t in text_norm.split() if len(t) > 1]
    if len(tokens) < min_ngram:
        return []

    candidates: Counter[str] = Counter()
    for n in range(min_ngram, max_ngram + 1):
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            candidates[ngram] += 1

    # 2회 이상 등장하는 것만 후보
    repeated = [(phrase, cnt) for phrase, cnt in candidates.items() if cnt >= 2]
    repeated.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    return repeated[:top_k]



