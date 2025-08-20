from __future__ import annotations

from typing import Dict, List


def build_master_prompt(
    service_name: str,
    slogan: str,
    target_customer: str,
    mood_tone: List[str],
    music_genre: str,
    version: str,
    trends: List[str] | None,
    examples: List[Dict] | None,
    extra: str | None,
) -> str:
    lines: list[str] = []
    lines.append("너는 최고의 로고송 프로듀서야. 아래 정보를 바탕으로 로고송 가사와 SUNO 프롬프트를 만들어줘.")
    lines.append("")
    lines.append("### 1. 사용자 입력 정보")
    lines.append(f"* 서비스 이름: {service_name}")
    lines.append(f"* 슬로건: {slogan}")
    lines.append(f"* 타겟 고객: {target_customer}")
    lines.append(f"* 요청 분위기: {', '.join(mood_tone)}")
    lines.append(f"* 음악 장르: {music_genre}")
    lines.append(f"* 분량: {version}")
    if extra:
        lines.append(f"* 추가 요청: {extra}")
    lines.append("")
    lines.append("### 2. 참고 데이터")
    lines.append("* 성공 로고송 패턴: 서비스 이름은 최소 2번 이상 반복, 슬로건은 후렴구에 배치.")
    if trends:
        lines.append(f"* 구글 트렌드 키워드: {', '.join(trends)}")
    if examples:
        lines.append("* 유사 사례 예시:")
        for ex in examples[:3]:
            lines.append(f"  - {ex.get('title','')} | {ex.get('url','')}")
    lines.append("")
    lines.append("### 3. 생성 작업 지시")
    lines.append("1. 위 정보를 모두 반영하여 중독성 있는 한국어 가사를 만들어줘. 서비스 이름과 슬로건을 자연스럽게 반복해줘.")
    if trends:
        lines.append("2. 가사에 트렌드 키워드를 자연스럽게 녹여줘.")
    else:
        lines.append("2. 타겟 고객과 장르에 맞는 한국어 어휘를 적절히 사용해줘.")
    lines.append("3. 가사를 바탕으로 SUNO에 입력할 상세한 프롬프트를 한국어로 작성해줘. 요청 분위기와 장르를 반영해.")
    return "\n".join(lines)


def build_suno_prompt_ko(
    service_name: str,
    mood_tone: List[str],
    music_genre: str,
    version: str,
) -> str:
    # 한국어 그대로 사용
    mood = ", ".join(mood_tone)
    
    return (
        f"'{service_name}'을 위한 중독성 있는 브랜드 로고송을 만들어주세요. "
        f"스타일: {music_genre}. 분위기: {mood}. 길이: {version}. "
        f"강력한 후크, 기억에 남는 후렴구, 현대적인 프로덕션에 집중해주세요."
    )


def build_suno_prompt_en(
    service_name: str,
    mood_tone: List[str],
    music_genre: str,
    version: str,
) -> str:
    # 영어 버전도 필요시 사용
    mood = ", ".join(mood_tone)
    
    return (
        f"Create a catchy brand jingle for '{service_name}'. "
        f"Style: {music_genre}. Mood: {mood}. Length: {version}. "
        f"Focus on strong hooks, memorable chorus, and modern production."
    )


