from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field


class Video(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: str = Field(index=True, unique=True)
    title: str
    view_count: int = 0
    url: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Embedding(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    kind: str = Field(description="type of embedding: lyrics, audio_emotion, pitch_profile, etc")
    vector: bytes = Field(description="binary-serialized float vector")
    metadata_json: str = Field(default="{}")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Transcript(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    text: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Prompt(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    request_id: Optional[str] = Field(default=None, index=True)
    master_prompt: str
    suno_prompt: str
    payload_json: str = Field(default="{}")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AnalysisSummary(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(foreign_key="video.id", index=True)
    kind: str = Field(description="summary kind: pitch_profile|audio_emotion")
    data_json: str = Field(default="{}")
    created_at: datetime = Field(default_factory=datetime.utcnow)


