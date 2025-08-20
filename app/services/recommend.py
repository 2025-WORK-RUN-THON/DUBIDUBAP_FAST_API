from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np
from sqlmodel import Session, select

from app.models import Video, Embedding
from app.services.embeddings import from_bytes, cosine_similarity


def recommend_videos(
    session: Session,
    kind_weights: Dict[str, float],
    query_vectors: Dict[str, np.ndarray],
    top_k: int = 3,
) -> List[Tuple[Video, float]]:
    videos = session.exec(select(Video)).all()
    if not videos:
        return []

    # Load embeddings per kind
    kind_to_embs: dict[str, list[Embedding]] = {}
    for kind in kind_weights.keys():
        emb_list = session.exec(select(Embedding).where(Embedding.kind == kind)).all()
        kind_to_embs[kind] = emb_list

    # Aggregate score per video
    scores: dict[int, float] = {v.id: 0.0 for v in videos}
    for kind, weight in kind_weights.items():
        embs = kind_to_embs.get(kind) or []
        if not embs or kind not in query_vectors:
            continue
        q = query_vectors[kind]
        for e in embs:
            ev = from_bytes(e.vector)
            # Pad/trim query to match embedding dimension defensively
            if q.shape[0] < ev.shape[0]:
                pad = np.zeros(ev.shape[0] - q.shape[0], dtype=np.float32)
                q_aligned = np.concatenate([q, pad], axis=0)
            elif q.shape[0] > ev.shape[0]:
                q_aligned = q[: ev.shape[0]]
            else:
                q_aligned = q
            v = cosine_similarity(q_aligned, ev)
            scores[e.video_id] = scores.get(e.video_id, 0.0) + float(weight) * float(v)

    # Rank
    ranked = sorted(((vid, score) for vid, score in scores.items()), key=lambda x: x[1], reverse=True)
    # Map ids to objects
    id_to_video: dict[int, Video] = {v.id: v for v in videos}
    out: list[Tuple[Video, float]] = []
    for vid, sc in ranked[:top_k]:
        video = id_to_video.get(vid)
        if video:
            out.append((video, sc))
    return out


