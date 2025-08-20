from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.services.embeddings import from_bytes, knn_cosine
from app.models import Embedding
from sqlmodel import Session, select

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore


def _get_qdrant_client() -> Optional[QdrantClient]:  # type: ignore
    if not getattr(settings, "ENABLE_QDRANT", False):
        return None
    if QdrantClient is None:
        return None
    url = getattr(settings, "QDRANT_URL", "")
    if not url:
        return None
    api_key = getattr(settings, "QDRANT_API_KEY", None)
    try:
        client = QdrantClient(url=url, api_key=api_key)
        return client
    except Exception:
        return None


def _collection_name(kind: str) -> str:
    return f"emb_{kind}"


def upsert_vector(
    *,
    session: Session,
    kind: str,
    embedding_id: int,
    video_id: int,
    vector: np.ndarray,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    client = _get_qdrant_client()
    if client is None or qmodels is None:
        # Fallback: nothing to do, we always keep SQLite as source-of-truth
        return
    dim = int(vector.shape[0])
    cname = _collection_name(kind)
    # Ensure collection
    try:
        client.get_collection(cname)
    except Exception:
        try:
            client.recreate_collection(
                collection_name=cname,
                vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            )
        except Exception:
            return
    # Upsert point
    payload_full = {"video_id": video_id, "embedding_id": embedding_id, "kind": kind}
    if payload:
        payload_full.update(payload)
    try:
        client.upsert(
            collection_name=cname,
            points=[
                qmodels.PointStruct(
                    id=int(embedding_id),
                    vector=vector.tolist(),
                    payload=payload_full,
                )
            ],
        )
    except Exception:
        # ignore
        return


def search_vectors(
    *,
    session: Session,
    kind: str,
    query: np.ndarray,
    top_k: int = 5,
) -> List[int]:
    client = _get_qdrant_client()
    if client is not None and qmodels is not None:
        cname = _collection_name(kind)
        try:
            hits = client.search(collection_name=cname, query_vector=query.tolist(), limit=top_k)
            # Return video_ids from payload when available
            ordered_video_ids: List[int] = []
            for h in hits:
                vid = h.payload.get("video_id") if isinstance(h.payload, dict) else None
                if isinstance(vid, int):
                    ordered_video_ids.append(vid)
            if ordered_video_ids:
                return ordered_video_ids
        except Exception:
            pass

    # Fallback to local embeddings
    embeddings = session.exec(select(Embedding).where(Embedding.kind == kind)).all()
    if not embeddings:
        return []
    candidate_vecs = [from_bytes(e.vector) for e in embeddings]
    order = knn_cosine(query, candidate_vecs, top_k=top_k)
    # map to video_ids
    ordered_video_ids = []
    for idx in order:
        e = embeddings[idx]
        ordered_video_ids.append(int(e.video_id))
    return ordered_video_ids


