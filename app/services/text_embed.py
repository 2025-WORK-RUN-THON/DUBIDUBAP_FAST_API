from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np

from app.core.config import settings
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _hash_bytes(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def embed_text(text: str, dim: int | None = None) -> np.ndarray:
    """Return embedding for text.

    If ENABLE_OPENAI_EMBEDDINGS is true and OPENAI_API_KEY available, use OpenAI. Otherwise fallback to
    deterministic hash-based embedding to avoid external dependency.
    """
    if settings.ENABLE_OPENAI_EMBEDDINGS and settings.OPENAI_API_KEY and OpenAI is not None:
        try:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            model = settings.OPENAI_EMBED_MODEL
            resp = client.embeddings.create(model=model, input=text)
            vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
            return vec
        except Exception:
            pass

    dimension = dim or settings.TEXT_EMBED_DIM
    seed = _hash_bytes(text)
    rng = np.random.default_rng(int.from_bytes(seed[:8], "little"))
    vec = rng.standard_normal(dimension).astype(np.float32)
    scale = 1.0 + (len(text) % 7) * 0.01
    return vec * scale


def embed_texts(texts: Iterable[str], dim: int | None = None) -> np.ndarray:
    arr = [embed_text(t, dim=dim) for t in texts]
    if not arr:
        return np.zeros((0, dim or settings.TEXT_EMBED_DIM), dtype=np.float32)
    return np.stack(arr, axis=0)


