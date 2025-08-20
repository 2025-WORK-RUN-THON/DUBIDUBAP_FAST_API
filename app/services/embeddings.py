from __future__ import annotations

from typing import Iterable, List

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def to_bytes(vector: Iterable[float]) -> bytes:
    return np.asarray(list(vector), dtype=np.float32).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def knn_cosine(query: np.ndarray, candidates: List[np.ndarray], top_k: int = 5) -> List[int]:
    if not candidates:
        return []
    sims = [cosine_similarity(query, c) for c in candidates]
    order = np.argsort(sims)[::-1]
    return list(map(int, order[:top_k]))


