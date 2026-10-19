import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from dynamor.features.catalog import index_of
from dynamor.models.base import Recommendation


def enemy_vector(enemy_ids, hero_catalog: list[int]) -> csr_matrix:
    positions = index_of(hero_catalog)
    columns: list[int] = []
    seen: set[int] = set()
    for hero_id in enemy_ids:
        if hero_id is None:
            continue
        hero_id = int(hero_id)
        if hero_id in seen or hero_id not in positions:
            continue
        seen.add(hero_id)
        columns.append(positions[hero_id])
    width = len(hero_catalog)
    if not columns:
        return csr_matrix((1, width), dtype=np.float64)
    values = np.ones(len(columns), dtype=np.float64)
    rows = np.zeros(len(columns), dtype=np.int32)
    return csr_matrix((values, (rows, columns)), shape=(1, width), dtype=np.float64)


def nearest_rows(
    index: csr_matrix,
    hero_row_mask: np.ndarray,
    query: csr_matrix,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(hero_row_mask, dtype=bool)
    if not mask.any() or query.nnz == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    positions = np.flatnonzero(mask)
    scores = cosine_similarity(query, index[positions]).ravel()
    order = np.argsort(-scores, kind="stable")[:top_k]
    return positions[order], scores[order]


def rank_items(
    targets: csr_matrix,
    neighbor_rows: np.ndarray,
    hero_rows: np.ndarray,
    item_catalog: list[int],
    top_items: int,
) -> list[Recommendation]:
    neighbor_rows = np.asarray(neighbor_rows)
    hero_rows = np.asarray(hero_rows)
    n_neighbors = len(neighbor_rows)
    if n_neighbors == 0:
        return []
    tf = np.asarray(targets[neighbor_rows].sum(axis=0)).ravel()
    df = np.asarray(targets[hero_rows].sum(axis=0)).ravel()
    n_hero = len(hero_rows)
    kept = np.flatnonzero(tf > 0)
    idf = np.log((1 + n_hero) / (1 + df[kept])) + 1
    scores = tf[kept] * idf
    confidence = tf[kept] / n_neighbors
    recommendations = [
        Recommendation(
            item_id=int(item_catalog[int(column)]),
            score=float(score),
            tf=int(count),
            confidence=float(share),
        )
        for column, score, count, share in zip(kept, scores, tf[kept], confidence, strict=True)
    ]
    recommendations.sort(key=lambda item: (-item.score, item.item_id))
    return recommendations[:top_items]


def recommend(
    hero_id: int,
    enemy_ids,
    index: csr_matrix,
    targets: csr_matrix,
    hero_ids: np.ndarray,
    hero_catalog: list[int],
    item_catalog: list[int],
    top_k: int,
    top_items: int,
) -> list[Recommendation]:
    query = enemy_vector(enemy_ids, hero_catalog)
    if query.nnz == 0:
        return []
    mask = np.asarray(hero_ids) == int(hero_id)
    neighbor_rows, _scores = nearest_rows(index, mask, query, top_k)
    return rank_items(targets, neighbor_rows, np.flatnonzero(mask), item_catalog, top_items)
