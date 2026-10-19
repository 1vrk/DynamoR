import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from dynamor.models.cosine import enemy_vector, nearest_rows, rank_items, recommend


def test_enemy_vector_ignores_unknown_and_duplicate_ids():
    vector = enemy_vector([60, 99, 60, 7], [1, 7, 60])

    assert vector.dtype == np.float64
    assert vector.shape == (1, 3)
    assert vector.toarray().tolist() == [[0.0, 1.0, 1.0]]


def test_nearest_rows_keeps_same_hero_and_ranks_by_cosine():
    index = csr_matrix(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    mask = np.array([True, True, False, True])
    query = csr_matrix(index.getrow(0))

    rows, scores = nearest_rows(index, mask, query, top_k=2)

    assert rows.tolist() == [0, 1]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1 / np.sqrt(2))

    all_rows, _ = nearest_rows(index, mask, query, top_k=10)
    assert all_rows.tolist() == [0, 1, 3]


def test_nearest_rows_returns_empty_without_a_hero_or_known_enemies():
    index = csr_matrix(np.eye(2, dtype=np.float64))
    query = enemy_vector([99], [1, 7])

    rows, scores = nearest_rows(index, np.array([False, False]), query, top_k=2)
    assert rows.tolist() == []
    assert scores.tolist() == []

    rows, scores = nearest_rows(index, np.array([True, True]), query, top_k=2)
    assert rows.tolist() == []
    assert scores.tolist() == []


def test_rank_items_prefers_the_less_common_item():
    targets = csr_matrix(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.uint8,
    )

    ranked = rank_items(targets, [0, 1], [0, 1, 2, 3], [10, 20, 30], top_items=15)

    assert [item.item_id for item in ranked] == [20, 10]
    assert [item.tf for item in ranked] == [2, 2]
    assert ranked[0].confidence == pytest.approx(1.0)
    assert ranked[1].confidence == pytest.approx(1.0)
    assert ranked[0].score == pytest.approx(2 * (math.log(5 / 3) + 1))
    assert ranked[1].score == pytest.approx(2.0)
    assert rank_items(targets, [0, 1], [0, 1, 2, 3], [10, 20, 30], top_items=1)[0].item_id == 20


def test_recommend_returns_the_situational_item_first():
    index = csr_matrix(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    targets = csr_matrix(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.uint8,
    )
    hero_ids = np.array([5, 5, 8, 5])

    recommendations = recommend(
        5,
        [1, 7],
        index,
        targets,
        hero_ids,
        [1, 7, 60],
        [10, 20, 30],
        top_k=2,
        top_items=15,
    )

    assert recommendations[0].item_id == 20
    assert recommend(
        5,
        [99],
        index,
        targets,
        hero_ids,
        [1, 7, 60],
        [10, 20, 30],
        top_k=2,
        top_items=15,
    ) == []
    assert recommend(
        0,
        [1, 7],
        index,
        targets,
        hero_ids,
        [1, 7, 60],
        [10, 20, 30],
        top_k=2,
        top_items=15,
    ) == []
