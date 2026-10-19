import numpy as np

from scipy.sparse import csr_matrix
from dynamor.services.recommender import group_items, suggest, validate_draft

def test_suggest_names_the_situational_item_and_groups_it():
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
    names = {20: ("Extra Spirit", "spirit"), 10: ("Close Quarters", "weapon")}
    shown = suggest(
        5,
        [1, 7],
        index,
        targets,
        np.array([5, 5, 8, 5]),
        [1, 7, 60],
        [10, 20, 30],
        names,
        top_k=2,
        top_items=15,
    )
    grouped = group_items(shown)
    assert shown[0].item_id == 20
    assert shown[0].category == "spirit"
    assert shown[0].confidence_pct == 100
    assert [item.item_id for item in grouped["spirit"]] == [20]
    assert [item.item_id for item in grouped["weapon"]] == [10]

def test_validate_draft_rejects_a_repeated_or_unknown_enemy():
    known = {1, 2, 3, 4, 5, 6}
    assert validate_draft(1, [2, 2, 3, 4, 5, 6], known) == "Enemy heroes must be different"
    assert validate_draft(1, [2, 3, 4, 5, 6, 7], known) == "Unknown hero"