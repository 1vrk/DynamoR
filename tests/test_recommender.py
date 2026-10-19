import numpy as np

from scipy.sparse import csr_matrix
from dynamor.services.recommender import ShownItem, format_item, group_items, group_phases, suggest, validate_draft

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
    names = {20: ("Extra Spirit", "spirit", 1600, None), 10: ("Close Quarters", "weapon", 800, None)}
    item_times = csr_matrix(
        ([100, 10, 300, 30], ([0, 0, 1, 1], [0, 1, 0, 1])),
        shape=(4, 3),
        dtype=np.uint32,
    )
    build = suggest(
        5,
        [1, 7],
        index,
        targets,
        np.array([5, 5, 8, 5]),
        [1, 7, 60],
        [10, 20, 30],
        names,
        item_times,
        top_k=2,
        top_items=15,
    )
    grouped = group_items(build.items)
    shown = build.items
    assert shown[0].item_id == 20
    assert shown[0].category == "spirit"
    assert shown[0].confidence_pct == 100
    assert [item.item_id for item in grouped["spirit"]] == [20]
    assert [item.item_id for item in grouped["weapon"]] == [10]


def test_suggest_sorts_the_earlier_purchase_first():
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
    item_times = csr_matrix(
        ([10, 100, 30, 300], ([0, 0, 1, 1], [0, 1, 0, 1])),
        shape=(4, 3),
        dtype=np.uint32,
    )
    names = {20: ("Extra Spirit", "spirit", 6400, None), 10: ("Close Quarters", "weapon", 800, None)}

    build = suggest(
        5,
        [1, 7],
        index,
        targets,
        np.array([5, 5, 8, 5]),
        [1, 7, 60],
        [10, 20, 30],
        names,
        item_times,
        top_k=2,
        top_items=15,
    )

    shown = build.items
    assert [item.item_id for item in shown] == [10, 20]
    assert shown[0].purchase_s == 20
    assert shown[0].cost == 800
    assert format_item(shown[0]) == "Close Quarters — 100% — 800 — 0:20"


def test_suggest_orders_skills_apart_from_shop_items():
    index = csr_matrix([[1, 1, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix([[1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    item_times = csr_matrix(
        ([50, 100, 10, 80, 300, 30], ([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])),
        shape=(2, 3),
        dtype=np.uint32,
    )
    names = {
        10: ("Close Quarters", "weapon", 800, None),
        30: ("Napalm", "skill", None, None),
        40: ("Flame Dash", "skill", None, None),
    }

    build = suggest(
        5,
        [1, 7],
        index,
        targets,
        np.array([5, 5]),
        [1, 7, 60],
        [10, 30, 40],
        names,
        item_times,
        top_k=2,
        top_items=15,
        skill_ids=[30, 40],
    )

    assert [item.item_id for item in build.items] == [10]
    assert [item.item_id for item in build.skills] == [40, 30]
    assert build.skills[0].purchase_s == 20


def test_group_phases_keeps_buy_order_inside_each_row():
    def shown(item_id: int, purchase_s: int | None) -> ShownItem:
        return ShownItem(item_id, str(item_id), "weapon", 50, 1, 1.0, 800, purchase_s, None)

    phases = group_phases(
        [shown(10, 5 * 60), shown(20, 12 * 60), shown(30, 25 * 60), shown(40, 8 * 60), shown(50, None)]
    )

    assert [item.item_id for item in phases["early"]] == [10, 40]
    assert [item.item_id for item in phases["mid"]] == [20]
    assert [item.item_id for item in phases["late"]] == [30, 50]


def test_validate_draft_rejects_a_repeated_or_unknown_enemy():
    known = {1, 2, 3, 4, 5, 6}
    assert validate_draft(1, [2, 2, 3, 4, 5, 6], known) == "Enemy heroes must be different"
    assert validate_draft(1, [2, 3, 4, 5, 6, 7], known) == "Unknown hero"