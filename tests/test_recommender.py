import numpy as np

from scipy.sparse import csr_matrix
from dynamor.services.recommender import (
    ShownItem,
    format_item,
    group_items,
    group_phases,
    investment_progress,
    suggest,
    validate_draft,
)

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


def _priced(item_id: int, category: str, cost: int, purchase_s: int, upgrade_from: int | None = None) -> ShownItem:
    return ShownItem(item_id, str(item_id), category, 50, 1, 1.0, cost, purchase_s, None, upgrade_from)


def test_suggest_puts_a_sold_lane_item_before_a_later_final():
    index = csr_matrix([[1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix([[1, 0], [1, 0]], dtype=np.uint8)
    item_times = csr_matrix(([300, 300], ([0, 1], [0, 0])), shape=(2, 2), dtype=np.uint32)
    names = {
        10: ("Close Quarters", "weapon", 800, None),
        99: ("Extra Spirit", "spirit", 800, None),
    }
    purchases = [
        [(99, 60, 90), (10, 300, 0)],
        [(99, 60, 80), (10, 300, 0)],
    ]

    build = suggest(
        5,
        [1],
        index,
        targets,
        np.array([5, 5]),
        [1, 7, 60],
        [10, 99],
        names,
        item_times,
        top_k=2,
        top_items=15,
        purchases=purchases,
    )

    assert [item.item_id for item in build.items] == [99, 10]
    assert build.items[0].purchase_s == 60
    assert build.items[0].confidence_pct == 100


def test_suggest_drops_a_purchase_below_forty_percent():
    index = csr_matrix([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix([[1], [1], [1]], dtype=np.uint8)
    item_times = csr_matrix(([10, 10, 10], ([0, 1, 2], [0, 0, 0])), shape=(3, 1), dtype=np.uint32)
    names = {10: ("Close Quarters", "weapon", 800, None), 98: ("Headshot Booster", "weapon", 800, None)}
    purchases = [[(98, 60, 70), (10, 10, 0)], [(10, 10, 0)], [(10, 10, 0)]]

    build = suggest(
        5,
        [1],
        index,
        targets,
        np.array([5, 5, 5]),
        [1, 7, 60],
        [10, 98],
        names,
        item_times,
        top_k=3,
        top_items=15,
        purchases=purchases,
    )

    assert [item.item_id for item in build.items] == [10]


def test_sold_component_in_the_route_counts_once_toward_the_upgrade():
    index = csr_matrix([[1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix([[0, 1], [0, 1]], dtype=np.uint8)
    item_times = csr_matrix(([200, 200], ([0, 1], [1, 1])), shape=(2, 2), dtype=np.uint32)
    names = {10: ("Extra Spirit", "spirit", 800, None), 30: ("Improved Spirit", "spirit", 3200, None)}
    purchases = [[(10, 60, 200), (30, 200, 0)], [(10, 60, 200), (30, 200, 0)]]
    upgrades = [[(30, 10)], [(30, 10)]]

    build = suggest(
        5,
        [1],
        index,
        targets,
        np.array([5, 5]),
        [1, 7, 60],
        [10, 30],
        names,
        item_times,
        top_k=2,
        top_items=15,
        upgrades=upgrades,
        purchases=purchases,
    )

    assert [item.item_id for item in build.items] == [10, 30]
    assert build.items[1].upgrade_from == 10
    assert investment_progress(build.items, ["spirit"]).totals["spirit"] == 3200


def test_upgrade_adds_only_the_price_difference():
    progress = investment_progress(
        [_priced(1, "spirit", 800, 10), _priced(2, "spirit", 3200, 20, 1)],
        ["spirit"],
    )

    assert progress.totals["spirit"] == 3200
    assert progress.crossed == {}


def test_separate_items_add_both_prices():
    progress = investment_progress(
        [_priced(1, "spirit", 1600, 10), _priced(2, "spirit", 3200, 20)],
        ["spirit"],
    )

    assert progress.totals["spirit"] == 4800
    assert progress.crossed[2] == "spirit"


def test_a_full_spirit_bar_is_not_a_close_without_a_spirit_scale():
    progress = investment_progress([_priced(1, "spirit", 6400, 10)], ["weapon", "vitality"])

    assert progress.totals["spirit"] == 6400
    assert progress.crossed == {}


def test_validate_draft_rejects_a_repeated_or_unknown_enemy():
    known = {1, 2, 3, 4, 5, 6}
    assert validate_draft(1, [2, 2, 3, 4, 5, 6], known) == "Enemy heroes must be different"
    assert validate_draft(1, [2, 3, 4, 5, 6, 7], known) == "Unknown hero"


def _preference_rows():
    index = csr_matrix([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix(
        [[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]],
        dtype=np.uint8,
    )
    item_times = csr_matrix(
        (
            [20, 20, 20, 20, 300, 300, 300, 60, 60],
            ([0, 1, 2, 3, 0, 1, 2, 0, 1], [0, 0, 0, 0, 1, 1, 1, 2, 2]),
        ),
        shape=(4, 3),
        dtype=np.uint32,
    )
    names = {
        10: ("Extra Health", "vitality", 800, None),
        20: ("Close Quarters", "weapon", 800, None),
        30: ("Boundless Spirit", "spirit", 3200, None),
    }
    return index, targets, item_times, names


def _preference_build(index, targets, item_times, names, **kwargs):
    return suggest(
        5,
        [1],
        index,
        targets,
        np.array([5, 5, 5, 5]),
        [1, 7, 60],
        [10, 20, 30],
        names,
        item_times,
        top_k=4,
        top_items=2,
        **kwargs,
    )


def test_spirit_stats_replace_a_higher_neighbor_score_and_time_still_orders_the_queue():
    index, targets, item_times, names = _preference_rows()

    build = _preference_build(
        index,
        targets,
        item_times,
        names,
        scale_weights={"spirit": 1.0},
        item_stats={30: {"spirit": 2}},
    )

    assert [item.item_id for item in build.items] == [10, 30]
    assert build.items[0].purchase_s == 20
    assert build.items[1].purchase_s == 60


def test_suggest_without_weights_keeps_the_neighbor_score_order():
    index, targets, item_times, names = _preference_rows()

    build = _preference_build(index, targets, item_times, names)

    assert [item.item_id for item in build.items] == [10, 20]


def test_a_6400_item_does_not_outweigh_an_equal_4800_close():
    index = csr_matrix([[1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix([[1, 1], [1, 1]], dtype=np.uint8)
    item_times = csr_matrix(([100, 100, 100, 100], ([0, 0, 1, 1], [0, 1, 0, 1])), shape=(2, 2), dtype=np.uint32)
    names = {
        10: ("Improved Spirit", "spirit", 4800, None),
        20: ("Boundless Spirit", "spirit", 6400, None),
    }

    build = suggest(
        5,
        [1],
        index,
        targets,
        np.array([5, 5]),
        [1, 7, 60],
        [10, 20],
        names,
        item_times,
        top_k=2,
        top_items=1,
        scale_weights={"spirit": 1.0},
    )

    assert [item.item_id for item in build.items] == [10]


def test_lane_share_ignores_header_stats():
    index = csr_matrix([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix([[1], [1], [1]], dtype=np.uint8)
    item_times = csr_matrix(([10, 10, 10], ([0, 1, 2], [0, 0, 0])), shape=(3, 1), dtype=np.uint32)
    names = {10: ("Close Quarters", "weapon", 800, None), 98: ("Extra Spirit", "spirit", 800, None)}
    purchases = [[(98, 60, 70), (10, 10, 0)], [(10, 10, 0)], [(10, 10, 0)]]

    build = suggest(
        5,
        [1],
        index,
        targets,
        np.array([5, 5, 5]),
        [1, 7, 60],
        [10, 98],
        names,
        item_times,
        top_k=3,
        top_items=15,
        purchases=purchases,
        scale_weights={"spirit": 1.0},
        item_stats={98: {"spirit": 2}},
    )

    assert [item.item_id for item in build.items] == [10]