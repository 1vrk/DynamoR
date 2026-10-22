import numpy as np

from scipy.sparse import csr_matrix
from dynamor.services.recommender import (
    ShownItem,
    curve_bonus,
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


def _heal_answer_build(*, hero_tags=None, item_answers=None, item_effects=None):
    index = csr_matrix([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix(
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0]],
        dtype=np.uint8,
    )
    item_times = csr_matrix(
        (
            [30, 30, 30, 30, 10, 10, 10, 90, 90, 90],
            ([0, 1, 2, 3, 0, 1, 2, 0, 1, 2], [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]),
        ),
        shape=(4, 3),
        dtype=np.uint32,
    )
    names = {
        10: ("Close Quarters", "weapon", 800, None),
        15: ("Healing Nova", "vitality", 1600, None),
        20: ("Healbane", "spirit", 1600, None),
    }
    return suggest(
        5,
        [1, 1, 1, 1],
        index,
        targets,
        np.array([5, 5, 5, 5]),
        [1, 7, 60],
        [10, 15, 20],
        names,
        item_times,
        top_k=4,
        top_items=2,
        hero_tags=hero_tags,
        item_answers=item_answers,
        item_effects=item_effects,
    )


def test_enemy_heal_raises_antiheal_and_time_still_orders_the_queue():
    answers = {15: ("heal",), 20: ("antiheal",), 40: ("antiheal",)}

    build = _heal_answer_build(hero_tags={1: ("heal",)}, item_answers=answers)

    assert [item.item_id for item in build.items] == [10, 20]
    assert build.items[0].purchase_s == 30
    assert build.items[1].purchase_s == 90


def test_suggest_without_threat_tags_keeps_the_neighbor_order():
    build = _heal_answer_build()

    assert [item.item_id for item in build.items] == [15, 10]


def test_curve_bonus_is_the_threshold_already_reached():
    assert curve_bonus("weapon", 0) == 0
    assert curve_bonus("weapon", 1600) == 12
    assert curve_bonus("weapon", 3200) == 18
    assert curve_bonus("weapon", 4000) == 18
    assert curve_bonus("weapon", 4800) == 46
    assert curve_bonus("weapon", 6400) == 54
    assert curve_bonus("weapon", 4800) - curve_bonus("weapon", 3200) == 28
    assert curve_bonus("vitality", 4800) == 38
    assert curve_bonus("spirit", 4800) == 38


def _two_item_build(names, item_times, *, top_items, enemies=None, **kwargs):
    index = csr_matrix([[1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix([[1, 1], [1, 1]], dtype=np.uint8)
    return suggest(
        5,
        [1, 1] if enemies is None else enemies,
        index,
        targets,
        np.array([5, 5]),
        [1, 7, 60],
        [10, 20],
        names,
        item_times,
        top_k=2,
        top_items=top_items,
        **kwargs,
    )


def test_a_later_1600_outranks_the_3200_step_and_time_stays_first():
    item_times = csr_matrix(
        ([100, 400, 100, 400], ([0, 0, 1, 1], [0, 1, 0, 1])),
        shape=(2, 2),
        dtype=np.uint32,
    )
    names = {
        10: ("Titanic Magazine", "weapon", 3200, None),
        20: ("Close Quarters", "weapon", 1600, None),
    }

    best = _two_item_build(names, item_times, top_items=1, scale_weights={"weapon": 1.0})
    shown = _two_item_build(names, item_times, top_items=2, scale_weights={"weapon": 1.0})

    assert [item.item_id for item in best.items] == [20]
    assert [item.item_id for item in shown.items] == [10, 20]
    assert shown.items[0].purchase_s == 100


def test_upgrade_reaches_the_3200_bonus_instead_of_4000_souls():
    index = csr_matrix([[1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix([[1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    item_times = csr_matrix(
        ([10, 40, 80, 10, 40, 80], ([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])),
        shape=(2, 3),
        dtype=np.uint32,
    )
    names = {
        10: ("Extra Spirit", "weapon", 800, None),
        20: ("Improved Spirit", "weapon", 3200, None),
        30: ("Close Quarters", "weapon", 800, None),
    }

    build = suggest(
        5,
        [1],
        index,
        targets,
        np.array([5, 5]),
        [1, 7, 60],
        [10, 20, 30],
        names,
        item_times,
        top_k=2,
        top_items=2,
        upgrades=[[(20, 10)], [(20, 10)]],
        scale_weights={"weapon": 1.0},
    )

    assert [item.item_id for item in build.items] == [10, 20]
    assert build.items[1].upgrade_from == 10


def test_weapon_damage_raises_an_item_and_penalties_lower_it():
    item_times = csr_matrix(
        ([100, 100, 100, 100], ([0, 0, 1, 1], [0, 1, 0, 1])),
        shape=(2, 2),
        dtype=np.uint32,
    )
    names = {
        10: ("Close Quarters", "weapon", 1600, None),
        20: ("Weighted Shots", "weapon", 1600, None),
    }

    raised = _two_item_build(
        names,
        item_times,
        top_items=1,
        scale_weights={"weapon": 1.0},
        item_effects={20: {"weapon_damage": 0.30}},
    )
    lowered = _two_item_build(
        names,
        item_times,
        top_items=1,
        scale_weights={"weapon": 1.0},
        item_effects={20: {"weapon_damage": 0.30, "stamina": -0.14, "move_speed": -0.5}},
    )

    assert [item.item_id for item in raised.items] == [20]
    assert [item.item_id for item in lowered.items] == [10]


def test_bullet_resist_answers_weapon_and_spirit_resist_answers_spirit():
    item_times = csr_matrix(
        ([100, 100, 100, 100], ([0, 0, 1, 1], [0, 1, 0, 1])),
        shape=(2, 2),
        dtype=np.uint32,
    )
    names = {
        10: ("Close Quarters", "weapon", 800, None),
        20: ("Bullet Resilience", "vitality", 800, None),
    }

    def picked(tags, effects):
        return _two_item_build(
            names,
            item_times,
            top_items=1,
            hero_tags={1: tags},
            item_effects=effects,
        )

    assert [item.item_id for item in picked(("weapon",), {20: {"bullet_resist": 0.30}}).items] == [20]
    assert [item.item_id for item in picked(("spirit",), {20: {"bullet_resist": 0.30}}).items] == [10]
    assert [item.item_id for item in picked(("spirit",), {20: {"spirit_resist": 0.30}}).items] == [20]
    assert [item.item_id for item in picked(("weapon",), {20: {"spirit_resist": 0.30}}).items] == [10]
    defense = _two_item_build(
        names,
        item_times,
        top_items=1,
        hero_tags={1: ("weapon",)},
        item_answers={20: ("defense",)},
        item_effects={},
    )
    assert [item.item_id for item in defense.items] == [10]


def test_heal_reduction_still_answers_heal_when_effects_are_present():
    build = _heal_answer_build(
        hero_tags={1: ("heal",)},
        item_answers={20: ("antiheal",)},
        item_effects={},
    )

    assert [item.item_id for item in build.items] == [10, 20]


def test_an_item_neighbors_did_not_keep_stays_out_of_the_fifteen():
    index = csr_matrix([[1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix([[1, 0], [1, 0]], dtype=np.uint8)
    item_times = csr_matrix(([10, 10], ([0, 1], [0, 0])), shape=(2, 2), dtype=np.uint32)
    names = {
        10: ("Close Quarters", "weapon", 800, None),
        40: ("Weighted Shots", "weapon", 3200, None),
    }

    build = suggest(
        5,
        [1, 1],
        index,
        targets,
        np.array([5, 5]),
        [1, 7, 60],
        [10, 40],
        names,
        item_times,
        top_k=2,
        top_items=15,
        scale_weights={"weapon": 1.0},
        hero_tags={1: ("weapon",)},
        item_effects={40: {"weapon_damage": 0.30, "bullet_resist": 0.30}},
    )

    assert [item.item_id for item in build.items] == [10]


def _counter_build(enemies, hero_tags):
    index = csr_matrix([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float64)
    targets = csr_matrix(
        [[1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
        dtype=np.uint8,
    )
    item_times = csr_matrix(
        ([40, 40, 40, 40, 40], ([0, 0, 0, 1, 2], [0, 1, 2, 1, 1])),
        shape=(4, 3),
        dtype=np.uint32,
    )
    names = {
        10: ("Extra Spirit", "spirit", 1600, None),
        20: ("Close Quarters", "weapon", 1600, None),
        30: ("Extra Health", "vitality", 1600, None),
        40: ("Healbane", "spirit", 1600, None),
    }
    return suggest(
        5,
        enemies,
        index,
        targets,
        np.array([5, 5, 5, 5]),
        [1, 2, 60],
        [10, 20, 30],
        names,
        item_times,
        top_k=4,
        top_items=15,
        hero_tags=hero_tags,
        item_answers={40: ("antiheal",)},
        hero_names={1: "Dynamo", 2: "Ivy"},
    )


def test_two_healers_raise_an_antiheal_missing_from_the_fifteen():
    build = _counter_build([1, 2], {1: ("heal",), 2: ("heal",)})

    assert [item.item_id for item in build.items] == [10, 20, 30]
    assert build.counters[0].item_id == 40
    assert build.counters[0].reason == "against Dynamo, Ivy — heal"
    assert build.counters[0].replaces == "Extra Spirit"
    assert build.counters[0].already is False


def test_one_healer_does_not_raise_a_counter():
    build = _counter_build([1], {1: ("heal",), 2: ("heal",)})

    assert build.counters == ()


def test_a_counter_already_in_the_queue_is_not_replaced():
    index = csr_matrix([[1, 1, 0], [1, 1, 0]], dtype=np.float64)
    targets = csr_matrix([[1], [1]], dtype=np.uint8)
    item_times = csr_matrix(([10, 10], ([0, 1], [0, 0])), shape=(2, 1), dtype=np.uint32)
    names = {40: ("Healbane", "spirit", 1600, None)}

    build = suggest(
        5,
        [1, 2],
        index,
        targets,
        np.array([5, 5]),
        [1, 2, 60],
        [40],
        names,
        item_times,
        top_k=2,
        top_items=15,
        hero_tags={1: ("heal",), 2: ("heal",)},
        item_answers={40: ("antiheal",)},
        hero_names={1: "Dynamo", 2: "Ivy"},
    )

    assert build.counters[0].item_id == 40
    assert build.counters[0].already is True
    assert build.counters[0].replaces is None


def test_bullet_resist_counters_weapon_and_lumped_defense_does_not():
    index = csr_matrix([[1, 0, 0], [1, 0, 0]], dtype=np.float64)
    targets = csr_matrix([[1], [1]], dtype=np.uint8)
    item_times = csr_matrix(([10, 10], ([0, 1], [0, 0])), shape=(2, 1), dtype=np.uint32)
    names = {
        10: ("Extra Spirit", "spirit", 1600, None),
        40: ("Bullet Resilience", "vitality", 1600, None),
        50: ("Metal Skin", "vitality", 1600, None),
    }

    build = suggest(
        5,
        [1, 1],
        index,
        targets,
        np.array([5, 5]),
        [1, 7, 60],
        [10],
        names,
        item_times,
        top_k=2,
        top_items=15,
        hero_tags={1: ("weapon",)},
        item_answers={50: ("defense",)},
        item_effects={40: {"bullet_resist": 0.45}, 50: {}},
        hero_names={1: "Haze"},
    )

    assert [counter.item_id for counter in build.counters] == [40]
    assert build.counters[0].reason == "against Haze — weapon"
    assert 40 not in {item.item_id for item in build.items}


def test_chart_time_and_keep_share_ignore_threats():
    count = 20
    index = csr_matrix(
        ([1] * count, (list(range(count)), [0] * count)),
        shape=(count, 3),
        dtype=np.float64,
    )
    targets = csr_matrix((count, 1), dtype=np.uint8)
    item_times = csr_matrix((count, 1), dtype=np.uint32)
    names = {
        10: ("Close Quarters", "weapon", 800, None),
        99: ("Rare", "spirit", 800, None),
    }
    purchases = []
    for row in range(count):
        sold = 0 if row < 10 else 50
        bought = [(10, 120, sold)]
        if row == 0:
            bought.append((99, 30, 0))
        purchases.append(bought)
    shared = dict(
        hero_id=5,
        index=index,
        targets=targets,
        hero_ids=np.array([5] * count),
        hero_catalog=[1, 7, 60],
        item_catalog=[10],
        names=names,
        item_times=item_times,
        top_k=count,
        top_items=15,
        purchases=purchases,
    )
    plain = suggest(enemy_ids=[1], **shared)
    marked = suggest(
        enemy_ids=[1, 1],
        hero_tags={1: ("heal",)},
        item_answers={10: ("antiheal",)},
        **shared,
    )

    point = plain.points[0]
    assert point.item_id == 10
    assert point.purchase_s == 120
    assert point.keep_share == 0.5
    assert 99 not in {row.item_id for row in plain.points}
    assert marked.points[0].keep_share == point.keep_share
    assert marked.points[0].purchase_s == point.purchase_s
    assert marked.points[0].answers is True