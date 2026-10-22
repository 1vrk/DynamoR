from datetime import datetime, timezone
from types import SimpleNamespace

import polars as pl

from dynamor.data.extract import build_rows, purchase_log, upgrade_pairs
from dynamor.data.schema import TEAM0, TEAM1

WHEN = datetime(2026, 10, 19, 17, 0, tzinfo=timezone.utc)


def player(match_id, hero_id, team, won, items, sold, bought=None):
    return {
        "match_id": match_id,
        "hero_id": hero_id,
        "team": team,
        "won": won,
        "average_badge": 111,
        "start_time": WHEN,
        "items.item_id": items,
        "items.sold_time_s": sold,
        "items.game_time_s": [0] * len(items) if bought is None else bought,
    }


def test_purchase_log_keeps_a_sold_item():
    items, times, sold = purchase_log([10, 30], [20, 0], [5, 20])

    assert items == [10, 30]
    assert times == [5, 20]
    assert sold == [20, 0]


def test_upgrade_pairs_link_one_item_bought_at_the_sale():
    children, parents = upgrade_pairs([10, 30], [20, 0], [5, 20], [1, 0])

    assert children == [30]
    assert parents == [10]


def test_upgrade_pairs_skip_two_purchases_at_the_same_second():
    children, parents = upgrade_pairs(
        [10, 20, 30, 40],
        [15, 15, 0, 0],
        [5, 6, 15, 15],
        [1, 1, 0, 0],
    )

    assert children == []
    assert parents == []


def test_upgrade_pairs_keep_the_last_step_of_a_chain():
    children, parents = upgrade_pairs([1, 2, 3], [10, 20, 0], [1, 10, 20], [1, 1, 0])

    assert children == [3]
    assert parents == [2]


def test_build_rows_keeps_full_winning_team_only():
    rows = []
    for hero_id in range(1, 7):
        items = [10, 20, 10, 30] if hero_id == 1 else [1]
        sold = [0, 7, 0, 0] if hero_id == 1 else [0]
        bought = [10, 20, 30, 40] if hero_id == 1 else None
        rows.append(player(100, hero_id, TEAM0, True, items, sold, bought))
    for hero_id in range(7, 13):
        rows.append(player(100, hero_id, TEAM1, False, [99], [0]))
    for hero_id in range(101, 106):
        rows.append(player(200, hero_id, TEAM0, True, [1], [0]))
    for hero_id in range(201, 207):
        rows.append(player(200, hero_id, TEAM1, False, [1], [0]))

    result = build_rows(pl.DataFrame(rows), SimpleNamespace(players_per_team=6))
    match = result.filter(pl.col("match_id") == 100)

    assert match.height == 6
    assert result.filter(pl.col("match_id") == 200).height == 0
    hero = match.filter(pl.col("hero_id_target") == 1)
    assert hero.get_column("winning_items").to_list() == [[10, 30]]
    assert hero.get_column("winning_item_times").to_list() == [[10, 40]]
    for record in match.iter_rows(named=True):
        assert len(record["ally_heroes"]) == 5
        assert len(record["enemy_heroes"]) == 6
        assert record["hero_id_target"] not in record["ally_heroes"]
        assert record["hero_id_target"] not in record["enemy_heroes"]