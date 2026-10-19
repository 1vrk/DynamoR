from datetime import datetime, timezone
from types import SimpleNamespace

import polars as pl

from dynamor.data.extract import build_rows
from dynamor.data.schema import TEAM0, TEAM1

WHEN = datetime(2026, 10, 19, 17, 0, tzinfo=timezone.utc)


def player(match_id, hero_id, team, won, items, sold):
    return {
        "match_id": match_id,
        "hero_id": hero_id,
        "team": team,
        "won": won,
        "average_badge": 111,
        "start_time": WHEN,
        "items.item_id": items,
        "items.sold_time_s": sold,
    }


def test_build_rows_keeps_full_winning_team_only():
    rows = []
    for hero_id in range(1, 7):
        items = [10, 20, 10, 30] if hero_id == 1 else [1]
        sold = [0, 7, 0, 0] if hero_id == 1 else [0]
        rows.append(player(100, hero_id, TEAM0, True, items, sold))
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
    assert match.filter(pl.col("hero_id_target") == 1).get_column("winning_items").to_list() == [[10, 30]]
    for record in match.iter_rows(named=True):
        assert len(record["ally_heroes"]) == 5
        assert len(record["enemy_heroes"]) == 6
        assert record["hero_id_target"] not in record["ally_heroes"]
        assert record["hero_id_target"] not in record["enemy_heroes"]