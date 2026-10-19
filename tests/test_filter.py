from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import polars as pl

from dynamor.data.filter import filter_matches

NOW = datetime(2026, 10, 19, 18, 0, tzinfo=timezone.utc)
RECENT = NOW - timedelta(hours=1)


def row(hero_id, **overrides):
    data = {
        "hero_id": hero_id,
        "won": False,
        "match_mode": "Ranked",
        "game_mode": "Normal",
        "match_outcome": "TeamWin",
        "average_badge": 111,
        "team": "Team1",
        "start_time": RECENT,
    }
    data.update(overrides)
    return data


def test_filter_keeps_loser_in_a_qualifying_match():
    frame = pl.DataFrame(
        [
            row(1),
            row(2, match_mode="Unranked"),
            row(3, average_badge=50),
            row(4, match_outcome="Draw"),
            row(5, team="Spectator"),
            row(6, won=True, team="Team0"),
            row(7, match_mode="4"),
        ],
        schema_overrides={"start_time": pl.Datetime("us", "UTC")},
    )
    settings = SimpleNamespace(lookback_hours=6, min_average_badge=91)

    kept = filter_matches(frame, settings, now=NOW)

    assert kept.get_column("hero_id").sort().to_list() == [1, 6, 7]