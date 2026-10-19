from datetime import datetime, timedelta, timezone

import polars as pl

from dynamor.data.schema import (
    GAME_MODE,
    MATCH_MODE,
    MATCH_OUTCOME,
    NORMAL,
    RANKED,
    SPECTATOR,
    TEAM,
    TEAM_WIN,
)


def _labels(mapping: dict) -> dict[str, str]:
    return {str(key): value for key, value in mapping.items()}


def filter_matches(frame, settings, *, now: datetime | None = None):
    if now is None:
        now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=int(settings.lookback_hours))
    normalized = frame.with_columns(
        pl.col("match_mode").cast(pl.Utf8).replace(_labels(MATCH_MODE)),
        pl.col("game_mode").cast(pl.Utf8).replace(_labels(GAME_MODE)),
        pl.col("match_outcome").cast(pl.Utf8).replace(_labels(MATCH_OUTCOME)),
        pl.col("team").cast(pl.Utf8).replace(_labels(TEAM)),
    )
    return normalized.filter(
        (pl.col("match_mode") == RANKED)
        & (pl.col("game_mode") == NORMAL)
        & (pl.col("match_outcome") == TEAM_WIN)
        & pl.col("average_badge").is_not_null()
        & (pl.col("average_badge") >= settings.min_average_badge)
        & (pl.col("team") != SPECTATOR)
        & (pl.col("start_time") >= pl.lit(window_start).cast(pl.Datetime("ms", "UTC")))
        & (pl.col("start_time") <= pl.lit(now).cast(pl.Datetime("ms", "UTC")))
    )