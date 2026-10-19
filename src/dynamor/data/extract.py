from pathlib import Path

import polars as pl

from dynamor.data.schema import SOURCE_COLUMNS, OUTPUT_COLUMNS, TEAM0, TEAM1

def scan_raw(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        raise ValueError("no parquet files")
    return pl.scan_parquet([str(path) for path in paths]).select(SOURCE_COLUMNS)

def final_items(item_ids, sold_time_s) -> list[int]:
    if item_ids is None or sold_time_s is None:
        return []
    if len(item_ids) != len(sold_time_s):
        return []
    kept = []
    seen = set()
    for item_id, sold_at in zip(item_ids, sold_time_s):
        if sold_at != 0 or item_id in seen:
            continue
        seen.add(item_id)
        kept.append(int(item_id))
    return kept

def build_rows(frame, settings) -> pl.DataFrame:
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    team_size = int(settings.players_per_team)
    output = []
    for key, match in frame.group_by("match_id"):
        match_id = key[0]
        teams = {
            TEAM0: match.filter(pl.col("team") == TEAM0),
            TEAM1: match.filter(pl.col("team") == TEAM1),
        }
        if teams[TEAM0].height != team_size or teams[TEAM1].height != team_size:
            continue
        heroes = {
            name: side.get_column("hero_id").to_list()
            for name, side in teams.items()
        }
        for team_name, side in teams.items():
            enemies = heroes[TEAM1 if team_name == TEAM0 else TEAM0]
            winners = side.filter(pl.col("won"))
            for record in winners.iter_rows(named=True):
                items = final_items(record["items.item_id"], record["items.sold_time_s"])
                if not items:
                    continue
                hero_id = int(record["hero_id"])
                output.append(
                    {
                        "match_id": int(match_id),
                        "hero_id_target": hero_id,
                        "ally_heroes": sorted(hero for hero in heroes[team_name] if hero != hero_id),
                        "enemy_heroes": sorted(int(hero) for hero in enemies),
                        "winning_items": items,
                        "average_badge": int(record["average_badge"]),
                        "start_time": record["start_time"],
                    }
                )
    if not output:
        return pl.DataFrame(
            schema={
                "match_id": pl.Int64,
                "hero_id_target": pl.Int32,
                "ally_heroes": pl.List(pl.Int32),
                "enemy_heroes": pl.List(pl.Int32),
                "winning_items": pl.List(pl.UInt32),
                "average_badge": pl.Int32,
                "start_time": pl.Datetime(time_zone="UTC"),
            }
        )
    return (
        pl.DataFrame(output)
        .select(OUTPUT_COLUMNS)
        .with_columns(
            pl.col("match_id").cast(pl.Int64),
            pl.col("hero_id_target").cast(pl.Int32),
            pl.col("ally_heroes").cast(pl.List(pl.Int32)),
            pl.col("enemy_heroes").cast(pl.List(pl.Int32)),
            pl.col("winning_items").cast(pl.List(pl.UInt32)),
            pl.col("average_badge").cast(pl.Int32),
        )
    )

def write_interim(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)