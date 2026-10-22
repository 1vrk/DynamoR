from pathlib import Path

import polars as pl

from dynamor.data.schema import SOURCE_COLUMNS, OUTPUT_COLUMNS, TEAM0, TEAM1

def scan_raw(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        raise ValueError("no parquet files")
    return pl.scan_parquet([str(path) for path in paths]).select(SOURCE_COLUMNS)

def upgrade_pairs(item_ids, sold_time_s, game_time_s, flags) -> tuple[list[int], list[int]]:
    if item_ids is None or sold_time_s is None or game_time_s is None or flags is None:
        return [], []
    if not (len(item_ids) == len(sold_time_s) == len(game_time_s) == len(flags)):
        return [], []
    bought_at: dict[int, list[int]] = {}
    for index, bought in enumerate(game_time_s):
        bought_at.setdefault(int(bought), []).append(index)
    sold_at_indexes: dict[int, list[int]] = {}
    for index, sold_at in enumerate(sold_time_s):
        sold_at = int(sold_at)
        if sold_at == 0 or int(flags[index]) != 1:
            continue
        sold_at_indexes.setdefault(sold_at, []).append(index)
    children: list[int] = []
    parents: list[int] = []
    kept = {int(item_id) for item_id, sold_at in zip(item_ids, sold_time_s) if int(sold_at) == 0}
    for sold_at, parent_indexes in sold_at_indexes.items():
        child_indexes = [
            index for index in bought_at.get(sold_at, []) if index not in parent_indexes
        ]
        if len(parent_indexes) != 1 or len(child_indexes) != 1:
            continue
        child = int(item_ids[child_indexes[0]])
        if child not in kept:
            continue
        children.append(child)
        parents.append(int(item_ids[parent_indexes[0]]))
    return children, parents


def purchase_log(item_ids, sold_time_s, game_time_s) -> tuple[list[int], list[int], list[int]]:
    if item_ids is None or sold_time_s is None or game_time_s is None:
        return [], [], []
    if not (len(item_ids) == len(sold_time_s) == len(game_time_s)):
        return [], [], []
    bought_ids: list[int] = []
    bought_times: list[int] = []
    sold_times: list[int] = []
    seen: set[int] = set()
    for item_id, sold_at, bought_at in zip(item_ids, sold_time_s, game_time_s):
        item_id = int(item_id)
        if item_id in seen:
            continue
        seen.add(item_id)
        bought_ids.append(item_id)
        bought_times.append(int(bought_at))
        sold_times.append(int(sold_at))
    return bought_ids, bought_times, sold_times


def final_items(item_ids, sold_time_s, game_time_s) -> tuple[list[int], list[int]]:
    if item_ids is None or sold_time_s is None or game_time_s is None:
        return [], []
    if not (len(item_ids) == len(sold_time_s) == len(game_time_s)):
        return [], []
    kept_ids: list[int] = []
    kept_times: list[int] = []
    seen: set[int] = set()
    for item_id, sold_at, bought_at in zip(item_ids, sold_time_s, game_time_s):
        if sold_at != 0 or item_id in seen:
            continue
        seen.add(int(item_id))
        kept_ids.append(int(item_id))
        kept_times.append(int(bought_at))
    return kept_ids, kept_times

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
                items, times = final_items(
                    record["items.item_id"],
                    record["items.sold_time_s"],
                    record["items.game_time_s"],
                )
                if not items:
                    continue
                bought_ids, bought_times, sold_times = purchase_log(
                    record["items.item_id"],
                    record["items.sold_time_s"],
                    record["items.game_time_s"],
                )
                children, parents = upgrade_pairs(
                    record["items.item_id"],
                    record["items.sold_time_s"],
                    record["items.game_time_s"],
                    record.get("items.flags"),
                )
                hero_id = int(record["hero_id"])
                output.append(
                    {
                        "match_id": int(match_id),
                        "hero_id_target": hero_id,
                        "ally_heroes": sorted(hero for hero in heroes[team_name] if hero != hero_id),
                        "enemy_heroes": sorted(int(hero) for hero in enemies),
                        "winning_items": items,
                        "winning_item_times": times,
                        "upgrade_child": children,
                        "upgrade_parent": parents,
                        "purchase_items": bought_ids,
                        "purchase_times": bought_times,
                        "purchase_sold": sold_times,
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
                "winning_item_times": pl.List(pl.UInt32),
                "upgrade_child": pl.List(pl.UInt32),
                "upgrade_parent": pl.List(pl.UInt32),
                "purchase_items": pl.List(pl.UInt32),
                "purchase_times": pl.List(pl.UInt32),
                "purchase_sold": pl.List(pl.UInt32),
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
            pl.col("winning_item_times").cast(pl.List(pl.UInt32)),
            pl.col("upgrade_child").cast(pl.List(pl.UInt32)),
            pl.col("upgrade_parent").cast(pl.List(pl.UInt32)),
            pl.col("purchase_items").cast(pl.List(pl.UInt32)),
            pl.col("purchase_times").cast(pl.List(pl.UInt32)),
            pl.col("purchase_sold").cast(pl.List(pl.UInt32)),
            pl.col("average_badge").cast(pl.Int32),
        )
    )

def chart_rows(frame, settings) -> pl.DataFrame:
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
        for side in teams.values():
            for record in side.iter_rows(named=True):
                bought_ids, bought_times, sold_times = purchase_log(
                    record["items.item_id"],
                    record["items.sold_time_s"],
                    record["items.game_time_s"],
                )
                output.append(
                    {
                        "match_id": int(match_id),
                        "hero_id": int(record["hero_id"]),
                        "won": bool(record["won"]),
                        "start_time": record["start_time"],
                        "purchase_items": bought_ids,
                        "purchase_times": bought_times,
                        "purchase_sold": sold_times,
                    }
                )
    schema = {
        "match_id": pl.Int64,
        "hero_id": pl.Int32,
        "won": pl.Boolean,
        "start_time": pl.Datetime(time_zone="UTC"),
        "purchase_items": pl.List(pl.UInt32),
        "purchase_times": pl.List(pl.UInt32),
        "purchase_sold": pl.List(pl.UInt32),
    }
    if not output:
        return pl.DataFrame(schema=schema)
    return (
        pl.DataFrame(output)
        .unique(subset=["match_id", "hero_id"], keep="first")
        .select(list(schema))
        .with_columns(
            pl.col("match_id").cast(pl.Int64),
            pl.col("hero_id").cast(pl.Int32),
            pl.col("won").cast(pl.Boolean),
            pl.col("purchase_items").cast(pl.List(pl.UInt32)),
            pl.col("purchase_times").cast(pl.List(pl.UInt32)),
            pl.col("purchase_sold").cast(pl.List(pl.UInt32)),
        )
    )


def write_interim(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)