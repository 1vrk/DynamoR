from pathlib import Path

import polars as pl


def _list_values(frame: pl.DataFrame, column: str) -> list[int]:
    values = frame.get_column(column).explode().drop_nulls().to_list()
    return [int(value) for value in values]


def hero_ids(frame: pl.DataFrame) -> list[int]:
    collected = [int(value) for value in frame.get_column("hero_id_target").drop_nulls().to_list()]
    collected.extend(_list_values(frame, "ally_heroes"))
    collected.extend(_list_values(frame, "enemy_heroes"))
    return sorted(set(collected))


def item_ids(frame: pl.DataFrame) -> list[int]:
    collected = set(_list_values(frame, "winning_items"))
    if "purchase_items" in frame.columns:
        collected.update(_list_values(frame, "purchase_items"))
    return sorted(collected)


def index_of(ids: list[int]) -> dict[int, int]:
    return {item_id: position for position, item_id in enumerate(ids)}


def write_ids(ids: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"id": ids}).with_columns(pl.col("id").cast(pl.UInt32)).write_parquet(path)


def read_ids(path: Path) -> list[int]:
    return [int(value) for value in pl.read_parquet(path).get_column("id").to_list()]
