import polars as pl

from dynamor.features.catalog import hero_ids, index_of, item_ids, read_ids, write_ids


def test_catalog_sorts_heroes_and_keeps_large_item_ids(tmp_path):
    frame = pl.DataFrame(
        {
            "hero_id_target": [60, 7],
            "ally_heroes": [[1], [1]],
            "enemy_heroes": [[60], [7]],
            "winning_items": [[3190916303], [10, 3190916303]],
        }
    )

    heroes = hero_ids(frame)
    items = item_ids(frame)
    path = tmp_path / "item_ids.parquet"
    write_ids(items, path)

    assert heroes == [1, 7, 60]
    assert items == [10, 3190916303]
    assert read_ids(path) == [10, 3190916303]
    assert index_of(items)[3190916303] == 1
