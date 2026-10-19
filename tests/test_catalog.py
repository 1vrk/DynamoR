import polars as pl

from dynamor.features.catalog import hero_ids, index_of, item_ids, read_ids, write_ids
from dynamor.catalog.assets import hero_rows, item_rows


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

def test_hero_rows_keeps_the_first_name_and_fills_missing_ids():
    payload = [
        {"id": 1, "name": "Infernus"},
        {"id": 2, "name": "Ignored"},
        {"id": 1, "name": "Second"},
    ]
    rows = hero_rows(payload, {1, 3})
    assert [(row["id"], row["name"]) for row in rows] == [(3, "3"), (1, "Infernus")]

def test_item_rows_maps_shop_slots_and_leaves_abilities_as_other():
    payload = [
        {"id": 10, "name": "Close Quarters", "type": "upgrade", "item_slot_type": "weapon"},
        {"id": 20, "name": "Extra Health", "type": "upgrade", "item_slot_type": "EItemSlotType_Armor"},
        {"id": 30, "name": "Extra Spirit", "type": "upgrade", "item_slot_type": "spirit"},
        {"id": 40, "name": "Dash", "type": "ability"},
    ]
    rows = item_rows(payload, {10, 20, 30, 40})
    assert [(row["name"], row["category"]) for row in rows] == [
        ("Close Quarters", "weapon"),
        ("Extra Health", "vitality"),
        ("Extra Spirit", "spirit"),
        ("Dash", "other"),
    ]