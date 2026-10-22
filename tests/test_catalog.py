import polars as pl

from dynamor.features.catalog import hero_ids, index_of, item_ids, read_ids, write_ids
from dynamor.catalog.assets import hero_close_categories, hero_rows, item_rows


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


def test_item_ids_include_purchases_that_were_sold():
    frame = pl.DataFrame(
        {
            "winning_items": [[10]],
            "purchase_items": [[10, 99]],
        }
    )

    assert item_ids(frame) == [10, 99]

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
        {"id": 10, "name": "Close Quarters", "type": "upgrade", "item_slot_type": "weapon", "item_tier": 1, "image_webp": "https://example/cq.webp"},
        {"id": 20, "name": "Extra Health", "type": "upgrade", "item_slot_type": "EItemSlotType_Armor", "item_tier": 2},
        {"id": 30, "name": "Extra Spirit", "type": "upgrade", "item_slot_type": "spirit", "item_tier": 3},
        {"id": 40, "name": "Dash", "type": "ability"},
        {"id": 50, "name": "Relic", "type": "upgrade", "item_slot_type": "weapon", "item_tier": 4},
    ]
    rows = item_rows(payload, {10, 20, 30, 40, 50})
    assert [(row["name"], row["category"]) for row in rows] == [
        ("Close Quarters", "weapon"),
        ("Extra Health", "vitality"),
        ("Extra Spirit", "spirit"),
        ("Dash", "other"),
        ("Relic", "weapon"),
    ]
    assert [row["cost"] for row in rows] == [800, 1600, 3200, None, 6400]
    assert rows[0]["image"] == "https://example/cq.webp"
    assert rows[3]["image"] is None


def test_item_rows_marks_signature_abilities_as_skills():
    payload = [
        {
            "id": 7,
            "name": "Napalm",
            "type": "ability",
            "class_name": "ability_napalm",
            "heroes": [1],
            "image_webp": "https://example/n.webp",
        }
    ]

    rows = item_rows(payload, {7}, {"ability_napalm"})

    assert rows[0]["category"] == "skill"
    assert rows[0]["owner"] == 1
    assert rows[0]["image"] == "https://example/n.webp"


def test_hero_close_categories_skip_a_weak_spirit_scale():
    heroes = [{"id": 84, "items": {"signature1": "ability_gun", "signature2": "ability_slow"}}]
    items = [
        {
            "class_name": "ability_gun",
            "properties": {
                "Damage": {
                    "scale_function": {
                        "class_name": "scale_function_ability_weapon_damage",
                        "specific_stat_scale_type": "EBaseWeaponDamageIncrease",
                        "stat_scale": 1.2,
                    }
                }
            },
        },
        {
            "class_name": "ability_slow",
            "properties": {
                "Slow": {"scale_function": {"class_name": "scale_function_tech_damage", "stat_scale": 0.08}}
            },
        },
    ]

    assert hero_close_categories(heroes, items)[84] == ("weapon",)