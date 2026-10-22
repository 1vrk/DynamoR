import polars as pl

from dynamor.features.catalog import hero_ids, index_of, item_ids, read_ids, write_ids
from dynamor.catalog.assets import (
    format_effects,
    format_scale_weights,
    format_stats,
    header_buckets,
    hero_close_categories,
    hero_rows,
    hero_scale_weights,
    hero_threat_tags,
    item_effects,
    item_rows,
    item_tag_sets,
    property_marks,
)


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


def _spirit_scale(value: float) -> dict:
    return {"scale_function": {"class_name": "scale_function_tech_damage", "stat_scale": value}}


def _weapon_scale(value: float) -> dict:
    return {
        "scale_function": {
            "class_name": "scale_function_ability_weapon_damage",
            "specific_stat_scale_type": "EBaseWeaponDamageIncrease",
            "stat_scale": value,
        }
    }


def test_five_weak_spirit_scales_stay_below_the_weapon_share():
    heroes = [{"id": 11, "items": {"signature1": "ability_mix"}}]
    items = [
        {
            "class_name": "ability_mix",
            "properties": {
                **{f"Spirit{index}": _spirit_scale(0.05) for index in range(5)},
                "Shot": _weapon_scale(1.2),
            },
        }
    ]

    assert hero_close_categories(heroes, items)[11] == ("weapon",)
    assert hero_scale_weights(heroes, items)[11] == {"weapon": 1.0}


def test_scale_weights_sum_magnitudes_instead_of_counting_properties():
    heroes = [{"id": 12, "items": {"signature1": "ability_mix"}}]
    items = [
        {
            "class_name": "ability_mix",
            "properties": {
                **{f"Spirit{index}": _spirit_scale(0.2) for index in range(5)},
                "Shot": _weapon_scale(6.0),
            },
        }
    ]

    weights = hero_scale_weights(heroes, items)[12]

    assert hero_close_categories(heroes, items)[12] == ("weapon",)
    assert set(weights) == {"weapon"}
    assert format_scale_weights(weights) == "weapon:0.86"


def test_header_buckets_count_bullet_resilience_and_skip_the_passive():
    item = {
        "tooltip_sections": [
            {
                "section_type": "innate",
                "section_attributes": [
                    {
                        "elevated_properties": ["BulletResist", "TechPowerPercent"],
                        "properties": [
                            "OutOfCombatHealthRegen",
                            "TechPower",
                            "BonusHealth",
                            "SpiritResist",
                            "AbilityCooldown",
                        ],
                    }
                ],
            },
            {
                "section_type": "passive",
                "section_attributes": [
                    {"important_properties": ["BulletResistBelowThreshold", "HealthThreshold"]}
                ],
            },
        ]
    }

    assert header_buckets(item) == {"vitality": 4, "spirit": 3}
    assert format_stats(header_buckets(item)) == "spirit:3,vitality:4"


def test_header_buckets_fall_back_to_elevated_flags_without_sections():
    item = {
        "properties": {
            "BulletResist": {"tooltip_is_elevated": True},
            "BonusMoveSpeed": {"tooltip_is_elevated": True},
            "BulletResistBelowThreshold": {"tooltip_is_elevated": False},
        }
    }

    assert header_buckets(item) == {"vitality": 1, "speed": 1}
    assert format_stats(header_buckets(item)) == "vitality:1,speed:1"


def test_item_rows_store_header_stats():
    payload = [
        {
            "id": 10,
            "name": "Bullet Resilience",
            "type": "upgrade",
            "item_slot_type": "vitality",
            "item_tier": 3,
            "tooltip_sections": [
                {
                    "section_type": "innate",
                    "section_attributes": [
                        {
                            "elevated_properties": ["BulletResist"],
                            "properties": ["OutOfCombatHealthRegen"],
                        }
                    ],
                },
                {
                    "section_type": "passive",
                    "section_attributes": [
                        {"important_properties": ["BulletResistBelowThreshold"]}
                    ],
                },
            ],
        }
    ]

    rows = item_rows(payload, {10})

    assert rows[0]["stats"] == "vitality:2"


def test_passive_slow_is_control_and_bullet_resist_is_defense():
    item = {
        "tooltip_sections": [
            {"section_type": "innate", "section_attributes": [{"properties": ["BulletResist"]}]},
            {
                "section_type": "passive",
                "section_attributes": [{"properties": ["SlowPercent", "BulletArmorReduction"]}],
            },
        ]
    }

    tags, answers = item_tag_sets(item)

    assert tags == {"defense", "control"}
    assert answers == {"defense"}
    assert property_marks("StunDuration") == ({"control"}, set())
    assert property_marks("BonusHealth") == (set(), set())
    assert property_marks("OutOfCombatHealthRegen") == (set(), set())


def test_heal_reduction_answers_heal_and_healing_does_not():
    item = {
        "tooltip_sections": [
            {
                "section_type": "passive",
                "section_attributes": [{"properties": ["HealAmpReceivePenaltyPercent", "HealOnKill"]}],
            }
        ]
    }

    tags, answers = item_tag_sets(item)

    assert tags == {"heal"}
    assert answers == {"antiheal"}


def test_ability_cooldown_and_range_are_not_spirit():
    heroes = [{"id": 5, "items": {"signature1": "dagger"}}]
    items = [
        {
            "class_name": "dagger",
            "properties": {
                "AbilityCooldown": {
                    "scale_function": {
                        "class_name": "scale_function_single_stat",
                        "specific_stat_scale_type": "ETechCooldown",
                    }
                },
                "AbilityCastRange": {
                    "scale_function": {
                        "class_name": "scale_function_single_stat",
                        "specific_stat_scale_type": "ETechRange",
                    }
                },
                "TechPower": {"value": 0},
                "WeaponPower": {"value": 0},
                "SlowPercent": {"value": 0},
                "BonusHealth": {"value": 100},
                "Damage": {
                    "value": 65,
                    "scale_function": {
                        "class_name": "scale_function_tech_damage",
                        "specific_stat_scale_type": "ETechPower",
                        "stat_scale": 2.2,
                    }
                },
            },
        }
    ]

    assert hero_threat_tags(heroes, items)[5] == {"spirit"}


def _stat(value, postfix="", prefix="{s:sign}"):
    return {"value": value, "prefix": prefix, "postfix": postfix}


def _section(section_type, elevated=(), properties=(), important=()):
    return {
        "section_type": section_type,
        "section_attributes": [
            {
                "elevated_properties": list(elevated),
                "properties": list(properties),
                "important_properties": list(important),
            }
        ],
    }


def test_effects_keep_the_sign_and_ignore_unlisted_properties():
    item = {
        "tooltip_sections": [
            _section(
                "innate",
                ["BaseAttackDamagePercent"],
                ["StatusResistancePercent", "StaminaCooldownReduction", "BonusMoveSpeed"],
            ),
            _section(
                "passive",
                properties=["GroundDashReductionPercent", "SlowDuration", "BuildUpPerShot"],
                important=["SlowPercent"],
            ),
        ],
        "properties": {
            "BaseAttackDamagePercent": _stat("30", "%"),
            "StatusResistancePercent": _stat("22", "%"),
            "StaminaCooldownReduction": _stat("-14", "%"),
            "BonusMoveSpeed": _stat("-0.5m", " m"),
            "GroundDashReductionPercent": _stat("-20", "%", ""),
            "SlowDuration": _stat("3.5", "s", ""),
            "BuildUpPerShot": _stat("0.7", "%", ""),
            "SlowPercent": _stat("24", "%", "-"),
            "TechPower": _stat("90"),
        },
    }

    effects = item_effects(item)

    assert effects == {
        "weapon_damage": 0.3,
        "debuff_resist": 0.22,
        "stamina": -0.14,
        "move_speed": -0.5,
    }
    assert format_effects(effects) == "weapon_damage:0.3,move_speed:-0.5,stamina:-0.14,debuff_resist:0.22"


def test_flat_spirit_and_health_use_their_own_divisors():
    item = {
        "tooltip_sections": [
            _section(
                "innate",
                ["TechPowerPercent"],
                ["TechPower", "BonusHealth", "OutOfCombatHealthRegen"],
            )
        ],
        "properties": {
            "TechPowerPercent": _stat("15", "%"),
            "TechPower": _stat("30"),
            "BonusHealth": _stat("75"),
            "OutOfCombatHealthRegen": _stat("4"),
            "HealOnKill": _stat("275"),
        },
    }

    assert item_effects(item) == {"spirit": 1.15, "health": 0.75}


def test_resist_shred_is_not_a_resist_effect():
    item = {
        "tooltip_sections": [
            _section("innate", properties=["BonusHealth", "BonusSprintSpeed"]),
            _section("passive", important=["BulletArmorReduction", "FireRateSlow"]),
        ],
        "properties": {
            "BonusHealth": _stat("100"),
            "BonusSprintSpeed": _stat("0.75m", "m"),
            "BulletArmorReduction": _stat("-10", "%"),
            "FireRateSlow": _stat("15", "%", "-"),
        },
    }

    assert item_effects(item) == {"health": 1.0, "move_speed": 0.75}


def test_bullet_and_spirit_resist_include_every_listed_line():
    bullet = {
        "tooltip_sections": [
            _section("innate", ["BulletResist"], ["OutOfCombatHealthRegen"]),
            _section("passive", important=["BulletResistBelowThreshold", "HealthThreshold"]),
        ],
        "properties": {
            "BulletResist": _stat("30", "%"),
            "OutOfCombatHealthRegen": _stat("3"),
            "HealthThreshold": {"value": "50"},
            "BulletResistBelowThreshold": _stat("15", "%"),
        },
    }
    spirit = {
        "tooltip_sections": [
            _section("innate", ["TechResist"]),
            _section("passive", important=["TechResistBelowThreshold"]),
        ],
        "properties": {
            "TechResist": _stat("30", "%"),
            "TechResistBelowThreshold": _stat("15", "%"),
        },
    }

    assert item_effects(bullet) == {"bullet_resist": 0.45}
    assert item_effects(spirit) == {"spirit_resist": 0.45}


def test_fire_rate_melee_and_lifesteal_are_signed_percents():
    item = {
        "tooltip_sections": [
            _section("innate", properties=["FireRate", "MeleeDamage", "BulletLifesteal"])
        ],
        "properties": {
            "FireRate": _stat("12", "%"),
            "MeleeDamage": _stat("20", "%"),
            "BulletLifesteal": _stat("8", "%"),
            "WeaponPower": _stat("99", "%"),
        },
    }

    assert item_effects(item) == {"fire_rate": 0.12, "melee": 0.2, "lifesteal": 0.08}


def test_item_rows_store_signed_effects():
    payload = [
        {
            "id": 10,
            "name": "Weighted Shots",
            "type": "upgrade",
            "item_slot_type": "weapon",
            "item_tier": 3,
            "tooltip_sections": [
                _section("innate", ["BaseAttackDamagePercent"], ["BonusMoveSpeed"]),
            ],
            "properties": {
                "BaseAttackDamagePercent": _stat("30", "%"),
                "BonusMoveSpeed": _stat("-0.5m", " m"),
            },
        }
    ]

    rows = item_rows(payload, {10, 11})

    assert rows[0]["effects"] == "weapon_damage:0.3,move_speed:-0.5"
    assert rows[1]["effects"] == ""