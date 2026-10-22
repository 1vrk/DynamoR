def category_of(slot) -> str:
    if slot is None:
        return "other"
    text = str(slot).strip().lower()
    if text == "":
        return "other"
    if "weapon" in text:
        return "weapon"
    if "vitality" in text or "armor" in text:
        return "vitality"
    if "spirit" in text or "tech" in text:
        return "spirit"
    return "other"

TIER_COST = {1: 800, 2: 1600, 3: 3200, 4: 6400}


def cost_of(item) -> int | None:
    tier = item.get("item_tier")
    if tier is None:
        return None
    try:
        return TIER_COST.get(int(tier))
    except (TypeError, ValueError):
        return None


_SCALE_MIN = 0.2
_WEAPON_STATS = {
    "EBaseWeaponDamageIncrease",
    "EWeaponPower",
    "EBulletDamage",
    "EWeaponDamageScale",
    "ELightMeleeDamage",
    "EHeavyMeleeDamage",
}


def categories_of_scale(scale) -> set[str]:
    if not isinstance(scale, dict):
        return set()
    raw = scale.get("stat_scale")
    if raw is not None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = None
        if value == 0 or (value is not None and value < _SCALE_MIN):
            return set()
    stats = []
    if scale.get("specific_stat_scale_type"):
        stats.append(str(scale["specific_stat_scale_type"]))
    stats.extend(str(item) for item in (scale.get("scaling_stats") or []))
    class_name = str(scale.get("class_name") or "")
    found: set[str] = set()
    if class_name in {"scale_function_tech_damage", "scale_function_healing_spirit_scale"} or "ETechPower" in stats:
        found.add("spirit")
    if "weapon" in class_name or any(stat in _WEAPON_STATS for stat in stats):
        found.add("weapon")
    if "EHealingOutput" in stats or any("Health" in stat and stat != "ELevelUpBoons" for stat in stats):
        found.add("vitality")
    return found


def scale_magnitudes(scale) -> dict[str, float]:
    if not isinstance(scale, dict):
        return {}
    try:
        value = float(scale.get("stat_scale"))
    except (TypeError, ValueError):
        return {}
    if value < _SCALE_MIN:
        return {}
    return {category: value for category in categories_of_scale(scale)}


def hero_scale_weights(heroes, items) -> dict[int, dict[str, float]]:
    by_class = {}
    for item in items:
        class_name = item.get("class_name")
        if class_name and class_name not in by_class:
            by_class[str(class_name)] = item
    result: dict[int, dict[str, float]] = {}
    for hero in heroes:
        totals: dict[str, float] = {}
        slots = hero.get("items") or {}
        if not isinstance(slots, dict):
            continue
        for key in ("signature1", "signature2", "signature3", "signature4"):
            ability = by_class.get(str(slots.get(key) or ""))
            if not ability:
                continue
            props = ability.get("properties") or {}
            if not isinstance(props, dict):
                continue
            for prop in props.values():
                if not isinstance(prop, dict):
                    continue
                for category, value in scale_magnitudes(prop.get("scale_function")).items():
                    totals[category] = totals.get(category, 0.0) + value
        total = sum(totals.values())
        if total <= 0:
            result[int(hero["id"])] = {}
            continue
        result[int(hero["id"])] = {
            name: amount / total
            for name, amount in totals.items()
            if amount / total >= _SCALE_MIN
        }
    return result


def hero_close_categories(heroes, items) -> dict[int, tuple[str, ...]]:
    return {
        hero_id: tuple(sorted(weights, key=lambda name: (-weights[name], name)))
        for hero_id, weights in hero_scale_weights(heroes, items).items()
    }


def format_scale_weights(weights: dict[str, float]) -> str:
    ranked = sorted(weights, key=lambda name: (-weights[name], name))
    return ",".join(f"{name}:{weights[name]:.2f}" for name in ranked)


_STAT_ORDER = ("spirit", "weapon", "vitality", "speed")


def bucket_of(name: str) -> str | None:
    key = str(name).lower().replace("_", "")
    if "movespeed" in key or "sprintspeed" in key:
        return "speed"
    if "resist" in key or "lifesteal" in key:
        return "vitality"
    if any(token in key for token in ("health", "heal", "regen")):
        return "vitality"
    if "cooldown" in key:
        return "spirit"
    if "spiritpower" in key or "techpower" in key:
        return "spirit"
    if any(token in key for token in ("weapon", "firerate", "bulletdamage", "melee")):
        return "weapon"
    return None


def _add_header_name(names: list[str], seen: set[str], name) -> None:
    text = str(name)
    if text in seen:
        return
    seen.add(text)
    names.append(text)


def _header_property_names(item) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    innate = [
        section
        for section in item.get("tooltip_sections") or []
        if isinstance(section, dict) and str(section.get("section_type") or "") == "innate"
    ]
    if innate:
        for section in innate:
            for attribute in section.get("section_attributes") or []:
                if not isinstance(attribute, dict):
                    continue
                for field in ("elevated_properties", "properties"):
                    for name in attribute.get(field) or []:
                        _add_header_name(names, seen, name)
        return names
    props = item.get("properties") or {}
    if isinstance(props, dict) and any(
        isinstance(prop, dict) and "tooltip_is_elevated" in prop for prop in props.values()
    ):
        for key, prop in props.items():
            if isinstance(prop, dict) and prop.get("tooltip_is_elevated"):
                _add_header_name(names, seen, key)
        return names
    for section in item.get("tooltip_sections") or []:
        if not isinstance(section, dict):
            continue
        for attribute in section.get("section_attributes") or []:
            if not isinstance(attribute, dict):
                continue
            for name in attribute.get("elevated_properties") or []:
                _add_header_name(names, seen, name)
    return names


def header_buckets(item) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in _header_property_names(item):
        bucket = bucket_of(name)
        if bucket is None:
            continue
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def format_stats(counts: dict[str, int]) -> str:
    return ",".join(f"{name}:{counts[name]}" for name in _STAT_ORDER if counts.get(name))


def signature_class_names(heroes) -> set[str]:
    names: set[str] = set()
    for hero in heroes:
        slots = hero.get("items") or {}
        if not isinstance(slots, dict):
            continue
        for key in ("signature1", "signature2", "signature3", "signature4"):
            value = slots.get(key)
            if value:
                names.add(str(value))
    return names


def owner_of(item) -> int | None:
    heroes = item.get("heroes") or []
    if len(heroes) == 1:
        return int(heroes[0])
    return None


def image_of(item) -> str | None:
    raw = item.get("image_webp")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _clean_name(raw_name, item_id: int) -> str:
    name = "" if raw_name is None else str(raw_name).strip()
    if name == "":
        return str(item_id)
    return name

def hero_rows(payload, allowed_ids) -> list[dict]:
    allowed = {int(item_id) for item_id in allowed_ids}
    rows: dict[int, str] = {}
    for hero in payload:
        hero_id = int(hero["id"])
        if hero_id not in allowed or hero_id in rows:
            continue
        rows[hero_id] = _clean_name(hero.get("name"), hero_id)
    for hero_id in allowed:
        if hero_id not in rows:
            rows[hero_id] = str(hero_id)
    return [
        {"id": hero_id, "name": name}
        for hero_id, name in sorted(rows.items(), key=lambda item: (item[1], item[0]))
    ]

def item_rows(payload, allowed_ids, signature_names=None) -> list[dict]:
    allowed = {int(item_id) for item_id in allowed_ids}
    signatures = set(signature_names or [])
    rows: dict[int, tuple[str, str, int | None, str | None, int | None, str]] = {}
    for item in payload:
        item_id = int(item["id"])
        if item_id not in allowed or item_id in rows:
            continue
        class_name = str(item.get("class_name") or "")
        owner = None
        if class_name in signatures:
            category = "skill"
            owner = owner_of(item)
        elif item.get("type") == "upgrade":
            category = category_of(item.get("item_slot_type"))
        else:
            category = "other"
        rows[item_id] = (
            _clean_name(item.get("name"), item_id),
            category,
            cost_of(item),
            image_of(item),
            owner,
            format_stats(header_buckets(item)),
        )
    for item_id in allowed:
        if item_id not in rows:
            rows[item_id] = (str(item_id), "other", None, None, None, "")
    return [
        {
            "id": item_id,
            "name": name,
            "category": category,
            "cost": cost,
            "image": image,
            "owner": owner,
            "stats": stats,
        }
        for item_id, (name, category, cost, image, owner, stats) in sorted(rows.items())
    ]