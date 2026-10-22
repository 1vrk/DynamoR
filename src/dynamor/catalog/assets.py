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


_BULLET_NAME_TOKENS = ("weapondamage", "bulletdamage", "attackdamage", "weaponpower")
_MELEE_STATS = {"ELightMeleeDamage", "EHeavyMeleeDamage"}


def _scale_stat_names(scale) -> list[str]:
    if not isinstance(scale, dict):
        return []
    stats = []
    if scale.get("specific_stat_scale_type"):
        stats.append(str(scale["specific_stat_scale_type"]))
    stats.extend(str(item) for item in (scale.get("scaling_stats") or []))
    return stats


def _is_melee(name: str, scale) -> bool:
    if "melee" in _compact_name(name):
        return True
    if not isinstance(scale, dict):
        return False
    if "melee" in str(scale.get("class_name") or "").lower():
        return True
    return any(stat in _MELEE_STATS for stat in _scale_stat_names(scale))


def _is_live_bullet_damage(name: str, prop) -> bool:
    if not isinstance(prop, dict) or not _property_is_live(prop):
        return False
    scale = prop.get("scale_function")
    if _is_melee(name, scale):
        return False
    key = _compact_name(name)
    if "firerate" in key or any(token in key for token in ("debuff", "reduction", "slow", "penalty", "shred")):
        return False
    if "bullet_damage" in str(prop.get("css_class") or "").lower():
        return True
    if "per" in key.replace("percent", ""):
        return False
    if any(token in key for token in _BULLET_NAME_TOKENS):
        return True
    if not isinstance(scale, dict):
        return False
    try:
        magnitude = float(scale.get("stat_scale"))
    except (TypeError, ValueError):
        return False
    return magnitude >= _SCALE_MIN and "weapon" in categories_of_scale(scale)


def _raise_weapon_bar(totals: dict[str, float]) -> None:
    others = sum(amount for category, amount in totals.items() if category != "weapon")
    floor = 1.0 if others <= 0 else others * 1.5
    totals["weapon"] = max(totals.get("weapon", 0.0), floor)


def hero_scale_weights(heroes, items) -> dict[int, dict[str, float]]:
    by_class = {}
    for item in items:
        class_name = item.get("class_name")
        if class_name and class_name not in by_class:
            by_class[str(class_name)] = item
    result: dict[int, dict[str, float]] = {}
    for hero in heroes:
        totals: dict[str, float] = {}
        bullet = False
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
            for name, prop in props.items():
                if not isinstance(prop, dict):
                    continue
                for category, value in scale_magnitudes(prop.get("scale_function")).items():
                    totals[category] = totals.get(category, 0.0) + value
                bullet = bullet or _is_live_bullet_damage(str(name), prop)
        if bullet:
            _raise_weapon_bar(totals)
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


_TAG_ORDER = ("weapon", "spirit", "heal", "control", "defense")
_ANSWER_ORDER = ("defense", "antiheal")
_HERO_THREATS = frozenset({"weapon", "spirit", "heal", "control"})


def format_tags(tags) -> str:
    owned = set(tags)
    return ",".join(name for name in _TAG_ORDER if name in owned)


def format_answers(answers) -> str:
    owned = set(answers)
    return ",".join(name for name in _ANSWER_ORDER if name in owned)


def _compact_name(name: str) -> str:
    return str(name).lower().replace("_", "")


def _scale_tag(scale) -> str | None:
    if not isinstance(scale, dict):
        return None
    class_name = str(scale.get("class_name") or "")
    stats = []
    if scale.get("specific_stat_scale_type"):
        stats.append(str(scale["specific_stat_scale_type"]))
    stats.extend(str(item) for item in (scale.get("scaling_stats") or []))
    if any(stat in {"ETechCooldown", "ETechDuration", "ETechRange", "ETechRadius"} for stat in stats):
        return None
    if "tech_damage" in class_name:
        return "spirit"
    if "weapon" in class_name or any(stat in _WEAPON_STATS for stat in stats):
        return "weapon"
    if "healing" in class_name or "EHealingOutput" in stats:
        return "heal"
    return None


def _mentions_heal(key: str) -> bool:
    if "lifesteal" in key or "healing" in key:
        return True
    return "heal" in key.replace("health", "")


def _property_is_live(prop) -> bool:
    if not isinstance(prop, dict):
        return False
    value = prop.get("value")
    if isinstance(value, str):
        text = value.strip().lower()
        if text.endswith("m") or text.endswith("s"):
            text = text[:-1]
        try:
            value = float(text)
        except ValueError:
            return text not in ("", "0", "0.0")
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None
    if number is not None and number != 0:
        return True
    scale = prop.get("scale_function")
    if isinstance(scale, dict):
        try:
            magnitude = float(scale.get("stat_scale"))
        except (TypeError, ValueError):
            magnitude = 0.0
        if magnitude >= _SCALE_MIN:
            return True
    return False


def property_marks(name: str, scale=None) -> tuple[set[str], set[str]]:
    key = _compact_name(name)
    generic = any(token in key for token in ("cooldown", "duration", "radius", "range"))
    threat = any(
        token in key
        for token in ("slow", "stun", "silence", "root", "immobil", "disarm", "heal", "resist", "weapon", "spirit", "techpower")
    )
    if generic and not threat:
        return set(), set()
    if ("movespeed" in key or "sprintspeed" in key) and "slow" not in key:
        return set(), set()
    if _mentions_heal(key) and ("penalty" in key or "disablehealing" in key):
        return set(), {"antiheal"}
    if ("resist" in key or "armor" in key) and ("reduction" in key or "shred" in key):
        return set(), set()
    if any(token in key for token in ("slow", "stun", "silence", "root", "immobil", "disarm")) and "resist" not in key:
        return {"control"}, set()
    if "resist" in key:
        return {"defense"}, {"defense"}
    if _mentions_heal(key):
        return {"heal"}, set()
    if any(token in key for token in ("weapondamage", "weaponpower", "attackdamage", "bulletdamage", "firerate", "meleedamage")) or (
        "melee" in key and "damage" in key
    ):
        return {"weapon"}, set()
    if any(token in key for token in ("techpower", "spiritpower", "spiritdamage", "techdamage", "bonusspirit")):
        return {"spirit"}, set()
    scaled = _scale_tag(scale)
    if scaled is None:
        return set(), set()
    return {scaled}, set()


def _listed_property_names(item) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for section in item.get("tooltip_sections") or []:
        if not isinstance(section, dict):
            continue
        for attribute in section.get("section_attributes") or []:
            if not isinstance(attribute, dict):
                continue
            for field in ("elevated_properties", "properties", "important_properties"):
                for raw in attribute.get(field) or []:
                    if isinstance(raw, dict):
                        raw = raw.get("important_property") or raw.get("name")
                    if raw is None:
                        continue
                    text = str(raw)
                    if text in seen:
                        continue
                    seen.add(text)
                    names.append(text)
    return names


_EFFECT_ORDER = (
    "weapon_damage",
    "fire_rate",
    "melee",
    "spirit",
    "health",
    "heal",
    "lifesteal",
    "move_speed",
    "stamina",
    "bullet_resist",
    "spirit_resist",
    "debuff_resist",
)


def _effect_channel(name: str) -> str | None:
    key = _compact_name(name)
    if ("resist" in key or "armor" in key) and ("reduction" in key or "shred" in key):
        return None
    if "bulletresist" in key or "bulletarmor" in key:
        return "bullet_resist"
    if "techresist" in key or "spiritresist" in key:
        return "spirit_resist"
    if any(token in key for token in ("statusresist", "debuffresist", "slowresist")):
        return "debuff_resist"
    if "stamina" in key:
        return "stamina"
    if ("movespeed" in key or "sprintspeed" in key) and "slow" not in key:
        return "move_speed"
    if "lifesteal" in key:
        return "lifesteal"
    if _mentions_heal(key) and "penalty" not in key and "disablehealing" not in key:
        return "heal"
    if "health" in key and "threshold" not in key and "regen" not in key:
        return "health"
    if "firerate" in key and "slow" not in key:
        return "fire_rate"
    if "meleedamage" in key or ("melee" in key and "damage" in key):
        return "melee"
    if any(token in key for token in ("weapondamage", "weaponpower", "attackdamage", "bulletdamage")):
        return "weapon_damage"
    if any(token in key for token in ("techpower", "spiritpower", "bonusspirit")):
        return "spirit"
    return None


def _signed_number(prop) -> tuple[float, str] | None:
    if not isinstance(prop, dict) or prop.get("value") is None:
        return None
    text = str(prop.get("value")).strip().lower().replace(" ", "")
    postfix = str(prop.get("postfix") or "").strip().lower()
    prefix = str(prop.get("prefix") or "").strip()
    if text.endswith("m/s"):
        text = text[:-3]
    elif text.endswith("m"):
        text = text[:-1]
    elif text.endswith("%"):
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    if prefix == "-" and number > 0:
        number = -number
    return number, postfix


def _effect_number(name: str, prop) -> float | None:
    channel = _effect_channel(name)
    parsed = _signed_number(prop)
    if channel is None or parsed is None:
        return None
    number, postfix = parsed
    if number == 0:
        return None
    percent = "percent" in _compact_name(name) or "%" in postfix or "%" in str(prop.get("value"))
    if channel == "move_speed":
        return number
    if channel == "spirit" and not percent:
        return number / 30.0
    if channel == "health" and not percent:
        return number / 100.0
    if channel in {"heal", "lifesteal"} and not percent:
        return None
    return number / 100.0


def item_effects(item) -> dict[str, float]:
    props = item.get("properties") or {}
    totals: dict[str, float] = {}
    for name in _listed_property_names(item):
        prop = props.get(name) if isinstance(props, dict) else None
        number = _effect_number(name, prop)
        channel = _effect_channel(name)
        if number is None or channel is None:
            continue
        totals[channel] = totals.get(channel, 0.0) + number
    return {key: round(value, 4) for key, value in totals.items() if round(value, 4)}


def format_effects(effects: dict[str, float]) -> str:
    parts = []
    for name in _EFFECT_ORDER:
        value = effects.get(name)
        if not value:
            continue
        parts.append(f"{name}:{value:.4f}".rstrip("0").rstrip("."))
    return ",".join(parts)


def item_tag_sets(item) -> tuple[set[str], set[str]]:
    tags: set[str] = set()
    answers: set[str] = set()
    props = item.get("properties") or {}
    for name in _listed_property_names(item):
        scale = None
        if isinstance(props, dict):
            prop = props.get(name)
            if isinstance(prop, dict):
                scale = prop.get("scale_function")
        found, found_answers = property_marks(name, scale)
        tags.update(found)
        answers.update(found_answers)
    return tags, answers


def hero_threat_tags(heroes, items) -> dict[int, set[str]]:
    by_class = {}
    for item in items:
        class_name = item.get("class_name")
        if class_name and class_name not in by_class:
            by_class[str(class_name)] = item
    result: dict[int, set[str]] = {}
    for hero in heroes:
        tags: set[str] = set()
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
            for name, prop in props.items():
                if not _property_is_live(prop):
                    continue
                scale = prop.get("scale_function") if isinstance(prop, dict) else None
                found, _answers = property_marks(str(name), scale)
                tags.update(tag for tag in found if tag in _HERO_THREATS)
        result[int(hero["id"])] = tags
    return result


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
    rows: dict[int, tuple] = {}
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
        tags, answers = item_tag_sets(item)
        rows[item_id] = (
            _clean_name(item.get("name"), item_id),
            category,
            cost_of(item),
            image_of(item),
            owner,
            format_stats(header_buckets(item)),
            format_tags(tags),
            format_answers(answers),
            format_effects(item_effects(item)),
        )
    for item_id in allowed:
        if item_id not in rows:
            rows[item_id] = (str(item_id), "other", None, None, None, "", "", "", "")
    return [
        {
            "id": item_id,
            "name": name,
            "category": category,
            "cost": cost,
            "image": image,
            "owner": owner,
            "stats": stats,
            "tags": tags,
            "answers": answers,
            "effects": effects,
        }
        for item_id, (name, category, cost, image, owner, stats, tags, answers, effects) in sorted(rows.items())
    ]