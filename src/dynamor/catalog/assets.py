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
    rows: dict[int, tuple[str, str, int | None, str | None, int | None]] = {}
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
        )
    for item_id in allowed:
        if item_id not in rows:
            rows[item_id] = (str(item_id), "other", None, None, None)
    return [
        {
            "id": item_id,
            "name": name,
            "category": category,
            "cost": cost,
            "image": image,
            "owner": owner,
        }
        for item_id, (name, category, cost, image, owner) in sorted(rows.items())
    ]