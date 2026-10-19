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
    
def item_rows(payload, allowed_ids) -> list[dict]:
    allowed = {int(item_id) for item_id in allowed_ids}
    rows: dict[int, tuple[str, str]] = {}
    for item in payload:
        item_id = int(item["id"])
        if item_id not in allowed or item_id in rows:
            continue
        category = "other"
        if item.get("type") == "upgrade":
            category = category_of(item.get("item_slot_type"))
        rows[item_id] = (_clean_name(item.get("name"), item_id), category)
    for item_id in allowed:
        if item_id not in rows:
            rows[item_id] = (str(item_id), "other")
    return [
        {"id": item_id, "name": name, "category": category}
        for item_id, (name, category) in sorted(rows.items())
    ]