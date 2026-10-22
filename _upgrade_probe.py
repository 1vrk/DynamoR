import requests
heroes = requests.get("https://api.deadlock-api.com/v1/assets/heroes?only_active=true", timeout=60, headers={"User-Agent": "DynamoR"}).json()
items = requests.get("https://api.deadlock-api.com/v1/assets/items", timeout=90, headers={"User-Agent": "DynamoR"}).json()
by_class = {item.get("class_name"): item for item in items}
hero = next(row for row in heroes if row["name"] == "Rat King")
slots = hero["items"]
for key in ("signature1", "signature2", "signature3", "signature4"):
    ability = by_class[slots[key]]
    print("==", key, ability.get("name"))
    for prop_key, prop in (ability.get("properties") or {}).items():
        if not isinstance(prop, dict):
            continue
        scale = prop.get("scale_function")
        if not isinstance(scale, dict):
            continue
        class_name = str(scale.get("class_name") or "")
        stat = str(scale.get("specific_stat_scale_type") or "")
        if "tech_damage" in class_name or stat == "ETechPower" or "weapon" in class_name or stat in {"EBaseWeaponDamageIncrease", "EWeaponPower", "EBulletDamage", "ELightMeleeDamage", "EHeavyMeleeDamage"} or "Health" in stat or stat == "EHealingOutput":
            print(" ", prop_key, class_name, stat, scale.get("stat_scale"), prop.get("label"))
