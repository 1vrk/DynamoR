import polars as pl
import requests

from dynamor.catalog.assets import hero_rows, item_rows, signature_class_names
from dynamor.features.catalog import read_ids
from dynamor.settings import settings

HEROES_URL = "https://api.deadlock-api.com/v1/assets/heroes?only_active=true"
ITEMS_URL = "https://api.deadlock-api.com/v1/assets/items"


def fetch_json(url: str) -> list:
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise SystemExit(f"assets request failed: {response.status_code} {url}")
    return response.json()


def write_table(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows)
    casts = [pl.col("id").cast(pl.UInt32)]
    if "cost" in frame.columns:
        casts.append(pl.col("cost").cast(pl.UInt32, strict=False))
    if "owner" in frame.columns:
        casts.append(pl.col("owner").cast(pl.UInt32, strict=False))
    frame.with_columns(casts).write_parquet(path)


def main() -> None:
    if not settings.hero_catalog_path.exists():
        raise SystemExit(f"hero catalog is missing: {settings.hero_catalog_path}")
    if not settings.item_catalog_path.exists():
        raise SystemExit(f"item catalog is missing: {settings.item_catalog_path}")

    hero_ids = read_ids(settings.hero_catalog_path)
    item_ids = read_ids(settings.item_catalog_path)
    heroes_payload = fetch_json(HEROES_URL)
    items_payload = fetch_json(ITEMS_URL)
    heroes = hero_rows(heroes_payload, hero_ids)
    items = item_rows(items_payload, item_ids, signature_class_names(heroes_payload))

    write_table(heroes, settings.hero_names_path)
    write_table(items, settings.item_names_path)

    slots = sorted(
        {
            str(item.get("item_slot_type"))
            for item in items_payload
            if item.get("type") == "upgrade"
        }
    )
    counts: dict[str, int] = {}
    for row in items:
        counts[row["category"]] = counts.get(row["category"], 0) + 1

    print(f"heroes={len(heroes)}")
    print(f"items={len(items)}")
    for category in ("weapon", "vitality", "spirit", "skill", "other"):
        print(f"{category}={counts.get(category, 0)}")
    print("slots", ", ".join(slots))
    print(settings.hero_names_path)
    print(settings.item_names_path)


if __name__ == "__main__":
    main()