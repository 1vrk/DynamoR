from dynamor.services.refresh import build_name_catalog
from dynamor.settings import settings


def main() -> None:
    summary = build_name_catalog(settings)
    print(f"heroes={summary['heroes']}")
    print(f"items={summary['items']}")
    for category in ("weapon", "vitality", "spirit", "skill", "other"):
        print(f"{category}={summary['counts'].get(category, 0)}")
    print("slots", ", ".join(summary["slots"]))
    print(settings.hero_names_path)
    print(settings.item_names_path)


if __name__ == "__main__":
    main()
