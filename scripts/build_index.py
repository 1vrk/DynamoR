from scipy import sparse

from dynamor.settings import settings


def main() -> None:
    if not settings.drafts_path.exists():
        raise SystemExit(f"draft matrix is missing: {settings.drafts_path}")
    drafts = sparse.load_npz(settings.drafts_path)
    hero_count = drafts.shape[1] // 2
    enemies = drafts[:, hero_count:]
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(settings.index_path, enemies)
    print(enemies.shape)
    print(settings.index_path)


if __name__ == "__main__":
    main()
