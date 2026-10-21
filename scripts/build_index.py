from dynamor.services.refresh import build_cosine_index
from dynamor.settings import settings


def main() -> None:
    enemies = build_cosine_index(settings)
    print(enemies.shape)
    print(settings.index_path)


if __name__ == "__main__":
    main()
