from dynamor.services.refresh import build_match_table
from dynamor.settings import settings


def main() -> None:
    frame = build_match_table(settings)
    matches = frame.get_column("match_id").n_unique()
    print(f"{frame.height} rows, {matches} matches")
    print(settings.interim_path)


if __name__ == "__main__":
    main()
