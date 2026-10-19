from dynamor.data.extract import build_rows, scan_raw, write_interim
from dynamor.data.filter import filter_matches
from dynamor.settings import settings


def main() -> None:
    paths = sorted(settings.raw_dir.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet files in {settings.raw_dir}")
    frame = build_rows(filter_matches(scan_raw(paths), settings), settings)
    write_interim(frame, settings.interim_path)
    matches = frame.get_column("match_id").n_unique()
    print(f"{frame.height} rows, {matches} matches")
    print(settings.interim_path)


if __name__ == "__main__":
    main()