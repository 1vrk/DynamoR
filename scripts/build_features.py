import polars as pl
from scipy import sparse

from dynamor.features.catalog import hero_ids, item_ids, write_ids
from dynamor.features.drafts import draft_matrix
from dynamor.features.targets import item_time_matrix, target_matrix
from dynamor.settings import settings


def main() -> None:
    if not settings.interim_path.exists():
        raise SystemExit(f"interim table is missing: {settings.interim_path}")
    frame = pl.read_parquet(settings.interim_path)
    heroes = hero_ids(frame)
    items = item_ids(frame)
    write_ids(heroes, settings.hero_catalog_path)
    write_ids(items, settings.item_catalog_path)
    drafts = draft_matrix(frame, heroes)
    targets = target_matrix(frame, items)
    item_times = item_time_matrix(frame, items)
    settings.rows_path.parent.mkdir(parents=True, exist_ok=True)
    frame.select("match_id", "hero_id_target", "average_badge", "start_time").write_parquet(
        settings.rows_path
    )
    sparse.save_npz(settings.drafts_path, drafts)
    sparse.save_npz(settings.targets_path, targets)
    sparse.save_npz(settings.item_times_path, item_times)
    print(f"n_rows={frame.height} N={len(heroes)} M={len(items)}")
    print(f"draft ones={drafts.nnz} target ones={targets.nnz}")
    print(settings.rows_path)
    print(settings.drafts_path)
    print(settings.targets_path)
    print(settings.item_times_path)


if __name__ == "__main__":
    main()
