from dynamor.services.refresh import build_feature_matrices
from dynamor.settings import settings


def main() -> None:
    summary = build_feature_matrices(settings)
    print(f"n_rows={summary['n_rows']} N={summary['heroes']} M={summary['items']}")
    print(f"draft ones={summary['draft_ones']} target ones={summary['target_ones']}")
    print(settings.rows_path)
    print(settings.drafts_path)
    print(settings.targets_path)
    print(settings.item_times_path)


if __name__ == "__main__":
    main()
